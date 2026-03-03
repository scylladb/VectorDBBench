"""Wrapper around the ScyllaDB vector database for VectorDB benchmarks."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Final

import cassandra
from cassandra import ConsistencyLevel
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, Session
from cassandra.policies import AddressTranslator as _BaseAddressTranslator
from cassandra.query import BatchStatement, BatchType, PreparedStatement

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import ScyllaDBIndexScope

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from .config import ScyllaDBIndexConfig

__all__ = ["ScyllaDB"]

log = logging.getLogger(__name__)

_INDEX_POLL_INTERVAL_SEC: Final[float] = 1.0
_INDEX_BUILD_TIMEOUT_SEC: Final[float] = 3600.0


class _ContactPointTranslator(_BaseAddressTranslator):
    """Translate discovered node addresses back to a known contact point.

    When ScyllaDB runs inside Docker the node may broadcast its internal
    container IP (e.g. ``172.18.0.2``) which is unreachable from the host.
    This translator maps any address that is *not* one of the original
    contact points back to the first contact point, keeping the driver
    connected to reachable addresses.
    """

    def __init__(self, contact_points: list[str]) -> None:
        self._known = set(contact_points)
        self._default = contact_points[0]

    def translate(self, addr: str) -> str:
        if addr in self._known:
            return addr
        log.debug(
            "Translating discovered address %s -> %s", addr, self._default
        )
        return self._default


class ScyllaDB(VectorDB):
    """ScyllaDB client for vector database operations.

    Manages connection lifecycle, schema creation, data ingestion,
    and ANN search against a ScyllaDB cluster with vector-search support.
    """

    supported_filter_types: ClassVar[list[FilterOp]] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    # -- construction --------------------------------------------------------

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ScyllaDBIndexConfig,
        collection_name: str = "vdb_bench_collection",
        id_col_name: str = "id",
        label_col_name: str = "filtering_label",
        vector_field: str = "vector",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ) -> None:
        self.name = "ScyllaDB"
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.id_col_name = id_col_name
        self.label_col_name = label_col_name
        self.vector_field = vector_field
        self.with_scalar_labels = with_scalar_labels

        self.auth_provider: PlainTextAuthProvider | None = self._build_auth_provider()

        # Mutable state — set by init() / prepare_filter()
        self.cluster: Cluster | None = None
        self.session: Session | None = None
        self.prepared_insert: PreparedStatement | None = None
        self.prepared_lookup: PreparedStatement | None = None
        self._filter_params: tuple[object, ...] = ()

        log.info(
            "%s using %s version of %s driver.",
            self.name,
            cassandra.__version__,
            self._get_driver_provider(),
        )
        log.info("%s index params: %s", self.name, self.case_config.index_param())

        with self._connect() as session:
            if drop_old:
                log.info("%s dropping old table: %s", self.name, self.table_name)
                session.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                self._create_table(session)
                self._create_index(session)

    # -- authentication & driver detection -----------------------------------

    @staticmethod
    def _build_auth_provider(
        env_path: str = ".env",
    ) -> PlainTextAuthProvider | None:
        """Build authentication provider from environment variables.

        Reads ``SCYLLADB_USERNAME`` and ``SCYLLADB_PASSWORD`` from a local
        ``.env`` file (if present).  Returns ``None`` when neither variable is
        set, and logs a warning when only one of the two is provided.

        Args:
            env_path: Path to the ``.env`` file.  Defaults to ``".env"``.
        """
        import environs  # optional dependency — imported lazily

        env = environs.Env()
        env.read_env(path=env_path, recurse=False)
        username: str | None = env("SCYLLADB_USERNAME", default=None)
        password: str | None = env("SCYLLADB_PASSWORD", default=None)

        if username and password:
            return PlainTextAuthProvider(username, password)
        if username or password:
            log.warning(
                "Only one of SCYLLADB_USERNAME / SCYLLADB_PASSWORD is set; "
                "authentication may fail."
            )
        return None

    @staticmethod
    def _get_driver_provider() -> str:
        """Detect whether the ScyllaDB-optimized or standard Cassandra driver is in use."""
        try:
            from cassandra.tablets import Tablet  # noqa: F401
        except ImportError:
            return "Cassandra"
        else:
            return "ScyllaDB"

    # -- connection helpers ---------------------------------------------------

    def _build_cluster(self, contact_points: list[str]) -> Cluster:
        """Create a :class:`Cluster` with address translation enabled."""
        return Cluster(
            contact_points,
            auth_provider=self.auth_provider,
            address_translator=_ContactPointTranslator(contact_points),
        )

    @contextmanager
    def _connect(self, keyspace: str | None = None) -> Generator[Session, None, None]:
        """Open a short-lived cluster connection and guarantee shutdown.

        If *keyspace* is ``None`` the configured keyspace is created (if
        needed) and set on the returned session automatically.
        """
        uri = self.db_config["cluster_uris"]
        ks = keyspace or self.db_config["keyspace"]

        cluster = self._build_cluster(uri)
        log.info("%s connecting to cluster at %s", self.name, uri)
        session = cluster.connect()
        log.info("%s shard awareness: %s", self.name, cluster.is_shard_aware())

        try:
            if keyspace is None:
                self._create_keyspace(session, ks)
            session.set_keyspace(ks)
            yield session
        finally:
            cluster.shutdown()

    def _ensure_session(self) -> Session:
        """Return the active session or raise if ``init()`` was not called."""
        if self.session is None:
            msg = (
                f"{self.name}: no active session — "
                "wrap operations inside `with self.init():`"
            )
            raise RuntimeError(msg)
        return self.session

    # -- schema management ---------------------------------------------------

    @property
    def _use_local_index(self) -> bool:
        """Whether to use a local (partition-level) secondary index."""
        return (
            self.with_scalar_labels
            and self.case_config.index_scope == ScyllaDBIndexScope.LOCAL
        )

    def _create_keyspace(self, session: Session, keyspace: str) -> None:
        """Create keyspace if it does not exist."""
        log.info("%s creating keyspace: %s", self.name, keyspace)
        replication_factor = self.db_config.get("replication_factor", 1)
        # Tablets require NetworkTopologyStrategy; SimpleStrategy is not supported.
        strategy = "NetworkTopologyStrategy"
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': '{strategy}', "
            f"'replication_factor': '{replication_factor}'}} "
            f"AND tablets = {{'enabled': 'true'}}"
        )

    def _create_table(self, session: Session) -> None:
        """Create table for vector storage."""
        if self._use_local_index:
            pk = f"PRIMARY KEY ({self.label_col_name}, {self.id_col_name})"
        elif self.with_scalar_labels:
            pk = f"PRIMARY KEY ({self.id_col_name}, {self.label_col_name})"
        else:
            pk = f"PRIMARY KEY ({self.id_col_name})"
        label_col = (
            f"{self.label_col_name} text,"
            if self.with_scalar_labels
            else ""
        )
        create_table_cql = (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            f"  {self.id_col_name} int,"
            f"  {label_col}"
            f"  {self.vector_field} vector<float, {self.dim}>,"
            f"  {pk}"
            f")"
        )
        session.execute(create_table_cql)
        log.info("%s created table: %s", self.name, self.table_name)

    def _create_index(self, session: Session) -> None:
        """Create vector search index on the table."""
        if self._use_local_index:
            target = f"(({self.label_col_name}), {self.vector_field})"
        else:
            target = f"({self.vector_field})"
        create_index_cql = (
            f"CREATE CUSTOM INDEX IF NOT EXISTS ON {self.table_name} "
            f"{target} USING 'vector_index' "
            f"WITH OPTIONS = {self.case_config.index_param()}"
        )
        session.execute(create_index_cql)
        log.info("%s created index on: %s", self.name, self.table_name)

    # -- lifecycle (per-process) ---------------------------------------------

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Create and destroy connections to the database.

        Must be used as a context manager before calling any data or search
        operations::

            with db.init():
                db.insert_embeddings(...)
                db.search_embedding(...)
        """
        uri = self.db_config["cluster_uris"]
        keyspace = self.db_config["keyspace"]
        self.cluster = self._build_cluster(uri)
        self.session = self.cluster.connect(keyspace)

        self._prepare_insert_statement()

        try:
            yield
        finally:
            self._reset_session_state()

    def _reset_session_state(self) -> None:
        """Shut down the cluster and clear all per-session state."""
        if self.cluster is not None:
            self.cluster.shutdown()
        self.cluster = None
        self.session = None
        self.prepared_insert = None
        self.prepared_lookup = None
        self._filter_params = ()

    def _prepare_insert_statement(self) -> None:
        """Prepare the CQL INSERT statement for the current session."""
        session = self._ensure_session()

        if self.with_scalar_labels:
            columns = f"{self.id_col_name}, {self.vector_field}, {self.label_col_name}"
            placeholders = "?, ?, ?"
        else:
            columns = f"{self.id_col_name}, {self.vector_field}"
            placeholders = "?, ?"

        self.prepared_insert = session.prepare(
            f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        )

    # -- data operations -----------------------------------------------------

    def need_normalize_cosine(self) -> bool:
        """Whether this database needs to normalize dataset to support COSINE."""
        return True

    def insert_embeddings(
        self,
        embeddings: Sequence[list[float]],
        metadata: Sequence[int],
        labels_data: Sequence[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings into ScyllaDB.

        Args:
            embeddings: Vectors to insert.
            metadata:   Integer keys (IDs) for each vector.
            labels_data: Optional string labels (only when scalar labels are enabled).

        Returns:
            Tuple of (inserted_count, error_or_None).
        """
        session = self._ensure_session()
        assert self.prepared_insert is not None, "prepared_insert not initialized"

        try:
            batch = BatchStatement(
                consistency_level=ConsistencyLevel.ONE,
                batch_type=BatchType.UNLOGGED,
            )
            if self.with_scalar_labels:
                if labels_data is None:
                    raise ValueError(
                        "labels_data is required when with_scalar_labels is True"
                    )
                for key, embedding, label in zip(
                    metadata, embeddings, labels_data, strict=True
                ):
                    batch.add(self.prepared_insert, (key, embedding, label))
            else:
                for key, embedding in zip(metadata, embeddings, strict=True):
                    batch.add(self.prepared_insert, (key, embedding))
            session.execute(batch)
        except Exception as e:
            log.warning("%s failed to insert data: %s", self.name, e)
            return 0, e
        return len(embeddings), None

    # -- search & filtering --------------------------------------------------

    def prepare_filter(self, filters: Filter) -> None:
        """Pre-prepare filter conditions to reduce redundancy during search.

        Filter values are bound via CQL prepared-statement parameters
        rather than interpolated into the query string.
        """
        session = self._ensure_session()

        if filters.type == FilterOp.NonFilter:
            where = ""
            allow_filtering = ""
            self._filter_params = ()
        elif filters.type == FilterOp.NumGE:
            where = f" WHERE {self.id_col_name} > ?"
            allow_filtering = " ALLOW FILTERING"
            self._filter_params = (filters.int_value,)
        elif filters.type == FilterOp.StrEqual:
            where = f" WHERE {self.label_col_name} = ?"
            allow_filtering = "" if self._use_local_index else " ALLOW FILTERING"
            self._filter_params = (filters.label_value,)
        else:
            msg = f"Unsupported filter for {self.name}: {filters}"
            raise ValueError(msg)

        self.prepared_lookup = session.prepare(
            f"SELECT {self.id_col_name} FROM {self.table_name}"
            f"{where} "
            f"ORDER BY {self.vector_field} ANN OF ? LIMIT ?"
            f"{allow_filtering}"
        )

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        """Get *k* most similar embeddings to *query*.

        Args:
            query: Query embedding to look up documents similar to.
            k:     Number of most similar embeddings to return.

        Returns:
            List of *k* most similar embedding IDs.

        Raises:
            RuntimeError: If ``prepare_filter`` was not called first.
        """
        session = self._ensure_session()
        if self.prepared_lookup is None:
            msg = (
                f"{self.name}: prepared_lookup is not set — "
                "call prepare_filter() before searching"
            )
            raise RuntimeError(msg)
        rows = session.execute(
            self.prepared_lookup, (*self._filter_params, query, k)
        )
        return [row[0] for row in rows] if rows else []

    # -- optimisation --------------------------------------------------------

    def _wait_for_index_build(
        self,
        timeout: float = _INDEX_BUILD_TIMEOUT_SEC,
        poll_interval: float = _INDEX_POLL_INTERVAL_SEC,
    ) -> None:
        """Block until the ANN index is queryable.

        Args:
            timeout:       Maximum seconds to wait before raising ``TimeoutError``.
            poll_interval: Seconds between successive probe queries.
        """
        session = self._ensure_session()
        log.info("%s waiting for index build to complete …", self.name)

        sample_vector = [0.0] * self.dim
        probe_cql = (
            f"SELECT * FROM {self.table_name} "
            f"ORDER BY {self.vector_field} ANN OF %s LIMIT 1"
        )

        deadline = time.monotonic() + timeout
        while True:
            try:
                session.execute(probe_cql, (sample_vector,))
            except Exception as e:
                if time.monotonic() >= deadline:
                    msg = f"{self.name}: index not ready after {timeout}s"
                    raise TimeoutError(msg) from e
                log.debug("%s index not ready yet: %s", self.name, e)
            else:
                log.info("%s index build completed.", self.name)
                return
            time.sleep(poll_interval)

    def optimize(self, data_size: int | None = None) -> None:
        """Wait for index to be fully built before search benchmarks."""
        self._wait_for_index_build()
