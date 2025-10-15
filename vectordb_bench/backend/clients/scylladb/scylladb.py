import logging
import time
import environs
from contextlib import contextmanager

import cassandra
from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, BatchType
from cassandra.auth import PlainTextAuthProvider

from ..api import VectorDB
from .config import ScyllaDBIndexConfig

log = logging.getLogger(__name__)
env = environs.Env()
env.read_env(path=".env", recurse=False)


class ScyllaDBError(Exception):
    """Custom exception class for ScyllaDB client errors."""


class ScyllaDB(VectorDB):
    """ScyllaDB client for vector database operations. (__init__ is called once per case, init is called in each process)"""
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: ScyllaDBIndexConfig,
        collection_name: str = "vdb_bench_collection",
        id_field: str = "id",
        vector_field: str = "vector",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.index_config = db_case_config
        self.table_name = collection_name
        self.id_field = id_field
        self.vector_field = vector_field
        self.drop_old_table = drop_old
        self.index_params = self.index_config.index_param()
        self.username = env("SCYLLADB_USERNAME", default=None)
        self.password = env("SCYLLADB_PASSWORD", default=None)
        self.auth_provider = None
        if self.username and self.password:
            self.auth_provider = PlainTextAuthProvider(self.username, self.password)
        elif self.username or self.password:
            log.warning("Only one of username or password is set. Authentication may fail.")

        log.info(f"Using {cassandra.__version__} version of Cassandra driver")
        log.info(f"index params: {self.index_params}")
        uri = self.db_config["cluster_uris"]
        keyspace = self.db_config["keyspace"]
        self.cluster = Cluster(uri, auth_provider=self.auth_provider)
        log.info(f"Connecting to ScyllaDB cluster at {uri}")
        self.session = self.cluster.connect()
        log.info(f"Shard awareness status: {self.cluster.is_shard_aware()}")

        log.info(f"Creating keyspace: {keyspace}")
        class_name = "SimpleStrategy" if self.db_config.get("replication_factor") == 1 else "NetworkTopologyStrategy"
        self.session.execute(f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
                             f"WITH replication = {{'class': '{class_name}', 'replication_factor': '{self.db_config["replication_factor"]}'}} "
                             f"AND tablets = {{'enabled': 'false'}}")
        self.session.set_keyspace(keyspace)
        
        if self.drop_old_table:
            log.info(f"Dropping old table: {self.table_name}")
            self.session.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self._create_table()
            self._create_index()

        self.cluster = None
        self.session = None

    @contextmanager
    def init(self):
        """Initialize ScyllaDB client and cleanup when done. Called for each stage of the test by the framework."""
        try:
            uri = self.db_config["cluster_uris"]
            keyspace = self.db_config["keyspace"]
            self.cluster = Cluster(uri, auth_provider=self.auth_provider)
            self.session = self.cluster.connect(keyspace)
            self.prepared_insert = self.session.prepare(f"INSERT INTO {self.table_name} ({self.id_field}, {self.vector_field}) VALUES (?, ?)")
            self.prepared_lookup = self.session.prepare(f"SELECT {self.id_field} FROM {self.table_name} ORDER BY {self.vector_field} ANN OF ? LIMIT ?")
            yield
        finally:
            if self.cluster is not None:
                self.cluster.shutdown()
                self.cluster = None
                self.session = None

    def _create_table(self):
        """Create table for vector storage"""
        create_table_cql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {self.id_field} int PRIMARY KEY,
            {self.vector_field} vector<float, {self.dim}>
        ) WITH CDC = {{'enabled' : true}}
        """
        self.session.execute(create_table_cql)
        log.info(f"Created table {self.table_name} if didn't exist")

    def need_normalize_cosine(self) -> bool:
        return True

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception | None):
        """Insert embeddings into ScyllaDB"""
        try:
            batch = BatchStatement(consistency_level=ConsistencyLevel.ONE, batch_type=BatchType.UNLOGGED)
            for key, embedding in zip(metadata, embeddings, strict=False):
                batch.add(self.prepared_insert, (key, embedding))
            self.session.execute(batch)
        except Exception as e:
            return 0, e
        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        """Search for similar vectors"""
        rows = self.session.execute(self.prepared_lookup, (query, k))
        if not rows:
            return []
        return [row[0] for row in rows]

    def _create_index(self):
        create_index_cql = f"""
        CREATE CUSTOM INDEX IF NOT EXISTS ON {self.table_name} ({self.vector_field})
        USING 'vector_index' WITH OPTIONS = {self.index_config.index_param()}
        """
        self.session.execute(create_index_cql)

    def _wait_for_build(self):
        """Wait for index build to complete"""
        log.info("Waiting for index build to complete...")
        while True:
            try:
                sample_vector = [0.0] * self.dim
                query = f"SELECT * FROM {self.table_name} ORDER BY {self.vector_field} ANN OF %s LIMIT 1"
                self.session.execute(query, (sample_vector,))
                log.info("Index build completed successfully.")
                break
            except Exception as e:
                log.error(f"Error checking index status: {e}")
            time.sleep(1)

    def optimize(self, data_size: int | None = None) -> None:
        self._wait_for_build()

    def ready_to_load(self) -> None:
        """ScyllaDB is always ready to load"""
