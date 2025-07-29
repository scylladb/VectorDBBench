
from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class ScyllaDBConfig(DBConfig, BaseModel):
    keyspace: str = "vdb_bench"
    cluster_uris: str = "127.0.0.1"
    def to_dict(self) -> dict:
        return {
            "keyspace": self.keyspace,
            "cluster_uris": self.cluster_uris.split(","),
        }


class ScyllaDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.HNSW  # ScyllaDB uses HNSW for vector search
    metric_type: MetricType | None  # Default metric type is L2
    m : int | None = 16  # Number of bi-directional links created for each element
    ef_construction: int | None = 128  # Size of the dynamic list for the construction
    ef_search: int | None = 128  # Size of the dynamic list for the search

    def get_similiarity_function_name(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        if self.metric_type == MetricType.IP:
            return "DOT_PRODUCT"
        return "EUCLIDEAN"  # Default to L2 similarity

    def index_param(self) -> dict:
        return {
                    "similarity_function": self.get_similiarity_function_name(),
                    "maximum_node_connections" : self.m,
                    "construction_beam_width": self.ef_construction,
                    "search_beam_width": self.ef_search,}

    def search_param(self) -> dict:
        return {}
