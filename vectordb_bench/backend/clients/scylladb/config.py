
from enum import Enum
from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

class Quantization(str, Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I8 = "i8"
    B1 = "b1"

class ScyllaDBConfig(DBConfig, BaseModel):
    keyspace: str = "vdb_bench"
    cluster_uris: str = "127.0.0.1"
    replication_factor: int = 1
    def to_dict(self) -> dict:
        return {
            "keyspace": self.keyspace,
            "cluster_uris": self.cluster_uris.split(","),
            "replication_factor": self.replication_factor,
        }


class ScyllaDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.HNSW  # ScyllaDB uses HNSW for vector search
    metric_type: MetricType | None  # Default metric type is L2
    m : int | None = 16  # Number of bi-directional links created for each element
    ef_construction: int | None = 128  # Size of the dynamic list for the construction
    ef_search: int | None = 128  # Size of the dynamic list for the search
    quantization: Quantization | None = Quantization.F32  # Quantization type
    rescoring: bool | None = False  # Whether to rescore search result with original vectors
    oversampling: float | None = 1.0  # Search for oversampling * LIMIT results to improve recall

    def get_similiarity_function_name(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "COSINE"
        if self.metric_type == MetricType.IP:
            return "DOT_PRODUCT"
        return "EUCLIDEAN"  # Default to L2 similarity

    def index_param(self) -> dict:
        params = {
                    "similarity_function": self.get_similiarity_function_name(),
                    "maximum_node_connections" : self.m,
                    "construction_beam_width": self.ef_construction,
                    "search_beam_width": self.ef_search,
                    }
        if self.quantization != Quantization.F32:
            params["quantization"] = self.quantization.value
        if self.rescoring != False:
            params["rescoring"] = self.rescoring
        if self.oversampling != 1.0:
            params["oversampling"] = self.oversampling
        return params

    def search_param(self) -> dict:
        return {}
