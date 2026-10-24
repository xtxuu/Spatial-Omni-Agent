"""Read-only access to a spatial counts release."""
from .store import SpatialStore
from .contracts import DataError, QueryResult
from .export import load_bundle

__all__ = ["SpatialStore", "DataError", "QueryResult", "load_bundle"]
