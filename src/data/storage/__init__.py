"""
Data storage module for Parquet and DuckDB.
"""

from src.data.storage.parquet_store import ParquetStore
from src.data.storage.duckdb_store import DuckDBStore

__all__ = ["ParquetStore", "DuckDBStore"]
