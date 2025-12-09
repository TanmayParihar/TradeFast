"""
Parquet file storage for market data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


class ParquetStore:
    """Parquet-based data storage."""

    def __init__(self, base_path: str | Path):
        """
        Initialize Parquet store.

        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(module="parquet_store")

    def save(
        self,
        df: pl.DataFrame,
        category: str,
        symbol: str | None = None,
        partition_by: str | None = None,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> Path:
        """
        Save DataFrame to Parquet.

        Args:
            df: DataFrame to save
            category: Data category ('ohlcv', 'orderbook', 'features', etc.)
            symbol: Trading symbol (optional)
            partition_by: Column to partition by (e.g., 'date')
            compression: Compression algorithm
            compression_level: Compression level

        Returns:
            Path to saved file/directory
        """
        if df.is_empty():
            self.logger.warning("Attempting to save empty DataFrame")
            return Path()

        # Build path
        if symbol:
            path = self.base_path / category / symbol
        else:
            path = self.base_path / category

        path.mkdir(parents=True, exist_ok=True)

        if partition_by and partition_by in df.columns:
            # Partitioned write
            file_path = path
            df.write_parquet(
                file_path,
                compression=compression,
                compression_level=compression_level,
                partition_by=partition_by,
            )
            self.logger.info(f"Saved partitioned data to {file_path}")
        else:
            # Single file write
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = path / f"{timestamp}.parquet"
            df.write_parquet(
                file_path,
                compression=compression,
                compression_level=compression_level,
            )
            self.logger.info(f"Saved {len(df)} rows to {file_path}")

        return file_path

    def load(
        self,
        category: str,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Load DataFrame from Parquet.

        Args:
            category: Data category
            symbol: Trading symbol (optional)
            start: Start datetime filter
            end: End datetime filter
            columns: Columns to load (None for all)

        Returns:
            Loaded DataFrame
        """
        if symbol:
            path = self.base_path / category / symbol
        else:
            path = self.base_path / category

        if not path.exists():
            self.logger.warning(f"Path does not exist: {path}")
            return pl.DataFrame()

        # Find all parquet files
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob("**/*.parquet"))

        if not files:
            self.logger.warning(f"No parquet files found in {path}")
            return pl.DataFrame()

        # Load and concatenate
        dfs = []
        for file in files:
            try:
                df = pl.read_parquet(file, columns=columns)
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")

        if not dfs:
            return pl.DataFrame()

        result = pl.concat(dfs)

        # Apply time filters
        if "timestamp" in result.columns:
            if start:
                result = result.filter(pl.col("timestamp") >= start)
            if end:
                result = result.filter(pl.col("timestamp") <= end)

        return result.sort("timestamp") if "timestamp" in result.columns else result

    def load_latest(
        self,
        category: str,
        symbol: str | None = None,
        n_files: int = 1,
    ) -> pl.DataFrame:
        """Load the most recent parquet file(s)."""
        if symbol:
            path = self.base_path / category / symbol
        else:
            path = self.base_path / category

        if not path.exists():
            return pl.DataFrame()

        files = sorted(path.glob("*.parquet"), key=lambda x: x.stat().st_mtime)

        if not files:
            return pl.DataFrame()

        files_to_load = files[-n_files:]
        dfs = [pl.read_parquet(f) for f in files_to_load]

        return pl.concat(dfs) if dfs else pl.DataFrame()

    def append(
        self,
        df: pl.DataFrame,
        category: str,
        symbol: str | None = None,
        dedupe_column: str = "timestamp",
    ) -> Path:
        """
        Append data to existing store, deduplicating.

        Args:
            df: New data to append
            category: Data category
            symbol: Trading symbol
            dedupe_column: Column to use for deduplication

        Returns:
            Path to updated file
        """
        existing = self.load(category, symbol)

        if existing.is_empty():
            return self.save(df, category, symbol)

        # Concatenate and dedupe
        combined = pl.concat([existing, df])
        combined = combined.unique(subset=[dedupe_column], keep="last")
        combined = combined.sort(dedupe_column)

        # Clear old files and save new
        self.clear(category, symbol)
        return self.save(combined, category, symbol)

    def clear(self, category: str, symbol: str | None = None) -> None:
        """Clear stored data for a category/symbol."""
        if symbol:
            path = self.base_path / category / symbol
        else:
            path = self.base_path / category

        if path.exists():
            for file in path.glob("*.parquet"):
                file.unlink()
            self.logger.info(f"Cleared data in {path}")

    def get_info(self, category: str, symbol: str | None = None) -> dict[str, Any]:
        """Get information about stored data."""
        if symbol:
            path = self.base_path / category / symbol
        else:
            path = self.base_path / category

        if not path.exists():
            return {"exists": False}

        files = list(path.glob("**/*.parquet"))

        info = {
            "exists": True,
            "path": str(path),
            "file_count": len(files),
            "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
        }

        if files:
            # Sample first file for schema
            sample = pl.read_parquet(files[0], n_rows=0)
            info["columns"] = sample.columns
            info["schema"] = {col: str(dtype) for col, dtype in sample.schema.items()}

            # Get row count
            total_rows = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
            info["total_rows"] = total_rows

        return info

    def list_categories(self) -> list[str]:
        """List all data categories."""
        return [
            d.name
            for d in self.base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def list_symbols(self, category: str) -> list[str]:
        """List all symbols for a category."""
        path = self.base_path / category
        if not path.exists():
            return []

        return [
            d.name
            for d in path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
