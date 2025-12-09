"""
Time-series cross-validation with purging and embargo.
"""

import numpy as np
from itertools import combinations
from typing import Iterator
from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):
    """
    K-Fold cross-validation with purging and embargo.

    Based on Marcos Lopez de Prado's methodology to prevent
    information leakage in time series data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
        purge_gap: int = 0,
    ):
        """
        Initialize PurgedKFold.

        Args:
            n_splits: Number of folds
            embargo_pct: Fraction of data to embargo after test set
            purge_gap: Number of samples to purge before test set
        """
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.embargo_pct = embargo_pct
        self.purge_gap = purge_gap

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for train/test splits.

        Args:
            X: Training data
            y: Target labels (unused)
            groups: Group labels (unused)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        embargo = int(n_samples * self.embargo_pct)

        # Create fold boundaries
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        test_starts = []
        for fold_size in fold_sizes:
            test_starts.append((current, current + fold_size))
            current += fold_size

        for test_start, test_end in test_starts:
            test_idx = indices[test_start:test_end]

            # Purge: remove samples too close before test
            train_before_end = max(0, test_start - self.purge_gap)
            train_before = indices[:train_before_end]

            # Embargo: skip samples right after test
            train_after_start = min(n_samples, test_end + embargo)
            train_after = indices[train_after_start:]

            train_idx = np.concatenate([train_before, train_after])

            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    Creates multiple backtest paths from one historical series
    by combining different test folds.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.02,
        purge_gap: int = 0,
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of folds
            n_test_splits: Number of folds to use as test set
            embargo_pct: Embargo fraction
            purge_gap: Purge gap
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        self.purge_gap = purge_gap

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        embargo = int(n_samples * self.embargo_pct)

        # Create fold groups
        fold_groups = np.array_split(indices, self.n_splits)

        # Generate all combinations of test folds
        for test_fold_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([fold_groups[i] for i in test_fold_indices])

            # Get training folds
            train_fold_indices = [i for i in range(self.n_splits) if i not in test_fold_indices]

            # Apply purging and embargo
            train_idx = []
            for train_fold_idx in train_fold_indices:
                fold = fold_groups[train_fold_idx]

                # Check if this fold is adjacent to any test fold
                is_before_test = any(
                    train_fold_idx == test_idx - 1 for test_idx in test_fold_indices
                )
                is_after_test = any(
                    train_fold_idx == test_idx + 1 for test_idx in test_fold_indices
                )

                if is_before_test:
                    # Purge end of fold
                    fold = fold[:-self.purge_gap] if self.purge_gap > 0 else fold
                if is_after_test:
                    # Embargo start of fold
                    fold = fold[embargo:] if embargo > 0 else fold

                train_idx.extend(fold)

            train_idx = np.array(train_idx)
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return number of splits."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


class TimeSeriesSplit:
    """Simple expanding window time series split."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int | None = None,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        test_size = self.test_size or n_samples // (self.n_splits + 1)
        min_train_size = test_size

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size

            train_end = test_start - self.gap

            if train_end < min_train_size:
                continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits
