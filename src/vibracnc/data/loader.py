from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd


def discover_csv_files(root: Path, pattern: str = "*.csv") -> list[Path]:
    return sorted(root.glob(pattern))


def load_csv(path: Path, column_names: Sequence[str] | None = None) -> pd.DataFrame:
    if column_names is not None:
        return pd.read_csv(path, names=column_names, header=None)
    return pd.read_csv(path)


def load_condition(
    dataset_root: Path,
    condition: str,
    column_names: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[pd.DataFrame]:
    condition_dir = dataset_root / condition
    if not condition_dir.exists():
        raise FileNotFoundError(f"{condition_dir} 경로가 존재하지 않습니다.")

    csv_files = discover_csv_files(condition_dir)
    if limit is not None:
        csv_files = csv_files[:limit]
    return [load_csv(path, column_names=column_names) for path in csv_files]


def load_multiple_conditions(
    dataset_root: Path,
    conditions: Iterable[str],
    column_names: Sequence[str] | None = None,
    limit: int | None = None,
) -> dict[str, list[pd.DataFrame]]:
    return {
        condition: load_condition(dataset_root, condition, column_names=column_names, limit=limit)
        for condition in conditions
    }

