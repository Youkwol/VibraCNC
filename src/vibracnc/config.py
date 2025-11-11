from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetConfig:
    root_dir: Path
    normal_conditions: tuple[str, ...] = ("c1", "c4", "c6")
    wear_file: str = "wear.csv"
    timestamp_column: str = "timestamp"
    sensor_columns: tuple[str, ...] = ("vx", "vy", "vz", "sx", "sy", "sz", "temp")
    fft_columns: tuple[str, ...] = ("vx", "vy", "vz")
    sampling_rate: float = 25600.0
    window_seconds: float = 0.1
    step_seconds: float = 0.05
    column_names: tuple[str, ...] | None = None

    def resolve_column_names(self) -> tuple[str, ...]:
        if self.column_names is not None:
            return self.column_names
        return (self.timestamp_column, *self.sensor_columns)


@dataclass(slots=True)
class ProjectPaths:
    data_dir: Path = Path("data")
    models_dir: Path = Path("artifacts/models")
    figures_dir: Path = Path("artifacts/figures")


