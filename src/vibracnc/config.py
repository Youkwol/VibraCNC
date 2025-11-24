from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RuleDefinition:
    name: str
    column: str
    metric: str
    operator: str
    threshold: float
    description: str | None = None


DEFAULT_RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        name="temp_high",
        column="temp",
        metric="max",
        operator="gt",
        threshold=65.0,
        description="온도가 65°C를 초과",
    ),
    RuleDefinition(
        name="vx_rms_high",
        column="vx",
        metric="rms",
        operator="gt",
        threshold=0.35,
        description="X축 진동 RMS 초과",
    ),
    RuleDefinition(
        name="vy_rms_high",
        column="vy",
        metric="rms",
        operator="gt",
        threshold=0.35,
        description="Y축 진동 RMS 초과",
    ),
    RuleDefinition(
        name="vz_rms_high",
        column="vz",
        metric="rms",
        operator="gt",
        threshold=0.35,
        description="Z축 진동 RMS 초과",
    ),
)


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
    rule_definitions: tuple[RuleDefinition, ...] = DEFAULT_RULES

    def resolve_column_names(self) -> tuple[str, ...]:
        if self.column_names is not None:
            return self.column_names
        return (self.timestamp_column, *self.sensor_columns)


@dataclass(slots=True)
class ProjectPaths:
    data_dir: Path = Path("data")
    models_dir: Path = Path("artifacts/models")
    figures_dir: Path = Path("artifacts/figures")


