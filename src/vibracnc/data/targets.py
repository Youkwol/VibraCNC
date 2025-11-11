from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_wear_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} 파일이 존재하지 않습니다.")
    return pd.read_csv(path)

