from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import kagglehub  # type: ignore
except ImportError as exc:  # pragma: no cover
    kagglehub = None  # type: ignore
    _IMPORT_ERROR: Optional[ImportError] = exc
else:
    _IMPORT_ERROR = None


DATASET_ID = "rabahba/phm-data-challenge-2010"


def download_phm2010_dataset(target_dir: Optional[Path | str] = None, force: bool = False) -> Path:
    """
    Download the PHM Data Challenge 2010 dataset using kagglehub.

    Parameters
    ----------
    target_dir:
        사용자 정의 저장 경로. 지정하지 않으면 kagglehub의 캐시 경로를 그대로 반환합니다.
    force:
        True로 설정하면 target_dir가 존재하더라도 내용을 덮어쓰고 최신 캐시 버전을 복사합니다.

    Returns
    -------
    Path
        데이터셋이 위치한 로컬 경로.
    """
    if kagglehub is None:
        raise RuntimeError(
            "kagglehub 패키지가 설치되어 있지 않습니다. `pip install kagglehub`으로 설치 후 다시 시도하세요."
        ) from _IMPORT_ERROR

    cache_path = Path(kagglehub.dataset_download(DATASET_ID))

    if target_dir is None:
        return cache_path

    target_path = Path(target_dir)
    if target_path.exists():
        if not force:
            return target_path
        for item in target_path.iterdir():
            if item.is_dir():
                for sub in item.glob("**/*"):
                    if sub.is_file():
                        sub.unlink()
                item.rmdir()
            else:
                item.unlink()
    else:
        target_path.mkdir(parents=True, exist_ok=True)

    for src in cache_path.rglob("*"):
        relative = src.relative_to(cache_path)
        dst = target_path / relative
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.write_bytes(src.read_bytes())

    return target_path

