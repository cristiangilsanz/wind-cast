from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd

TARGET = "energy"
TIME = "datetime"
SOTAVENTO = "13"
TEST_SIZE = 921


def _bundled(name: str) -> Path:
    return Path(str(files("windcast") / "data" / name))


def to_sotavento(frame: pd.DataFrame) -> pd.DataFrame:
    located = [column for column in frame.columns if column.endswith(f".{SOTAVENTO}")]
    if not located:
        return frame
    kept = [column for column in (TIME, TARGET) if column in frame.columns] + located
    return frame[kept].rename(columns=lambda column: column.removesuffix(f".{SOTAVENTO}"))


def read(path: Path | str) -> pd.DataFrame:
    frame = to_sotavento(pd.read_csv(path))
    if TIME not in frame.columns:
        raise ValueError(f"{path} has no {TIME} column")
    frame[TIME] = pd.to_datetime(frame[TIME])
    return frame.set_index(TIME).sort_index()


def load_available() -> pd.DataFrame:
    return read(_bundled("sotavento.csv.gz"))


def load_competition() -> pd.DataFrame:
    return read(_bundled("competition.csv.gz"))


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[TARGET]), frame[TARGET]


def temporal_split(
    frame: pd.DataFrame, test_size: int = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < len(frame):
        raise ValueError("the test period must leave rows for training")
    return frame.iloc[:-test_size], frame.iloc[-test_size:]
