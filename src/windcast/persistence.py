from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline


def save(model: Pipeline, name: str, columns: list[str], path: Path) -> None:
    joblib.dump(
        {"model": model, "name": name, "columns": columns, "sklearn": sklearn.__version__}, path
    )


def load(path: Path) -> tuple[Pipeline, str, list[str]]:
    bundle = joblib.load(path)
    if bundle["sklearn"] != sklearn.__version__:
        warnings.warn(
            f"{path} was trained with scikit-learn {bundle['sklearn']}, "
            "retrain it with wind-cast train",
            stacklevel=2,
        )
    return bundle["model"], bundle["name"], bundle["columns"]


def predict(model: Pipeline, columns: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"the data is missing these forecast columns: {', '.join(missing)}")
    energy = model.predict(frame[columns]).clip(min=0)
    return pd.DataFrame({"energy": energy.round(2)}, index=frame.index)
