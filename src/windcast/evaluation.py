from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from windcast.data import features_and_target, temporal_split
from windcast.models import CANDIDATES, Candidate

FOLDS = 5
SCORING = "neg_root_mean_squared_error"


@dataclass(frozen=True)
class Result:
    name: str
    default_inner: float
    tuned_inner: float
    outer_rmse: float
    outer_r2: float
    fit_seconds: float
    params: dict


def rmse(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def inner_rmse(model: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int, jobs: int) -> float:
    scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(folds), scoring=SCORING, n_jobs=jobs)
    return float(-scores.mean())


def tune(candidate: Candidate, X: pd.DataFrame, y: pd.Series, folds: int, jobs: int):
    if not candidate.grid:
        model = candidate.build()
        return model, inner_rmse(model, X, y, folds, jobs), {}
    search = GridSearchCV(
        candidate.build(),
        candidate.grid,
        cv=TimeSeriesSplit(folds),
        scoring=SCORING,
        n_jobs=jobs,
        refit=False,
    )
    search.fit(X, y)
    model = clone(candidate.build()).set_params(**search.best_params_)
    return model, float(-search.best_score_), search.best_params_


def evaluate(
    candidate: Candidate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: int = FOLDS,
    jobs: int = -1,
) -> Result:
    X_train, y_train = features_and_target(train)
    X_test, y_test = features_and_target(test)
    default_inner = inner_rmse(candidate.build(), X_train, y_train, folds, jobs)
    model, tuned_inner, params = tune(candidate, X_train, y_train, folds, jobs)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    predicted = model.predict(X_test)
    return Result(
        name=candidate.name,
        default_inner=default_inner,
        tuned_inner=tuned_inner,
        outer_rmse=rmse(y_test, predicted),
        outer_r2=float(r2_score(y_test, predicted)),
        fit_seconds=fit_seconds,
        params=params,
    )


def compare(
    frame: pd.DataFrame,
    candidates: Sequence[Candidate] = CANDIDATES,
    folds: int = FOLDS,
    jobs: int = -1,
) -> list[Result]:
    train, test = temporal_split(frame)
    return [evaluate(candidate, train, test, folds, jobs) for candidate in candidates]


def select(results: Sequence[Result]) -> Result:
    return min(results, key=lambda result: result.tuned_inner)


def train_final(
    candidate: Candidate, frame: pd.DataFrame, folds: int = FOLDS, jobs: int = -1
) -> Pipeline:
    X, y = features_and_target(frame)
    model, _, _ = tune(candidate, X, y, folds, jobs)
    return model.fit(X, y)
