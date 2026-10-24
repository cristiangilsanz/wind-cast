from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from windcast.features import WindFeatures

SEED = 42


def pipeline(model: RegressorMixin, scaled: bool = True) -> Pipeline:
    steps = [("features", WindFeatures())]
    if scaled:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def tuned_svr() -> Pipeline:
    return pipeline(SVR(C=1800, gamma=0.01, epsilon=10))


def tuned_boosting() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20, random_state=SEED
    )


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[[], Pipeline]
    grid: dict[str, list] = field(default_factory=dict)


CANDIDATES = [
    Candidate("Mean Baseline", lambda: pipeline(DummyRegressor(strategy="mean"), scaled=False)),
    Candidate(
        "K-Nearest Neighbours",
        lambda: pipeline(KNeighborsRegressor()),
        {"model__n_neighbors": [5, 10, 20, 40], "model__weights": ["uniform", "distance"]},
    ),
    Candidate(
        "Decision Tree",
        lambda: pipeline(DecisionTreeRegressor(random_state=SEED), scaled=False),
        {"model__max_depth": [4, 6, 8, 12], "model__min_samples_leaf": [5, 20, 50]},
    ),
    Candidate("Linear Regression", lambda: pipeline(LinearRegression())),
    Candidate("Ridge", lambda: pipeline(Ridge()), {"model__alpha": [0.1, 1, 10, 100]}),
    Candidate(
        "Lasso",
        lambda: pipeline(Lasso(max_iter=20000)),
        {"model__alpha": [0.1, 1, 10, 50]},
    ),
    Candidate(
        "Elastic Net",
        lambda: pipeline(ElasticNet(max_iter=20000)),
        {"model__alpha": [0.1, 1, 10], "model__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    Candidate(
        "Support Vector Regression",
        lambda: pipeline(SVR()),
        {"model__C": [300, 1000, 1800, 3000], "model__gamma": [0.005, 0.01, 0.02]},
    ),
    Candidate(
        "Gradient Boosting",
        lambda: pipeline(HistGradientBoostingRegressor(random_state=SEED), scaled=False),
        {"model__learning_rate": [0.03, 0.06, 0.1], "model__max_leaf_nodes": [15, 31]},
    ),
    Candidate(
        "SVR + Gradient Boosting",
        lambda: Pipeline(
            [
                (
                    "model",
                    VotingRegressor(
                        [
                            ("svr", tuned_svr()),
                            ("boosting", pipeline(tuned_boosting(), scaled=False)),
                        ]
                    ),
                )
            ]
        ),
    ),
]

BEST = "SVR + Gradient Boosting"


def candidate(name: str) -> Candidate:
    for item in CANDIDATES:
        if item.name.lower() == name.lower():
            return item
    names = ", ".join(item.name for item in CANDIDATES)
    raise ValueError(f"unknown model {name!r}, choose one of: {names}")
