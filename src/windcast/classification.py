from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from windcast.data import features_and_target, temporal_split
from windcast.features import WindFeatures
from windcast.models import SEED

HIGH = "high"
LOW = "low"
QUANTILE = 0.75


@dataclass(frozen=True)
class ClassResult:
    name: str
    inner_balanced_accuracy: float
    balanced_accuracy: float
    roc_auc: float


def threshold(energy: pd.Series) -> float:
    return float(energy.quantile(QUANTILE))


def labels(energy: pd.Series, cut: float) -> pd.Series:
    return pd.Series(np.where(energy > cut, HIGH, LOW), index=energy.index)


def high_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, classes.index(HIGH)]
    decision = model.decision_function(X)
    return decision if classes[1] == HIGH else -decision


def classifier(model) -> Pipeline:
    return Pipeline([("features", WindFeatures()), ("scale", StandardScaler()), ("model", model)])


CLASSIFIERS = {
    "Majority Baseline": lambda: classifier(DummyClassifier(strategy="most_frequent")),
    "K-Nearest Neighbours": lambda: classifier(KNeighborsClassifier(n_neighbors=15)),
    "Decision Tree": lambda: classifier(
        DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=SEED)
    ),
    "Logistic Regression": lambda: classifier(
        LogisticRegression(class_weight="balanced", max_iter=2000)
    ),
    "Support Vector Machine": lambda: classifier(SVC(class_weight="balanced", random_state=SEED)),
}


def classify(frame: pd.DataFrame, jobs: int = -1) -> tuple[float, list[ClassResult]]:
    train, test = temporal_split(frame)
    X_train, energy_train = features_and_target(train)
    X_test, energy_test = features_and_target(test)
    cut = threshold(energy_train)
    y_train = labels(energy_train, cut)
    y_test = labels(energy_test, cut)
    results = []
    for name, build in CLASSIFIERS.items():
        inner = cross_val_score(
            build(),
            X_train,
            y_train,
            cv=TimeSeriesSplit(5),
            scoring="balanced_accuracy",
            n_jobs=jobs,
        )
        model = build().fit(X_train, y_train)
        results.append(
            ClassResult(
                name=name,
                inner_balanced_accuracy=float(inner.mean()),
                balanced_accuracy=float(balanced_accuracy_score(y_test, model.predict(X_test))),
                roc_auc=float(roc_auc_score(y_test == HIGH, high_score(model, X_test))),
            )
        )
    return cut, results
