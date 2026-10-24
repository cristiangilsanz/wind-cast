from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HEIGHTS = ("10", "100")
CALM = 1e-9
SHEAR_OFFSET = 0.1


class WindFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y=None) -> WindFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        enriched = X.copy()
        for height in HEIGHTS:
            u = enriched[f"u{height}"]
            v = enriched[f"v{height}"]
            speed = np.hypot(u, v)
            enriched[f"speed{height}"] = speed
            enriched[f"direction_sin{height}"] = v / (speed + CALM)
            enriched[f"direction_cos{height}"] = u / (speed + CALM)
        enriched["speed100_cubed"] = enriched["speed100"] ** 3
        enriched["shear"] = enriched["speed100"] / (enriched["speed10"] + SHEAR_OFFSET)
        return enriched

    def get_feature_names_out(self, input_features=None):
        derived = [
            f"{name}{height}"
            for height in HEIGHTS
            for name in ("speed", "direction_sin", "direction_cos")
        ]
        return np.asarray(
            [*(input_features or []), *derived, "speed100_cubed", "shear"], dtype=object
        )
