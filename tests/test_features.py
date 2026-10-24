import numpy as np
import pandas as pd

from windcast.features import WindFeatures


def frame():
    return pd.DataFrame(
        {"u10": [3.0, 0.0], "v10": [4.0, -2.0], "u100": [6.0, 1.0], "v100": [8.0, 0.0]}
    )


def test_speed_comes_from_the_wind_components():
    enriched = WindFeatures().fit_transform(frame())
    assert enriched["speed10"].tolist() == [5.0, 2.0]
    assert enriched["speed100"].tolist() == [10.0, 1.0]
    assert enriched["speed100_cubed"].tolist() == [1000.0, 1.0]


def test_directions_lie_on_the_unit_circle():
    enriched = WindFeatures().fit_transform(frame())
    radius = np.hypot(enriched["direction_sin100"], enriched["direction_cos100"])
    assert np.allclose(radius, 1.0)


def test_calm_air_does_not_divide_by_zero():
    calm = pd.DataFrame({"u10": [0.0], "v10": [0.0], "u100": [0.0], "v100": [0.0]})
    enriched = WindFeatures().fit_transform(calm)
    assert np.isfinite(enriched.to_numpy()).all()


def test_the_original_columns_are_kept():
    enriched = WindFeatures().fit_transform(frame())
    assert list(enriched.columns[:4]) == ["u10", "v10", "u100", "v100"]
    names = WindFeatures().get_feature_names_out(list(frame().columns))
    assert list(names) == list(enriched.columns)
