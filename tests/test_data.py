import pandas as pd
import pytest

from windcast.data import (
    TARGET,
    features_and_target,
    load_available,
    load_competition,
    read,
    temporal_split,
    to_sotavento,
)


def test_the_bundled_data_keeps_only_sotavento_forecasts():
    frame = load_available()
    assert frame.shape == (4748, 23)
    assert TARGET in frame.columns
    assert not any(column.endswith(".13") for column in frame.columns)
    assert frame.isna().sum().sum() == 0


def test_rows_are_in_time_order():
    frame = load_available()
    assert frame.index.is_monotonic_increasing
    assert isinstance(frame.index, pd.DatetimeIndex)


def test_competition_forecasts_have_the_same_columns_without_energy():
    competition = load_competition()
    features, _ = features_and_target(load_available())
    assert list(competition.columns) == list(features.columns)
    assert competition.index.min() > load_available().index.max()


def test_raw_grids_are_reduced_to_location_13():
    raw = pd.DataFrame(
        {"datetime": ["2009-01-01"], "energy": [1.0], "u100.12": [9.0], "u100.13": [3.0]}
    )
    assert list(to_sotavento(raw).columns) == ["datetime", "energy", "u100"]


def test_files_without_dates_are_rejected(tmp_path):
    path = tmp_path / "forecasts.csv"
    pd.DataFrame({"u100": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="datetime"):
        read(path)


def test_the_test_period_comes_after_training():
    train, test = temporal_split(load_available())
    assert len(test) == 921
    assert train.index.max() < test.index.min()


def test_the_split_needs_rows_on_both_sides():
    with pytest.raises(ValueError):
        temporal_split(load_available(), test_size=0)
