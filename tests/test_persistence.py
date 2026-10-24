import numpy as np
import pytest

from windcast.data import features_and_target, load_available, load_competition
from windcast.models import candidate
from windcast.persistence import load, predict, save


@pytest.fixture(scope="module")
def trained():
    X, y = features_and_target(load_available().iloc[:800])
    return candidate("Ridge").build().fit(X, y), list(X.columns)


def test_saved_models_predict_the_same(trained, tmp_path):
    model, columns = trained
    path = tmp_path / "model.joblib"
    save(model, "Ridge", columns, path)
    loaded, name, _ = load(path)
    competition = load_competition()[columns]
    assert name == "Ridge"
    assert np.allclose(loaded.predict(competition), model.predict(competition))


def test_old_models_ask_to_be_retrained(trained, tmp_path, monkeypatch):
    model, columns = trained
    path = tmp_path / "model.joblib"
    save(model, "Ridge", columns, path)
    monkeypatch.setattr("sklearn.__version__", "0.1")
    with pytest.warns(UserWarning, match="retrain"):
        load(path)


def test_predictions_are_never_negative(trained):
    model, columns = trained
    predictions = predict(model, columns, load_competition())
    assert (predictions["energy"] >= 0).all()
    assert len(predictions) == 1189


def test_missing_forecast_columns_are_reported(trained):
    model, columns = trained
    with pytest.raises(ValueError, match="u100"):
        predict(model, columns, load_competition().drop(columns=["u100"]))
