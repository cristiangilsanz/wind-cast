import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from windcast.data import features_and_target, load_available, temporal_split
from windcast.evaluation import Result, compare, rmse, select
from windcast.models import BEST, CANDIDATES


@pytest.fixture(scope="module")
def results():
    return {result.name: result for result in compare(load_available())}


def test_the_original_model_still_scores_as_in_the_report():
    train, test = temporal_split(load_available())
    X_train, y_train = features_and_target(train)
    X_test, y_test = features_and_target(test)
    original = make_pipeline(StandardScaler(), SVR(C=1800, gamma=0.02, epsilon=0.0))
    original.fit(X_train, y_train)
    assert rmse(y_test, original.predict(X_test)) == pytest.approx(386.7, abs=1.0)


def test_the_blend_is_chosen_by_cross_validation(results):
    assert select(list(results.values())).name == BEST


def test_the_blend_beats_the_original_on_the_held_out_period(results):
    assert results[BEST].outer_rmse == pytest.approx(379.4, abs=1.0)
    assert results[BEST].outer_rmse < 386.5
    assert results[BEST].outer_r2 > 0.67


def test_every_model_beats_the_mean_baseline(results):
    baseline = results["Mean Baseline"].outer_rmse
    assert baseline == pytest.approx(666.8, abs=1.0)
    others = [result for name, result in results.items() if name != "Mean Baseline"]
    assert all(result.outer_rmse < baseline for result in others)


def test_tuning_improves_the_support_vector_machine(results):
    svr = results["Support Vector Regression"]
    assert svr.tuned_inner < svr.default_inner - 100


def test_selection_uses_cross_validation_not_the_test_period():
    honest = Result("honest", 1.0, 300.0, 500.0, 0.1, 0.0, {})
    lucky = Result("lucky", 1.0, 310.0, 100.0, 0.9, 0.0, {})
    assert select([honest, lucky]).name == "honest"


def test_results_cover_every_candidate(results):
    assert list(results) == [item.name for item in CANDIDATES]
