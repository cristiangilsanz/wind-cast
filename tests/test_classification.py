import pandas as pd
import pytest

from windcast.classification import HIGH, LOW, classify, labels, threshold
from windcast.data import load_available


def test_high_production_is_the_top_quarter():
    energy = pd.Series([0.0, 100.0, 200.0, 300.0, 400.0])
    cut = threshold(energy)
    assert cut == 300.0
    assert labels(energy, cut).tolist() == [LOW, LOW, LOW, LOW, HIGH]


@pytest.fixture(scope="module")
def results():
    return classify(load_available())


def test_the_threshold_comes_from_training_data_only(results):
    cut, _ = results
    assert cut == pytest.approx(1092.8, abs=0.1)


def test_the_support_vector_machine_separates_the_classes(results):
    _, rows = results
    scores = {row.name: row for row in rows}
    assert scores["Majority Baseline"].balanced_accuracy == 0.5
    assert scores["Support Vector Machine"].balanced_accuracy > 0.82
    assert scores["Support Vector Machine"].roc_auc > 0.9
