import numpy as np
import pytest

from windcast.data import features_and_target, load_available
from windcast.models import BEST, CANDIDATES, candidate


@pytest.mark.parametrize("item", CANDIDATES, ids=lambda item: item.name)
def test_every_model_fits_and_predicts(item):
    X, y = features_and_target(load_available().iloc[:600])
    predictions = item.build().fit(X, y).predict(X.iloc[:20])
    assert np.isfinite(predictions).all()


def test_models_are_found_by_name_in_any_case():
    assert candidate("ridge").name == "Ridge"
    assert candidate(BEST).name == BEST


def test_unknown_models_list_the_choices():
    with pytest.raises(ValueError, match="choose one of"):
        candidate("Neural Network")
