import pandas as pd

from windcast.cli import main


def test_train_then_predict(tmp_path, capsys):
    model = tmp_path / "model.joblib"
    out = tmp_path / "predictions.csv"
    assert main(["train", "--model", "Ridge", "--out", str(model)]) == 0
    assert main(["predict", "--model-file", str(model), "--out", str(out)]) == 0
    predictions = pd.read_csv(out)
    assert list(predictions.columns) == ["datetime", "energy"]
    assert len(predictions) == 1189
    assert "Wrote 1189 predictions from Ridge" in capsys.readouterr().out


def test_compare_prints_a_table(capsys):
    assert main(["compare", "--models", "Mean Baseline", "Ridge"]) == 0
    out = capsys.readouterr().out
    assert "Test RMSE" in out
    assert "Best by cross-validation: Ridge" in out


def test_classify_reports_every_classifier(capsys):
    assert main(["classify"]) == 0
    out = capsys.readouterr().out
    assert "Support Vector Machine" in out
    assert "top quarter" in out


def test_plot_writes_an_image(tmp_path):
    chart = tmp_path / "forecast.png"
    assert main(["plot", "--model", "Ridge", "--out", str(chart)]) == 0
    assert chart.stat().st_size > 10_000


def test_unknown_models_fail_cleanly(capsys):
    assert main(["train", "--model", "Magic"]) == 1
    assert "unknown model" in capsys.readouterr().err


def test_missing_model_files_fail_cleanly(tmp_path, capsys):
    assert main(["predict", "--model-file", str(tmp_path / "none.joblib")]) == 1
    assert "wind-cast:" in capsys.readouterr().err
