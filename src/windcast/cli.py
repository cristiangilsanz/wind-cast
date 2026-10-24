from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from windcast.classification import classify
from windcast.data import (
    features_and_target,
    load_available,
    load_competition,
    read,
    temporal_split,
)
from windcast.evaluation import FOLDS, compare, evaluate, rmse, select, train_final
from windcast.models import BEST, CANDIDATES, candidate
from windcast.persistence import load, predict, save
from windcast.plotting import plot_forecast

DEFAULT_MODEL = Path("wind-cast.joblib")


def table(header: Sequence[str], rows: list[Sequence[str]]) -> str:
    lines = [header, *rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]

    def render(line: Sequence[str]) -> str:
        cells = [line[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(line[1:], widths[1:], strict=True)]
        return "  ".join(cells)

    return "\n".join(render(line) for line in lines)


def run_compare(args: argparse.Namespace) -> int:
    chosen = [candidate(name) for name in args.models] if args.models else CANDIDATES
    print(f"Comparing {len(chosen)} models with {args.folds} time-ordered folds...")
    results = compare(load_available(), chosen, folds=args.folds)
    rows = [
        (
            result.name,
            f"{result.default_inner:.1f}",
            f"{result.tuned_inner:.1f}",
            f"{result.outer_rmse:.1f}",
            f"{result.outer_r2:.3f}",
            f"{result.fit_seconds:.2f}s",
        )
        for result in results
    ]
    print(table(("Model", "Default CV", "Tuned CV", "Test RMSE", "Test R2", "Fit"), rows))
    best = select(results)
    print(
        f"\nBest by cross-validation: {best.name} "
        f"(test RMSE {best.outer_rmse:.1f}, R2 {best.outer_r2:.3f})"
    )
    return 0


def run_train(args: argparse.Namespace) -> int:
    chosen = candidate(args.model)
    frame = load_available()
    print(f"Training {chosen.name} on all {len(frame)} forecasts...")
    model = train_final(chosen, frame, folds=args.folds)
    columns = list(features_and_target(frame)[0].columns)
    save(model, chosen.name, columns, args.out)
    print(f"Saved the model to {args.out}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    model, name, columns = load(args.model_file)
    frame = read(args.data) if args.data else load_competition()
    predictions = predict(model, columns, frame)
    predictions.to_csv(args.out, index_label="datetime")
    print(f"Wrote {len(predictions)} predictions from {name} to {args.out}")
    return 0


def run_classify(args: argparse.Namespace) -> int:
    cut, results = classify(load_available())
    print(f"High production means more than {cut:.1f}, the top quarter of the training period\n")
    rows = [
        (
            result.name,
            f"{result.inner_balanced_accuracy:.3f}",
            f"{result.balanced_accuracy:.3f}",
            f"{result.roc_auc:.3f}",
        )
        for result in results
    ]
    print(table(("Classifier", "CV Balanced Acc.", "Test Balanced Acc.", "Test ROC AUC"), rows))
    return 0


def run_plot(args: argparse.Namespace) -> int:
    frame = load_available()
    train, test = temporal_split(frame)
    chosen = candidate(args.model)
    result = evaluate(chosen, train, test, folds=args.folds)
    model = train_final(chosen, train, folds=args.folds)
    X_test, y_test = features_and_target(test)
    forecast = y_test.copy()
    forecast[:] = model.predict(X_test)
    figure = plot_forecast(y_test, forecast, rmse(y_test, forecast), start=args.start)
    figure.savefig(args.out, dpi=150, facecolor=figure.get_facecolor())
    print(f"Saved the forecast chart to {args.out} (test RMSE {result.outer_rmse:.1f})")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="wind-cast",
        description="Forecast the energy of the Sotavento wind farm from weather forecasts.",
    )
    commands = root.add_subparsers(dest="command", required=True, metavar="command")

    compare_command = commands.add_parser(
        "compare", help="compare every model on the held-out period"
    )
    compare_command.add_argument(
        "--models", nargs="+", metavar="NAME", help="only compare these models"
    )
    compare_command.set_defaults(run=run_compare)

    train = commands.add_parser("train", help="train the final model on all the data")
    train.add_argument("--model", default=BEST)
    train.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    train.set_defaults(run=run_train)

    predict_command = commands.add_parser(
        "predict", help="forecast energy for new weather forecasts"
    )
    predict_command.add_argument("--model-file", type=Path, default=DEFAULT_MODEL)
    predict_command.add_argument(
        "--data", type=Path, help="CSV of forecasts, the bundled competition set by default"
    )
    predict_command.add_argument("--out", type=Path, default=Path("predictions.csv"))
    predict_command.set_defaults(run=run_predict)

    classify_command = commands.add_parser(
        "classify", help="predict high or low production instead"
    )
    classify_command.set_defaults(run=run_classify)

    plot = commands.add_parser("plot", help="chart the forecast against the measured energy")
    plot.add_argument("--model", default=BEST)
    plot.add_argument("--start", type=int, help="first test forecast to draw")
    plot.add_argument("--out", type=Path, default=Path("forecast.png"))
    plot.set_defaults(run=run_plot)

    for command in (compare_command, train, plot):
        command.add_argument("--folds", type=int, default=FOLDS)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"wind-cast: {error}", file=sys.stderr)
        return 1


def entry() -> None:
    sys.exit(main())
