from __future__ import annotations

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

BACKGROUND = "#0d1117"
GRID = "#2a3146"
TEXT = "#c9d1e3"
ACTUAL = "#9aa3b8"
FORECAST = "#4fb3ff"
WINDOW = 4 * 28
STEP = pd.Timedelta(hours=6)


def longest_stretch(index: pd.DatetimeIndex) -> int:
    breaks = [0, *[i for i in range(1, len(index)) if index[i] - index[i - 1] > STEP], len(index)]
    starts = breaks[:-1]
    lengths = [end - begin for begin, end in pairwise(breaks)]
    return starts[lengths.index(max(lengths))]


def regular(series: pd.Series) -> pd.Series:
    grid = pd.date_range(series.index[0], series.index[-1], freq=STEP)
    return series.reindex(grid)


def plot_forecast(
    actual: pd.Series, forecast: pd.Series, rmse: float, start: int | None = None
) -> Figure:
    first = longest_stretch(actual.index) if start is None else start
    span = slice(first, first + WINDOW)
    actual, forecast = regular(actual.iloc[span]), regular(forecast.iloc[span])
    figure, axes = plt.subplots(figsize=(12, 5.5), facecolor=BACKGROUND)
    axes.set_facecolor(BACKGROUND)
    axes.fill_between(
        actual.index, actual.fillna(0), where=actual.notna(), color=ACTUAL, alpha=0.18, linewidth=0
    )
    axes.plot(actual.index, actual, color=ACTUAL, linewidth=2.2, label="Measured energy")
    axes.plot(forecast.index, forecast, color=FORECAST, linewidth=2.6, label="Forecast a day ahead")
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes.grid(color=GRID, linewidth=0.8)
    for spine in axes.spines.values():
        spine.set_color(GRID)
    axes.tick_params(colors=TEXT, labelsize=12)
    axes.set_ylabel("Energy", color=TEXT, fontsize=13)
    axes.legend(
        facecolor=BACKGROUND, edgecolor=GRID, labelcolor=TEXT, loc="upper left", fontsize=12
    )
    axes.set_title(
        f"Sotavento wind farm, held-out weeks of 2009  ·  RMSE {rmse:.0f}",
        color=TEXT,
        fontsize=15,
        loc="left",
        pad=12,
    )
    axes.margins(x=0.01)
    figure.tight_layout()
    return figure
