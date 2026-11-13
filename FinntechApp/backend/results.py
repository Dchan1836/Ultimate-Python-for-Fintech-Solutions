"""Collect metrics, tables, and chart images for one demo run."""

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from backend.plotting import figure_image


def format_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "—"
        magnitude = abs(number)
        if magnitude >= 1000:
            return f"{number:,.2f}"
        if magnitude >= 1:
            return f"{number:,.4f}"
        return f"{number:.6f}"
    return str(value)


def format_cell(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.isoformat()
    return format_value(value)


class DemoResult:
    def __init__(self) -> None:
        self.plots: list[dict] = []
        self.metrics: list[dict] = []
        self.tables: list[dict] = []
        self.notes: list[str] = []
        self.report: str | None = None
        self.data_sample: dict | None = None

    def plot(self, title: str, fig) -> None:
        self.plots.append({"title": title, "image": figure_image(fig)})

    def metric(self, label: str, value, help: dict | None = None) -> None:
        item = {"label": label, "value": format_value(value)}
        if help:
            item["help"] = help
        self.metrics.append(item)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def table(self, title: str, frame: pd.DataFrame, limit: int = 12) -> None:
        self.tables.append(self._frame(title, frame, limit))

    def add_sample(self, frame: pd.DataFrame, caption: str, limit: int = 6) -> None:
        packed = self._frame("Data sample", frame, limit)
        packed["caption"] = caption
        self.data_sample = packed

    def _frame(self, title: str, frame: pd.DataFrame, limit: int) -> dict:
        view = frame.copy()
        if len(view) > limit:
            view = view.tail(limit)
        if not isinstance(view.index, pd.RangeIndex):
            index_name = view.index.name or "Date"
            view.index.name = index_name
            view = view.reset_index()
        columns = [str(column) for column in view.columns]
        rows = [
            [format_cell(value) for value in record]
            for record in view.itertuples(index=False, name=None)
        ]
        return {"title": title, "columns": columns, "rows": rows}

    def to_dict(self) -> dict:
        return {
            "plots": self.plots,
            "metrics": self.metrics,
            "tables": self.tables,
            "notes": self.notes,
            "report": self.report,
            "sample": self.data_sample,
        }
