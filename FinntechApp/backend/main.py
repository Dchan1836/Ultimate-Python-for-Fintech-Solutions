"""HTTP API for the chapter 7 and chapter 8 demos."""

import os
import re
import threading
import traceback
from datetime import date

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.catalog import DEMOS
from backend.plotting import apply_style
from backend.runners.linear_regression import run as run_linear
from backend.runners.lstm_demo import run as run_lstm
from backend.runners.neural_network import run as run_neural
from backend.runners.supervised import run as run_supervised
from backend.runners.time_series import run as run_time_series
from backend.runners.unsupervised import run as run_unsupervised
from backend.runners.xgboost_demo import run as run_xgboost

apply_style()

RUNNERS = {
    "linear-regression": run_linear,
    "lstm": run_lstm,
    "xgboost": run_xgboost,
    "supervised-learning": run_supervised,
    "neural-network": run_neural,
    "unsupervised-learning": run_unsupervised,
    "time-series": run_time_series,
}

TICKER = re.compile(r"\^?[A-Z][A-Z0-9.\-]{0,9}")
_lock = threading.Lock()

app = FastAPI(title="Finntech Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _demo(demo_id: str) -> dict:
    for demo in DEMOS:
        if demo["id"] == demo_id:
            return demo
    raise HTTPException(status_code=404, detail="That demo is not in this lab.")


def _coerce(demo: dict, raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object of parameters.")
    cleaned = {}
    for field in demo["fields"]:
        name = field["name"]
        incoming = raw.get(name, field["default"])
        if incoming is None or incoming == "":
            incoming = field["default"]
        kind = field["type"]
        label = field["label"]
        if kind in ("text", "symbols"):
            text = str(incoming).strip()
            if not text:
                raise ValueError(f"{label} is required.")
            if kind == "symbols":
                parts = [part.strip().upper() for part in text.replace(";", ",").split(",") if part.strip()]
                if len(parts) < 2:
                    raise ValueError("Enter at least two tickers.")
                if len(parts) > 20:
                    raise ValueError("Use 20 tickers or fewer.")
                invalid = [part for part in parts if not TICKER.fullmatch(part)]
                if invalid:
                    raise ValueError(f"These tickers look invalid: {', '.join(invalid)}.")
                cleaned[name] = parts
            else:
                text = text.upper() if field.get("upper") else text
                if field.get("upper") and not TICKER.fullmatch(text):
                    raise ValueError(f"{label} should be a ticker such as MSFT.")
                cleaned[name] = text
        elif kind == "number":
            try:
                number = float(incoming)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} must be a number.") from exc
            if field.get("kind") == "int":
                number = int(number)
            if field.get("min") is not None and number < field["min"]:
                raise ValueError(f"{label} must be at least {field['min']}.")
            if field.get("max") is not None and number > field["max"]:
                raise ValueError(f"{label} must be at most {field['max']}.")
            cleaned[name] = number
        elif kind == "checkbox":
            cleaned[name] = bool(incoming)
        elif kind == "select":
            allowed = [option["value"] for option in field["options"]]
            if str(incoming) not in allowed:
                raise ValueError(f"{label} must be one of: {', '.join(allowed)}.")
            cleaned[name] = str(incoming)
        elif kind == "date":
            text = str(incoming).strip()
            try:
                date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(f"{label} must be a YYYY-MM-DD date.") from exc
            cleaned[name] = text
        else:
            cleaned[name] = incoming
    return cleaned


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demos")
def demos():
    return DEMOS


@app.post("/api/run/{demo_id}")
def run_demo(demo_id: str, params: dict = Body(...)):
    demo = _demo(demo_id)
    runner = RUNNERS.get(demo_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="That demo is not in this lab.")
    try:
        cleaned = _coerce(demo, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A model is already running. Wait for it to finish.")
    try:
        return runner(cleaned)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            plt.close("all")
        finally:
            _lock.release()
