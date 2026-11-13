"""Shared price-history downloads for the chapter demos."""

from datetime import date

import pandas as pd


def require_range(start: str, end: str) -> None:
    if date.fromisoformat(start) >= date.fromisoformat(end):
        raise ValueError("The start date must be before the end date.")


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(column) for column in frame.columns]
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index(level=0, drop=True)
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Price history is missing {', '.join(missing)}.")


def download_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    require_range(start, end)
    try:
        frame = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            multi_level_index=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise ValueError(f"Could not download {symbol}: {exc}") from exc
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"No prices for {symbol} between {start} and {end}.")
    frame = clean_frame(frame)
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError(f"No prices for {symbol} between {start} and {end}.")
    return frame


def download_yahooquery(symbol: str, start: str, end: str) -> pd.DataFrame:
    from yahooquery import Ticker

    require_range(start, end)
    try:
        frame = Ticker(symbol).history(start=start, end=end, interval="1d")
    except Exception as exc:
        raise ValueError(f"Could not download {symbol}: {exc}") from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"No prices for {symbol} between {start} and {end}.")
    frame = clean_frame(frame)
    rename = {column: column.capitalize() for column in frame.columns}
    rename.update({"adjclose": "Adj Close"})
    frame = frame.rename(columns=rename)
    return frame


def download_period(symbol: str, period: str, interval: str) -> pd.DataFrame:
    from yahooquery import Ticker

    try:
        frame = Ticker(symbol).history(period=period, interval=interval)
    except Exception as exc:
        raise ValueError(f"Could not download {symbol}: {exc}") from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"No {period} history for {symbol}.")
    frame = clean_frame(frame)
    if "dividends" not in frame.columns:
        frame["dividends"] = 0.0
    return frame
