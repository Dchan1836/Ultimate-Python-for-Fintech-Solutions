"""Stationarity checks and an ARIMA forecast of the closing price."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

from backend.market import download_yahooquery
from backend.plotting import BLUE, INK, RED, TEAL, new_figure, style_axis
from backend.results import DemoResult


def _adf(series: pd.Series):
    try:
        return adfuller(series, autolag="AIC", result_object=False)
    except TypeError:
        return adfuller(series, autolag="AIC")


def run(params: dict) -> dict:
    result = DemoResult()
    symbol = params["symbol"]
    history = download_yahooquery(symbol, params["start"], params["end"])
    close_column = "Close" if "Close" in history.columns else "close"
    if close_column not in history.columns:
        raise ValueError(f"No close prices for {symbol}.")
    close = history[close_column].dropna().astype(float)
    close = close[close > 0]
    if len(close) < 40:
        raise ValueError("Need at least 40 closing prices. Widen the date range.")
    price_sample = pd.DataFrame({"Close": close.round(2)})
    if params["use_log"]:
        price_sample["Log close"] = np.log(close).round(4)
    result.add_sample(
        price_sample,
        "The most recent closes in the selected range. The ARIMA model is fit on the earlier part of this series.",
    )

    window = int(params["rolling_window"])
    if window >= len(close) / 2:
        raise ValueError("The rolling window is too long for this series.")

    density, density_axis = new_figure(10, 4.2)
    close.plot(kind="kde", ax=density_axis, color=BLUE, linewidth=1.6)
    style_axis(density_axis, f"{symbol} close distribution", "Close", "Density")
    result.plot("Close distribution", density)

    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    stationarity, stationarity_axis = new_figure(12, 4.6)
    stationarity_axis.plot(close.index, close, color=BLUE, linewidth=1.2, label="Close")
    stationarity_axis.plot(rolling_mean.index, rolling_mean, color=RED, linewidth=1.4, label=f"{window}-day mean")
    stationarity_axis.plot(rolling_std.index, rolling_std, color=INK, linewidth=1.2, label=f"{window}-day std")
    style_axis(stationarity_axis, "Rolling mean and standard deviation", "Date", "Price")
    stationarity_axis.legend(loc="best")
    result.plot("Stationarity", stationarity)

    adf = _adf(close)
    adf_table = pd.Series(adf[0:4], index=["Test statistic", "p-value", "Lags used", "Observations"])
    for key, value in adf[4].items():
        adf_table[f"Critical value {key}"] = value
    result.table("Augmented Dickey-Fuller test", adf_table.to_frame("Value"), limit=10)
    p_value = float(adf[1])
    if p_value > 0.05:
        result.note("The ADF p-value is above 0.05, so this close series looks non-stationary. ARIMA differences it.")
    else:
        result.note("The ADF p-value is at or below 0.05, so the test rejects a unit root in the close.")

    modeled = np.log(close) if params["use_log"] else close.copy()
    log_mean = modeled.rolling(window).mean()
    log_std = modeled.rolling(window).std()
    moving, moving_axis = new_figure(12, 4.6)
    moving_axis.plot(modeled.index, modeled, color=BLUE, linewidth=1.1, label="Log close" if params["use_log"] else "Close")
    moving_axis.plot(log_mean.index, log_mean, color=RED, linewidth=1.4, label="Rolling mean")
    moving_axis.plot(log_std.index, log_std, color=INK, linewidth=1.2, label="Rolling std")
    style_axis(moving_axis, "Series used by the ARIMA model", "Date", "Log close" if params["use_log"] else "Close")
    moving_axis.legend(loc="best")
    result.plot("Model series", moving)

    if params["decompose"]:
        period = int(params["seasonal_period"])
        if len(modeled.dropna()) > period * 2:
            from statsmodels.tsa.seasonal import seasonal_decompose

            decomposed = seasonal_decompose(modeled.dropna(), model="additive", period=period)
            decompose_figure = decomposed.plot()
            decompose_figure.set_size_inches(12, 8)
            result.plot("Seasonal decomposition", decompose_figure)
        else:
            result.note("The series is too short for that seasonal period, so decomposition was skipped.")

    skip = int(params["skip_start"])
    usable = modeled.iloc[skip:]
    split_at = int(len(usable) * float(params["train_fraction"]))
    if split_at < 30 or len(usable) - split_at < 5:
        raise ValueError("The train fraction does not leave both a training sample and a holdout.")
    train = usable.iloc[:split_at]
    test = usable.iloc[split_at:]
    # A business-day index has no fixed frequency, which blocks statsmodels forecasts.
    train_values = pd.Series(train.to_numpy(dtype=float), name="close")

    order = (int(params["p"]), int(params["d"]), int(params["q"]))
    summaries = []
    if params["run_auto"]:
        from pmdarima.arima import auto_arima

        searched = auto_arima(
            train_values,
            start_p=0,
            start_q=0,
            test="adf",
            max_p=int(params["max_p"]),
            max_q=int(params["max_q"]),
            m=1,
            d=None,
            seasonal=False,
            start_P=0,
            D=0,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        summaries.append("Automatic order search\n" + str(searched.summary()))
        diagnostics = searched.plot_diagnostics(figsize=(12, 8))
        result.plot("Automatic ARIMA diagnostics", diagnostics)
        chosen = tuple(int(value) for value in searched.order)
        result.metric(
            "Auto order",
            str(chosen),
            help={
                "what": "The (p, d, q) order the automatic search preferred.",
                "how": "Stepwise auto ARIMA tries combinations up to the maximum p and q, differences the series until the ADF test is satisfied, and keeps the order with the lowest AIC. The search is not seasonal.",
                "why": "It is a data-chosen alternative to typing p, d, and q yourself. The forecast uses it only when that option is turned on.",
            },
        )
        if params["use_auto_order"]:
            order = chosen
            result.note(f"The forecast uses the automatic order {order}.")
        else:
            result.note(f"The search preferred {chosen}. The forecast uses the order you set, {order}.")

    steps = int(params["forecast_steps"]) or len(test)
    try:
        from statsmodels.tsa.arima.model import ARIMA

        fitted = ARIMA(train_values, order=order).fit()
    except Exception as exc:
        raise ValueError(f"ARIMA{order} did not fit this series. Try another order or use the automatic order. ({exc})") from exc
    summaries.append(f"Fitted ARIMA{order}\n" + str(fitted.summary()))
    if not params["run_auto"]:
        result.plot("ARIMA diagnostics", fitted.plot_diagnostics(figsize=(12, 8)))

    forecast = fitted.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    interval = forecast.conf_int(alpha=float(params["alpha"]))
    lower = interval.iloc[:, 0]
    upper = interval.iloc[:, 1]
    def to_price(values):
        array = np.asarray(values, dtype=float)
        return np.exp(array) if params["use_log"] else array

    mean_price = to_price(mean)
    lower_price = to_price(lower)
    upper_price = to_price(upper)
    test_price = to_price(test)

    compare = min(steps, len(test))
    if steps != len(test):
        result.note(
            f"Errors use the first {compare} holdout days. Set forecast steps to 0 to cover all {len(test)} holdout days."
        )
    forecast_index = test.index[:compare]
    chart, axis = new_figure(12, 5)
    tail = train.iloc[-min(80, len(train)) :]
    tail_price = np.exp(tail.to_numpy(dtype=float)) if params["use_log"] else tail.to_numpy(dtype=float)
    axis.plot(tail.index, tail_price, color=BLUE, linewidth=1.2, label="Train")
    axis.plot(test.index, test_price, color=INK, linewidth=1.3, label="Holdout")
    axis.plot(forecast_index, mean_price[:compare], color=TEAL, linewidth=1.6, label="Forecast")
    axis.fill_between(forecast_index, lower_price[:compare], upper_price[:compare], color=TEAL, alpha=0.18, label="Confidence band")
    style_axis(axis, f"{symbol} ARIMA{order} forecast", "Date", "Close")
    axis.legend(loc="best")
    result.plot("Forecast", chart)

    actual = test_price[:compare]
    predicted = mean_price[:compare]
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mse)
    mape = float(np.mean(np.abs(predicted - actual) / np.abs(actual)))
    result.metric(
        "Observations",
        len(close),
        help={
            "what": "The number of closing prices in the selected range.",
            "how": "Count the closes after dropping missing values and any price that is not positive. A non-positive price cannot be logged.",
            "why": "The stationarity test, the split, and the forecast all start from this series. A short range cannot support the rolling window and a holdout.",
        },
    )
    result.metric(
        "Train rows",
        len(train),
        help={
            "what": "The earlier prices used to fit the ARIMA model.",
            "how": "Drop the first few points, then keep the training fraction of what remains. The chapter drops 3 points and trains on 90% of the rest.",
            "why": "The order is estimated only on these rows. The error numbers are not scored on them.",
        },
    )
    result.metric(
        "Holdout rows",
        len(test),
        help={
            "what": "The later prices reserved for the forecast comparison.",
            "how": "These are the rows after the training cut. With forecast steps set to 0, the model predicts one step for each of them.",
            "why": "MAE, RMSE, and MAPE compare the forecast with these actual closes. That is the check that the model was not scored on the data it was fit to.",
        },
    )
    result.metric(
        "Order",
        str(order),
        help={
            "what": "The (p, d, q) order actually used to forecast.",
            "how": "p is the number of lagged prices, d is how many times the series is differenced, and q is the number of lagged forecast errors. This is the order you set, unless the forecast is switched to the automatic order.",
            "why": "The chapter forecasts with (1, 1, 2). A different order changes both the forecast path and the confidence band.",
        },
    )
    result.metric(
        "Forecast steps",
        steps,
        help={
            "what": "How many closes ahead the model predicted.",
            "how": "Use the number you entered. Zero means one step for every holdout day. Errors use only the steps that overlap the holdout.",
            "why": "A longer forecast reaches further from the last training price and usually becomes less reliable. The band on the chart widens for that reason.",
        },
    )
    result.metric(
        "ADF p-value",
        p_value,
        help={
            "what": "The augmented Dickey-Fuller test's evidence against a unit root in the close.",
            "how": "The test compares a regression of the price change on the lagged price with the critical values from the Dickey-Fuller distribution. The p-value is the probability of a test statistic at least this extreme if the series has a unit root. The chapter runs it with the AIC lag rule.",
            "why": "A p-value above 0.05 means the close looks non-stationary, so the level wanders and a differenced model such as ARIMA is appropriate. A value at or below 0.05 rejects that unit root.",
        },
    )
    result.metric(
        "MAE",
        mae,
        help={
            "what": "The average absolute miss between the forecast and the holdout close, in dollars.",
            "how": "If the model was fit to the log close, the forecast is converted back with the exponential before this comparison. Then average the absolute differences over the overlapping holdout days.",
            "why": "It is the typical forecast error in price units. A miss of $2 above and a miss of $2 below both count as $2.",
        },
    )
    result.metric(
        "RMSE",
        rmse,
        help={
            "what": "The square root of the average squared miss, in dollars.",
            "how": "Square each overlapping holdout error in price units, average those squares, and take the square root. Large misses count more than they do in MAE.",
            "why": "It quotes the typical error in dollars while giving extra weight to the days the forecast was furthest off.",
        },
    )
    result.metric(
        "MAPE",
        f"{mape * 100:.2f}%",
        help={
            "what": "The average miss as a percent of the actual close.",
            "how": "For each overlapping holdout day, divide the absolute error by the absolute actual close, then average those ratios and multiply by 100.",
            "why": "A $2 miss is a different forecast quality at $50 than at $300. MAPE puts the error on a percent scale so those days can be compared.",
        },
    )
    preview = pd.DataFrame(
        {"Actual": actual, "Forecast": predicted, "Lower": lower_price[:compare], "Upper": upper_price[:compare]},
        index=forecast_index,
    )
    result.table("Forecast sample", preview.round(2), limit=12)
    result.report = "\n\n".join(summaries)
    return result.to_dict()
