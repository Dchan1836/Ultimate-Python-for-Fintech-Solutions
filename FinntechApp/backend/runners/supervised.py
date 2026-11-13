"""Random forest next-day direction model and a long/flat return strategy."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from backend.market import download_ohlcv, require_columns
from backend.plotting import GOLD, TEAL, new_figure, style_axis
from backend.results import DemoResult


def run(params: dict) -> dict:
    result = DemoResult()
    symbol = params["symbol"]
    window = int(params["rolling_window"])
    prices = download_ohlcv(symbol, params["start"], params["end"])
    require_columns(prices, ["Open", "High", "Low", "Close"])
    price_column = "Adj Close" if "Adj Close" in prices.columns else "Close"
    if price_column != "Adj Close":
        result.note("Adjusted close was missing, so the model uses the raw close.")

    frame = prices.dropna(subset=["Open", "High", "Low", price_column]).copy()
    frame["Open-Close"] = (frame["Open"] - frame["Close"]) / frame["Open"]
    frame["High-Low"] = (frame["High"] - frame["Low"]) / frame["Low"]
    frame["percent_change"] = frame[price_column].pct_change()
    frame["std_5"] = frame["percent_change"].rolling(window).std()
    frame["ret_5"] = frame["percent_change"].rolling(window).mean()
    frame = frame.dropna()
    sample = frame[[price_column, "Open-Close", "High-Low", "std_5", "ret_5"]].copy()
    sample[price_column] = sample[price_column].round(2)
    result.add_sample(
        sample,
        "The most recent rows after the rolling window. The forest uses the four feature columns to guess whether the next day is up.",
    )
    if len(frame) < 40:
        raise ValueError("Not enough rows after the rolling window. Widen the date range or shorten the window.")

    features = frame[["Open-Close", "High-Low", "std_5", "ret_5"]]
    target = np.where(frame[price_column].shift(-1) > frame[price_column], 1, -1)
    split = int(len(frame) * float(params["split_ratio"]))
    if split < 20 or len(frame) - split < 8:
        raise ValueError("That split leaves too few train or test rows.")

    train_x, test_x = features.iloc[:split], features.iloc[split:]
    train_y, test_y = target[:split], target[split:]
    depth = int(params["max_depth"])
    forest = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=depth if depth > 0 else None,
        random_state=int(params["random_state"]),
    )
    forest.fit(train_x, train_y)
    predictions = forest.predict(features)
    accuracy = accuracy_score(test_y, forest.predict(test_x))
    report = classification_report(test_y, forest.predict(test_x), zero_division=0)

    frame = frame.copy()
    frame["strategy_returns"] = frame["percent_change"].shift(-1) * predictions
    test_returns = frame["strategy_returns"].iloc[split:].dropna()
    hold_returns = frame["percent_change"].shift(-1).iloc[split:].dropna()
    shared = test_returns.index.intersection(hold_returns.index)
    test_returns = test_returns.loc[shared]
    hold_returns = hold_returns.loc[shared]

    histogram, hist_axis = new_figure(10, 4.4)
    hist_axis.hist(test_returns * 100, bins=18, color=TEAL, edgecolor="white")
    style_axis(hist_axis, "Test strategy returns", "Strategy return (%)", "Days")
    result.plot("Return distribution", histogram)

    growth, growth_axis = new_figure(12, 4.6)
    growth_axis.plot(shared, (test_returns + 1).cumprod(), color=TEAL, linewidth=1.7, label="Strategy")
    growth_axis.plot(shared, (hold_returns + 1).cumprod(), color=GOLD, linewidth=1.5, label="Buy and hold")
    style_axis(growth_axis, "Growth of 1 over the test window", "Date", "Growth of 1")
    growth_axis.legend(loc="best")
    result.plot("Cumulative growth", growth)

    importance = pd.DataFrame(
        {"Feature": features.columns, "Importance": forest.feature_importances_}
    ).sort_values("Importance", ascending=False)
    result.table("Feature importance", importance, limit=10)
    preview = pd.DataFrame(
        {
            "Signal": predictions[split:],
            "Strategy return": frame["strategy_returns"].iloc[split:].to_numpy(),
        },
        index=frame.index[split:],
    ).dropna()
    result.table("Test signals", preview.round(4), limit=12)

    result.metric(
        "Price column",
        price_column,
        help={
            "what": "The price series used to decide whether the next day is up or down.",
            "how": "Use adjusted close when Yahoo includes it, because the chapter script does. Otherwise use the raw close.",
            "why": "Adjusted close removes the jump from a dividend or split, so the up-or-down label is about the investment return rather than a corporate action.",
        },
    )
    result.metric(
        "Train rows",
        split,
        help={
            "what": "The earlier days used to grow the forest.",
            "how": "Keep the first part of the series. The train fraction, 75% in the chapter, sets where that cut falls. The split is in time order, not random.",
            "why": "The forest learns the relationship between the features and the next day's direction only from these days.",
        },
    )
    result.metric(
        "Test rows",
        len(frame) - split,
        help={
            "what": "The later days used to score the forest and to run the strategy.",
            "how": "These are the rows after the time split. Accuracy, the return histogram, and both growth numbers use this stretch.",
            "why": "Judging the strategy on later days shows whether the forest's signals would have worked after it was trained, rather than on days it already saw.",
        },
    )
    result.metric(
        "Trees",
        int(params["n_estimators"]),
        help={
            "what": "How many decision trees vote in the forest.",
            "how": "Each tree is trained on a bootstrap sample of the training rows and a random subset of the features. The forest's signal is the majority vote. The chapter uses 100 trees.",
            "why": "More trees usually make the vote more stable. They also take longer to fit, and after a point extra trees stop changing the test result.",
        },
    )
    result.metric(
        "Test accuracy",
        f"{accuracy * 100:.2f}%",
        help={
            "what": "The share of test days whose next-day direction the forest got right.",
            "how": "Compare the forest's call, 1 for a higher next close and -1 otherwise, with the actual direction on each test day. Divide the matches by the number of test days.",
            "why": "It says how often the signal pointed the right way. It does not say how large the resulting gain or loss was; the growth numbers do that.",
        },
    )
    result.metric(
        "Strategy growth of 1",
        float((test_returns + 1).prod()) if len(test_returns) else None,
        help={
            "what": "What $1 would have grown to by following the forest over the test window.",
            "how": "On each test day, multiply the next day's percent change by the signal. A signal of 1 keeps the return and a signal of -1 reverses it. Add 1 to each result and multiply them together.",
            "why": "This is the chapter's strategy result in dollars of growth. Above 1 means the signaled returns compounded to a gain. Below 1 means they compounded to a loss.",
        },
    )
    result.metric(
        "Buy-and-hold growth of 1",
        float((hold_returns + 1).prod()) if len(hold_returns) else None,
        help={
            "what": "What $1 would have grown to by simply holding the stock over the same test days.",
            "how": "Take the next day's percent change with no signal, add 1, and multiply those factors across the same days used for the strategy.",
            "why": "It is the baseline on the cumulative chart. The strategy is only an improvement when its growth ends higher than this one.",
        },
    )
    result.report = report
    result.note(
        "The signal is 1 when the model expects a higher adjusted close and -1 otherwise. "
        "The strategy multiplies the next day's percent change by that signal, so -1 reverses the day's return."
    )
    return result.to_dict()
