"""Next-day direction models: logistic regression, SVC, and XGBoost."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from backend.market import download_ohlcv, require_columns
from backend.plotting import BLUE, GOLD, TEAL, new_figure, style_axis
from backend.results import DemoResult

FEATURE_COLUMNS = ["open-close", "low-high", "is_quarter_end"]


def run(params: dict) -> dict:
    result = DemoResult()
    symbol = params["symbol"]
    prices = download_ohlcv(symbol, params["start"], params["end"])
    require_columns(prices, ["Open", "High", "Low", "Close"])
    prices = prices.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(prices) < 80:
        raise ValueError("Need at least 80 trading days. Widen the date range.")

    close_chart, close_axis = new_figure(12, 4.2)
    close_axis.plot(prices.index, prices["Close"], color=TEAL, linewidth=1.4)
    style_axis(close_axis, f"{symbol} closing price", "Date", "Close")
    result.plot("Closing price", close_chart)

    plot_columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in prices.columns]
    distribution, distribution_axes = new_figure(12, 7, rows=2, cols=3)
    for axis, column in zip(np.ravel(distribution_axes), plot_columns):
        sns.histplot(prices[column].dropna(), kde=True, ax=axis, color=TEAL)
        style_axis(axis, column)
    for axis in np.ravel(distribution_axes)[len(plot_columns) :]:
        axis.axis("off")
    result.plot("Feature distributions", distribution)

    boxes, box_axes = new_figure(12, 7, rows=2, cols=3)
    for axis, column in zip(np.ravel(box_axes), plot_columns):
        sns.boxplot(x=prices[column].dropna(), ax=axis, color="#b7e4d4")
        style_axis(axis, column)
    for axis in np.ravel(box_axes)[len(plot_columns) :]:
        axis.axis("off")
    result.plot("Feature box plots", boxes)

    indexed = prices.copy()
    indexed["month"] = indexed.index.month
    indexed["is_quarter_end"] = np.where(indexed["month"] % 3 == 0, 1, 0)
    indexed["open-close"] = indexed["Open"] - indexed["Close"]
    indexed["low-high"] = indexed["Low"] - indexed["High"]
    indexed["target"] = np.where(indexed["Close"].shift(-1) > indexed["Close"], 1, 0)
    indexed = indexed.dropna(subset=FEATURE_COLUMNS + ["target"])
    sample = indexed[["Close", "open-close", "low-high", "is_quarter_end", "target"]].copy()
    sample[["Close", "open-close", "low-high"]] = sample[["Close", "open-close", "low-high"]].round(2)
    result.add_sample(
        sample,
        "The most recent days. Target is 1 when the next close is higher than that day's close, and 0 otherwise.",
    )

    features = indexed[FEATURE_COLUMNS]
    target = indexed["target"].astype(int)
    if target.nunique() < 2:
        raise ValueError("Every day moved the same direction. Widen the date range.")

    scaled = StandardScaler().fit_transform(features)
    train_x, valid_x, train_y, valid_y = train_test_split(
        scaled,
        target,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
    )
    if train_y.nunique() < 2 or valid_y.nunique() < 2:
        raise ValueError("One of the splits contains a single class. Change the test size or the date range.")

    models = []
    if params["logistic"]:
        models.append(("Logistic regression", LogisticRegression(max_iter=1000)))
    if params["svc"]:
        models.append(
            (
                "Support vector classifier",
                SVC(kernel=params["svc_kernel"], probability=True),
            )
        )
    booster = None
    if params["xgboost"]:
        booster = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            random_state=int(params["random_state"]),
            objective="binary:logistic",
            eval_metric="auc",
        )
        models.append(("XGBoost", booster))
    if not models:
        raise ValueError("Choose at least one model.")

    scores = []
    for name, model in models:
        model.fit(train_x, train_y)
        train_auc = roc_auc_score(train_y, model.predict_proba(train_x)[:, 1])
        valid_auc = roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1])
        scores.append({"Model": name, "Train AUC": train_auc, "Validation AUC": valid_auc})

    score_frame = pd.DataFrame(scores)
    bars, bar_axis = new_figure(10, 4.6)
    positions = np.arange(len(score_frame))
    width = 0.36
    bar_axis.bar(positions - width / 2, score_frame["Train AUC"], width=width, color=BLUE, label="Train AUC")
    bar_axis.bar(positions + width / 2, score_frame["Validation AUC"], width=width, color=GOLD, label="Validation AUC")
    bar_axis.set_xticks(positions, score_frame["Model"])
    bar_axis.set_ylim(0, 1)
    style_axis(bar_axis, "ROC AUC by model", ylabel="AUC")
    bar_axis.legend(loc="best")
    result.plot("Model comparison", bars)

    if booster is not None:
        importance, importance_axis = new_figure(8, 4.2)
        importance_axis.barh(FEATURE_COLUMNS, booster.feature_importances_, color=TEAL)
        style_axis(importance_axis, "XGBoost feature importance", "Importance")
        result.plot("XGBoost feature importance", importance)

    result.table("ROC AUC", score_frame, limit=10)
    result.metric(
        "Trading days",
        len(indexed),
        help={
            "what": "The days that have both the three model inputs and a next-day direction.",
            "how": "Build the open-close spread, the low-high spread, and the quarter-end flag, then drop the last day because it has no following close to label.",
            "why": "This is the sample that gets split. The up-day rate and both row counts are shares of this number.",
        },
    )
    result.metric(
        "Up days",
        f"{float(target.mean()) * 100:.1f}%",
        help={
            "what": "The share of days whose next close was higher than that day's close.",
            "how": "Label a day 1 when the following close is higher and 0 otherwise. Divide the number of 1s by the number of trading days.",
            "why": "Near 50% means the two outcomes are balanced, so a model cannot look good by always guessing the common direction. Far from 50%, accuracy alone becomes misleading; AUC does not.",
        },
    )
    result.metric(
        "Train rows",
        len(train_y),
        help={
            "what": "The days used to fit every checked model.",
            "how": "Draw a random share of the trading days for training. The test size sets how much is left out, and the random state makes that draw repeatable.",
            "why": "The models learn only from these days. A score on them can look strong because the model has already seen those outcomes.",
        },
    )
    result.metric(
        "Validation rows",
        len(valid_y),
        help={
            "what": "The days held out so the models can be scored on outcomes they did not fit.",
            "how": "These are the remaining rows after the random training split. The chapter holds out 10%.",
            "why": "Validation AUC is the number that says whether a model ranks new days correctly. Train AUC does not.",
        },
    )
    for row in scores:
        result.metric(
            f"{row['Model']} validation AUC",
            row["Validation AUC"],
            help={
                "what": f"How well {row['Model']} ranks unseen up days ahead of unseen down days.",
                "how": "The model gives each validation day a probability that the next close is higher. AUC is the chance that a true up day gets a higher probability than a true down day. The scale runs from 0 to 1, and 0.5 is a coin flip.",
                "why": "This is the fair comparison between the models. Above 0.5 means better than chance on new days. A much higher training AUC in the chart means the model memorized the training days.",
            },
        )
    result.note(
        "The label is 1 when the next close is higher. Features are the open-close spread, "
        "the low-high spread, and a quarter-end flag, scaled before the split."
    )
    return result.to_dict()
