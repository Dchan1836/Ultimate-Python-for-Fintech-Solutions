"""Linear regression of the close on same-day open, high, low, and volume."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from backend.market import download_ohlcv, require_columns
from backend.plotting import BLUE, RED, TEAL, new_figure, style_axis
from backend.results import DemoResult


def run(params: dict) -> dict:
    result = DemoResult()
    symbol = params["symbol"]
    prices = download_ohlcv(symbol, params["start"], params["end"])
    require_columns(prices, ["Open", "High", "Low", "Close", "Volume"])
    prices = prices.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    sample = prices[["Open", "High", "Low", "Close", "Volume"]].copy()
    sample[["Open", "High", "Low", "Close"]] = sample[["Open", "High", "Low", "Close"]].round(2)
    result.add_sample(
        sample,
        "The most recent days in the selected range. The model predicts that day's close from its open, high, low, and volume.",
    )
    if len(prices) < 40:
        raise ValueError("Need at least 40 trading days. Widen the date range.")

    history, axis = new_figure(12, 4.2)
    axis.plot(prices.index, prices["Close"], color=RED, linewidth=1.4)
    style_axis(axis, f"{symbol} closing price", "Date", "Close")
    result.plot("Closing price", history)

    features = prices[["Open", "High", "Low", "Volume"]]
    target = prices["Close"]
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=float(params["test_size"]),
        shuffle=bool(params["shuffle"]),
        random_state=int(params["random_state"]),
    )
    if len(test_y) < 5 or len(train_y) < 10:
        raise ValueError("The test split left too few rows. Lower the test size or widen the dates.")

    model = LinearRegression()
    model.fit(train_x, train_y)
    predicted = pd.Series(model.predict(test_x), index=test_y.index, name="Predicted")
    ordered = pd.DataFrame({"Actual": test_y, "Predicted": predicted}).sort_index()

    comparison, comparison_axis = new_figure(12, 4.2)
    comparison_axis.plot(ordered.index, ordered["Actual"], color=RED, linewidth=1.5, label="Actual close")
    comparison_axis.plot(ordered.index, ordered["Predicted"], color=TEAL, linewidth=1.5, label="Predicted close")
    style_axis(comparison_axis, "Holdout close: actual and predicted", "Date", "Close")
    comparison_axis.legend(loc="best")
    result.plot("Holdout forecast", comparison)

    scatter, scatter_axis = new_figure(7.2, 6.2)
    scatter_axis.scatter(ordered["Actual"], ordered["Predicted"], color=BLUE, alpha=0.75, s=28)
    low = float(min(ordered["Actual"].min(), ordered["Predicted"].min()))
    high = float(max(ordered["Actual"].max(), ordered["Predicted"].max()))
    scatter_axis.plot([low, high], [low, high], color=GOLD_LINE, linewidth=1.2, label="Perfect fit")
    style_axis(scatter_axis, "Actual close vs predicted close", "Actual close", "Predicted close")
    scatter_axis.legend(loc="best")
    result.plot("Actual versus predicted", scatter)

    coefficients = np.ravel(model.coef_)
    coefficient_frame = pd.DataFrame({"Feature": features.columns, "Coefficient": coefficients})
    result.table("Regression coefficients", coefficient_frame, limit=10)
    result.table("Holdout prices", ordered.round(2), limit=12)

    mae = mean_absolute_error(test_y, predicted)
    mse = mean_squared_error(test_y, predicted)
    rmse = float(np.sqrt(mse))
    r2 = float(model.score(test_x, test_y))
    mean_actual = float(test_y.mean())
    mean_predicted = float(predicted.mean())
    ratio = (mean_actual / mean_predicted) * 100 if mean_predicted else float("nan")

    result.metric(
        "Rows",
        len(prices),
        help={
            "what": "The number of trading days in the selected range that have an open, high, low, close, and volume.",
            "how": "Count the downloaded daily bars after dropping any day that is missing one of those fields.",
            "why": "This is the sample the split is drawn from. A short range leaves too few days for a stable fit and a meaningful holdout.",
        },
    )
    result.metric(
        "Train rows",
        len(train_y),
        help={
            "what": "The days used to estimate the regression equation.",
            "how": "Set aside the test fraction, then count what remains. With shuffle off, these are the earlier days in the range.",
            "why": "The coefficients and intercept are learned only from these days. The error numbers below are not scored on them.",
        },
    )
    result.metric(
        "Test rows",
        len(test_y),
        help={
            "what": "The holdout days used to judge the forecast.",
            "how": "The test size is the fraction of rows held out. With shuffle off, that block is the end of the series, which is how the chapter script splits the data.",
            "why": "Scoring days the model did not fit shows whether the equation holds up on new prices, rather than repeating the training days.",
        },
    )
    result.metric(
        "R²",
        r2,
        help={
            "what": "The share of holdout closing-price variation explained by the same day's open, high, low, and volume.",
            "how": "One minus the sum of squared prediction errors, divided by the sum of squared gaps between each actual close and the average actual close. Both sums use the holdout only. A value of 1 is a perfect match, 0 is no better than predicting the average close, and a negative value is worse than that average.",
            "why": "It is the summary of fit on unseen days. It is often high here because the same day's open, high, and low already sit very close to the close.",
        },
    )
    result.metric(
        "Intercept",
        float(np.ravel(model.intercept_)[0]),
        help={
            "what": "The constant term in the fitted price equation.",
            "how": "The close is modeled as intercept + (a coefficient × open) + (a coefficient × high) + (a coefficient × low) + (a coefficient × volume). The intercept is the value of that equation when every feature is zero.",
            "why": "It shifts the whole prediction so the other coefficients can fit real prices. It is not itself a realistic price, because a trading day never has zero open, high, low, and volume.",
        },
    )
    result.metric(
        "MAE",
        mae,
        help={
            "what": "The average absolute miss on the holdout, in dollars.",
            "how": "For each holdout day, take the absolute difference between the actual close and the predicted close, then average those differences.",
            "why": "It is the typical size of the error in the same units as the price. A close that is $2 high and a close that is $2 low both count as a $2 miss.",
        },
    )
    result.metric(
        "MSE",
        mse,
        help={
            "what": "The average squared miss on the holdout.",
            "how": "Square each holdout error, actual close minus predicted close, then take the mean. Fitting the line minimizes this quantity on the training rows; the number shown is the same calculation on the holdout.",
            "why": "Squaring makes a large miss count much more than a small one. The units are dollars squared, so read it with RMSE rather than as a price.",
        },
    )
    result.metric(
        "RMSE",
        rmse,
        help={
            "what": "The square root of the mean squared error, back in dollars.",
            "how": "Take the square root of the holdout MSE. Like MAE, it is an average error in price units, but large misses pull it up more because they were squared first.",
            "why": "It quotes the typical error in dollars while still giving extra weight to the worst days.",
        },
    )
    result.metric(
        "Mean actual / mean predicted",
        f"{ratio:.2f}%",
        help={
            "what": "How the average holdout close compares with the average predicted close, as a percent.",
            "how": "Divide the mean of the actual holdout closes by the mean of the predicted holdout closes, then multiply by 100. The chapter script prints this ratio as the model's accuracy. 100% means the two averages match.",
            "why": "It only checks the average level. Daily misses can cancel, so a value near 100% can sit next to a large MAE or RMSE. Use MAE and RMSE for the size of the day-by-day error.",
        },
    )
    if params["shuffle"]:
        result.note("The holdout rows were drawn at random, so the forecast chart is sorted back into date order.")
    else:
        result.note("The holdout is the last part of the series, matching the chapter script.")
    return result.to_dict()


GOLD_LINE = "#a15c07"
