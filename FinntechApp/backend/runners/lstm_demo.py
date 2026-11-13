"""Stacked LSTM that forecasts the next price from a sliding window."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from backend.market import download_ohlcv, require_columns
from backend.plotting import BLUE, RED, TEAL, new_figure, style_axis
from backend.results import DemoResult


def run(params: dict) -> dict:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from keras import Sequential
    from keras.layers import Dense, Dropout, Input, LSTM

    result = DemoResult()
    symbol = params["symbol"]
    column = params["price_column"]
    lookback = int(params["lookback"])
    layers = int(params["layers"])
    units = int(params["units"])
    dropout = float(params["dropout"])
    epochs = int(params["epochs"])
    batch_size = int(params["batch_size"])
    validation_split = float(params["validation_split"])

    training = download_ohlcv(symbol, params["train_start"], params["train_end"])
    testing = download_ohlcv(symbol, params["test_start"], params["test_end"])
    require_columns(training, [column])
    require_columns(testing, [column])
    training = training.dropna(subset=[column])
    testing = testing.dropna(subset=[column])
    if len(training) <= lookback + 10:
        raise ValueError(
            f"The training window needs more than {lookback + 10} trading days for a lookback of {lookback}."
        )
    if len(testing) < 5:
        raise ValueError("The test window needs at least 5 trading days.")
    train_rows = training[[column]].tail(3).copy()
    train_rows["Window"] = "Train"
    test_rows = testing[[column]].head(3).copy()
    test_rows["Window"] = "Test"
    price_sample = pd.concat([train_rows, test_rows])
    price_sample[column] = price_sample[column].round(2)
    result.add_sample(
        price_sample,
        f"Three {column.lower()} prices at the end of training and three at the start of the test window.",
    )
    if params["test_start"] < params["train_end"]:
        result.note("The test window overlaps the training window, so some forecast days were also used to fit the network.")

    train_values = training[[column]].to_numpy(dtype=float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_values)

    windows = []
    targets = []
    for end in range(lookback, len(scaled_train)):
        windows.append(scaled_train[end - lookback : end, 0])
        targets.append(scaled_train[end, 0])
    x_train = np.array(windows).reshape(-1, lookback, 1)
    y_train = np.array(targets)

    model = Sequential()
    model.add(Input(shape=(lookback, 1)))
    for index in range(layers):
        model.add(
            LSTM(
                units=units,
                return_sequences=index < layers - 1,
                kernel_initializer="glorot_uniform",
            )
        )
        model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(optimizer=params["optimizer"], loss="mean_squared_error")

    fit_kwargs = {"batch_size": batch_size, "epochs": epochs, "verbose": 0}
    if validation_split > 0:
        fit_kwargs["validation_split"] = validation_split
    history = model.fit(x_train, y_train, **fit_kwargs)

    combined = pd.concat([training[column], testing[column]])
    combined = combined[~combined.index.duplicated(keep="last")]
    scaled_all = scaler.transform(combined.to_numpy(dtype=float).reshape(-1, 1))
    positions = {stamp: position for position, stamp in enumerate(combined.index)}
    windows = []
    actuals = []
    dates = []
    for stamp in testing.index:
        position = positions.get(stamp)
        if position is None or position < lookback:
            continue
        windows.append(scaled_all[position - lookback : position, 0])
        actuals.append(float(testing.loc[stamp, column]))
        dates.append(stamp)
    if len(windows) < 3:
        raise ValueError("Not enough test days fell after the lookback window.")
    predicted_scale = model.predict(np.array(windows).reshape(-1, lookback, 1), verbose=0)
    predictions = scaler.inverse_transform(predicted_scale).ravel()
    forecast = pd.DataFrame({"Actual": actuals, "Predicted": predictions}, index=pd.Index(dates, name="Date"))
    chart, axis = new_figure(12, 4.8)
    axis.plot(forecast.index, forecast["Actual"], color=RED, linewidth=1.6, label="Actual")
    axis.plot(forecast.index, forecast["Predicted"], color=BLUE, linewidth=1.6, label="Predicted")
    style_axis(axis, f"{symbol} {column.lower()}: actual and LSTM forecast", "Date", column)
    axis.legend(loc="best")
    result.plot("Actual and predicted prices", chart)

    loss_figure, loss_axis = new_figure(12, 4)
    loss_axis.plot(history.history["loss"], color=TEAL, linewidth=1.6, label="Training loss")
    if "val_loss" in history.history:
        loss_axis.plot(history.history["val_loss"], color=GOLD, linewidth=1.6, label="Validation loss")
    style_axis(loss_axis, "Training loss", "Epoch", "Mean squared error")
    loss_axis.legend(loc="best")
    result.plot("Training loss", loss_figure)

    mae = mean_absolute_error(forecast["Actual"], forecast["Predicted"])
    rmse = float(np.sqrt(mean_squared_error(forecast["Actual"], forecast["Predicted"])))
    result.metric(
        "Training days",
        len(training),
        help={
            "what": "The number of trading days used to build the sliding windows the network learns from.",
            "how": "Count the rows in the training download after dropping days with a missing price. Each window, except the first lookback days, becomes one training example.",
            "why": "The network only sees this history. A short range cannot support a long lookback and leaves too few windows to learn a pattern.",
        },
    )
    result.metric(
        "Test days",
        len(forecast),
        help={
            "what": "The number of later days for which the network produced a price.",
            "how": "Count test dates that still have a full lookback of earlier prices. Days at the start of a very short test window are skipped when that history does not exist.",
            "why": "MAE and RMSE are averages over these days only. This is the forecast you can compare with prices the network was not trained on.",
        },
    )
    result.metric(
        "Lookback",
        lookback,
        help={
            "what": "How many prior trading days the network reads before predicting the next price.",
            "how": "Each example is a sequence of this many scaled prices. The chapter script uses 60.",
            "why": "A longer lookback gives the network more context and fewer training windows. Too short a window cannot see a multi-week move.",
        },
    )
    result.metric(
        "LSTM layers",
        layers,
        help={
            "what": "How many LSTM layers are stacked.",
            "how": "Every layer except the last passes its full sequence to the next layer. The last layer outputs one vector, and a dense layer turns that into the next price. The chapter stacks four layers.",
            "why": "More layers can learn a more complicated pattern, and they also take longer to train and are easier to overfit on a few hundred days.",
        },
    )
    result.metric(
        "Epochs",
        epochs,
        help={
            "what": "How many times the network passed through the training windows.",
            "how": "One epoch is one update cycle over every training window, in batches of the chosen batch size. The loss chart has one point per epoch.",
            "why": "Too few epochs leaves the forecast undertrained. Too many lets the network memorize the training path. The chapter uses 100.",
        },
    )
    result.metric(
        "Final training loss",
        float(history.history["loss"][-1]),
        help={
            "what": "The network's average squared error on the training windows after the last epoch.",
            "how": "Prices are scaled between 0 and 1 before training. The loss is the mean of (scaled actual − scaled prediction) squared, measured on the windows used to fit the network.",
            "why": "A falling training loss means the network is fitting the history it can see. It does not, by itself, say the forecast will be accurate on later days.",
        },
    )
    if "val_loss" in history.history:
        result.metric(
            "Final validation loss",
            float(history.history["val_loss"][-1]),
            help={
                "what": "The same squared error on a slice of training windows that was held out of the weight updates.",
                "how": "The validation split takes that fraction from the end of the training windows. The number is the mean squared error on those windows after the last epoch, still in scaled units.",
                "why": "If this stays high while the training loss falls, the network is memorizing the training path instead of learning a pattern that carries forward.",
            },
        )
    result.metric(
        "Test MAE",
        mae,
        help={
            "what": "The average absolute miss between the forecast and the actual test price, in dollars.",
            "how": "For each test day, take the absolute difference between the actual price and the predicted price, then average those differences.",
            "why": "It is the typical size of the forecast error in the same units as the price. A miss of $2 above and a miss of $2 below both count as $2.",
        },
    )
    result.metric(
        "Test RMSE",
        rmse,
        help={
            "what": "The square root of the average squared miss on the test days, in dollars.",
            "how": "Square each test-day error, average those squares, then take the square root. Large misses pull this number up more than they pull MAE.",
            "why": "It quotes the typical forecast error in dollars while giving extra weight to the worst days.",
        },
    )
    result.table("Forecast sample", forecast.round(2), limit=12)
    result.note(
        f"Each point is the next {column.lower()} predicted from the previous {lookback} trading days. "
        "The chapter script uses four LSTM layers, dropout 0.2, and the Adam optimizer."
    )
    return result.to_dict()


GOLD = "#a15c07"
