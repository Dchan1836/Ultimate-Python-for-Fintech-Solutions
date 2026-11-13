"""Feed-forward nets swept across how many prior closes they can see."""

import numpy as np
import pandas as pd

from backend.market import download_period
from backend.plotting import BLUE, GOLD, TEAL, new_figure, style_axis
from backend.results import DemoResult


def _windows(close: np.ndarray, dividends: np.ndarray, length: int):
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

    generator = TimeseriesGenerator(close, close, length=length, batch_size=len(close))
    features, targets = generator[0]
    flags = np.zeros(len(targets))
    cursor = length
    for row in range(len(targets)):
        flags[row] = 1.0 if dividends[cursor] > 0 else 0.0
        cursor += 1
    combined = np.concatenate([np.asarray(features, dtype=float), flags.reshape(-1, 1)], axis=1)
    return combined, np.asarray(targets, dtype=float).reshape(-1)


def run(params: dict) -> dict:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from keras import Sequential, backend, layers

    tf.random.set_seed(int(params["random_state"]))
    np.random.seed(int(params["random_state"]))

    result = DemoResult()
    symbol = params["symbol"]
    input_start = int(params["input_start"])
    input_end = int(params["input_end"])
    if input_end < input_start:
        raise ValueError("The longest input window must be at least the shortest one.")

    history = download_period(symbol, params["period"], params["interval"])
    if "close" not in history.columns:
        raise ValueError(f"No close column in the {symbol} history.")
    history = history.dropna(subset=["close"]).copy()
    dividends = history["dividends"].fillna(0).to_numpy(dtype=float) if "dividends" in history.columns else np.zeros(len(history))
    close = history["close"].to_numpy(dtype=float)
    price_sample = history[["close", "dividends"]].copy()
    price_sample["close"] = price_sample["close"].round(2)
    price_sample = price_sample.rename(columns={"close": "Close", "dividends": "Dividend"})
    result.add_sample(
        price_sample,
        "The most recent closes in the downloaded history. Each model sees a window of these closes plus whether that day paid a dividend.",
    )
    if len(close) <= input_end + 30:
        raise ValueError("Not enough history for that input window. Choose a longer period or a shorter window.")

    raw_min = float(close.min())
    raw_max = float(close.max())
    span = raw_max - raw_min or 1.0
    normalized_close = (close - raw_min) / span
    dividend_span = float(dividends.max() - dividends.min()) or 1.0
    normalized_dividends = (dividends - float(dividends.min())) / dividend_span

    usable = len(normalized_close) - input_end
    val_size = int(params["val_size"])
    if val_size >= usable - 20:
        val_size = max(20, usable // 5)
        result.note(f"Validation rows were reduced to {val_size} so each model still has a training set.")

    epochs = int(params["epochs"])
    hidden = int(params["hidden_units"])
    batch_size = int(params["batch_size"])
    rows = []
    histories = {}

    for length in range(input_start, input_end + 1):
        backend.clear_session()
        features, targets = _windows(normalized_close, normalized_dividends, length)
        if len(features) <= val_size + 10:
            raise ValueError(f"A window of {length} days does not leave enough rows to train.")
        model = Sequential(
            [
                layers.Input(shape=(length + 1,)),
                layers.Dense(hidden, activation="relu"),
                layers.Dense(hidden, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        fitted = model.fit(
            features[:-val_size],
            targets[:-val_size],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(features[-val_size:], targets[-val_size:]),
            verbose=0,
        )
        train_loss = float(fitted.history["loss"][-1])
        val_loss = float(fitted.history["val_loss"][-1])
        scaled = (raw_max - raw_min) * val_loss + raw_min
        rows.append(
            {
                "Inputs": length,
                "Training loss": train_loss,
                "Validation loss": val_loss,
                "Scaled validation loss": scaled,
            }
        )
        histories[length] = fitted.history
        del model

    stats = pd.DataFrame(rows)
    best = stats.sort_values("Validation loss").iloc[0]
    curve, curve_axis = new_figure(10, 4.6)
    curve_axis.plot(stats["Inputs"], stats["Validation loss"], color=TEAL, marker="o", linewidth=1.6)
    style_axis(curve_axis, "Validation loss by input length", "Prior closes in the window", "Validation loss")
    result.plot("Validation loss by window", curve)

    best_history = histories[int(best["Inputs"])]
    loss_figure, loss_axis = new_figure(10, 4.4)
    loss_axis.plot(best_history["loss"], color=BLUE, linewidth=1.6, label="Training loss")
    loss_axis.plot(best_history["val_loss"], color=GOLD, linewidth=1.6, label="Validation loss")
    style_axis(loss_axis, f"Best window ({int(best['Inputs'])} inputs)", "Epoch", "Mean squared error")
    loss_axis.legend(loc="best")
    result.plot("Best model loss", loss_figure)

    result.table("Window sweep", stats, limit=20)
    result.metric(
        "History rows",
        len(close),
        help={
            "what": "The number of closing prices downloaded for the selected history and interval.",
            "how": "Count the daily or weekly closes returned for the ticker after dropping missing closes.",
            "why": "Every window has to be cut from this history. A short history cannot support a long input window and a large validation holdout.",
        },
    )
    result.metric(
        "Validation rows",
        val_size,
        help={
            "what": "How many of the most recent windows are held out of training for every window length.",
            "how": "Take the requested number of rows from the end of each window set. If that would leave almost nothing to train on, the count is reduced and a note says so. The chapter holds out the last 1,000 rows.",
            "why": "The validation loss is measured on these recent windows, so the comparison between input lengths is about newer prices rather than prices the network just trained on.",
        },
    )
    result.metric(
        "Epochs",
        epochs,
        help={
            "what": "How many times each network passes through its training windows.",
            "how": "A separate network is trained for every input length, and each one runs for this many epochs. The chapter uses 20.",
            "why": "The sweep compares the loss after the same number of passes. Too few passes can make a longer window look worse only because it had less time to fit.",
        },
    )
    result.metric(
        "Hidden units",
        hidden,
        help={
            "what": "The width of each of the two hidden layers.",
            "how": "Every network has two dense layers of this size, with a ReLU activation, and then one unit that predicts the next scaled close. The chapter uses 64.",
            "why": "Wider layers can fit a more detailed pattern. They also have more weights to learn from the same history, so a wider network is easier to overfit.",
        },
    )
    result.metric(
        "Best input length",
        int(best["Inputs"]),
        help={
            "what": "The number of prior closes that produced the lowest validation loss.",
            "how": "Train one network for each length from the shortest window to the longest. After the last epoch, keep the length whose validation loss is smallest.",
            "why": "It is the window the sweep prefers for this stock and history. The second chart shows that network's training and validation loss by epoch.",
        },
    )
    result.metric(
        "Best validation loss",
        float(best["Validation loss"]),
        help={
            "what": "The winning network's average squared error on the held-out windows.",
            "how": "Closes are scaled between 0 and 1. The loss is the mean of (scaled actual close − scaled prediction) squared on the validation rows after the last epoch.",
            "why": "Lower is a closer fit on recent prices the network did not train on. Compare lengths by this number, not by the training loss.",
        },
    )
    result.metric(
        "Scaled validation loss",
        float(best["Scaled validation loss"]),
        help={
            "what": "The chapter's translation of the winning validation loss back toward the price scale.",
            "how": "Multiply the validation loss by the range of the raw close, then add the minimum close: (maximum close − minimum close) × loss + minimum close.",
            "why": "The raw loss is in scaled units and is hard to compare with a stock price. This is the figure the chapter prints beside each window length. It is not itself a dollar forecast error.",
        },
    )
    result.note(
        "Each window is the prior closes plus a flag for whether that day paid a dividend. "
        "Scaled validation loss uses the chapter formula: (max close − min close) × loss + min close."
    )
    return result.to_dict()
