"""Parameter forms for the chapter 7 and chapter 8 demos."""

TICKERS = ["MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "JPM"]
CLUSTER_TICKERS = ["MSFT", "AAPL", "IBM", "GE", "F", "DELL", "JPM", "XOM", "JNJ", "KO"]


def text(name, label, default, group, help="", suggestions=None, symbols=False):
    field = {
        "name": name,
        "label": label,
        "type": "symbols" if symbols else "text",
        "default": default,
        "group": group,
        "help": help,
        "upper": True,
    }
    if suggestions:
        field["suggestions"] = suggestions
    return field


def date_field(name, label, default, group, help=""):
    return {"name": name, "label": label, "type": "date", "default": default, "group": group, "help": help}


def number(name, label, default, group, minimum, maximum, step=1, kind="int", help=""):
    return {
        "name": name,
        "label": label,
        "type": "number",
        "default": default,
        "group": group,
        "min": minimum,
        "max": maximum,
        "step": step,
        "kind": kind,
        "help": help,
    }


def check(name, label, default, group, help=""):
    return {"name": name, "label": label, "type": "checkbox", "default": default, "group": group, "help": help}


def select(name, label, default, group, options, help=""):
    return {
        "name": name,
        "label": label,
        "type": "select",
        "default": default,
        "group": group,
        "options": [{"value": value, "label": text_label} for value, text_label in options],
        "help": help,
    }


DEMOS = [
    {
        "id": "linear-regression",
        "chapter": "Chapter 7",
        "title": "Linear regression",
        "script": "chapter7/code/LinearRegressionExample.py ",
        "blurb": "Predicts the closing price from the same day's open, high, low, and volume, then compares the holdout with the actual close.",
        "runtime": "Usually finishes in a few seconds.",
        "outputs": ["Closing price history", "Holdout actual and predicted close", "Actual-versus-predicted scatter", "Error metrics and coefficients", "A short sample of the prices"],
        "fields": [
            text("symbol", "Ticker", "MSFT", "Prices", "Yahoo Finance symbol.", TICKERS),
            date_field("start", "Start", "2021-06-21", "Prices"),
            date_field("end", "End", "2022-12-25", "Prices"),
            number("test_size", "Test size", 0.15, "Model", 0.05, 0.5, 0.05, "float", "Fraction of rows held out. The chapter uses 0.15, without shuffling."),
            check("shuffle", "Shuffle the split", False, "Model", "Off keeps the holdout as the end of the series."),
            number("random_state", "Random state", 0, "Model", 0, 9999, 1, "int"),
        ],
    },
    {
        "id": "lstm",
        "chapter": "Chapter 7",
        "title": "LSTM forecast",
        "script": "chapter7/code/LSTMExample.py",
        "blurb": "Trains a stacked LSTM on a sliding window of prices, then overlays the forecast on a later test window.",
        "runtime": "A short run is about a minute. The first neural run also loads TensorFlow. The chapter's 100 epochs take several minutes.",
        "outputs": ["Actual and predicted prices", "Training loss by epoch", "Test MAE and RMSE", "A short sample of the train and test prices"],
        "presets": [
            {
                "id": "book",
                "label": "Book settings",
                "values": {"epochs": 100, "batch_size": 30, "layers": 4, "units": 50, "dropout": 0.2, "lookback": 60, "optimizer": "adam"},
            }
        ],
        "fields": [
            text("symbol", "Ticker", "MSFT", "Prices", "Yahoo Finance symbol.", TICKERS),
            select("price_column", "Price", "Close", "Prices", [("Open", "Open"), ("High", "High"), ("Low", "Low"), ("Close", "Close")], "Series the network learns. The chapter reads a single price column."),
            date_field("train_start", "Train start", "2021-06-21", "Prices"),
            date_field("train_end", "Train end", "2022-12-25", "Prices"),
            date_field("test_start", "Test start", "2023-01-21", "Prices"),
            date_field("test_end", "Test end", "2023-09-25", "Prices"),
            number("lookback", "Lookback days", 60, "Network", 10, 120, 1, "int", "Prior days used to predict the next price. The chapter uses 60."),
            number("layers", "LSTM layers", 4, "Network", 1, 4, 1, "int", "The chapter stacks four LSTM layers."),
            number("units", "Units per layer", 50, "Network", 8, 128, 1, "int"),
            number("dropout", "Dropout", 0.2, "Network", 0, 0.8, 0.05, "float"),
            number("epochs", "Epochs", 8, "Network", 1, 200, 1, "int", "The chapter trains for 100 epochs."),
            number("batch_size", "Batch size", 30, "Network", 8, 256, 1, "int"),
            select("optimizer", "Optimizer", "adam", "Network", [("adam", "Adam"), ("rmsprop", "RMSprop"), ("sgd", "SGD")]),
            number("validation_split", "Validation split", 0, "Network", 0, 0.4, 0.05, "float", "0 matches the chapter, which does not hold out validation rows."),
        ],
    },
    {
        "id": "xgboost",
        "chapter": "Chapter 7",
        "title": "XGBoost direction",
        "script": "chapter7/code/XGBoostExample.py",
        "blurb": "Labels each day by whether the next close is higher, then compares logistic regression, a support vector classifier, and XGBoost.",
        "runtime": "Usually finishes in a few seconds.",
        "outputs": ["Closing price", "Feature distributions and box plots", "ROC AUC comparison", "XGBoost feature importance", "A short sample of the feature rows"],
        "fields": [
            text("symbol", "Ticker", "MSFT", "Prices", "Yahoo Finance symbol.", TICKERS),
            date_field("start", "Start", "2021-06-21", "Prices"),
            date_field("end", "End", "2023-10-20", "Prices"),
            number("test_size", "Test size", 0.1, "Split", 0.05, 0.4, 0.05, "float", "The chapter holds out 10% at random."),
            number("random_state", "Random state", 2022, "Split", 0, 9999, 1, "int"),
            check("logistic", "Logistic regression", True, "Models"),
            check("svc", "Support vector classifier", True, "Models"),
            select("svc_kernel", "SVC kernel", "poly", "Models", [("poly", "Polynomial"), ("rbf", "RBF"), ("linear", "Linear")], "The chapter uses a polynomial kernel."),
            check("xgboost", "XGBoost", True, "Models"),
            number("n_estimators", "XGBoost trees", 100, "Models", 10, 500, 1, "int"),
            number("max_depth", "XGBoost max depth", 6, "Models", 1, 12, 1, "int"),
            number("learning_rate", "XGBoost learning rate", 0.3, "Models", 0.01, 0.5, 0.01, "float", "0.3 is the library default the chapter relies on."),
        ],
    },
    {
        "id": "supervised-learning",
        "chapter": "Chapter 8",
        "title": "Random forest strategy",
        "script": "chapter8/code/Supervised_Learning_Example.py",
        "blurb": "A random forest classifies the next adjusted close as higher or lower. Those signals are applied to the following day's return.",
        "runtime": "Usually finishes in a few seconds.",
        "outputs": ["Distribution of strategy returns", "Strategy growth against buy and hold", "Accuracy and classification report", "A short sample of the feature rows"],
        "fields": [
            text("symbol", "Ticker", "MSFT", "Prices", "Yahoo Finance symbol.", TICKERS),
            date_field("start", "Start", "2021-06-21", "Prices"),
            date_field("end", "End", "2022-12-25", "Prices"),
            number("n_estimators", "Trees", 100, "Forest", 10, 500, 1, "int", "The chapter uses 100 trees."),
            number("max_depth", "Max depth", 0, "Forest", 0, 30, 1, "int", "0 leaves the depth unlimited."),
            number("split_ratio", "Train fraction", 0.75, "Forest", 0.5, 0.9, 0.05, "float", "The chapter trains on the first 75% of rows."),
            number("rolling_window", "Rolling window", 5, "Forest", 2, 40, 1, "int", "Days used for the return mean and standard deviation."),
            number("random_state", "Random state", 0, "Forest", 0, 9999, 1, "int"),
        ],
    },
    {
        "id": "neural-network",
        "chapter": "Chapter 8",
        "title": "Neural network windows",
        "script": "chapter8/code/Neural_Network_Example.py",
        "blurb": "Sweeps how many prior closes a small feed-forward network can see, and plots validation loss against that window. Each window also carries a dividend flag.",
        "runtime": "A short sweep takes a minute or two. Book settings train nine networks on the full history and take much longer. The first run also loads TensorFlow.",
        "outputs": ["Validation loss by window length", "Loss curves for the best window", "Table of training and validation loss", "A short sample of the closes"],
        "presets": [
            {
                "id": "book",
                "label": "Book settings",
                "values": {"period": "max", "interval": "1d", "input_start": 2, "input_end": 10, "epochs": 20, "batch_size": 32, "hidden_units": 64, "val_size": 1000},
            }
        ],
        "fields": [
            text("symbol", "Ticker", "MSFT", "Data", "Yahoo Finance symbol.", TICKERS),
            select("period", "History", "2y", "Data", [("1y", "1 year"), ("2y", "2 years"), ("5y", "5 years"), ("10y", "10 years"), ("max", "Full history")], "The chapter downloads the full history."),
            select("interval", "Interval", "1d", "Data", [("1d", "Daily"), ("1wk", "Weekly")]),
            number("input_start", "Shortest window", 2, "Sweep", 1, 30, 1, "int", "The chapter starts at 2 prior closes."),
            number("input_end", "Longest window", 4, "Sweep", 1, 30, 1, "int", "The chapter trains every length through 10."),
            number("epochs", "Epochs", 8, "Sweep", 1, 100, 1, "int", "The chapter uses 20 epochs."),
            number("batch_size", "Batch size", 32, "Sweep", 8, 256, 1, "int"),
            number("hidden_units", "Hidden units", 64, "Sweep", 8, 256, 1, "int", "Both hidden layers use this width. The chapter uses 64."),
            number("val_size", "Validation rows", 200, "Sweep", 20, 2000, 1, "int", "The chapter holds out the last 1,000 rows."),
            number("random_state", "Random state", 7, "Sweep", 0, 9999, 1, "int"),
        ],
    },
    {
        "id": "unsupervised-learning",
        "chapter": "Chapter 8",
        "title": "ROE and beta clusters",
        "script": "chapter8/code/UnSupervised_Learning_Example.py",
        "blurb": "Clusters stocks by average return on equity and beta. The scatter is the chapter's k-means chart, drawn with the k you choose.",
        "runtime": "Fetches fundamentals for every ticker. Symbols with no statements are skipped and listed under the results.",
        "outputs": ["ROE versus beta scatter, colored by cluster", "Optional k-means elbow curve", "Table of ROE, beta, and cluster", "A short sample of the ROE and beta rows"],
        "presets": [
            {"id": "four", "label": "Four clusters", "values": {"n_clusters": 4, "elbow": True}},
        ],
        "fields": [
            text(
                "symbols",
                "Tickers",
                "MSFT, AAPL, IBM, GE, FORD, DELL",
                "Universe",
                "Comma-separated symbols. The chapter list includes FORD, which has no Yahoo listing; Ford's ticker is F.",
                CLUSTER_TICKERS,
                symbols=True,
            ),
            select("algorithm", "Algorithm", "kmeans", "Clustering", [("kmeans", "K-means"), ("dbscan", "DBSCAN")], "The chapter fits k-means. DBSCAN is imported beside it."),
            number("n_clusters", "Clusters (k)", 2, "Clustering", 2, 8, 1, "int", "Used by k-means. The chapter fits k = 2."),
            number("eps", "DBSCAN eps", 0.9, "Clustering", 0.2, 3, 0.1, "float", "Neighborhood radius after the features are scaled."),
            number("min_samples", "DBSCAN min samples", 2, "Clustering", 2, 8, 1, "int"),
            check("elbow", "Draw the elbow curve", False, "Clustering", "Inertia for k = 1 upward. The chapter leaves this plot commented out."),
            number("random_state", "Random state", 0, "Clustering", 0, 9999, 1, "int"),
        ],
    },
    {
        "id": "time-series",
        "chapter": "Chapter 8",
        "title": "ARIMA time series",
        "script": "chapter8/code/Time_Series_Example.py",
        "blurb": "Checks whether the close is stationary, plots the series the ARIMA model sees, and forecasts the holdout with a confidence band.",
        "runtime": "The automatic order search usually finishes in under a minute. Turn it off for a faster forecast.",
        "outputs": ["Close distribution", "Rolling mean and standard deviation", "Seasonal decomposition", "Forecast with confidence band", "Model diagnostics", "A short sample of the closes"],
        "presets": [
            {"id": "fast", "label": "Skip the order search", "values": {"run_auto": False, "decompose": False}},
        ],
        "fields": [
            text("symbol", "Ticker", "MSFT", "Prices", "Yahoo Finance symbol.", TICKERS),
            date_field("start", "Start", "2021-06-21", "Prices"),
            date_field("end", "End", "2023-10-25", "Prices"),
            check("use_log", "Model the log close", True, "Prices", "The chapter differences the log price."),
            number("rolling_window", "Rolling window", 12, "Prices", 5, 60, 1, "int", "The chapter uses 12 observations."),
            number("train_fraction", "Train fraction", 0.9, "Prices", 0.6, 0.95, 0.05, "float"),
            number("skip_start", "Drop first points", 3, "Prices", 0, 20, 1, "int", "The chapter drops the first 3 log prices before the split."),
            check("decompose", "Seasonal decomposition", True, "Forecast"),
            number("seasonal_period", "Seasonal period", 12, "Forecast", 2, 60, 1, "int"),
            check("run_auto", "Search for an order", True, "Forecast", "Runs the chapter's stepwise auto_arima search."),
            check("use_auto_order", "Forecast with the searched order", False, "Forecast", "Off uses p, d, and q below. The chapter forecasts with (1, 1, 2)."),
            number("max_p", "Search max p", 3, "Forecast", 1, 5, 1, "int"),
            number("max_q", "Search max q", 3, "Forecast", 1, 5, 1, "int"),
            number("p", "p", 1, "Forecast", 0, 5, 1, "int"),
            number("d", "d", 1, "Forecast", 0, 2, 1, "int"),
            number("q", "q", 2, "Forecast", 0, 5, 1, "int"),
            number("alpha", "Interval alpha", 0.05, "Forecast", 0.01, 0.2, 0.01, "float", "0.05 is a 95% band."),
            number("forecast_steps", "Forecast steps", 0, "Forecast", 0, 500, 1, "int", "0 forecasts one step for each holdout day."),
        ],
    },
]
