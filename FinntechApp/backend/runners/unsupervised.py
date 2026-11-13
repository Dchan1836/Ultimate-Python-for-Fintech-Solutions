"""Cluster stocks by return on equity and beta."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from backend.plotting import RED, new_figure, style_axis
from backend.results import DemoResult


def _fundamentals(symbol: str) -> dict:
    from yahooquery import Ticker

    ticker = Ticker(symbol)
    statements = ticker.all_financial_data()
    if not isinstance(statements, pd.DataFrame) or statements.empty:
        raise ValueError("no financial statements")
    if "NetIncome" not in statements.columns or "StockholdersEquity" not in statements.columns:
        raise ValueError("statements are missing net income or equity")
    roe = (statements["NetIncome"] / statements["StockholdersEquity"] * 100).replace([np.inf, -np.inf], np.nan).dropna()
    if roe.empty:
        raise ValueError("ROE could not be computed")
    stats = ticker.key_stats
    payload = stats.get(symbol) if isinstance(stats, dict) else None
    if not isinstance(payload, dict) or payload.get("beta") is None:
        raise ValueError("beta is missing")
    beta = payload["beta"]
    if isinstance(beta, dict):
        beta = beta.get("raw", beta.get("beta"))
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta is missing")
    return {"stock": symbol, "ROE(%)": round(float(roe.mean()), 2), "Beta": round(beta, 2)}


def _cluster_label(label: int, algorithm: str) -> str:
    if algorithm == "dbscan" and int(label) == -1:
        return "Noise"
    if algorithm == "dbscan":
        return f"Cluster {int(label)}"
    return f"Cluster {int(label) + 1}"


def run(params: dict) -> dict:
    result = DemoResult()
    symbols = params["symbols"]
    algorithm = params["algorithm"]
    rows = []
    for symbol in symbols:
        try:
            rows.append(_fundamentals(symbol))
        except Exception as exc:
            result.note(f"{symbol} was skipped: {exc}")
    if len(rows) < 2:
        raise ValueError("Need fundamentals for at least two tickers. Check the symbols and try again.")

    frame = pd.DataFrame(rows).set_index("stock")
    result.add_sample(
        frame[["ROE(%)", "Beta"]].sort_index(),
        "Average return on equity and beta for the tickers that had both figures. Clustering uses only these two columns.",
        limit=8,
    )
    scaled = StandardScaler().fit_transform(frame[["ROE(%)", "Beta"]].to_numpy(dtype=float))
    if algorithm == "kmeans":
        clusters = int(params["n_clusters"])
        if clusters >= len(frame):
            raise ValueError(f"k must be smaller than the number of stocks with data ({len(frame)}).")
        labels = KMeans(n_clusters=clusters, n_init=10, random_state=int(params["random_state"])).fit_predict(scaled)
        title = f"K-means clusters (k = {clusters})"
    else:
        labels = DBSCAN(eps=float(params["eps"]), min_samples=int(params["min_samples"])).fit_predict(scaled)
        title = "DBSCAN clusters"
        clusters = len({int(label) for label in labels if int(label) != -1})

    frame["cluster"] = labels
    frame["Cluster"] = [_cluster_label(label, algorithm) for label in labels]
    chart, axis = new_figure(11, 7)
    sns.scatterplot(
        data=frame.reset_index(),
        x="Beta",
        y="ROE(%)",
        hue="Cluster",
        s=90,
        palette="bright",
        ax=axis,
        edgecolor="white",
    )
    for symbol, row in frame.iterrows():
        axis.annotate(symbol, (row["Beta"], row["ROE(%)"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    style_axis(axis, title, "Beta", "ROE (%)")
    result.plot("ROE and beta clusters", chart)

    if params["elbow"] and algorithm == "kmeans" and len(frame) >= 4:
        limit = min(8, len(frame) - 1)
        inertias = []
        ks = list(range(1, limit + 1))
        for k in ks:
            model = KMeans(n_clusters=k, n_init=10, random_state=int(params["random_state"]))
            model.fit(scaled)
            inertias.append(model.inertia_)
        elbow, elbow_axis = new_figure(10, 4.4)
        elbow_axis.plot(ks, inertias, color=RED, marker="o", linewidth=1.5)
        style_axis(elbow_axis, "K-means elbow", "k", "Inertia")
        result.plot("Elbow curve", elbow)
    elif params["elbow"] and algorithm != "kmeans":
        result.note("The elbow curve is only drawn for k-means.")

    table = frame[["ROE(%)", "Beta", "Cluster"]].sort_values(["Cluster", "ROE(%)"], ascending=[True, False])
    result.table("Cluster members", table, limit=20)
    result.metric(
        "Stocks clustered",
        len(frame),
        help={
            "what": "How many tickers had both an average return on equity and a beta.",
            "how": "For each symbol, divide net income by stockholders' equity, average the years that have both numbers, and read beta from the key statistics. Keep the symbol only when both values exist.",
            "why": "The scatter and the cluster fit use only these stocks. A ticker with no statements never appears in the chart.",
        },
    )
    result.metric(
        "Stocks skipped",
        len(symbols) - len(frame),
        help={
            "what": "How many requested tickers were left out.",
            "how": "Subtract the number of stocks that produced both ROE and beta from the number of symbols you entered. The notes name each one and the reason it failed.",
            "why": "The chapter list includes FORD, which has no Yahoo listing. A skipped name means the clusters were fit without that company.",
        },
    )
    result.metric(
        "Algorithm",
        "K-means" if algorithm == "kmeans" else "DBSCAN",
        help={
            "what": "The method that assigned each stock to a group.",
            "how": "K-means places a chosen number of centers and assigns each stock to the nearest center. DBSCAN groups stocks that sit close together and labels isolated stocks as noise. Both run after ROE and beta are standardized.",
            "why": "The chapter fits k-means. DBSCAN does not ask for a number of groups, so its cluster count can change with the neighborhood radius.",
        },
    )
    result.metric(
        "Clusters",
        clusters,
        help={
            "what": "How many groups the fit produced.",
            "how": "For k-means this is the k you set. For DBSCAN it is the number of groups after noise points, labeled −1, are excluded.",
            "why": "It is the number of colors on the scatter. A useful k separates stocks that really differ in ROE and beta; the elbow curve shows where adding another group stops helping.",
        },
    )
    result.note("ROE is the average of net income divided by stockholders' equity. Both axes are clustered after standard scaling, then plotted in the original units.")
    return result.to_dict()
