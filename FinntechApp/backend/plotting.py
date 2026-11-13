"""Matplotlib helpers that return PNG charts instead of opening windows."""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1c1915"
TEAL = "#0f6e56"
BLUE = "#1d4e89"
GOLD = "#a15c07"
RED = "#9f2d2d"
GRID = "#e6e0d4"

_STYLED = False


def apply_style() -> None:
    global _STYLED
    if _STYLED:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "#faf7f1",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d9d0c1",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": "#3f3a33",
            "ytick.color": "#3f3a33",
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "medium",
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "legend.frameon": False,
        }
    )
    _STYLED = True


def new_figure(width: float, height: float, rows: int = 1, cols: int = 1):
    apply_style()
    fig, axes = plt.subplots(rows, cols, figsize=(width, height))
    return fig, axes


def style_axis(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ensure_figure(fig):
    if fig is None:
        return plt.gcf()
    if hasattr(fig, "savefig"):
        return fig
    axes = getattr(fig, "ravel", lambda: [])()
    if len(axes):
        return axes[0].figure
    return plt.gcf()


def figure_image(fig) -> str:
    fig = ensure_figure(fig)
    try:
        fig.tight_layout()
    except Exception:
        pass
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
