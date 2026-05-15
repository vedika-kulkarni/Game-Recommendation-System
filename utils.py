"""
utils.py
========
Shared helper utilities for the Hybrid Game Recommendation System.

Contains:
  - Session-state caching (for Streamlit)
  - Formatted display helpers
  - Visualisation builders (returned as matplotlib figures)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# Colour palette (consistent across all plots)
# ─────────────────────────────────────────────────────────────────

PALETTE = {
    "primary":   "#6C63FF",
    "secondary": "#FF6584",
    "accent":    "#43E97B",
    "bg":        "#0F0F1A",
    "surface":   "#1A1A2E",
    "text":      "#EAEAEA",
    "muted":     "#7A7A9D",
    "Casual Player":   "#43E97B",
    "Moderate Player": "#F9A826",
    "Hardcore Player": "#FF6584",
}

CLUSTER_COLORS = [PALETTE["Casual Player"],
                  PALETTE["Moderate Player"],
                  PALETTE["Hardcore Player"]]


def _apply_dark_theme(fig: plt.Figure, ax) -> None:
    """Apply consistent dark theme to any matplotlib figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    if isinstance(ax, np.ndarray):
        axes = ax.flatten()
    elif isinstance(ax, (list, tuple)):
        axes = ax
    else:
        axes = [ax]
    for a in axes:
        a.set_facecolor(PALETTE["surface"])
        a.tick_params(colors=PALETTE["text"], labelsize=9)
        a.xaxis.label.set_color(PALETTE["text"])
        a.yaxis.label.set_color(PALETTE["text"])
        a.title.set_color(PALETTE["text"])
        for spine in a.spines.values():
            spine.set_edgecolor(PALETTE["muted"])


# ─────────────────────────────────────────────────────────────────
# Plot 1 – Top Games by Popularity
# ─────────────────────────────────────────────────────────────────

def plot_top_games(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of the most-played games (total playtime).
    """
    agg = (
        df.groupby("game_name")["playtime"]
        .sum()
        .nlargest(top_n)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.plasma(np.linspace(0.25, 0.85, len(agg)))
    bars = ax.barh(agg.index, agg.values, color=colors, edgecolor="none", height=0.7)

    # Value labels
    for bar, val in zip(bars, agg.values):
        ax.text(
            bar.get_width() + agg.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}h",
            va="center", fontsize=8, color=PALETTE["text"]
        )

    ax.set_xlabel("Total Playtime (hours)", fontsize=10)
    ax.set_title(f"Top {top_n} Games by Total Playtime", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, agg.max() * 1.15)

    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Plot 2 – User Clusters Scatter
# ─────────────────────────────────────────────────────────────────

def plot_user_clusters(cluster_df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot of users coloured by cluster label.
    X-axis: game variety  |  Y-axis: total playtime
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, color in [
        ("Casual Player",   PALETTE["Casual Player"]),
        ("Moderate Player", PALETTE["Moderate Player"]),
        ("Hardcore Player", PALETTE["Hardcore Player"]),
    ]:
        sub = cluster_df[cluster_df["cluster_label"] == label]
        ax.scatter(
            sub["game_variety"],
            sub["total_playtime"],
            c=color, label=label, alpha=0.55, s=12, edgecolors="none",
        )

    ax.set_xlabel("Game Variety (# unique games)", fontsize=10)
    ax.set_ylabel("Total Playtime (hours)", fontsize=10)
    ax.set_title("User Segmentation by Playing Behaviour", fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(framealpha=0.2, labelcolor=PALETTE["text"], fontsize=9)
    legend.get_frame().set_facecolor(PALETTE["surface"])

    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Plot 3 – Cluster Distribution Pie
# ─────────────────────────────────────────────────────────────────

def plot_cluster_distribution(cluster_df: pd.DataFrame) -> plt.Figure:
    """Pie chart showing the percentage of users in each cluster."""
    counts = cluster_df["cluster_label"].value_counts()
    colors = [PALETTE.get(l, "#888") for l in counts.index]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
    )
    for t in texts:
        t.set_color(PALETTE["text"])
        t.set_fontsize(10)
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title("Player Segment Distribution", fontsize=13, fontweight="bold",
                 color=PALETTE["text"], pad=12)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Plot 4 – Recommendation Score Bar Chart
# ─────────────────────────────────────────────────────────────────

def plot_recommendation_scores(
    recs_df: pd.DataFrame, score_col: str, title: str = "Recommendations"
) -> plt.Figure:
    """Generic horizontal bar chart for any recommendation results."""
    if recs_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No recommendations found.", ha="center", va="center",
                color=PALETTE["text"], fontsize=12)
        _apply_dark_theme(fig, ax)
        return fig

    fig, ax = plt.subplots(figsize=(9, max(3, len(recs_df) * 0.6)))

    colors = plt.cm.cool(np.linspace(0.3, 0.9, len(recs_df)))
    bars = ax.barh(
        recs_df["game_name"][::-1],
        recs_df[score_col][::-1],
        color=colors[::-1], edgecolor="none", height=0.6
    )
    for bar, val in zip(bars, recs_df[score_col][::-1]):
        ax.text(
            bar.get_width() + recs_df[score_col].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=8, color=PALETTE["text"]
        )

    ax.set_xlabel("Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, recs_df[score_col].max() * 1.2)
    _apply_dark_theme(fig, ax)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Plot 5 – Trending Games
# ─────────────────────────────────────────────────────────────────

def plot_trending_games(trending_df: pd.DataFrame) -> plt.Figure:
    """
    Dual-axis bar chart:  total playtime (left) + user count (right)
    """
    games = trending_df["game_name"]
    playtime = trending_df["total_playtime"]
    users = trending_df["user_count"]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(games))
    w = 0.4

    ax1.bar(x - w / 2, playtime, width=w,
            color=PALETTE["primary"], alpha=0.85, label="Total Playtime (h)", zorder=3)
    ax2.bar(x + w / 2, users, width=w,
            color=PALETTE["secondary"], alpha=0.85, label="# Players", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(games, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Total Playtime (hours)", color=PALETTE["primary"], fontsize=9)
    ax2.set_ylabel("Number of Players", color=PALETTE["secondary"], fontsize=9)
    ax1.set_title("Trending Games", fontsize=13, fontweight="bold", pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                        loc="upper right", framealpha=0.2, fontsize=9)
    legend.get_frame().set_facecolor(PALETTE["surface"])
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])

    _apply_dark_theme(fig, [ax1, ax2])
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────

def format_recs_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up a recommendations DataFrame for display."""
    df = df.copy()
    if "game_name" in df.columns:
        df["game_name"] = df["game_name"].str.title()
    for col in df.select_dtypes(include=[float]).columns:
        df[col] = df[col].round(4)
    return df


def user_exists(user_id: int, matrix: pd.DataFrame) -> bool:
    return user_id in matrix.index


def game_exists(game_name: str, game_sim_df: pd.DataFrame) -> bool:
    return (
        game_name in game_sim_df.index or
        game_sim_df.index.str.lower().str.contains(
            game_name.lower(), regex=False
        ).any()
    )
