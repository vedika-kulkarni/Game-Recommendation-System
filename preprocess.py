"""
preprocess.py
=============
Handles all data loading, cleaning, and preparation for the
Hybrid Game Recommendation System.

Performance note
----------------
The raw Steam-200K dataset creates an ~11 000 × 3 500 user-game matrix.
Dense cosine similarity on that scale takes several minutes and ~4 GB RAM.
We therefore filter to the TOP_USERS most-active users and TOP_GAMES most
popular games before building the matrix.  This keeps response times<30 s
while preserving recommendation quality.
"""

import os
import glob
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Tuneable limits
# ─────────────────────────────────────────────
TOP_USERS = 5_000   # keep the N most-active users
TOP_GAMES = 1_500   # keep the N most-popular games


# ─────────────────────────────────────────────
# 1. Dataset Discovery
# ─────────────────────────────────────────────

def find_dataset(data_dir: str = "data") -> str:
    """
    Dynamically locate the first CSV file inside `data_dir`.
    Raises FileNotFoundError if none is found.
    """
    patterns = [
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "steam*.csv"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No CSV dataset found in '{data_dir}/'. "
        "Please place your steam CSV file inside the data/ folder."
    )


# ─────────────────────────────────────────────
# 2. Raw Loading
# ─────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """
    Load the raw steam CSV.
    Handles the steam-200k format:
        user_id | game_name | behavior ('purchase'/'play') | value | 0
    Also handles simpler 3-column formats:
        user_id | game_name | playtime
    """
    df = pd.read_csv(path, header=None)

    if df.shape[1] == 5:
        df.columns = ["user_id", "game_name", "behavior", "value", "_zero"]
        # Keep only 'play' rows so `value` represents hours played
        df = df[df["behavior"] == "play"].copy()
        df = df[["user_id", "game_name", "value"]].rename(columns={"value": "playtime"})
    elif df.shape[1] == 3:
        df.columns = ["user_id", "game_name", "playtime"]
    else:
        raise ValueError(
            f"Unexpected number of columns: {df.shape[1]}. "
            "Expected 3 (user_id, game_name, playtime) or "
            "5 (steam-200k format)."
        )

    return df


# ─────────────────────────────────────────────
# 3. Cleaning & Normalisation
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows with missing values
    - Remove duplicates (keep max playtime per user-game pair)
    - Ensure correct dtypes
    - Strip whitespace from game names
    """
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Strip leading/trailing whitespace from string columns
    df["game_name"] = df["game_name"].str.strip()

    # Drop nulls
    df.dropna(subset=["user_id", "game_name", "playtime"], inplace=True)

    # Cast types
    df["user_id"] = df["user_id"].astype(int)
    df["playtime"] = pd.to_numeric(df["playtime"], errors="coerce")
    df.dropna(subset=["playtime"], inplace=True)

    # Remove zero-playtime rows (user purchased but never played)
    df = df[df["playtime"] > 0]

    # Deduplicate: keep max playtime per user-game pair
    df = (
        df.groupby(["user_id", "game_name"], as_index=False)
        .agg({"playtime": "max"})
    )

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. User-Game Matrix
# ─────────────────────────────────────────────

def filter_active(df: pd.DataFrame,
                  top_users: int = TOP_USERS,
                  top_games: int = TOP_GAMES) -> pd.DataFrame:
    """
    Reduce the dataset to the most-active users and most-popular games
    so that the dense similarity matrix fits comfortably in RAM and
    can be computed in <30 s on a laptop.

    Selection criteria:
      - top_games  : games played by the most unique users
      - top_users  : users with the highest total playtime
    """
    # Most popular games (by unique-player count)
    popular_games = (
        df.groupby("game_name")["user_id"]
        .nunique()
        .nlargest(top_games)
        .index
    )
    df = df[df["game_name"].isin(popular_games)]

    # Most active users (by total playtime) — among remaining records
    active_users = (
        df.groupby("user_id")["playtime"]
        .sum()
        .nlargest(top_users)
        .index
    )
    df = df[df["user_id"].isin(active_users)]

    return df.reset_index(drop=True)


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the cleaned DataFrame into a User × Game matrix.
    Missing values (games not played by a user) are filled with 0.
    """
    matrix = df.pivot_table(
        index="user_id",
        columns="game_name",
        values="playtime",
        aggfunc="max",
        fill_value=0,
    )
    return matrix


# ─────────────────────────────────────────────
# 5. Log-normalisation Helper
# ─────────────────────────────────────────────

def log_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p normalisation so that extreme playtimes don't
    dominate cosine similarity calculations.
    """
    return np.log1p(matrix)


# ─────────────────────────────────────────────
# 6. Master Pipeline
# ─────────────────────────────────────────────

def load_and_preprocess(data_dir: str = "data",
                        top_users: int = TOP_USERS,
                        top_games: int = TOP_GAMES):
    """
    Full preprocessing pipeline.
    Returns:
        df          – filtered long-format DataFrame
        matrix      – raw User × Game pivot table
        matrix_norm – log-normalised User × Game pivot table
    """
    path = find_dataset(data_dir)
    print(f"[preprocess] Loading dataset from : {path}")

    raw = load_raw(path)
    df_full = clean(raw)

    print(f"[preprocess] Records after cleaning  : {len(df_full):,}")
    print(f"[preprocess] Unique users (raw)       : {df_full['user_id'].nunique():,}")
    print(f"[preprocess] Unique games (raw)       : {df_full['game_name'].nunique():,}")

    # ── Filter to keep matrix tractable ──────────────────────────
    df = filter_active(df_full, top_users=top_users, top_games=top_games)

    print(f"[preprocess] Records after filtering : {len(df):,}")
    print(f"[preprocess] Unique users (filtered) : {df['user_id'].nunique():,}")
    print(f"[preprocess] Unique games (filtered) : {df['game_name'].nunique():,}")

    matrix = build_matrix(df)
    matrix_norm = log_normalize(matrix)

    return df, matrix, matrix_norm
