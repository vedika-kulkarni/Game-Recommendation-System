"""
model.py
========
Core recommendation engine for the Hybrid Game Recommendation System.

Includes:
  - User-based collaborative filtering
  - Item-based (game) similarity
  - Hybrid recommendations
  - Cold-start handling
  - Trending games
  - User clustering (KMeans)
  - Evaluation metrics (Precision@K, Recall@K)
  - Explainable AI snippets
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 – Similarity Matrices (cached at module level)
# ═══════════════════════════════════════════════════════════════════

_user_sim_df: pd.DataFrame = None
_game_sim_df: pd.DataFrame = None


def compute_user_similarity(matrix_norm: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between all users."""
    global _user_sim_df
    sim = cosine_similarity(matrix_norm)
    _user_sim_df = pd.DataFrame(
        sim, index=matrix_norm.index, columns=matrix_norm.index
    )
    return _user_sim_df


def compute_game_similarity(matrix_norm: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between all games (transpose first)."""
    global _game_sim_df
    sim = cosine_similarity(matrix_norm.T)
    _game_sim_df = pd.DataFrame(
        sim, index=matrix_norm.columns, columns=matrix_norm.columns
    )
    return _game_sim_df


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 – User-Based Collaborative Filtering
# ═══════════════════════════════════════════════════════════════════

def recommend_by_user(
    user_id: int,
    matrix: pd.DataFrame,
    user_sim_df: pd.DataFrame,
    top_n: int = 5,
    n_similar_users: int = 10,
) -> pd.DataFrame:
    """
    Recommend top-N games for `user_id` based on what similar
    users have played but this user hasn't.

    Returns a DataFrame with columns:
        game_name | score | reason
    """
    if user_id not in matrix.index:
        raise ValueError(f"User ID {user_id} not found in the dataset.")

    # Games already played by this user (playtime > 0)
    played = set(matrix.columns[matrix.loc[user_id] > 0])

    # Top similar users (exclude the user themselves)
    sim_scores = user_sim_df.loc[user_id].drop(index=user_id, errors="ignore")
    similar_users = sim_scores.nlargest(n_similar_users).index.tolist()

    if not similar_users:
        return pd.DataFrame(columns=["game_name", "score", "reason"])

    # Weighted average of similar-users' playtimes
    weights = sim_scores[similar_users].values
    sub_matrix = matrix.loc[similar_users]
    weighted_scores = sub_matrix.T.dot(weights) / (weights.sum() + 1e-9)

    # Remove games already played
    candidates = weighted_scores.drop(index=list(played), errors="ignore")
    top_games = candidates.nlargest(top_n)

    result = pd.DataFrame({
        "game_name": top_games.index,
        "score": top_games.values,
        "reason": (
            f"Recommended because {n_similar_users} similar users "
            "played this game for many hours"
        ),
    })
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 – Item-Based (Game) Similarity
# ═══════════════════════════════════════════════════════════════════

def recommend_similar_games(
    game_name: str,
    game_sim_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend games similar to `game_name` using item-item cosine
    similarity.

    Returns a DataFrame with columns:
        game_name | similarity | reason
    """
    if game_name not in game_sim_df.index:
        # Fuzzy fallback: find closest match
        candidates = game_sim_df.index[
            game_sim_df.index.str.lower().str.contains(
                game_name.lower(), regex=False
            )
        ].tolist()
        if not candidates:
            raise ValueError(
                f"Game '{game_name}' not found. "
                "Try a different name or check spelling."
            )
        game_name = candidates[0]

    sims = game_sim_df.loc[game_name].drop(index=game_name, errors="ignore")
    top_games = sims.nlargest(top_n)

    result = pd.DataFrame({
        "game_name": top_games.index,
        "similarity": top_games.values.round(4),
        "reason": f"Similar to '{game_name}' based on shared user playtime patterns",
    })
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 – Hybrid Recommendation
# ═══════════════════════════════════════════════════════════════════

def hybrid_recommendation(
    user_id: int,
    matrix: pd.DataFrame,
    user_sim_df: pd.DataFrame,
    game_sim_df: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 5,
    user_weight: float = 0.6,
    item_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Combine user-based and item-based signals into a single ranked list.

    Strategy:
      1. Get user-based recommendations (weighted score).
      2. For each game the user HAS played, collect top item-similar games.
      3. Merge both lists and compute a blended score.

    Returns a DataFrame with columns:
        game_name | hybrid_score | reason
    """
    if user_id not in matrix.index:
        raise ValueError(f"User ID {user_id} not found in the dataset.")

    # ── User-based signal ──────────────────────────────────────────
    user_recs = recommend_by_user(
        user_id, matrix, user_sim_df, top_n=top_n * 3
    )
    user_scores = dict(zip(user_recs["game_name"], user_recs["score"]))

    # ── Item-based signal ──────────────────────────────────────────
    played_games = list(matrix.columns[matrix.loc[user_id] > 0])
    played_games_present = [g for g in played_games if g in game_sim_df.index]

    item_scores: dict = {}
    played_set = set(played_games)

    for game in played_games_present[:10]:          # limit to 10 seed games
        sims = game_sim_df.loc[game].drop(index=game, errors="ignore")
        for g, s in sims.nlargest(top_n).items():
            if g not in played_set:
                item_scores[g] = item_scores.get(g, 0) + s

    # Normalize item scores
    if item_scores:
        max_item = max(item_scores.values()) + 1e-9
        item_scores = {g: v / max_item for g, v in item_scores.items()}

    # Normalize user scores
    if user_scores:
        max_user = max(user_scores.values()) + 1e-9
        user_scores_norm = {g: v / max_user for g, v in user_scores.items()}
    else:
        user_scores_norm = {}

    # ── Blend ──────────────────────────────────────────────────────
    all_games = set(user_scores_norm) | set(item_scores)
    blended = {}
    for g in all_games:
        u = user_scores_norm.get(g, 0)
        i = item_scores.get(g, 0)
        blended[g] = user_weight * u + item_weight * i

    top_hybrid = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result_rows = []
    for game, score in top_hybrid:
        sources = []
        if game in user_scores_norm:
            sources.append("similar users' playtime patterns")
        if game in item_scores:
            sources.append("similarity to games you've played")
        reason = "Recommended based on: " + " & ".join(sources)
        result_rows.append({"game_name": game, "hybrid_score": round(score, 4), "reason": reason})

    return pd.DataFrame(result_rows)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 – Cold-Start Handling
# ═══════════════════════════════════════════════════════════════════

def recommend_from_favorites(
    game_list: list,
    game_sim_df: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    For new users with no history, recommend games based on a list
    of favourite games they provide.

    Returns a DataFrame with columns:
        game_name | score | reason
    """
    if not game_list:
        raise ValueError("Please provide at least one favourite game.")

    # Validate games; use fuzzy match fallback
    valid_games = []
    for g in game_list:
        if g in game_sim_df.index:
            valid_games.append(g)
        else:
            matches = game_sim_df.index[
                game_sim_df.index.str.lower().str.contains(
                    g.lower(), regex=False
                )
            ].tolist()
            if matches:
                valid_games.append(matches[0])

    if not valid_games:
        raise ValueError(
            "None of the provided games were found in the dataset. "
            "Please check the game names."
        )

    avoid = set(valid_games)
    aggregated: dict = {}

    for game in valid_games:
        sims = game_sim_df.loc[game].drop(index=game, errors="ignore")
        for g, s in sims.nlargest(top_n * 3).items():
            if g not in avoid:
                aggregated[g] = aggregated.get(g, 0) + s

    top_games = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(top_games, columns=["game_name", "score"])
    result["reason"] = (
        "Recommended because it shares player communities with your favourites: "
        + ", ".join(valid_games)
    )
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 – Trending Games
# ═══════════════════════════════════════════════════════════════════

def get_trending_games(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Rank games by:
      - total playtime across all users
      - number of distinct users who played them

    Trending score = normalised(total_playtime) + normalised(user_count)

    Returns a DataFrame with columns:
        game_name | total_playtime | user_count | trending_score
    """
    agg = df.groupby("game_name").agg(
        total_playtime=("playtime", "sum"),
        user_count=("user_id", "nunique"),
    ).reset_index()

    # Normalise each metric to [0, 1]
    agg["norm_playtime"] = agg["total_playtime"] / (agg["total_playtime"].max() + 1e-9)
    agg["norm_users"] = agg["user_count"] / (agg["user_count"].max() + 1e-9)
    agg["trending_score"] = (agg["norm_playtime"] + agg["norm_users"]) / 2

    return (
        agg[["game_name", "total_playtime", "user_count", "trending_score"]]
        .sort_values("trending_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 – User Clustering (KMeans)
# ═══════════════════════════════════════════════════════════════════

CLUSTER_LABELS = {0: "Casual Player", 1: "Moderate Player", 2: "Hardcore Player"}


def cluster_users(
    df: pd.DataFrame,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """
    Segment users into Casual / Moderate / Hardcore using KMeans on:
      - total_playtime
      - game_variety (number of distinct games played)
      - avg_playtime

    Returns a DataFrame indexed by user_id with a 'cluster_label' column.
    """
    user_stats = df.groupby("user_id").agg(
        total_playtime=("playtime", "sum"),
        game_variety=("game_name", "nunique"),
        avg_playtime=("playtime", "mean"),
    ).reset_index()

    features = user_stats[["total_playtime", "game_variety", "avg_playtime"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_stats["cluster"] = kmeans.fit_predict(features_scaled)

    # Map cluster ids to meaningful labels based on average total_playtime
    cluster_means = user_stats.groupby("cluster")["total_playtime"].mean().sort_values()
    label_map = {old: new for new, old in enumerate(cluster_means.index)}
    user_stats["cluster"] = user_stats["cluster"].map(label_map)
    user_stats["cluster_label"] = user_stats["cluster"].map(CLUSTER_LABELS)

    return user_stats.set_index("user_id")


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 – Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════

def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    """
    Precision@K = |recommended ∩ relevant| / K

    Args:
        recommended : ordered list of recommended game names
        relevant    : list of games the user actually played (ground truth)
        k           : cut-off rank

    Returns:
        float in [0, 1]
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for g in recommended_k if g in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    """
    Recall@K = |recommended ∩ relevant| / |relevant|

    Args:
        recommended : ordered list of recommended game names
        relevant    : list of games the user actually played
        k           : cut-off rank

    Returns:
        float in [0, 1]
    """
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for g in recommended_k if g in relevant_set)
    return hits / len(relevant_set)


def evaluate_model(
    matrix: pd.DataFrame,
    user_sim_df: pd.DataFrame,
    sample_size: int = 100,
    k: int = 5,
) -> dict:
    """
    Hold-out evaluation on a random sample of users.

    For each sampled user:
      - Use 80% of their games as training signal (already in matrix)
      - Treat remaining 20% as ground truth
      - Compute precision@K and recall@K

    Returns dict with mean precision and recall.
    """
    # Users with enough games for a meaningful split
    active_users = matrix.index[matrix.gt(0).sum(axis=1) >= 5].tolist()
    if not active_users:
        return {"precision@k": 0.0, "recall@k": 0.0, "k": k, "n_users": 0}

    sample = np.random.choice(
        active_users, min(sample_size, len(active_users)), replace=False
    )

    precisions, recalls = [], []

    for uid in sample:
        played_games = list(matrix.columns[matrix.loc[uid] > 0])
        if len(played_games) < 2:
            continue
        split = max(1, int(len(played_games) * 0.2))
        ground_truth = played_games[:split]

        try:
            recs_df = recommend_by_user(uid, matrix, user_sim_df, top_n=k)
            recs = recs_df["game_name"].tolist()
            precisions.append(precision_at_k(recs, ground_truth, k))
            recalls.append(recall_at_k(recs, ground_truth, k))
        except Exception:
            continue

    return {
        "precision@k": round(np.mean(precisions), 4) if precisions else 0.0,
        "recall@k": round(np.mean(recalls), 4) if recalls else 0.0,
        "k": k,
        "n_users": len(precisions),
    }
