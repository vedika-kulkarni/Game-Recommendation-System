"""
app.py
======
Streamlit UI for the Hybrid Game Recommendation System.

Run with:
    streamlit run app.py
"""

import time
import streamlit as st
import pandas as pd
import numpy as np

# ─── Local modules ────────────────────────────────────────────────
from preprocess import load_and_preprocess
from model import (
    compute_user_similarity,
    compute_game_similarity,
    recommend_by_user,
    recommend_similar_games,
    hybrid_recommendation,
    recommend_from_favorites,
    get_trending_games,
    cluster_users,
    evaluate_model,
)
from utils import (
    PALETTE,
    format_recs_table,
    user_exists,
    game_exists,
    plot_top_games,
    plot_user_clusters,
    plot_cluster_distribution,
    plot_recommendation_scores,
    plot_trending_games,
)

# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GameMind · Hybrid Recommendation System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F0F1A;
    color: #EAEAEA;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122A 0%, #0F0F1A 100%);
    border-right: 1px solid #2A2A4A;
}
[data-testid="stSidebar"] * { color: #EAEAEA !important; }

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    font-size: 0.92rem;
    transition: transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 4px 15px rgba(108,99,255,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(108,99,255,0.55);
}

/* ── Cards ───────────────────────────────────────── */
.card {
    background: linear-gradient(145deg, #1A1A2E, #16213E);
    border: 1px solid #2A2A4A;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.5); }

/* ── Metric cards ────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1E1B4B, #312E81);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    border: 1px solid #4338CA44;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #A5B4FC; line-height: 1.1; }
.metric-label { font-size: 0.82rem; color: #7C7CA0; margin-top: 4px; }

/* ── Badges ──────────────────────────────────────── */
.badge {
    display: inline-block;
    background: rgba(108,99,255,0.18);
    color: #A5B4FC;
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
.badge-green  { background: rgba(67,233,123,0.12); color:#6EE7B7; border-color:rgba(67,233,123,0.3); }
.badge-orange { background: rgba(249,168,38,0.12); color:#FCD34D; border-color:rgba(249,168,38,0.3); }
.badge-red    { background: rgba(255,101,132,0.12); color:#FCA5A5; border-color:rgba(255,101,132,0.3); }

/* ── Section headers ─────────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    border-left: 4px solid #6C63FF;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}
.section-header h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #A5B4FC, #6C63FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Reason box ──────────────────────────────────── */
.reason-box {
    background: rgba(108,99,255,0.08);
    border-left: 3px solid #6C63FF;
    border-radius: 0 8px 8px 0;
    padding: 6px 12px;
    font-size: 0.82rem;
    color: #A5B4FC;
    margin-top: 4px;
}

/* ── DataFrame styling ───────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Hero banner ─────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #0F0F1A 60%, #1A0B2E 100%);
    border: 1px solid #2A2A4A;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.15), transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #FFFFFF, #A5B4FC, #6C63FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #8B8BB5; font-size: 1.05rem; margin-top: 0.5rem; }

/* ── Tabs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #1A1A2E;
    border-radius: 10px 10px 0 0;
    border: 1px solid #2A2A4A;
    color: #7A7A9D;
    font-weight: 600;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1E1B4B, #2D2A6E) !important;
    color: #A5B4FC !important;
    border-color: #6C63FF !important;
}

/* ── Input fields ────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #1A1A2E !important;
    border: 1px solid #2A2A4A !important;
    border-radius: 10px !important;
    color: #EAEAEA !important;
}
.stMultiSelect > div { background: #1A1A2E !important; border-radius: 10px !important; }

/* ── Spinner ─────────────────────────────────────── */
.stSpinner { color: #6C63FF !important; }

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0F0F1A; }
::-webkit-scrollbar-thumb { background: #2A2A4A; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_data():
    df, matrix, matrix_norm = load_and_preprocess()
    user_sim_df   = compute_user_similarity(matrix_norm)
    game_sim_df   = compute_game_similarity(matrix_norm)
    cluster_df    = cluster_users(df)
    trending_df   = get_trending_games(df, top_n=15)
    return df, matrix, matrix_norm, user_sim_df, game_sim_df, cluster_df, trending_df


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🎮</div>
        <h2 style='font-family:Space Grotesk; font-size:1.4rem; font-weight:800;
                   background:linear-gradient(90deg,#A5B4FC,#6C63FF);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   margin:0;'>GameMind</h2>
        <p style='color:#5A5A7D; font-size:0.78rem; margin:4px 0 0;'>
            Hybrid Recommendation Engine
        </p>
    </div>
    <hr style='border-color:#2A2A4A; margin: 0.8rem 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "🤝 User Recommendations",
            "🎯 Similar Games",
            "🌟 Hybrid Engine",
            "🆕 New User (Cold Start)",
            "🔥 Trending Games",
            "👥 User Clusters",
            "📊 Model Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2A2A4A;'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#5A5A7D; padding: 0.5rem 0;'>
        <b style='color:#7A7A9D;'>Data</b><br>Steam 200K Dataset<br><br>
        <b style='color:#7A7A9D;'>Algorithms</b><br>
        • Cosine Similarity<br>
        • KMeans Clustering<br>
        • Hybrid Scoring<br><br>
        <b style='color:#7A7A9D;'>Tech Stack</b><br>
        Python · Pandas · Scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════

with st.spinner("⚙️ Initialising GameMind — loading & computing similarities …"):
    try:
        df, matrix, matrix_norm, user_sim_df, game_sim_df, cluster_df, trending_df = load_data()
        load_error = None
    except Exception as e:
        load_error = str(e)

if load_error:
    st.error(f"**Failed to load data:** {load_error}")
    st.stop()

# Useful constants
all_game_names = sorted(game_sim_df.index.tolist())
all_user_ids   = sorted(matrix.index.tolist())


# ═══════════════════════════════════════════════════════════════════
#  HELPER: section header
# ═══════════════════════════════════════════════════════════════════

def sh(icon: str, title: str):
    st.markdown(
        f"""<div class='section-header'><span style='font-size:1.4rem'>{icon}</span>
        <h2>{title}</h2></div>""",
        unsafe_allow_html=True,
    )


def rec_card(rank: int, game: str, score: float, reason: str, badge_class: str = ""):
    star = "⭐" * min(rank, 1)
    st.markdown(f"""
    <div class='card'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <span style='font-size:0.78rem; color:#5A5A7D'>#{rank}</span>
                <span style='font-size:1.05rem; font-weight:700; margin-left:8px;'>{game}</span>
            </div>
            <span class='badge {badge_class}'>Score: {score:.4f}</span>
        </div>
        <div class='reason-box'>💡 {reason}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown("""
    <div class='hero'>
        <h1>🎮 GameMind</h1>
        <p>Production-ready Hybrid Game Recommendation Engine powered by<br>
           <b>Collaborative Filtering</b> · <b>Item Similarity</b> · <b>KMeans Clustering</b></p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df['user_id'].nunique():,}</div>
            <div class='metric-label'>👤 Unique Users</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df['game_name'].nunique():,}</div>
            <div class='metric-label'>🎮 Unique Games</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>📋 Play Records</div></div>""", unsafe_allow_html=True)
    with c4:
        avg_h = df["playtime"].mean()
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{avg_h:.1f}h</div>
            <div class='metric-label'>⏱️ Avg Playtime</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        sh("📊", "Top Games by Popularity")
        st.pyplot(plot_top_games(df, top_n=12))
    with col_b:
        sh("🔥", "Trending Right Now")
        t = trending_df[["game_name", "total_playtime", "user_count"]].head(8).copy()
        t.columns = ["Game", "Total Hours", "# Players"]
        t["Total Hours"] = t["Total Hours"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(t, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sh("🕹️", "Feature Guide")
    fc1, fc2, fc3 = st.columns(3)
    features = [
        ("🤝", "User Recommendations", "Collaborative filtering based on users with similar taste"),
        ("🎯", "Similar Games",         "Item-based similarity to find games like one you love"),
        ("🌟", "Hybrid Engine",         "Blends user + item signals for the best results"),
        ("🆕", "Cold-Start",            "New user? Pick favourite games → get instant recs"),
        ("🔥", "Trending Games",        "See what's hot right now by playtime & players"),
        ("👥", "User Clusters",         "Visualise Casual / Moderate / Hardcore player groups"),
    ]
    for idx, (icon, title, desc) in enumerate(features):
        col = [fc1, fc2, fc3][idx % 3]
        with col:
            st.markdown(f"""
            <div class='card' style='min-height:110px;'>
                <div style='font-size:1.6rem'>{icon}</div>
                <div style='font-weight:700; margin:6px 0 4px;'>{title}</div>
                <div style='font-size:0.82rem; color:#7A7A9D;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: USER RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════

elif page == "🤝 User Recommendations":
    sh("🤝", "User-Based Collaborative Filtering")
    st.markdown(
        "<p style='color:#7A7A9D'>Finds users with similar playtime patterns and recommends "
        "games they loved but you haven't tried yet.</p>", unsafe_allow_html=True
    )

    with st.form("user_rec_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            uid_input = st.number_input(
                "Enter User ID", min_value=int(min(all_user_ids)),
                max_value=int(max(all_user_ids)),
                value=int(all_user_ids[0]),
                step=1, help="Must be a valid user ID in the dataset"
            )
        with col2:
            top_n = st.slider("Top N games", 3, 20, 5)
        submitted = st.form_submit_button("🔍 Get Recommendations", use_container_width=True)

    if submitted:
        if not user_exists(int(uid_input), matrix):
            st.error(f"❌ User ID {uid_input} not found in the dataset.")
        else:
            with st.spinner("Finding your recommendations…"):
                recs = recommend_by_user(
                    int(uid_input), matrix, user_sim_df, top_n=top_n
                )

            if recs.empty:
                st.warning("No recommendations found for this user.")
            else:
                st.success(f"✅ Top {len(recs)} recommendations for User **{uid_input}**")
                tab1, tab2 = st.tabs(["📋 List View", "📊 Chart View"])
                with tab1:
                    for i, row in recs.iterrows():
                        rec_card(i + 1, row["game_name"], row["score"], row["reason"])
                with tab2:
                    st.pyplot(plot_recommendation_scores(
                        recs, "score", f"Recommendations for User {uid_input}"
                    ))

                # Also show games the user has already played
                with st.expander("🕹️ Games already played by this user"):
                    played = matrix.columns[matrix.loc[int(uid_input)] > 0].tolist()
                    played_df = pd.DataFrame({
                        "Game": played,
                        "Playtime (h)": matrix.loc[int(uid_input), played].values
                    }).sort_values("Playtime (h)", ascending=False)
                    st.dataframe(format_recs_table(played_df), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: SIMILAR GAMES
# ═══════════════════════════════════════════════════════════════════

elif page == "🎯 Similar Games":
    sh("🎯", "Item-Based Game Similarity")
    st.markdown(
        "<p style='color:#7A7A9D'>Finds games with similar player communities — "
        "players who loved one tend to love the others.</p>", unsafe_allow_html=True
    )

    with st.form("game_sim_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            game_input = st.selectbox(
                "Select a game", options=all_game_names,
                help="Choose a game to find similar titles"
            )
        with col2:
            top_n_g = st.slider("Top N", 3, 20, 5, key="topn_g")
        submitted_g = st.form_submit_button("🔍 Find Similar Games", use_container_width=True)

    if submitted_g:
        with st.spinner("Computing game similarity…"):
            try:
                sims = recommend_similar_games(game_input, game_sim_df, top_n=top_n_g)
                tab1, tab2 = st.tabs(["📋 List View", "📊 Chart View"])
                with tab1:
                    st.success(f"✅ Games similar to **{game_input}**")
                    for i, row in sims.iterrows():
                        rec_card(
                            i + 1, row["game_name"],
                            row["similarity"], row["reason"], "badge-green"
                        )
                with tab2:
                    st.pyplot(plot_recommendation_scores(
                        sims, "similarity", f"Games Similar to {game_input}"
                    ))
            except ValueError as e:
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════════
#  PAGE: HYBRID ENGINE
# ═══════════════════════════════════════════════════════════════════

elif page == "🌟 Hybrid Engine":
    sh("🌟", "Hybrid Recommendation Engine")
    st.markdown(
        "<p style='color:#7A7A9D'>Combines <b>user-based</b> and <b>item-based</b> signals "
        "with configurable weights for the most accurate results.</p>", unsafe_allow_html=True
    )

    with st.form("hybrid_form"):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            h_uid = st.number_input(
                "User ID", min_value=int(min(all_user_ids)),
                max_value=int(max(all_user_ids)),
                value=int(all_user_ids[0]), step=1
            )
        with col2:
            h_topn = st.slider("Top N", 3, 20, 5, key="h_topn")
        with col3:
            u_weight = st.slider("User weight", 0.0, 1.0, 0.6, 0.05)
        with col4:
            i_weight = st.slider("Item weight", 0.0, 1.0, 0.4, 0.05)
        submitted_h = st.form_submit_button("🚀 Generate Hybrid Recs", use_container_width=True)

    if submitted_h:
        if not user_exists(int(h_uid), matrix):
            st.error(f"❌ User ID {h_uid} not found.")
        else:
            with st.spinner("Blending recommendations…"):
                try:
                    h_recs = hybrid_recommendation(
                        int(h_uid), matrix, user_sim_df, game_sim_df, df,
                        top_n=h_topn, user_weight=u_weight, item_weight=i_weight,
                    )
                    st.success(f"✅ Hybrid recommendations for User **{h_uid}**")

                    # Weight info
                    st.markdown(f"""
                    <div style='display:flex; gap:10px; margin-bottom:1rem;'>
                        <span class='badge'>👤 User weight: {u_weight:.0%}</span>
                        <span class='badge badge-green'>🎯 Item weight: {i_weight:.0%}</span>
                        <span class='badge badge-orange'>Top {h_topn} results</span>
                    </div>""", unsafe_allow_html=True)

                    tab1, tab2 = st.tabs(["📋 List View", "📊 Chart View"])
                    with tab1:
                        for i, row in h_recs.iterrows():
                            rec_card(i+1, row["game_name"], row["hybrid_score"],
                                     row["reason"], "badge-orange")
                    with tab2:
                        st.pyplot(plot_recommendation_scores(
                            h_recs, "hybrid_score", f"Hybrid Recs for User {h_uid}"
                        ))
                except Exception as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════════
#  PAGE: COLD START
# ═══════════════════════════════════════════════════════════════════

elif page == "🆕 New User (Cold Start)":
    sh("🆕", "Cold-Start Recommendations")
    st.markdown(
        "<p style='color:#7A7A9D'>New here? Select games you enjoy and we'll find similar ones "
        "— no history needed.</p>", unsafe_allow_html=True
    )

    with st.form("cold_form"):
        fav_games = st.multiselect(
            "Select your favourite games (pick 1–10)",
            options=all_game_names,
            max_selections=10,
            help="Pick games you enjoy and we'll find what else you'd love.",
        )
        c_topn = st.slider("Recommendations to show", 3, 20, 5, key="c_topn")
        submitted_c = st.form_submit_button("✨ Recommend For Me", use_container_width=True)

    if submitted_c:
        if not fav_games:
            st.warning("Please select at least one game.")
        else:
            with st.spinner("Finding recommendations based on your favourites…"):
                try:
                    c_recs = recommend_from_favorites(
                        fav_games, game_sim_df, df, top_n=c_topn
                    )
                    st.success(f"✅ Based on your love of: {', '.join(fav_games)}")
                    for i, row in c_recs.iterrows():
                        rec_card(i+1, row["game_name"], row["score"],
                                 row["reason"], "badge-green")
                    st.pyplot(plot_recommendation_scores(
                        c_recs, "score", "Cold-Start Recommendations"
                    ))
                except ValueError as e:
                    st.error(str(e))


# ═══════════════════════════════════════════════════════════════════
#  PAGE: TRENDING GAMES
# ═══════════════════════════════════════════════════════════════════

elif page == "🔥 Trending Games":
    sh("🔥", "Trending Games")
    st.markdown(
        "<p style='color:#7A7A9D'>Games ranked by <b>total playtime</b> + "
        "<b>number of unique players</b>.</p>", unsafe_allow_html=True
    )

    t_topn = st.slider("Show top N games", 5, 30, 15)
    t_df = get_trending_games(df, top_n=t_topn)

    tab1, tab2 = st.tabs(["📊 Chart", "📋 Table"])
    with tab1:
        st.pyplot(plot_trending_games(t_df))
    with tab2:
        display_df = t_df.copy()
        display_df["total_playtime"] = display_df["total_playtime"].apply(lambda x: f"{x:,.0f}h")
        display_df["trending_score"] = display_df["trending_score"].round(4)
        display_df.columns = ["Game", "Total Playtime", "# Players", "Trending Score"]
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: USER CLUSTERS
# ═══════════════════════════════════════════════════════════════════

elif page == "👥 User Clusters":
    sh("👥", "User Segmentation & Clustering")
    st.markdown(
        "<p style='color:#7A7A9D'>KMeans clustering groups users into Casual, Moderate, "
        "and Hardcore players based on total playtime and game variety.</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_user_clusters(cluster_df))
    with col2:
        st.pyplot(plot_cluster_distribution(cluster_df))

    # Summary stats per cluster
    sh("📊", "Cluster Statistics")
    summary = cluster_df.groupby("cluster_label").agg(
        Users=("total_playtime", "count"),
        Avg_Total_Playtime=("total_playtime", "mean"),
        Avg_Game_Variety=("game_variety", "mean"),
        Avg_Session_Hours=("avg_playtime", "mean"),
    ).round(1).reset_index()
    summary.columns = ["Segment", "# Users", "Avg Total Hours", "Avg Games Played", "Avg Hours/Game"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Lookup a specific user
    with st.expander("🔎 Look up a user's cluster"):
        lu_uid = st.number_input(
            "User ID", min_value=int(min(all_user_ids)),
            max_value=int(max(all_user_ids)),
            value=int(all_user_ids[0]), step=1, key="lu_uid"
        )
        if st.button("Check Cluster"):
            if lu_uid in cluster_df.index:
                row = cluster_df.loc[lu_uid]
                badge = {
                    "Casual Player":   "badge-green",
                    "Moderate Player": "badge-orange",
                    "Hardcore Player": "badge-red",
                }.get(row["cluster_label"], "")
                st.markdown(f"""
                <div class='card'>
                    <b>User {lu_uid}</b> &mdash;
                    <span class='badge {badge}'>{row['cluster_label']}</span><br><br>
                    Total Playtime: <b>{row['total_playtime']:,.0f}h</b> &nbsp;|&nbsp;
                    Games Played: <b>{int(row['game_variety'])}</b> &nbsp;|&nbsp;
                    Avg Hours/Game: <b>{row['avg_playtime']:.1f}h</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("User not found in clustering data.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════

elif page == "📊 Model Evaluation":
    sh("📊", "Model Evaluation — Precision@K & Recall@K")
    st.markdown(
        "<p style='color:#7A7A9D'>Evaluates the recommendation model using a hold-out "
        "strategy on a sample of users.</p>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        k_val = st.slider("K (cut-off rank)", 1, 20, 5)
    with col2:
        sample_size = st.slider("Sample users", 10, 500, 100)

    if st.button("▶️ Run Evaluation", use_container_width=False):
        with st.spinner(f"Evaluating on {sample_size} users with K={k_val}…"):
            results = evaluate_model(matrix, user_sim_df, sample_size=sample_size, k=k_val)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{results['precision@k']:.4f}</div>
                <div class='metric-label'>Precision@{k_val}</div></div>""",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{results['recall@k']:.4f}</div>
                <div class='metric-label'>Recall@{k_val}</div></div>""",
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{results['n_users']}</div>
                <div class='metric-label'>Users Evaluated</div></div>""",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <h4 style='margin:0 0 10px; color:#A5B4FC;'>📖 What do these numbers mean?</h4>
            <p style='color:#7A7A9D; margin:0;'>
            <b>Precision@K</b> — of the K games recommended, what fraction did the user
            actually play? Higher = fewer irrelevant recommendations.<br><br>
            <b>Recall@K</b> — of all games the user played (ground truth), what fraction
            appeared in the top-K recommendations? Higher = fewer missed relevant games.<br><br>
            <i>Note: Steam playtime data has implicit feedback (play ≠ love), so moderate
            scores are expected and normal for collaborative filtering on this dataset.</i>
            </p>
        </div>""", unsafe_allow_html=True)
