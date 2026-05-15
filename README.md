# 🎮 GameMind — Hybrid Game Recommendation System

A **production-ready, hybrid game recommendation engine** built with Python and Streamlit. It analyses the Steam 200K dataset to deliver personalised, explainable game recommendations using multiple AI/ML algorithms.

---

## 📌 Project Description

GameMind combines **collaborative filtering**, **item-based similarity**, and **user clustering** into a unified Streamlit dashboard. Every recommendation comes with a human-readable explanation of *why* it was suggested.

---

## 📂 Project Structure

```
game-recommender/
│
├── data/
│   └── steam-200k.csv          # Local dataset (not downloaded)
│
├── preprocess.py               # Data loading, cleaning, matrix building
├── model.py                    # All recommendation algorithms + metrics
├── utils.py                    # Charts, formatting helpers
├── app.py                      # Streamlit UI
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📊 Dataset Details

| Column      | Description                                |
|-------------|--------------------------------------------|
| `user_id`   | Unique Steam user identifier               |
| `game_name` | Name of the game                           |
| `behavior`  | Either `purchase` or `play`                |
| `value`     | Playtime in hours (only when behavior=play)|

**Source:** Steam 200K dataset ([Kaggle](https://www.kaggle.com/tamber/steam-video-games))  
**Size:** 200,000 rows → ~70,000 unique play records after filtering  
**Playtime** is used as an implicit rating proxy (higher playtime = higher preference).

---

## ✨ Features

| # | Feature | Algorithm |
|---|---------|-----------|
| 1 | **User-Based Recommendations** | Cosine similarity on user-game matrix |
| 2 | **Item-Based (Game) Similarity** | Cosine similarity on transposed matrix |
| 3 | **Hybrid Recommendations** | Weighted blend of user + item signals |
| 4 | **Cold-Start Handling** | Favourite game selection → item similarity |
| 5 | **Trending Games** | Ranked by total playtime + unique player count |
| 6 | **User Clustering** | KMeans (Casual / Moderate / Hardcore) |
| 7 | **Evaluation Metrics** | Precision@K, Recall@K with hold-out strategy |
| 8 | **Explainable AI** | Every rec includes a natural-language reason |
| 9 | **Rich Visualisations** | Dark-themed charts via Matplotlib |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place dataset
Ensure the dataset is at:
```
data/steam-200k.csv
```
*(The app auto-detects any `.csv` file in the `data/` folder.)*

### 3. Launch the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 Algorithm Details

### User-Based Collaborative Filtering
1. Build a User × Game playtime matrix.
2. Apply `log1p` normalisation to reduce extreme playtime dominance.
3. Compute pairwise cosine similarity between users.
4. For a target user, find the top-10 most similar users.
5. Compute weighted-average playtime scores across unseen games.
6. Return top-N games with the highest scores.

### Item-Based Similarity
1. Transpose the matrix to get a Game × User view.
2. Compute cosine similarity between all games.
3. For a seed game, return the top-N most similar games.

### Hybrid Engine
- `hybrid_score = 0.6 × user_score_norm + 0.4 × item_score_norm`
- Weights are configurable in the UI.

### Cold Start
- New users pick favourite games from a dropdown.
- Item-based similarity is applied to each favourite.
- Scores are aggregated and top-N returned.

### Trending Games
- `trending_score = (norm_total_playtime + norm_user_count) / 2`

### User Clustering (KMeans, k=3)
Features: `total_playtime`, `game_variety`, `avg_playtime`
Labels assigned by ascending cluster centroid playtime:
- Cluster 0 → Casual Player
- Cluster 1 → Moderate Player
- Cluster 2 → Hardcore Player

---

## 📈 Sample Outputs

```
Top 5 Hybrid Recommendations for User 151603712:
  1. Counter-Strike        score: 0.8421  reason: similar users' playtime patterns & games you've played
  2. Left 4 Dead 2         score: 0.7935  reason: similarity to games you've played
  3. Portal 2              score: 0.7210  reason: similar users' playtime patterns
  4. Team Fortress 2       score: 0.6854  reason: similar users' playtime patterns & games you've played
  5. Garry's Mod           score: 0.6540  reason: similarity to games you've played
```

```
Evaluation Results (K=5, 100 users):
  Precision@5 : 0.0340
  Recall@5    : 0.0180
```
*(Low values are expected for implicit feedback datasets — they do not reflect real-world CTR.)*

---

## 🛠️ Tech Stack

| Technology   | Purpose                         |
|--------------|---------------------------------|
| Python 3.9+  | Core language                   |
| Pandas       | Data manipulation               |
| NumPy        | Numerical operations            |
| Scikit-learn | Cosine similarity, KMeans       |
| Streamlit    | Interactive web UI              |
| Matplotlib   | Visualisations                  |

---

## 🎯 For GitHub & Resume

This project demonstrates:
- End-to-end ML pipeline (data → model → UI)
- Multiple recommendation paradigms
- Clean modular code with error handling
- Production-ready Streamlit dashboard with explainability

---

*Built with ❤️ using Python & Streamlit*
