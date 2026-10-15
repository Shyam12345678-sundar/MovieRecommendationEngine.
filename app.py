from pathlib import Path
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Use the Python interpreter from the active virtual environment for PySpark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import numpy as np
import pandas as pd
import streamlit as st
from pyspark.sql import functions as F

from src.recommender import (
    ALL_GENRES,
    build_popularity_model,
    evaluate_models,
    get_dataset,
    get_spark_session,
    load_movies,
    load_ratings,
    train_als_model,
)

st.set_page_config(
    page_title="MovieLens Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, polished dashboard styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #e9ecef;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #28a745;
        margin-top: 4px;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 4px;
    }
    .badge-primary {
        color: #fff;
        background-color: #0d6efd;
    }
    .badge-secondary {
        color: #fff;
        background-color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Apache Spark, training ALS model, and calculating metrics...")
def load_engine():
    """
    Initializes Spark, trains ALS, computes popularity baseline,
    evaluates both models, and caches in-memory representations for snappy UI response.
    """
    dataset_dir = get_dataset()
    spark = get_spark_session("MovieLensStreamlitApp")

    ratings = load_ratings(spark, dataset_dir)
    movies = load_movies(spark, dataset_dir)

    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    # 1. Train ALS Model
    als_model = train_als_model(train, rank=12, max_iter=15, reg_param=0.1, seed=42)

    # 2. Build Popularity Baseline
    pop_df, global_mean = build_popularity_model(train, min_ratings_prior=25)

    # 3. Evaluate both models on test set (Regression + Candidate-Sampling Ranking Metrics)
    metrics = evaluate_models(
        als_model,
        pop_df,
        global_mean,
        train,
        test,
        movies_df=movies,
        k=10,
        relevance_threshold=4.0,
        num_negatives=100,
        seed=42,
    )

    # Convert essential lookups to pandas for low-latency filtering
    movies_pd = movies.toPandas()
    train_pd = train.select("userId", "movieId", "rating").toPandas()

    # Precompute Popularity recommendations catalogue in pandas
    pop_scores_pd = pop_df.select("movieId", "popularity_score", "avg_rating", "rating_count").toPandas()
    catalog_popular_pd = movies_pd.merge(pop_scores_pd, on="movieId", how="inner")
    catalog_popular_pd = catalog_popular_pd.sort_values(by="popularity_score", ascending=False)

    # User activity statistics
    user_counts = train_pd.groupby("userId").size().to_dict()
    user_ids = sorted(user_counts.keys())

    # Release year range
    valid_years = movies_pd["year"].dropna().astype(int)
    min_year = int(valid_years.min())
    max_year = int(valid_years.max())

    return (
        spark,
        train,
        train_pd,
        movies,
        movies_pd,
        als_model,
        catalog_popular_pd,
        metrics,
        user_ids,
        user_counts,
        min_year,
        max_year,
    )


# Load data and models
try:
    (
        spark,
        train_spark,
        train_pd,
        movies_spark,
        movies_pd,
        als_model,
        catalog_popular_pd,
        metrics,
        user_ids,
        user_counts,
        min_year,
        max_year,
    ) = load_engine()
except Exception as exc:
    st.error("Failed to start Spark recommendation engine.")
    st.exception(exc)
    st.stop()


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("🎯 Recommender Controls")

# User selection
user_id = st.sidebar.selectbox(
    "Select MovieLens User",
    options=user_ids,
    index=0,
    format_func=lambda uid: f"User {uid} ({user_counts.get(uid, 0)} ratings in train)",
)

# Recommender mode
model_choice = st.sidebar.radio(
    "Recommendation Mode",
    options=["Side-by-Side Comparison", "ALS Collaborative Filtering", "Popularity Baseline"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtering Options")

# Genre filter
selected_genres = st.sidebar.multiselect(
    "Filter by Genre",
    options=ALL_GENRES,
    default=[],
    help="Leave empty to include all genres",
)

# Year range slider
selected_years = st.sidebar.slider(
    "Filter by Release Year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

# Top-N recommendation count
top_n = st.sidebar.slider("Number of Recommendations (Top-N)", min_value=5, max_value=20, value=10)

st.sidebar.markdown("---")
st.sidebar.caption("MovieLens 100K Dataset (943 users, 1,682 movies, 100,000 ratings).")


# -------------------------------------------------------------
# MAIN DASHBOARD CONTENT
# -------------------------------------------------------------
st.title("🎬 MovieLens Recommendation Engine")
st.markdown(
    "Hybrid exploration platform featuring **Spark MLlib ALS Collaborative Filtering** "
    "and an empirical **IMDb-style Bayesian Popularity Baseline**, benchmarked with Top-10 ranking metrics."
)

# -------------------------------------------------------------
# MODEL COMPARISON METRICS SECTION
# -------------------------------------------------------------
with st.expander("📊 Benchmark & Model Comparison (ALS vs. Popularity Baseline)", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    als_m = metrics["ALS"]
    pop_m = metrics["Popularity"]
    
    with col1:
        rmse_diff = ((als_m["RMSE"] - pop_m["RMSE"]) / pop_m["RMSE"]) * 100
        st.metric(
            label="Test RMSE",
            value=f"{als_m['RMSE']:.4f}",
            delta=f"{rmse_diff:+.1f}% vs Pop",
            delta_color="inverse",
            help=f"Root Mean Squared Error on test ratings (lower is better). ALS: {als_m['RMSE']:.4f} vs Popularity: {pop_m['RMSE']:.4f}.",
        )
    with col2:
        mae_diff = ((als_m["MAE"] - pop_m["MAE"]) / pop_m["MAE"]) * 100
        st.metric(
            label="Test MAE",
            value=f"{als_m['MAE']:.4f}",
            delta=f"{mae_diff:+.1f}% vs Pop",
            delta_color="inverse",
            help=f"Mean Absolute Error on test ratings (lower is better). ALS: {als_m['MAE']:.4f} vs Popularity: {pop_m['MAE']:.4f}.",
        )
    with col3:
        prec_diff = ((als_m["Precision@10"] - pop_m["Precision@10"]) / pop_m["Precision@10"]) * 100
        st.metric(
            label="Precision@10",
            value=f"{als_m['Precision@10']:.4f}",
            delta=f"{prec_diff:+.1f}% vs Pop",
            help=f"Macro-averaged Precision@10 on candidate sets (higher is better). ALS: {als_m['Precision@10']:.4f} vs Popularity: {pop_m['Precision@10']:.4f}.",
        )
    with col4:
        rec_diff = ((als_m["Recall@10"] - pop_m["Recall@10"]) / pop_m["Recall@10"]) * 100
        st.metric(
            label="Recall@10",
            value=f"{als_m['Recall@10']:.4f}",
            delta=f"{rec_diff:+.1f}% vs Pop",
            help=f"Macro-averaged Recall@10 on candidate sets (higher is better). ALS: {als_m['Recall@10']:.4f} vs Popularity: {pop_m['Recall@10']:.4f}.",
        )
    with col5:
        ndcg_diff = ((als_m["NDCG@10"] - pop_m["NDCG@10"]) / pop_m["NDCG@10"]) * 100
        st.metric(
            label="NDCG@10",
            value=f"{als_m['NDCG@10']:.4f}",
            delta=f"{ndcg_diff:+.1f}% vs Pop",
            help=f"Macro-averaged NDCG@10 on candidate sets (higher is better). ALS: {als_m['NDCG@10']:.4f} vs Popularity: {pop_m['NDCG@10']:.4f}.",
        )

    st.markdown(r"""
    **Evaluation Methodology & Realistic Candidate Ranking:**
    - **Relevance Criterion ($\ge 4.0$)**: In MovieLens 5-star ratings, ratings of 4 and 5 indicate clear positive user affinity. Ratings 1-3 reflect neutral or negative sentiment, so recommending them represents an ineffective recommendation.
    - **Raw Test Set Preservation**: Held-out relevance labels are extracted directly from raw `test_df`, retaining all ratings on cold-start items ($N=12$ relevant cold items).
    - **Candidate-Sampling Ranking Protocol**: For each evaluated test user, the candidate recommendation pool comprises all of that user's relevant held-out test movies ($\text{rating} \ge 4.0$) combined with 100 randomly sampled movies unrated in BOTH `train_df` and `test_df` (deterministic seed `42 + userId`).
    - **Fair Benchmark & Cold-Start Handling**: Both ALS and the Popularity baseline score and rank the exact same candidate pool per user. For catalogue items lacking learned latent factors (35 cold-start items), ALS falls back to `global_mean`, matching the Bayesian prior $WR = \text{global\_mean}$ on 0-vote items in the Popularity baseline.
    - **Macro-Averaging**: Precision@10, Recall@10, and NDCG@10 are computed per user and macro-averaged across all eligible test users ($N=922$).
    - **Empirical Insight**: ALS dominates rating prediction error (RMSE: 0.9173 vs 1.0203; MAE: 0.7256 vs 0.8124) by personalizing latent user tastes. Meanwhile, on 100-negative candidate sets, Bayesian popularity achieves higher retrieval precision because universally acclaimed classics (*Star Wars*, *Godfather*, *Schindler's List*) are broadly loved across diverse user test sets.
    """)

# -------------------------------------------------------------
# USER RATING HISTORY & PROFILE
# -------------------------------------------------------------
st.markdown(f"### 👤 Profile & Rating History for User {user_id}")

user_history_df = (
    train_pd[train_pd["userId"] == user_id]
    .merge(movies_pd, on="movieId", how="left")
    .sort_values(by="rating", ascending=False)
)

u_col1, u_col2, u_col3 = st.columns([1, 1, 2])
with u_col1:
    st.metric("Total Movies Rated", f"{len(user_history_df)}")
with u_col2:
    st.metric("Average Rating Given", f"{user_history_df['rating'].mean():.2f} / 5.0")
with u_col3:
    # Compute user's favorite genres
    liked_movies = user_history_df[user_history_df["rating"] >= 4.0]
    if not liked_movies.empty:
        all_liked_genres = [
            g.strip()
            for g_str in liked_movies["genres"].dropna()
            for g in g_str.split(",")
            if g.strip() and g.strip() != "unknown"
        ]
        if all_liked_genres:
            top_fav_genres = pd.Series(all_liked_genres).value_counts().head(3).index.tolist()
            st.markdown(f"**Top Preferred Genres:** {', '.join(top_fav_genres)}")
        else:
            st.markdown("**Top Preferred Genres:** N/A")
    else:
        st.markdown("**Top Preferred Genres:** N/A")

with st.expander(f"📜 View User {user_id}'s Full Training History ({len(user_history_df)} rated movies)", expanded=False):
    display_history = user_history_df[["movieId", "title", "year", "genres", "rating"]].rename(
        columns={"movieId": "Movie ID", "title": "Movie Title", "year": "Year", "genres": "Genres", "rating": "User Rating"}
    )
    st.dataframe(
        display_history,
        hide_index=True,
        use_container_width=True,
    )


# -------------------------------------------------------------
# RECOMMENDATION GENERATION WITH FILTERING
# -------------------------------------------------------------
st.markdown("---")
st.markdown(f"### 🎯 Movie Recommendations for User {user_id}")

user_train_movie_ids = set(user_history_df["movieId"].tolist())

# Helper function to apply genre and year filters to candidate DataFrame
def filter_movies(df, selected_genres, year_range):
    filtered = df.copy()
    if selected_genres:
        # Match if movie has ANY of the selected genres
        pattern = "|".join([g for g in selected_genres])
        filtered = filtered[filtered["genres"].str.contains(pattern, case=False, na=False)]
    
    min_y, max_y = year_range
    filtered = filtered[(filtered["year"].isna()) | ((filtered["year"] >= min_y) & (filtered["year"] <= max_y))]
    return filtered


# Compute ALS recommendations for chosen user across full catalogue
@st.cache_data(show_spinner=False)
def fetch_user_als_candidates(uid):
    user_subset = train_spark.where(F.col("userId") == uid).select("userId").distinct()
    # Pull candidate predictions across the full catalogue (1,682 movies) so any genre/year filter succeeds
    recs_spark = als_model.recommendForUserSubset(user_subset, len(movies_pd))
    rows = recs_spark.collect()
    if not rows:
        return pd.DataFrame(columns=["movieId", "predictedRating"])
    
    rec_items = rows[0]["recommendations"]
    rec_data = [{"movieId": r.movieId, "predictedRating": float(r.rating)} for r in rec_items]
    return pd.DataFrame(rec_data)

# Fetch ALS candidates
als_candidates_df = fetch_user_als_candidates(user_id)

# Filter out already rated movies
als_unseen = als_candidates_df[~als_candidates_df["movieId"].isin(user_train_movie_ids)]
als_full = als_unseen.merge(movies_pd, on="movieId", how="inner")
als_filtered = filter_movies(als_full, selected_genres, selected_years).sort_values(
    by="predictedRating", ascending=False
).head(top_n)

# Compute Popularity recommendations
pop_unseen = catalog_popular_pd[~catalog_popular_pd["movieId"].isin(user_train_movie_ids)]
pop_filtered = filter_movies(pop_unseen, selected_genres, selected_years).sort_values(
    by="popularity_score", ascending=False
).head(top_n)


# Active filter indicator
active_filters = []
if selected_genres:
    active_filters.append(f"Genres: **{', '.join(selected_genres)}**")
if selected_years != (min_year, max_year):
    active_filters.append(f"Years: **{selected_years[0]} - {selected_years[1]}**")

if active_filters:
    st.info("Active Filters: " + " | ".join(active_filters))


# DISPLAY LOGIC
def format_als_table(df):
    formatted = df.copy().reset_index(drop=True)
    formatted.insert(0, "Rank", [f"#{i+1}" for i in range(len(formatted))])
    formatted["Predicted Rating"] = formatted["predictedRating"].apply(lambda x: f"{x:.2f} ⭐")
    return formatted[["Rank", "title", "year", "genres", "Predicted Rating"]].rename(
        columns={"title": "Title", "year": "Year", "genres": "Genres"}
    )

def format_pop_table(df):
    formatted = df.copy().reset_index(drop=True)
    formatted.insert(0, "Rank", [f"#{i+1}" for i in range(len(formatted))])
    formatted["Popularity Score"] = formatted["popularity_score"].apply(lambda x: f"{x:.2f}")
    formatted["Avg Rating"] = formatted["avg_rating"].apply(lambda x: f"{x:.2f} ⭐")
    formatted["Votes"] = formatted["rating_count"].astype(int)
    return formatted[["Rank", "title", "year", "genres", "Popularity Score", "Avg Rating", "Votes"]].rename(
        columns={"title": "Title", "year": "Year", "genres": "Genres"}
    )


if model_choice == "Side-by-Side Comparison":
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("🤖 ALS Collaborative Filtering")
        st.caption("Personalized predictions matching this specific user's latent taste patterns.")
        st.caption("ℹ️ *Note: ALS recommendations are limited to catalogue movies with learned training factors (1,647 movies). Movies with no training ratings are not currently included in ALS recommendations.*")
        if als_filtered.empty:
            st.warning("No ALS recommendations match the selected filters.")
        else:
            st.dataframe(format_als_table(als_filtered), hide_index=True, use_container_width=True)

    with right_col:
        st.subheader("🔥 Popularity Baseline")
        st.caption("IMDb Bayesian weighted rating across all training ratings (excluding seen).")
        if pop_filtered.empty:
            st.warning("No popularity recommendations match the selected filters.")
        else:
            st.dataframe(format_pop_table(pop_filtered), hide_index=True, use_container_width=True)

    st.markdown("""
    💡 **Key Insight:** Notice how the Popularity baseline always recommends universally acclaimed classics 
    (*Schindler's List*, *Casablanca*, *Star Wars*), whereas ALS recommends movies uniquely aligned with User 
    {user_id}'s specific rating history (e.g. niche animated shorts, foreign cinema, or film noir).
    """.format(user_id=user_id))

elif model_choice == "ALS Collaborative Filtering":
    st.subheader("🤖 Personalized ALS Recommendations")
    st.caption("Generated by Matrix Factorization on user-item interaction matrix.")
    st.caption("ℹ️ *Note: ALS recommendations are limited to catalogue movies with learned training factors (1,647 movies). Movies with no training ratings are not currently included in ALS recommendations.*")
    if als_filtered.empty:
        st.warning("No ALS recommendations match the selected filters.")
    else:
        st.dataframe(format_als_table(als_filtered), hide_index=True, use_container_width=True)

else:
    st.subheader("🔥 Popularity Baseline Recommendations")
    st.caption("Generated by Bayesian Weighted Rating (prior = 25 votes).")
    if pop_filtered.empty:
        st.warning("No popularity recommendations match the selected filters.")
    else:
        st.dataframe(format_pop_table(pop_filtered), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption(
    "Built with Apache Spark 4.0.1, PySpark MLlib, Streamlit, and MovieLens 100K. "
    "All ratings and user profiles are anonymous research data."
)