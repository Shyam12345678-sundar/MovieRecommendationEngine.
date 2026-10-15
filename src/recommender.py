from pathlib import Path
import os
import sys
import urllib.request
import zipfile
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Use the Python interpreter from the active virtual environment.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StructField,
    StructType,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ZIP_PATH = DATA_DIR / "ml-100k.zip"
DATASET_DIR = DATA_DIR / "ml-100k"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), nullable=False),
    StructField("movieId", IntegerType(), nullable=False),
    StructField("rating", FloatType(), nullable=False),
    StructField("timestamp", LongType(), nullable=True),
])

GENRE_MAP = [
    ("_c5", "unknown"),
    ("_c6", "Action"),
    ("_c7", "Adventure"),
    ("_c8", "Animation"),
    ("_c9", "Children's"),
    ("_c10", "Comedy"),
    ("_c11", "Crime"),
    ("_c12", "Documentary"),
    ("_c13", "Drama"),
    ("_c14", "Fantasy"),
    ("_c15", "Film-Noir"),
    ("_c16", "Horror"),
    ("_c17", "Musical"),
    ("_c18", "Mystery"),
    ("_c19", "Romance"),
    ("_c20", "Sci-Fi"),
    ("_c21", "Thriller"),
    ("_c22", "War"),
    ("_c23", "Western"),
]

ALL_GENRES = [name for _, name in GENRE_MAP if name != "unknown"]


def get_dataset():
    """Unpack MovieLens 100K, downloading it if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not (DATASET_DIR / "u.data").exists():
        if not ZIP_PATH.exists():
            print("Downloading MovieLens 100K...")
            urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

        print("Unpacking dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(DATA_DIR)

    return DATASET_DIR


def get_spark_session(app_name="MovieLensRecommendationEngine"):
    """Initialize and return a configured SparkSession."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_ratings(spark, dataset_dir):
    """Load ratings data from u.data."""
    return (
        spark.read
        .option("sep", "\t")
        .schema(RATINGS_SCHEMA)
        .csv(str(dataset_dir / "u.data"))
        .select("userId", "movieId", "rating", "timestamp")
    )


def load_movies(spark, dataset_dir):
    """
    Load movie catalog from u.item with extracted release year and genre lists.
    Uses native Spark SQL expressions to avoid JVM-Python worker overhead.
    """
    genre_exprs = [
        F.when(F.col(c) == "1", F.lit(name))
        for c, name in GENRE_MAP
    ]

    return (
        spark.read
        .option("sep", "|")
        .option("encoding", "ISO-8859-1")
        .csv(str(dataset_dir / "u.item"))
        .select(
            F.col("_c0").cast("int").alias("movieId"),
            F.col("_c1").alias("title"),
            F.coalesce(
                F.nullif(F.regexp_extract(F.col("_c1"), r"\((\d{4})\)", 1), F.lit("")),
                F.nullif(F.regexp_extract(F.col("_c2"), r"(\d{4})$", 1), F.lit(""))
            ).cast("int").alias("year"),
            F.concat_ws(", ", *genre_exprs).alias("genres")
        )
    )


def train_als_model(train_df, rank=12, max_iter=15, reg_param=0.1, seed=42):
    """Train an Alternating Least Squares (ALS) collaborative filtering model."""
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        coldStartStrategy="drop",
        seed=seed,
    )
    return als.fit(train_df)


def build_popularity_model(train_df, min_ratings_prior=25):
    """
    Build a popularity baseline using training ratings.
    Computes per-movie vote counts, average rating, and Bayesian weighted rating:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where v is movie rating count, m is the prior threshold, R is movie mean rating,
    and C is the global mean rating across all training ratings.
    """
    global_mean = float(train_df.select(F.avg("rating")).first()[0])

    stats = train_df.groupBy("movieId").agg(
        F.avg("rating").alias("avg_rating"),
        F.count("rating").alias("rating_count")
    )

    m = float(min_ratings_prior)
    c_lit = F.lit(global_mean)
    v_col = F.col("rating_count")
    r_col = F.col("avg_rating")

    weighted_score = (v_col / (v_col + m)) * r_col + (m / (v_col + m)) * c_lit

    popularity_df = stats.withColumn("popularity_score", weighted_score)
    return popularity_df, global_mean


def compute_ranking_metrics_for_user(ranked_items, relevant_items, k=10):
    """
    Compute Precision@K, Recall@K, and NDCG@K for a single user given a ranked list
    and a ground-truth set of relevant items.
    """
    top_k = ranked_items[:k]
    hits = [1 if item in relevant_items else 0 for item in top_k]

    precision = sum(hits) / float(k)
    recall = sum(hits) / float(len(relevant_items)) if len(relevant_items) > 0 else 0.0

    # Discounted Cumulative Gain (DCG@K)
    dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))

    # Ideal Discounted Cumulative Gain (IDCG@K)
    ideal_hits_count = min(k, len(relevant_items))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits_count))

    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, ndcg


def evaluate_models(
    als_model,
    popularity_df,
    global_mean,
    train_df,
    test_df,
    movies_df=None,
    k=10,
    relevance_threshold=4.0,
    num_negatives=100,
    seed=42,
):
    """
    Evaluate ALS and Popularity Baseline on:
    1. Rating Regression: RMSE and MAE on test set.
       Evaluated strictly on test rows for which ALS produced a prediction.
       ALS uses coldStartStrategy="drop", excluding test rows for cold-start movies
       (movies with no training ratings). Popularity is evaluated on the exact same
       test rows for a fair, controlled comparison.
    2. Top-K Ranking: Macro-averaged Precision@K, Recall@K, and NDCG@K using a
       realistic candidate-sampling protocol.
       - Ground-truth relevance: Held-out test ratings >= relevance_threshold (default 4.0)
         drawn directly from raw test_df (preserving all held-out ratings).
         Rationale: In MovieLens 1-5 star scale, ratings 4-5 reflect positive affinity/satisfaction,
         whereas ratings 1-3 reflect neutral or negative sentiment.
       - Candidate set per user: Evaluates sampled candidates rather than the entire catalogue:
         all relevant held-out items (test rating >= 4.0) plus a reproducible random sample
         of num_negatives movies (default 100) that the user rated in neither the training split
         nor the test split (unobserved in both splits).
       - Both models score and rank the exact same candidate set per user.
       - ALS handles cold-start candidate movies by falling back to the global training mean.
       - Evaluates macro-average across all users with at least 1 relevant held-out item.
    """
    # 1. Rating Regression Evaluation on test set
    als_test_preds = als_model.transform(test_df)

    evaluator_rmse = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
    als_rmse = evaluator_rmse.evaluate(als_test_preds)

    evaluator_mae = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="mae")
    als_mae = evaluator_mae.evaluate(als_test_preds)

    # Note: als_test_preds drops cold-start rows because coldStartStrategy='drop'.
    # For rating regression, compute popularity RMSE/MAE on the same evaluated test rows:
    test_eval_pd = als_test_preds.select("userId", "movieId", "rating", "prediction").toPandas()
    train_pd = train_df.select("userId", "movieId", "rating").toPandas()

    pop_pd = popularity_df.select("movieId", "avg_rating", "popularity_score").toPandas()
    pop_score_map = dict(zip(pop_pd["movieId"], pop_pd["popularity_score"]))
    pop_avg_map = dict(zip(pop_pd["movieId"], pop_avg_rating := pop_pd["avg_rating"]))

    test_eval_pd["pop_rating_pred"] = test_eval_pd["movieId"].map(pop_avg_map).fillna(global_mean)
    pop_rmse = float(np.sqrt(((test_eval_pd["rating"] - test_eval_pd["pop_rating_pred"]) ** 2).mean()))
    pop_mae = float(np.abs(test_eval_pd["rating"] - test_eval_pd["pop_rating_pred"]).mean())

    # 2. Ranking Evaluation with Realistic Candidate Sampling
    # Raw test ratings preserve ALL held-out items, including cold-start items
    raw_test_pd = test_df.select("userId", "movieId", "rating").toPandas()

    if movies_df is not None:
        all_movie_ids = np.array(sorted([int(r["movieId"]) for r in movies_df.select("movieId").distinct().collect()]))
    else:
        all_movie_ids = np.array(sorted(list(set(train_pd["movieId"]).union(set(raw_test_pd["movieId"])))))

    train_mids = set(train_pd["movieId"].unique())
    cold_items = set(all_movie_ids) - train_mids
    cold_start_items_count = len(cold_items)

    # Pre-extract latent factor representations for fast, exact dot-product scoring
    u_factors = {int(row["id"]): np.array(row["features"]) for row in als_model.userFactors.collect()}
    i_factors = {int(row["id"]): np.array(row["features"]) for row in als_model.itemFactors.collect()}

    train_user_items = train_pd.groupby("userId")["movieId"].apply(set).to_dict()
    test_user_items = raw_test_pd.groupby("userId")["movieId"].apply(set).to_dict()
    grouped_test = raw_test_pd.groupby("userId")

    als_precisions, als_recalls, als_ndcgs = [], [], []
    pop_precisions, pop_recalls, pop_ndcgs = [], [], []
    cold_in_relevant_count = 0
    cold_in_candidates_count = 0

    for uid, group in grouped_test:
        # Build held-out relevance labels from the raw test_df (relevance threshold >= 4.0)
        rel_items = set(group[group["rating"] >= relevance_threshold]["movieId"])
        if not rel_items:
            continue

        cold_rel = rel_items.intersection(cold_items)
        if cold_rel:
            cold_in_relevant_count += len(cold_rel)

        # Exclude every movie the user rated in BOTH train_df and test_df when sampling negatives
        train_seen = train_user_items.get(uid, set())
        test_seen = test_user_items.get(uid, set())
        all_seen = train_seen.union(test_seen)

        # Unobserved catalogue pool (completely unrated by this user)
        unobserved = np.setdiff1d(all_movie_ids, list(all_seen))

        # Draw a reproducible random sample of num_negatives unobserved movies
        rng = np.random.RandomState(seed + int(uid))
        if len(unobserved) <= num_negatives:
            negatives = unobserved
        else:
            negatives = rng.choice(unobserved, size=num_negatives, replace=False)

        # Candidate pool: user's relevant held-out items + sampled unobserved negatives
        candidates = list(rel_items) + list(negatives)

        cold_cand = set(candidates).intersection(cold_items)
        if cold_cand:
            cold_in_candidates_count += len(cold_cand)

        # Score candidates with ALS:
        # Explicit cold-start handling: if mid lacks learned item factors, fallback to global_mean
        u_vec = u_factors.get(uid)
        if u_vec is not None:
            als_scores = [
                float(np.dot(u_vec, i_factors[mid])) if mid in i_factors else float(global_mean)
                for mid in candidates
            ]
        else:
            als_scores = [float(global_mean) for _ in candidates]

        # Score candidates with Popularity:
        # Bayesian weighted rating (for 0-rating cold items, mathematically equals global_mean)
        pop_scores = [float(pop_score_map.get(mid, global_mean)) for mid in candidates]

        # Rank candidate set with ALS
        als_ranked = [candidates[idx] for idx in np.argsort(als_scores)[::-1]]
        p_a, r_a, n_a = compute_ranking_metrics_for_user(als_ranked, rel_items, k=k)
        als_precisions.append(p_a)
        als_recalls.append(r_a)
        als_ndcgs.append(n_a)

        # Rank exact same candidate set with Popularity
        pop_ranked = [candidates[idx] for idx in np.argsort(pop_scores)[::-1]]
        p_p, r_p, n_p = compute_ranking_metrics_for_user(pop_ranked, rel_items, k=k)
        pop_precisions.append(p_p)
        pop_recalls.append(r_p)
        pop_ndcgs.append(n_p)

    metrics = {
        "ALS": {
            "RMSE": als_rmse,
            "MAE": als_mae,
            f"Precision@{k}": float(np.mean(als_precisions)),
            f"Recall@{k}": float(np.mean(als_recalls)),
            f"NDCG@{k}": float(np.mean(als_ndcgs)),
            "evaluated_users": len(als_precisions),
        },
        "Popularity": {
            "RMSE": pop_rmse,
            "MAE": pop_mae,
            f"Precision@{k}": float(np.mean(pop_precisions)),
            f"Recall@{k}": float(np.mean(pop_recalls)),
            f"NDCG@{k}": float(np.mean(pop_ndcgs)),
            "evaluated_users": len(pop_precisions),
        },
        "metadata": {
            "total_test_rows": len(raw_test_pd),
            "evaluated_regression_rows": len(test_eval_pd),
            "dropped_cold_start_rows": len(raw_test_pd) - len(test_eval_pd),
            "eligible_users": len(als_precisions),
            "cold_start_items_in_catalogue": cold_start_items_count,
            "cold_start_in_relevant_sets": cold_in_relevant_count,
            "cold_start_in_candidate_sets": cold_in_candidates_count,
            "relevance_threshold": relevance_threshold,
            "num_negatives": num_negatives,
            "seed": seed,
            "k": k,
        }
    }
    return metrics


def get_als_recommendations(model, user_id, train_df, movies_df, top_n=10, genre=None, min_year=None, max_year=None):
    """
    Generate top-N personalized movie recommendations for a user with ALS,
    excluding movies already rated in the training set, with optional genre and year filters.
    """
    user_subset = train_df.where(F.col("userId") == user_id).select("userId").distinct()
    already_rated = train_df.where(F.col("userId") == user_id).select("movieId").distinct()

    catalogue_count = movies_df.count()
    recs = (
        model.recommendForUserSubset(user_subset, catalogue_count)
        .select("userId", F.explode("recommendations").alias("rec"))
        .select(
            F.col("rec.movieId").alias("movieId"),
            F.col("rec.rating").alias("predictedRating"),
        )
        .join(already_rated, on="movieId", how="left_anti")
        .join(movies_df, on="movieId", how="inner")
    )

    if genre and genre != "All":
        recs = recs.where(F.col("genres").contains(genre))
    if min_year is not None:
        recs = recs.where(F.col("year") >= min_year)
    if max_year is not None:
        recs = recs.where(F.col("year") <= max_year)

    return recs.orderBy(F.desc("predictedRating")).limit(top_n)


def get_popular_recommendations(popularity_df, user_id, train_df, movies_df, top_n=10, genre=None, min_year=None, max_year=None):
    """
    Generate top-N popularity-based movie recommendations for a user,
    excluding movies already rated in the training set, with optional genre and year filters.
    """
    already_rated = train_df.where(F.col("userId") == user_id).select("movieId").distinct()

    recs = (
        popularity_df
        .join(already_rated, on="movieId", how="left_anti")
        .join(movies_df, on="movieId", how="inner")
        .select(
            "movieId",
            "title",
            "year",
            "genres",
            F.col("popularity_score").alias("score"),
            "avg_rating",
            "rating_count"
        )
    )

    if genre and genre != "All":
        recs = recs.where(F.col("genres").contains(genre))
    if min_year is not None:
        recs = recs.where(F.col("year") >= min_year)
    if max_year is not None:
        recs = recs.where(F.col("year") <= max_year)

    return recs.orderBy(F.desc("score")).limit(top_n)


def main():
    dataset_dir = get_dataset()
    spark = get_spark_session()

    try:
        ratings = load_ratings(spark, dataset_dir)
        movies = load_movies(spark, dataset_dir)

        total_ratings = ratings.count()
        total_movies = movies.count()
        total_users = ratings.select("userId").distinct().count()

        print("=" * 70)
        print("MOVIELENS 100K RECOMMENDATION ENGINE")
        print("=" * 70)
        print(f"Dataset summary: {total_ratings:,} ratings | {total_users:,} users | {total_movies:,} movies")

        # 80/20 train/test split with fixed seed for reproducible evaluation
        train, test = ratings.randomSplit([0.8, 0.2], seed=42)
        print(f"Training split:  {train.count():,} rows")
        print(f"Held-out test:   {test.count():,} rows")

        # 1. Train ALS Collaborative Filtering
        print("\n[1/3] Training ALS Collaborative Filtering Model...")
        als_model = train_als_model(train, rank=12, max_iter=15, reg_param=0.1, seed=42)

        # 2. Build Popularity Baseline
        print("[2/3] Building Popularity-based Baseline (Bayesian Weighted Rating)...")
        pop_df, global_mean = build_popularity_model(train, min_ratings_prior=25)

        # 3. Comprehensive Model Evaluation
        print("[3/3] Evaluating Models (Regression + Top-10 Candidate-Ranking Metrics)...")
        print("      * Ground-truth relevant: held-out test ratings >= 4.0 from raw test split.")
        print("      * Unobserved negatives: 100 randomly sampled movies unrated in BOTH train & test (seed=42).")
        print("      * Candidate set per user: relevant held-out items + 100 sampled unobserved negatives.")
        print("      * ALS cold-start candidate fallback: global training mean (retains all candidate items).")
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

        meta = metrics["metadata"]
        print(f"      * Evaluated regression test rows: {meta['evaluated_regression_rows']:,} / {meta['total_test_rows']:,} ({meta['dropped_cold_start_rows']} cold-start rows dropped by ALS coldStartStrategy)")
        print(f"      * Evaluated eligible users: {meta['eligible_users']} (users with >= 1 held-out test rating >= 4.0)")
        print(f"      * Cold-start catalogue items: {meta['cold_start_items_in_catalogue']} (0 training ratings; scored with global-mean fallback)")
        print(f"      * Cold-start items in relevant sets: {meta['cold_start_in_relevant_sets']} | in candidate pools: {meta['cold_start_in_candidate_sets']}")

        # Print Model Comparison Table
        print("\n" + "=" * 70)
        print("MODEL EVALUATION & COMPARISON SUMMARY")
        print("=" * 70)
        header = f"{'Metric':<18} | {'ALS (Collaborative)':<22} | {'Popularity Baseline':<20} | {'Winner'}"
        print(header)
        print("-" * len(header))

        for metric in ["RMSE", "MAE", "Precision@10", "Recall@10", "NDCG@10"]:
            val_als = metrics["ALS"][metric]
            val_pop = metrics["Popularity"][metric]
            
            # For error metrics, lower is better; for ranking metrics, higher is better
            if metric in ["RMSE", "MAE"]:
                winner = "ALS (Lower error)" if val_als < val_pop else "Popularity"
            else:
                winner = "ALS (Higher quality)" if val_als > val_pop else "Popularity"

            print(f"{metric:<18} | {val_als:<22.4f} | {val_pop:<20.4f} | {winner}")

        print("=" * 70)

        # Sample user recommendations
        sample_user_id = 1
        print(f"\n[+] Top 10 Personalized Recommendations for User {sample_user_id} (ALS):")
        als_recs = get_als_recommendations(als_model, sample_user_id, train, movies, top_n=10)
        als_recs.select(
            "movieId",
            "title",
            "year",
            "genres",
            F.round("predictedRating", 2).alias("predictedRating")
        ).show(truncate=False)

        print(f"[+] Top 10 Popularity-Based Recommendations for User {sample_user_id} (Baseline):")
        pop_recs = get_popular_recommendations(pop_df, sample_user_id, train, movies, top_n=10)
        pop_recs.select(
            "movieId",
            "title",
            "year",
            "genres",
            F.round("score", 2).alias("bayesianScore"),
            F.round("avg_rating", 2).alias("avgRating"),
            "rating_count"
        ).show(truncate=False)

    finally:
        spark.stop()


