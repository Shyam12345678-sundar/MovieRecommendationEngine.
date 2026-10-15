# 🎬 MovieLens Recommendation Engine

An end-to-end, high-performance movie recommendation engine built on **Apache Spark 4.0.1 (PySpark)** and **Streamlit**, using the **MovieLens 100K** benchmark dataset.

The system features **Alternating Least Squares (ALS)** collaborative filtering alongside an empirical **IMDb-style Bayesian Popularity baseline**, rigorously evaluated across both rating regression metrics (RMSE, MAE) and realistic candidate-sampling ranking metrics (**Precision@10**, **Recall@10**, and **NDCG@10**). An interactive Streamlit dashboard enables real-time exploration, user profile inspection, side-by-side model comparison, and dynamic genre/year filtering.

---

## 🚀 Key Features

- **Collaborative Filtering with Spark MLlib ALS**: Learns latent user and movie factor representations ($k=12$ latent dimensions, $\lambda=0.1$ regularization, 15 iterations) with cold-start drop strategy.
- **Bayesian Popularity Baseline**: Ranks movies using an IMDb-style weighted shrinkage formula ($m=25$ vote prior), preventing single-vote obscure movies from artificially outranking well-established classics.
- **Realistic Candidate-Sampling Offline Evaluation**:
  - **Rating Regression**: Test RMSE and MAE across held-out ratings.
  - **Top-10 Ranking Evaluation**: Evaluates each user by ranking a candidate set consisting of their relevant held-out movies ($r \ge 4.0$) plus 100 randomly sampled unrated catalogue movies.
  - **Reproducible & Controlled**: Identical candidate set scored and ranked by both models with a deterministic random seed; training-rated items strictly excluded.
  - **Macro-Averaging**: Precision@10, Recall@10, and NDCG@10 averaged across all eligible test users ($N=922$).
- **Interactive Streamlit Dashboard**:
  - **Dynamic Benchmark Cards**: Live comparison of ALS vs. Popularity with performance deltas and metric tooltips.
  - **User Selector & Profile Viewer**: Search through all 943 users, view total ratings, average rating given, and top preferred genres.
  - **Side-by-Side Comparison**: Visually inspect personalized ALS recommendations vs. Popularity baseline recommendations for any user.
  - **Catalog Filtering**: Instant multi-genre filtering and release year range slider (1922 – 1998).
  - **Historical Rating Inspector**: Collapsible table displaying the user's full training history with ratings and movie metadata.

---

## 📊 Benchmark & Evaluation Results

Evaluated on an 80/20 train/test split of the 100,000 ratings across 943 users and 1,682 movies (evaluated on all 922 eligible test users with at least one relevant held-out movie):

| Metric | ALS Collaborative Filtering | Popularity Baseline | Delta / Superior Model |
| :--- | :---: | :---: | :---: |
| **RMSE** (Rating Error) | **0.9173** | 1.0203 | **ALS (-10.1% error)** |
| **MAE** (Absolute Error) | **0.7256** | 0.8124 | **ALS (-10.7% error)** |
| **Precision@10** | 0.3017 | **0.3300** | **Popularity (+9.4% precision)** |
| **Recall@10** | 0.2891 | **0.3214** | **Popularity (+11.2% recall)** |
| **NDCG@10** | 0.3589 | **0.4165** | **Popularity (+16.0% ranking gain)** |

### 🔍 Empirical Analysis & Takeaways
1. **Rating Prediction (RMSE / MAE)**:
   - **Evaluated Test Rows**: Calculated strictly on the **20,053 test rows** for which ALS produced a prediction (out of 20,099 total test rows). Because ALS is configured with `coldStartStrategy="drop"`, test rows corresponding to movies with no training ratings (46 cold-start test ratings across 35 movies) are excluded.
   - **Fair Controlled Comparison**: To ensure a strictly fair and controlled comparison, the Popularity baseline is evaluated on the exact same 20,053 test rows.
   - **Performance**: ALS achieves substantially lower error than the popularity baseline (RMSE: $0.9173$ vs $1.0203$, MAE: $0.7256$ vs $0.8124$), showing that matrix factorization personalizes rating predictions far better than a non-personalized average.
2. **Candidate Ranking (Precision@10, Recall@10, NDCG@10)**:
   - **Sampled Candidates (Not Full Catalogue)**: Ranking metrics evaluate a **sampled candidate pool** for each user rather than the entire 1,682-movie catalogue. Each user's candidate set consists of their relevant held-out test movies ($r \ge 4.0$) plus 100 unobserved movies that the user rated in **neither the training split nor the test split**.
   - **Baseline Competitiveness**: Under this protocol, the Bayesian Popularity baseline proves to be a formidable competitor. In MovieLens, widely acclaimed classics (*Schindler's List*, *Star Wars*, *Casablanca*, *Pulp Fiction*) have high prior probabilities of being rated $\ge 4.0$ across most users. This is a well-documented phenomenon in recommender systems literature (e.g., Cremonesi, Koren & Turrin, *RecSys 2010*): rating-optimized explicit ALS does not directly optimize ranking loss, giving Bayesian popularity an advantage on top-N accuracy for universally popular items, while ALS excels at uncovering user-specific niche items.

---

## 📐 Detailed Methodology & Metric Definitions

### 1. Realistic Candidate-Sampling Ranking Protocol
Evaluating recommendations purely on movies a user already rated in the test set artificially constrains the recommendation pool. To simulate real-world recommender conditions:
1. **Ground-Truth Relevant Set from Raw Test Data**:
   For each test user $u$, relevant held-out items are defined directly from the **raw `test_df`** as movies with test rating $\ge 4.0$:
   $$\text{Relevant}_u = \{i \in \text{Test}_u \mid r_{ui} \ge 4.0\}$$
   Building relevance from raw `test_df` rather than `als_model.transform(test_df)` ensures that cold-start items in the test split are preserved rather than dropped. There are **922 eligible users** having at least one test rating $\ge 4.0$ (21 users had no test ratings $\ge 4.0$).
2. **Exclusion of User's Observed Movies**:
   Every movie the user rated in **both `train_df` and `test_df`** is excluded when sampling unobserved negatives:
   $$\text{Observed}_u = \text{Train}_u \cup \text{Test}_u$$
3. **Reproducible Negative Sampling**:
   From all catalogue movies that user $u$ never rated in either split ($\text{Catalogue} \setminus \text{Observed}_u$), a sample of $N=100$ unobserved movies is drawn using a deterministic seed (`seed = 42 + userId`):
   $$\text{UnobservedPool}_u = \text{Catalogue} \setminus (\text{Train}_u \cup \text{Test}_u)$$
   $$\text{Candidates}_u = \text{Relevant}_u \cup \text{Sample}(\text{UnobservedPool}_u, 100)$$
4. **Controlled Scoring & Cold-Start Candidate Handling**:
   Both ALS and the Popularity baseline score the **exact same candidate set** $\text{Candidates}_u$:
   - **Catalogue Cold-Start Items**: There are **35 items** in the catalogue with 0 training ratings (12 of which appear in relevant held-out sets across users).
   - **ALS Scoring**: Scores items via latent factor inner product $p_u^T q_i$. For cold-start items lacking learned latent factors in `itemFactors`, ALS explicitly falls back to the global training mean `global_mean` ($C \approx 3.53$). This ensures no items are silently dropped and all candidates are ranked.
   - **Popularity Scoring**: Scores items via the IMDb Bayesian weighted score. For a 0-rating cold item ($v=0$), the formula mathematically reduces to the Bayesian prior $C = \text{global\_mean}$, matching ALS's uninformed prior.
   - The top 10 highest-scoring items from each model are selected for evaluation.
5. **Macro-Averaging**:
   Metrics are computed per user and then averaged unweighted across all 922 eligible users:
   $$\text{Macro-Metric} = \frac{1}{|U_{\text{eval}}|} \sum_{u \in U_{\text{eval}}} \text{Metric}_u$$

### 2. Relevance Threshold Rationale ($r \ge 4.0$)
On MovieLens's 1-to-5 star scale:
- Ratings of **4 and 5** indicate clear user enjoyment and satisfaction (positive affinity).
- Ratings of **1, 2, and 3** signify dissatisfaction or indifference.
- Recommending a movie that a user would rate 2 or 3 stars is a false positive that degrades user experience. Therefore, only held-out items with $r_{ui} \ge 4.0$ are treated as relevant hits.

### 3. Metric Formulations
- **Precision@10**:
  $$\text{Precision}@10 = \frac{|\text{Top10}_u \cap \text{Relevant}_u|}{10}$$
- **Recall@10**:
  $$\text{Recall}@10 = \frac{|\text{Top10}_u \cap \text{Relevant}_u|}{|\text{Relevant}_u|}$$
- **NDCG@10 (Normalized Discounted Cumulative Gain)**:
  $$\text{DCG}@10 = \sum_{k=1}^{10} \frac{\mathbb{I}(\text{Top10}_u[k-1] \in \text{Relevant}_u)}{\log_2(k + 1)}, \quad \text{NDCG}@10 = \frac{\text{DCG}@10}{\text{IDCG}@10}$$
  where $\text{IDCG}@10 = \sum_{k=1}^{\min(10, |\text{Relevant}_u|)} \frac{1}{\log_2(k + 1)}$. Rewards models that place relevant items higher up the top-10 list.
- **RMSE & MAE**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{u,i} (r_{ui} - \hat{r}_{ui})^2}, \quad \text{MAE} = \frac{1}{N} \sum_{u,i} |r_{ui} - \hat{r}_{ui}|$$
  Evaluated on the $N=20,053$ test rows where ALS produced predictions.

### 4. Bayesian Weighted Popularity Baseline
To prevent single-vote movies from dominating the baseline ranking, we compute an **IMDb-style Bayesian shrinkage score**:
$$\text{WR} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
where $v$ is movie rating count, $m=25$ is the prior vote threshold, $R$ is average movie rating, and $C \approx 3.53$ is the global training mean. For rating predictions, each movie defaults to its training average rating (with fallback to global mean).

---

## 🛠️ Project Structure

```text
MovieRecommendationEngine/
├── app.py                  # Interactive Streamlit dashboard application
├── src/
│   └── recommender.py      # Core PySpark pipeline: ALS training, baseline, evaluation & metrics
├── data/
│   ├── ml-100k.zip         # Local MovieLens 100K archive (reused automatically)
│   └── ml-100k/            # Extracted MovieLens dataset (u.data, u.item, u.genre, etc.)
├── .venv/                  # Python 3.12 virtual environment (ARM64 Windows)
└── README.md               # Project documentation and metric guide
```

---

## 💻 Setup & Run Instructions

### Prerequisites
- **Operating System**: Windows (ARM64 or x64) / macOS / Linux
- **Python**: Python 3.12 (in `.venv`)
- **Java**: Java 17 (OpenJDK / Oracle JDK) configured in `JAVA_HOME`
- **Apache Spark**: PySpark 4.0.1

### 1. Activate Virtual Environment
Open PowerShell in the project root:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run Offline Training & Candidate Evaluation Pipeline
Execute the full PySpark pipeline:
```powershell
python src/recommender.py
```
This script will:
1. Load local MovieLens 100K files from `data/` without external downloads.
2. Extract release years and genres from `u.item`.
3. Train ALS collaborative filtering and compute the Bayesian popularity baseline.
4. Run rating error regression (RMSE, MAE) and candidate-sampling ranking evaluation (Precision@10, Recall@10, NDCG@10).
5. Output the comparison summary table and top-10 sample recommendations for User 1.

### 3. Launch Streamlit Interactive Dashboard
Run the dashboard:
```powershell
streamlit run app.py
```
Or directly:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Navigate to `http://localhost:8501` (or next available port) to explore the interface.
