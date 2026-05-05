"""
analyzer.py : Core analytical engine.

Pipeline:
  1. Load papers from CSV
  2. TF-IDF + LDA topic modeling on abstracts
  3. Temporal binning (quarterly by default)
  4. Keyword frequency trends + Mann-Kendall test
  5. Topic distribution drift via Jensen-Shannon divergence
  6. Output: trend_data.json + drift_events.json + topic_assignments.csv
"""

import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    DATA_DIR, DRIFT_FILE, DRIFT_THRESHOLD, LDA_MAX_ITER, MAX_DF,
    MAX_FEATURES, MIN_PAPERS_IN_BIN, MK_P_THRESHOLD, N_TOPICS,
    N_TOP_WORDS, MIN_DF, PAPERS_FILE, RANDOM_STATE, TIME_BIN,
    TOPIC_FILE, TREND_FILE,
)

warnings.filterwarnings("ignore")


# Text preprocessing 

_STOP = {
    "the", "a", "an", "and", "or", "in", "to", "of", "for", "with", "on",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "this", "that", "these", "those", "we", "our", "their", "from", "as",
    "by", "at", "it", "its", "not", "but", "also", "which", "such", "can",
    "any", "all", "one", "two", "more", "less", "using", "based", "use",
    "used", "show", "shows", "paper", "propose", "proposed", "results",
    "method", "approach", "work", "model", "models", "demonstrate",
    "however", "therefore", "thus", "may", "large", "different", "new",
    "recent", "existing", "provides", "provide", "task", "tasks",
}


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(t for t in text.split() if len(t) > 2 and t not in _STOP)


# 1. Data loading 

def load_papers() -> pd.DataFrame:
    df = pd.read_csv(PAPERS_FILE, parse_dates=["published"])
    df = df.dropna(subset=["abstract", "published"])
    df["period"] = df["published"].dt.to_period(TIME_BIN)
    df["clean_text"] = (
        df["title"].fillna("") + " " + df["abstract"].fillna("")
    ).apply(_clean)
    print(f"✓  Loaded {len(df)} papers  "
          f"({df['published'].min().date()} → {df['published'].max().date()})")
    return df


# 2. Topic modeling 

def run_topic_model(df: pd.DataFrame):
    """
    Fit TF-IDF + LDA on cleaned abstracts.
    Returns: (lda, vectorizer, doc_topic_matrix, topic_labels, topic_top_words)
    """
    print(f"\n Fitting TF-IDF + LDA  ({N_TOPICS} topics) …")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    print(f" Vocabulary: {X.shape[1]} terms  |  Documents: {X.shape[0]}")

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=LDA_MAX_ITER,
        random_state=RANDOM_STATE,
        learning_method="batch",
        doc_topic_prior=0.1,      
        topic_word_prior=0.01,    
    )
    doc_topic = lda.fit_transform(X)
    print(f"   LDA perplexity: {lda.perplexity(X):.1f}")

    vocab = vectorizer.get_feature_names_out()
    topic_labels, topic_top_words = [], []
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-N_TOP_WORDS:][::-1]
        words   = [vocab[j] for j in top_idx]
        label   = f"T{i+1}: {' · '.join(words[:3])}"
        topic_labels.append(label)
        topic_top_words.append(words)
        print(f"   {label}")

    return lda, vectorizer, doc_topic, topic_labels, topic_top_words


# 3. Temporal trend computation 
def compute_trends(df: pd.DataFrame, doc_topic: np.ndarray) -> tuple:
    """
    Compute per-time-bin:
      - Paper volume
      - Keyword frequencies (from matched_keywords column)
      - Topic distributions (mean LDA probabilities)

    Returns (trend_data dict, enriched df).
    """
    df = df.copy()
    df["dominant_topic"] = doc_topic.argmax(axis=1)

    periods = sorted(df["period"].unique())
    valid   = [p for p in periods if (df["period"] == p).sum() >= MIN_PAPERS_IN_BIN]
    print(f"\n Valid time bins: {len(valid)}  ({TIME_BIN}, ≥{MIN_PAPERS_IN_BIN} papers each)")

    volume = {str(p): int((df["period"] == p).sum()) for p in valid}

    kw_freq: dict = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        if row["period"] not in valid:
            continue
        for kw in str(row.get("matched_keywords", "")).split(";"):
            kw = kw.strip().lower()
            if kw:
                kw_freq[kw][str(row["period"])] += 1

    topic_dist = {}
    for p in valid:
        mask = (df["period"] == p).to_numpy()
        dist = doc_topic[mask].mean(axis=0)
        topic_dist[str(p)] = dist.tolist()

    trend_data = {
        "periods"   : [str(p) for p in valid],
        "volume"    : volume,
        "kw_freq"   : {k: dict(v) for k, v in kw_freq.items()},
        "topic_dist": topic_dist,
    }
    return trend_data, df


# 4. Mann-Kendall trend test 
def mann_kendall(series: list) -> tuple:
    """
    Non-parametric monotonic trend test on a numeric sequence.
    """
    n = len(series)
    if n < 4:
        return 0.0, 1.0, "stable"

    s = sum(
        int(series[j] > series[i]) - int(series[j] < series[i])
        for i in range(n - 1)
        for j in range(i + 1, n)
    )
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s != 0:
        z = (s - np.sign(s)) / np.sqrt(var_s)
    else:
        z = 0.0

    p     = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    tau   = s / (n * (n - 1) / 2)
    trend = ("increasing" if s > 0 else "decreasing") if p < MK_P_THRESHOLD else "stable"
    return float(tau), float(p), trend


# 5. Keyword classification 

def classify_keywords(trend_data: dict) -> list:
    """
    Apply Mann-Kendall to each keyword's time series, returns sorted list with trend classification + growth rate.
    """
    periods = trend_data["periods"]
    results = []

    for kw, freq_map in trend_data["kw_freq"].items():
        series = [freq_map.get(p, 0) for p in periods]
        if sum(series) < 5:
            continue

        tau, p_val, direction = mann_kendall(series)
        mid        = max(1, len(series) // 2)
        first_half = np.mean(series[:mid])
        second_half= np.mean(series[mid:])
        growth     = (second_half - first_half) / (first_half + 1e-6)

        results.append({
            "keyword"    : kw,
            "total_count": int(sum(series)),
            "trend"      : direction,
            "mk_tau"     : round(tau, 4),
            "mk_p"       : round(p_val, 4),
            "growth_rate": round(float(growth), 4),
            "series"     : series,
        })

    results.sort(key=lambda x: x["growth_rate"], reverse=True)
    emerging  = sum(1 for r in results if r["trend"] == "increasing")
    declining = sum(1 for r in results if r["trend"] == "decreasing")
    stable    = sum(1 for r in results if r["trend"] == "stable")
    print(f"\n Keyword trends - emerging: {emerging}  |  "
          f"declining: {declining}  |  stable: {stable}")
    return results


#  6. Drift detection 

def detect_drift(trend_data: dict, topic_labels: list) -> list:
    """
    Detect topic distribution drift between consecutive time windows, uses Jensen-Shannon divergence on topic probability vectors.
    """
    periods    = trend_data["periods"]
    topic_dist = trend_data["topic_dist"]
    events     = []

    for i in range(1, len(periods)):
        p_prev, p_curr = periods[i - 1], periods[i]
        if p_prev not in topic_dist or p_curr not in topic_dist:
            continue

        prev = np.array(topic_dist[p_prev]) + 1e-10
        curr = np.array(topic_dist[p_curr]) + 1e-10
        prev /= prev.sum()
        curr /= curr.sum()

        js = float(jensenshannon(prev, curr) ** 2)
        if js > DRIFT_THRESHOLD:
            delta        = curr - prev
            gain_idx     = int(np.argmax(delta))
            lose_idx     = int(np.argmin(delta))
            events.append({
                "period"        : p_curr,
                "js_divergence" : round(js, 4),
                "gaining_topic" : topic_labels[gain_idx],
                "losing_topic"  : topic_labels[lose_idx],
                "delta_gaining" : round(float(delta[gain_idx]), 4),
                "delta_losing"  : round(float(delta[lose_idx]), 4),
            })

    print(f"\n Drift events: {len(events)}  "
          f"(JS-divergence threshold = {DRIFT_THRESHOLD})")
    for e in events:
        print(f"   [{e['period']}] JS={e['js_divergence']:.3f}  "
              f"↑ {e['gaining_topic']}")
    return events


# 7. Save topic assignments 

def save_topic_assignments(df: pd.DataFrame, topic_labels: list,
                            topic_top_words: list, doc_topic: np.ndarray):
    rows = []
    for i, row in df.iterrows():
        tid   = int(doc_topic[i].argmax())
        conf  = float(doc_topic[i].max())
        rows.append({
            "arxiv_id"       : row["arxiv_id"],
            "dominant_topic" : topic_labels[tid],
            "confidence"     : round(conf, 4),
            "top_words"      : ", ".join(topic_top_words[tid]),
        })
    pd.DataFrame(rows).to_csv(TOPIC_FILE, index=False)
    print(f" Topic assignments saved - {TOPIC_FILE}")


#  Main entry point

def run_analysis():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1 — load
    df = load_papers()

    # Step 2 — topic modeling
    lda, vec, doc_topic, topic_labels, topic_top_words = run_topic_model(df)

    # Step 3 — temporal trends
    trend_data, df_enriched = compute_trends(df, doc_topic)

    # Step 4 - keyword classification
    kw_results = classify_keywords(trend_data)
    trend_data["kw_trends"] = kw_results

    # Step 5 — drift detection
    drift_events = detect_drift(trend_data, topic_labels)
    trend_data["topic_labels"]    = topic_labels
    trend_data["topic_top_words"] = topic_top_words

    with open(TREND_FILE, "w") as f:
        json.dump(trend_data, f, indent=2)
    with open(DRIFT_FILE, "w") as f:
        json.dump(drift_events, f, indent=2)
    save_topic_assignments(df_enriched, topic_labels, topic_top_words, doc_topic)

    print(f"\n Trend data : {TREND_FILE}")
    print(f" Drift events : {DRIFT_FILE}")
    return trend_data, drift_events
