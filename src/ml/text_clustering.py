"""
评论 K-Means 聚类 + 主题词提取
================================

设计依据
- 老师 WS4 的下游没做"无监督发现主题"，本模块补上。
- K-Means 跑在 TF-IDF 矩阵上（最适合主题词解读），
  对每个簇提取代表性词（簇质心权重最高的 token）→ 直接生成报告里能用的"主题关键词表"。
- 同时跑一份 Doc2Vec 矩阵的聚类，与 TF-IDF 结果对比 ARI / NMI，
  作为"两种向量化方法在下游任务的差异"的实证证据。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

from src.utils import get_logger, get_path, load_config

log = get_logger("clustering")


def _top_terms_per_cluster(
    km: KMeans, feature_names: list[str], top_n: int = 10
) -> dict[int, list[str]]:
    """从 KMeans 质心反查最重要的词（仅适用于 TF-IDF）"""
    centers = km.cluster_centers_
    out: dict[int, list[str]] = {}
    for i, center in enumerate(centers):
        top_idx = center.argsort()[::-1][:top_n]
        out[i] = [feature_names[j] for j in top_idx]
    return out


def cluster_with_tfidf(
    df: pd.DataFrame,
    n_clusters: int = 6,
    text_col: str = "text",
    label_col: str | None = "genre",
) -> dict:
    """
    在 TF-IDF 矩阵上跑 KMeans，输出每个簇的主题词 + 与真实标签的对齐指标。
    """
    tfidf_path = get_path("outputs/models/text_vec/tfidf_matrix.npy")
    tfidf_model_path = get_path("outputs/models/text_vec/tfidf_model.joblib")
    if not tfidf_path.exists() or not tfidf_model_path.exists():
        raise FileNotFoundError("请先跑 src.vectorize.text_vectorize.compare_vectorizers")

    X = np.load(tfidf_path)
    vectorizer = joblib.load(tfidf_model_path)
    feature_names = vectorizer.get_feature_names_out().tolist()
    log.info(f"TF-IDF 聚类输入：{X.shape}")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    out_df = df.copy().iloc[: len(labels)].reset_index(drop=True)
    out_df["cluster_tfidf"] = labels

    metrics: dict = {
        "n_clusters": n_clusters,
        "silhouette": float(
            silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
        ),
    }
    if label_col and label_col in out_df.columns:
        y_true = LabelEncoder().fit_transform(out_df[label_col].astype(str))
        metrics["ari_vs_genre"] = float(adjusted_rand_score(y_true, labels))
        metrics["nmi_vs_genre"] = float(normalized_mutual_info_score(y_true, labels))

    top_terms = _top_terms_per_cluster(km, feature_names, top_n=12)

    save_dir = get_path("outputs/models/clustering")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_dir / "reviews_with_tfidf_clusters.csv", index=False, encoding="utf-8-sig")
    joblib.dump(km, save_dir / "kmeans_tfidf.joblib")

    with open(save_dir / "tfidf_top_terms.txt", "w", encoding="utf-8") as f:
        for cid, terms in top_terms.items():
            f.write(f"Cluster {cid}: {', '.join(terms)}\n")

    log.info(f"TF-IDF 聚类指标：{metrics}")
    log.info("各簇代表性主题词：")
    for cid, terms in top_terms.items():
        log.info(f"  cluster {cid}: {', '.join(terms[:8])}")

    return {"metrics": metrics, "top_terms": top_terms, "labels": labels}


def cluster_with_doc2vec(
    df: pd.DataFrame,
    n_clusters: int = 6,
    label_col: str | None = "genre",
) -> dict:
    d2v_path = get_path("outputs/models/text_vec/doc2vec_matrix.npy")
    if not d2v_path.exists():
        raise FileNotFoundError("请先跑 text_vectorize")

    X = np.load(d2v_path)
    log.info(f"Doc2Vec 聚类输入：{X.shape}")
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    out_df = df.copy().iloc[: len(labels)].reset_index(drop=True)
    out_df["cluster_doc2vec"] = labels

    metrics: dict = {
        "n_clusters": n_clusters,
        "silhouette": float(
            silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=42)
        ),
    }
    if label_col and label_col in out_df.columns:
        y_true = LabelEncoder().fit_transform(out_df[label_col].astype(str))
        metrics["ari_vs_genre"] = float(adjusted_rand_score(y_true, labels))
        metrics["nmi_vs_genre"] = float(normalized_mutual_info_score(y_true, labels))

    save_dir = get_path("outputs/models/clustering")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(save_dir / "reviews_with_doc2vec_clusters.csv", index=False, encoding="utf-8-sig")
    joblib.dump(km, save_dir / "kmeans_doc2vec.joblib")
    log.info(f"Doc2Vec 聚类指标：{metrics}")
    return {"metrics": metrics, "labels": labels}


def run_full_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """同时跑 TF-IDF 和 Doc2Vec 两组聚类，输出对比表 + 合并簇标签"""
    cfg = load_config()
    n_clusters = cfg["ml"]["kmeans_n_clusters"]

    r1 = cluster_with_tfidf(df, n_clusters=n_clusters)
    r2 = cluster_with_doc2vec(df, n_clusters=n_clusters)

    cmp_df = pd.DataFrame(
        [
            {"vectorizer": "tfidf", **r1["metrics"]},
            {"vectorizer": "doc2vec", **r2["metrics"]},
        ]
    )
    cmp_path = get_path("outputs/models/clustering/comparison.csv")
    cmp_df.to_csv(cmp_path, index=False)
    log.info("\n" + cmp_df.to_string(index=False))

    out_df = df.copy().iloc[: len(r1["labels"])].reset_index(drop=True)
    out_df["cluster_tfidf"] = r1["labels"]
    out_df["cluster_doc2vec"] = r2["labels"][: len(out_df)]
    out_df.to_csv(
        get_path("data/processed/reviews_with_clusters.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return out_df


if __name__ == "__main__":
    df = pd.read_csv(get_path("data/processed/all_reviews.csv"))
    run_full_clustering(df)
