"""
文本向量化模块
================================

设计依据
- 直接复用老师 WS4 的三种文本向量化方法（TF-IDF / Word2Vec / Doc2Vec），
  把"epub.read_epub(...)"读书的输入替换为"YouTube + IMDb 评论"的字符串列表。
- 在老师代码之上额外做的事：
    1) 统一封装为一个 TextVectorizer 抽象，方便对比；
    2) 实现一组对比指标（聚类纯度、KNN 检索 mAP、t-SNE 可分性），
       兑现作业要求"基于自己的结果而非理论"。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import get_logger, get_path, load_config

log = get_logger("text_vec")


# ---------- 预处理 ----------
def tokenize(text: str, min_len: int = 3) -> list[str]:
    """老师 WS4 用 gensim.utils.simple_preprocess(min_len=3, deacc=True)"""
    return simple_preprocess(text or "", min_len=min_len, deacc=True)


def prepare_corpus(df: pd.DataFrame, text_col: str = "text") -> tuple[list[str], list[list[str]]]:
    docs = df[text_col].astype(str).tolist()
    tokens = [tokenize(d) for d in docs]
    return docs, tokens


# ---------- 向量化器 ----------
@dataclass
class VectorizationResult:
    name: str
    matrix: np.ndarray            # shape: (n_docs, dim)
    model: object                 # 训练好的模型
    extra: dict | None = None


class TfidfVectorize:
    """老师 WS4：TfidfVectorizer(min_df=2)"""

    def __init__(self, max_features: int = 3000, min_df: int = 2):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words="english",
            ngram_range=(1, 2),
        )

    def fit_transform(self, docs: list[str]) -> VectorizationResult:
        X = self.vectorizer.fit_transform(docs).toarray().astype(np.float32)
        log.info(f"TF-IDF 矩阵: {X.shape}, 词表大小 {len(self.vectorizer.vocabulary_)}")
        return VectorizationResult("tfidf", X, self.vectorizer)


class Word2VecMean:
    """老师 WS4：Word2Vec(vector_size=50, window=3, min_count=5)
    评论级别表示用"词向量平均"，是工程上最常见的简单基线。
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

    def fit_transform(self, tokens: list[list[str]]) -> VectorizationResult:
        model = Word2Vec(
            sentences=tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=20,
        )
        # 文档向量 = 平均词向量
        mat = np.zeros((len(tokens), self.vector_size), dtype=np.float32)
        for i, toks in enumerate(tokens):
            vecs = [model.wv[t] for t in toks if t in model.wv]
            if vecs:
                mat[i] = np.mean(vecs, axis=0)
        log.info(f"Word2Vec 平均向量: {mat.shape}, 词表 {len(model.wv)}")
        return VectorizationResult("word2vec_mean", mat, model)


class Doc2VecVectorize:
    """老师 WS4 没直接用 Doc2Vec，但作业要求"两种方法对比"，加上它正好覆盖经典的 doc-level embedding。"""

    def __init__(self, vector_size: int = 100, epochs: int = 30, min_count: int = 2):
        self.vector_size = vector_size
        self.epochs = epochs
        self.min_count = min_count

    def fit_transform(self, tokens: list[list[str]]) -> VectorizationResult:
        tagged = [TaggedDocument(words=t, tags=[str(i)]) for i, t in enumerate(tokens)]
        model = Doc2Vec(
            documents=tagged,
            vector_size=self.vector_size,
            window=5,
            min_count=self.min_count,
            workers=4,
            epochs=self.epochs,
            dm=1,  # PV-DM
        )
        mat = np.vstack([model.dv[str(i)] for i in range(len(tokens))]).astype(np.float32)
        log.info(f"Doc2Vec 向量: {mat.shape}")
        return VectorizationResult("doc2vec", mat, model)


# ---------- 对比指标 ----------
def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """聚类纯度：每个簇中占比最大的真实类别比例之和 / 总数。
    
    数学定义：purity = (1/N) Σ_k max_j |C_k ∩ T_j|
    """
    n = len(labels_true)
    purity = 0.0
    for k in np.unique(labels_pred):
        mask = labels_pred == k
        if not mask.any():
            continue
        true_in_cluster = labels_true[mask]
        # 该簇里出现最多的真实标签的次数
        counts = np.bincount(true_in_cluster.astype(int))
        purity += counts.max()
    return purity / n


def knn_retrieval_map(
    matrix: np.ndarray, labels: np.ndarray, k: int = 10
) -> float:
    """KNN 检索的 mean Average Precision @ k。
    
    评估"语义相近的评论是否在向量空间里也相邻"。值越高越好。
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(matrix)
    np.fill_diagonal(sims, -1.0)  # 排除自身
    n = len(matrix)
    aps = []
    for i in range(n):
        topk = np.argsort(-sims[i])[:k]
        hits = (labels[topk] == labels[i]).astype(np.float32)
        if hits.sum() == 0:
            aps.append(0.0)
            continue
        # AP@k
        precisions = np.cumsum(hits) / (np.arange(k) + 1)
        ap = (precisions * hits).sum() / hits.sum()
        aps.append(ap)
    return float(np.mean(aps))


def compare_vectorizers(
    df: pd.DataFrame,
    label_col: str = "genre",
    text_col: str = "text",
    n_clusters: int | None = None,
    save_dir: str | Path | None = None,
) -> tuple[dict[str, VectorizationResult], pd.DataFrame]:
    """
    在同一份评论上跑 TF-IDF / Word2Vec / Doc2Vec，输出对比表格。

    返回
    ----
    results : {name: VectorizationResult}
    metrics : 对比表格 DataFrame
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder

    cfg = load_config()["vectorize"]["text"]
    save_dir = Path(save_dir) if save_dir else get_path("outputs/models/text_vec")
    save_dir.mkdir(parents=True, exist_ok=True)

    docs, tokens = prepare_corpus(df, text_col)
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str).values)
    if n_clusters is None:
        n_clusters = len(le.classes_)

    log.info(f"对比向量化 | n_docs={len(docs)} | n_classes={len(le.classes_)} | n_clusters={n_clusters}")

    vectorizers = {
        "tfidf": TfidfVectorize(
            max_features=cfg["tfidf_max_features"], min_df=cfg["tfidf_min_df"]
        ).fit_transform(docs),
        "word2vec_mean": Word2VecMean(
            vector_size=cfg["word2vec_vector_size"]
        ).fit_transform(tokens),
        "doc2vec": Doc2VecVectorize(
            vector_size=cfg["doc2vec_vector_size"], epochs=cfg["doc2vec_epochs"]
        ).fit_transform(tokens),
    }

    rows = []
    for name, res in vectorizers.items():
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = km.fit_predict(res.matrix)
        purity = cluster_purity(y, cluster_labels)
        map_at_10 = knn_retrieval_map(res.matrix, y, k=10)
        rows.append(
            {
                "vectorizer": name,
                "dim": res.matrix.shape[1],
                "cluster_purity": round(purity, 4),
                "knn_mAP@10": round(map_at_10, 4),
            }
        )
        # 保存矩阵和模型
        np.save(save_dir / f"{name}_matrix.npy", res.matrix)
        joblib.dump(res.model, save_dir / f"{name}_model.joblib")

    metrics = pd.DataFrame(rows).sort_values("knn_mAP@10", ascending=False)
    metrics.to_csv(save_dir / "comparison.csv", index=False)
    log.info("\n" + metrics.to_string(index=False))
    return vectorizers, metrics


if __name__ == "__main__":
    csv = get_path("data/processed/all_reviews.csv")
    if not csv.exists():
        csv = get_path("data/raw/youtube_comments.csv")
    df = pd.read_csv(csv)
    compare_vectorizers(df, label_col="genre", text_col="text")
