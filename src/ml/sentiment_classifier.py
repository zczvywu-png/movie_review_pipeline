"""
情感分类（监督学习）
================================

设计依据
- 用 GPT 打的标签（正/负/中）做 Y，TF-IDF / Doc2Vec 向量做 X，
  对比 LogisticRegression / LinearSVC / RandomForest 三种经典分类器。
- 这是经典 ML pipeline，对应作业要求"机器学习：训练模型以得出见解（分析）"。
- 同时输出混淆矩阵、分类报告、跨向量化方法的准确率对比 → 报告里能直接放图。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.utils import get_logger, get_path, load_config

log = get_logger("sentiment_clf")

CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
    "linsvc": lambda: LinearSVC(C=1.0),
    "rf": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}


def _load_matrix(name: str) -> np.ndarray:
    p = get_path(f"outputs/models/text_vec/{name}_matrix.npy")
    if not p.exists():
        raise FileNotFoundError(f"缺向量矩阵：{p}，请先跑 text_vectorize")
    return np.load(p)


def train_one(
    X: np.ndarray, y: np.ndarray, clf_name: str, test_size: float = 0.2
) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    clf = CLASSIFIERS[clf_name]()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "clf": clf,
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1_macro": float(f1_score(y_te, pred, average="macro")),
        "report": classification_report(y_te, pred, zero_division=0, output_dict=True),
        "confusion": confusion_matrix(y_te, pred).tolist(),
        "y_test": y_te,
        "y_pred": pred,
    }


def benchmark_sentiment(
    df: pd.DataFrame,
    label_col: str = "sentiment",
    vectorizers: list[str] = ("tfidf", "doc2vec"),
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(
            f"DataFrame 缺 '{label_col}' 列。请先跑 src.api.sentiment_analysis 给评论打标。"
        )
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str))
    classes = list(le.classes_)
    log.info(f"标签分布：{pd.Series(y).map(dict(enumerate(classes))).value_counts().to_dict()}")

    cfg = load_config()["ml"]
    save_dir = get_path("outputs/models/sentiment_clf")
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for vec_name in vectorizers:
        X = _load_matrix(vec_name)
        # 对齐长度
        n = min(len(X), len(y))
        X_, y_ = X[:n], y[:n]
        for clf_name in CLASSIFIERS:
            log.info(f"--- {vec_name} + {clf_name} ---")
            res = train_one(X_, y_, clf_name, test_size=cfg["classifier_test_size"])
            rows.append(
                {
                    "vectorizer": vec_name,
                    "classifier": clf_name,
                    "accuracy": round(res["accuracy"], 4),
                    "f1_macro": round(res["f1_macro"], 4),
                }
            )
            joblib.dump(res["clf"], save_dir / f"{vec_name}_{clf_name}.joblib")
            # 保存混淆矩阵
            np.savez(
                save_dir / f"{vec_name}_{clf_name}_eval.npz",
                confusion=np.array(res["confusion"]),
                classes=np.array(classes),
                y_test=res["y_test"],
                y_pred=res["y_pred"],
            )
            log.info(f"  acc={res['accuracy']:.4f}  f1_macro={res['f1_macro']:.4f}")

    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    summary.to_csv(save_dir / "benchmark.csv", index=False)
    log.info("\n" + summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    df = pd.read_csv(get_path("data/processed/reviews_with_sentiment.csv"))
    benchmark_sentiment(df)
