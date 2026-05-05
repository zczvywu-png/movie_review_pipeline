"""
统一可视化模块（matplotlib + seaborn）
================================

设计依据
- 作业明确要求"使用 Matplotlib 和 Seaborn 对数据集进行各种分析"。
- 本模块产出 7 张图：
    1) 评论数量按类型分布（柱状图）
    2) 情感分布堆叠条形图（按类型）
    3) 词云（按类型分别一张）
    4) 文本向量降维 t-SNE 散点图（按类型着色，对比 TF-IDF / Doc2Vec）
    5) 图像向量降维 t-SNE 散点图（CLIP vs ResNet50）
    6) 海报缩略图聚类网格
    7) 跨模态相关性热力图（评论情感 × 海报视觉特征）
- 所有图统一存到 outputs/figures/，命名清晰便于贴报告。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.utils import get_logger, get_path

log = get_logger("viz")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.family"] = "DejaVu Sans"


def _save(fig, name: str):
    out = get_path(f"outputs/figures/{name}")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"已保存 {out}")


# ---------- 1. 评论量分布 ----------
def plot_review_counts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = (
        df.groupby(["genre", "source"]).size().unstack(fill_value=0)
        if "source" in df.columns
        else df["genre"].value_counts().to_frame("count")
    )
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Review Count by Genre and Source", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    _save(fig, "01_review_counts.png")


# ---------- 2. 情感分布 ----------
def plot_sentiment_distribution(df: pd.DataFrame):
    if "sentiment" not in df.columns:
        log.warning("缺 sentiment 列，跳过")
        return

    # 仅保留 3 类核心情绪（GPT 偶尔会返回 off_topic 等扩展标签，可视化里忽略）
    known = ["positive", "neutral", "negative"]
    df_plot = df[df["sentiment"].isin(known)].copy()
    if df_plot.empty:
        log.warning("过滤后无可绘制 sentiment 数据，跳过")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pct = (
        df_plot.groupby(["genre", "sentiment"]).size().unstack(fill_value=0)
    )
    pct = pct.div(pct.sum(axis=1), axis=0) * 100
    pct = pct.reindex(columns=[c for c in known if c in pct.columns])
    palette = {"positive": "#5cb85c", "neutral": "#bfbfbf", "negative": "#d9534f"}
    pct.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=[palette[c] for c in pct.columns],
        edgecolor="white",
    )
    axes[0].set_title("Sentiment Composition (%) by Genre", fontweight="bold")
    axes[0].set_ylabel("Percentage")
    axes[0].legend(title="Sentiment", loc="upper right")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    if "intensity" in df_plot.columns:
        sns.violinplot(
            data=df_plot,
            x="genre",
            y="intensity",
            hue="sentiment",
            hue_order=[c for c in known if c in df_plot["sentiment"].unique()],
            split=False,
            ax=axes[1],
            palette=palette,
        )
        axes[1].set_title("Sentiment Intensity Distribution", fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_sentiment_distribution.png")


# ---------- 3. 词云 ----------
def plot_wordclouds(df: pd.DataFrame, text_col: str = "text"):
    try:
        from wordcloud import STOPWORDS, WordCloud
    except ImportError:
        log.warning("wordcloud 未安装，跳过")
        return

    genres = sorted(df["genre"].unique())
    n = len(genres)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    stop = set(STOPWORDS) | {"movie", "film", "trailer", "watch", "going", "see", "look"}
    for i, g in enumerate(genres):
        text = " ".join(df[df["genre"] == g][text_col].astype(str).tolist())
        if not text.strip():
            continue
        wc = WordCloud(
            width=600, height=400, background_color="white", stopwords=stop, max_words=80
        ).generate(text)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Genre: {g}", fontweight="bold")
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    _save(fig, "03_wordclouds.png")


# ---------- 4. 文本向量 t-SNE ----------
def plot_text_tsne(df: pd.DataFrame):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    le = LabelEncoder()
    y = le.fit_transform(df["genre"].astype(str))

    palette = sns.color_palette("tab10", n_colors=len(le.classes_))

    for ax, name in zip(axes, ["tfidf", "doc2vec"]):
        p = get_path(f"outputs/models/text_vec/{name}_matrix.npy")
        if not p.exists():
            ax.text(0.5, 0.5, f"{name}_matrix.npy not found", ha="center", va="center")
            ax.set_title(name)
            continue
        X = np.load(p)
        n = min(len(X), len(y))
        # t-SNE 慢，限制规模
        if n > 1500:
            idx = np.random.RandomState(42).choice(n, 1500, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X[:n], y[:n]
        ts = TSNE(n_components=2, init="pca", random_state=42, perplexity=30).fit_transform(Xs)
        for i, cls in enumerate(le.classes_):
            mask = ys == i
            ax.scatter(ts[mask, 0], ts[mask, 1], s=12, alpha=0.6, color=palette[i], label=cls)
        ax.set_title(f"t-SNE on {name.upper()} ({Xs.shape[1]}-d)", fontweight="bold")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    _save(fig, "04_text_tsne.png")


# ---------- 5. 图像向量 t-SNE ----------
def plot_image_tsne():
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder

    idx_csv = get_path("outputs/models/image_vec/image_index.csv")
    if not idx_csv.exists():
        log.warning("缺 image_index.csv，跳过")
        return
    df = pd.read_csv(idx_csv)
    le = LabelEncoder()
    y = le.fit_transform(df["genre"].astype(str))
    palette = sns.color_palette("tab10", n_colors=len(le.classes_))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name in zip(axes, ["clip", "resnet50"]):
        p = get_path(f"outputs/models/image_vec/{name}_matrix.npy")
        if not p.exists():
            continue
        X = np.load(p)
        n = min(len(X), len(y))
        ts = TSNE(n_components=2, init="pca", random_state=42, perplexity=min(30, n // 4)).fit_transform(X[:n])
        for i, cls in enumerate(le.classes_):
            mask = y[:n] == i
            ax.scatter(ts[mask, 0], ts[mask, 1], s=30, alpha=0.7, color=palette[i], label=cls, edgecolors="white", linewidth=0.5)
        ax.set_title(f"Posters t-SNE ({name.upper()}, {X.shape[1]}-d)", fontweight="bold")
        ax.legend(loc="best")
    plt.tight_layout()
    _save(fig, "05_image_tsne.png")


# ---------- 6. 海报网格 ----------
def plot_poster_grid(per_genre: int = 6):
    posters_root = get_path("data/posters")
    if not posters_root.exists():
        return
    genres = sorted([d.name for d in posters_root.iterdir() if d.is_dir()])
    rows = len(genres)
    cols = per_genre
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i, g in enumerate(genres):
        files = sorted((posters_root / g).glob("*.jpg"))[:per_genre]
        for j in range(cols):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(files):
                ax.imshow(Image.open(files[j]).convert("RGB"))
                if j == 0:
                    ax.set_ylabel(g, fontsize=12, fontweight="bold", rotation=0, labelpad=40, va="center")
    plt.suptitle("Poster Samples by Genre", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    _save(fig, "06_poster_grid.png")


# ---------- 7. 跨模态热力图 ----------
def plot_cross_modal_heatmap():
    """评论情感 vs 海报视觉特征（CLIP）的相关性"""
    sentiment_csv = get_path("data/processed/reviews_with_sentiment.csv")
    img_csv = get_path("outputs/models/image_vec/image_index.csv")
    clip_path = get_path("outputs/models/image_vec/clip_matrix.npy")
    if not all([sentiment_csv.exists(), img_csv.exists(), clip_path.exists()]):
        log.warning("跨模态热力图依赖未就绪，跳过")
        return

    rev = pd.read_csv(sentiment_csv)
    img_df = pd.read_csv(img_csv)
    clip_mat = np.load(clip_path)

    # 每部电影：① 正向评论比例 ② CLIP 海报特征均值
    sent_per_movie = (
        rev.groupby("tmdb_id")["sentiment"]
        .apply(lambda s: (s == "positive").mean())
        .rename("positive_ratio")
    )
    intensity_per_movie = rev.groupby("tmdb_id")["intensity"].mean().rename("avg_intensity")

    # 海报特征 → 取每部电影所有海报特征的均值，再降到 8 维 PCA
    from sklearn.decomposition import PCA

    img_df = img_df.reset_index(drop=True)
    movie_feats: dict[int, np.ndarray] = {}
    for tid in img_df["tmdb_id"].unique():
        mask = (img_df["tmdb_id"] == tid).values
        movie_feats[int(tid)] = clip_mat[mask].mean(axis=0)
    feat_df = pd.DataFrame.from_dict(movie_feats, orient="index")
    feat_df.index.name = "tmdb_id"
    pca = PCA(n_components=min(8, feat_df.shape[1], feat_df.shape[0]))
    pcs = pca.fit_transform(feat_df.values)
    pc_df = pd.DataFrame(pcs, index=feat_df.index, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])

    merged = pc_df.join(sent_per_movie).join(intensity_per_movie).dropna()
    if len(merged) < 3:
        log.warning("可用电影数过少，跳过热力图")
        return

    corr = merged.corr().loc[
        ["positive_ratio", "avg_intensity"], [c for c in merged.columns if c.startswith("PC")]
    ]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Cross-Modal Correlation: Review Sentiment ↔ Poster Visual PCs", fontweight="bold")
    _save(fig, "07_cross_modal_heatmap.png")


# ---------- 一键画完 ----------
def plot_all():
    log.info("============ 可视化开始 ============")
    review_csv = get_path("data/processed/reviews_with_sentiment.csv")
    if not review_csv.exists():
        review_csv = get_path("data/processed/all_reviews.csv")
    df = pd.read_csv(review_csv)

    plot_review_counts(df)
    plot_sentiment_distribution(df)
    plot_wordclouds(df)
    plot_text_tsne(df)
    plot_image_tsne()
    plot_poster_grid()
    plot_cross_modal_heatmap()
    log.info("============ 可视化完成 ============")


if __name__ == "__main__":
    plot_all()
