"""
将 YouTube + IMDb 两个评论源合并为统一长表
================================

设计依据
- 后续向量化 / ML / 可视化都需要一张"统一 schema"的评论表。
- 字段对齐：text / source / genre / movie_title / tmdb_id / imdb_id / rating / votes
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_logger, get_path

log = get_logger("merge")

UNIFIED_COLS = [
    "text",
    "raw_text",
    "source",
    "genre",
    "movie_title",
    "movie_year",
    "tmdb_id",
    "imdb_id",
    "rating",
    "votes",
    "author",
]


def _safe_read_csv(path) -> pd.DataFrame | None:
    """读取 CSV，文件不存在 / 文件为空 / 仅有 header 时返回 None。"""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        log.warning(f"{path.name} 为空文件，跳过")
        return None
    if df.empty:
        log.warning(f"{path.name} 没有数据行，跳过")
        return None
    return df


def merge_reviews() -> pd.DataFrame:
    yt_csv = get_path("data/raw/youtube_comments.csv")
    imdb_csv = get_path("data/raw/imdb_reviews.csv")

    frames = []

    yt = _safe_read_csv(yt_csv)
    if yt is not None:
        log.info(f"YouTube: {len(yt)} 条")
        yt["rating"] = pd.NA
        frames.append(yt)

    im = _safe_read_csv(imdb_csv)
    if im is not None:
        log.info(f"IMDb: {len(im)} 条")
        im["votes"] = im.get("helpful_yes", pd.Series(dtype=float))
        frames.append(im)

    if not frames:
        raise FileNotFoundError(
            "没有可用的评论数据：YouTube / IMDb 原始 CSV 都为空或不存在，"
            "请先跑 `python main.py scrape`"
        )

    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in UNIFIED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[UNIFIED_COLS + [c for c in df.columns if c not in UNIFIED_COLS]]

    # 去重 + 过滤过短
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.len() >= 10].reset_index(drop=True)
    df = df.drop_duplicates(subset=["text", "movie_title"]).reset_index(drop=True)

    out = get_path("data/processed/all_reviews.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"合并后：{len(df)} 条 → {out}")
    log.info("按来源:\n" + df["source"].value_counts().to_string())
    log.info("按类型:\n" + df["genre"].value_counts().to_string())
    return df


if __name__ == "__main__":
    merge_reviews()
