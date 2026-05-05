"""
用 GPT 生成"合成影评"做数据增强
================================

设计依据
- 作业要求"使用生成 API 来增强您的数据集或生成与您的主题相关的合成内容"。
- 给 GPT 一段 few-shot 真实评论作风格示例 + 电影元数据，让它生成新评论。
- 生成的评论会和真实评论合并到一张表，新增列 `is_synthetic=True` 标识来源，
  避免数据泄漏到测试集。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.api.gpt_client import chat_json, make_client
from src.utils import get_logger, get_path, load_config

log = get_logger("synthetic")

GEN_SYSTEM_PROMPT = """You are a movie fan writing short YouTube comments under a film trailer.
Mimic the casual style of real YouTube comments (short, emotional, sometimes with mild slang).
Avoid quoting the example reviews verbatim. Each comment must be 1-3 sentences."""

GEN_USER_TEMPLATE = """Movie: {title} ({year})
Genre: {genre}
Synopsis: {overview}

Here are 3 real fan comments for style reference:
1. {ex1}
2. {ex2}
3. {ex3}

Now generate {n} NEW, ORIGINAL fan comments for this trailer.
Return strictly as JSON: {{"comments": ["...", "...", ...]}}"""


def generate_for_movie(
    client,
    movie: dict,
    examples: list[str],
    n: int = 5,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """给单部电影生成 n 条合成评论"""
    ex = random.sample(examples, k=min(3, len(examples))) if examples else ["Looks great!", "Can't wait!", "This will be epic"]
    while len(ex) < 3:
        ex.append("Excited for this one")

    prompt = GEN_USER_TEMPLATE.format(
        title=movie.get("title", "Unknown"),
        year=movie.get("movie_year") or movie.get("release_date", "")[:4],
        genre=movie.get("genre", ""),
        overview=(movie.get("overview") or "")[:400],
        ex1=ex[0][:200],
        ex2=ex[1][:200],
        ex3=ex[2][:200],
        n=n,
    )
    try:
        result = chat_json(
            client,
            messages=[
                {"role": "system", "content": GEN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.9,
            max_tokens=600,
        )
        comments = result.get("comments", []) if isinstance(result, dict) else []
        return [c for c in comments if isinstance(c, str) and 5 < len(c) < 600][:n]
    except Exception as e:
        log.warning(f"[{movie.get('title')}] 生成失败：{e}")
        return []


def augment_dataset(
    real_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
    per_genre: int = 20,
) -> pd.DataFrame:
    """
    给每个类型的每部电影生成若干合成评论，最后拼接到真实评论 DataFrame。
    """
    cfg = load_config()
    client = make_client()
    model = cfg["openai"]["model"]

    # 把元数据按 tmdb_id 索引
    meta_lookup: dict[int, dict] = {}
    if metadata_df is not None:
        for _, row in metadata_df.iterrows():
            meta_lookup[int(row["tmdb_id"])] = row.to_dict()

    rows: list[dict] = []
    movie_keys = real_df[["genre", "movie_title", "tmdb_id"]].drop_duplicates()

    # 平摊到每部电影：per_genre 是每个类型总量
    n_movies_per_genre = movie_keys.groupby("genre").size().to_dict()

    for _, m in tqdm(list(movie_keys.iterrows()), desc="合成评论"):
        genre = m["genre"]
        n_for_movie = max(1, per_genre // max(n_movies_per_genre[genre], 1))
        examples = (
            real_df[real_df["tmdb_id"] == m["tmdb_id"]]["text"].astype(str).tolist()
        )
        movie_info = {
            "title": m["movie_title"],
            "tmdb_id": m["tmdb_id"],
            "genre": genre,
            "movie_year": meta_lookup.get(int(m["tmdb_id"]), {}).get("release_date", "")[:4],
            "overview": meta_lookup.get(int(m["tmdb_id"]), {}).get("overview", ""),
        }
        synthetic = generate_for_movie(client, movie_info, examples, n=n_for_movie, model=model)
        for s in synthetic:
            rows.append(
                {
                    "text": s,
                    "raw_text": s,
                    "genre": genre,
                    "movie_title": m["movie_title"],
                    "tmdb_id": m["tmdb_id"],
                    "source": "gpt_synthetic",
                    "is_synthetic": True,
                }
            )

    syn_df = pd.DataFrame(rows)
    syn_df.to_csv(get_path("data/processed/synthetic_reviews.csv"), index=False, encoding="utf-8-sig")
    log.info(f"生成 {len(syn_df)} 条合成评论")

    # 给真实评论加 is_synthetic=False
    real_marked = real_df.copy()
    if "is_synthetic" not in real_marked.columns:
        real_marked["is_synthetic"] = False

    combined = pd.concat([real_marked, syn_df], ignore_index=True, sort=False)
    out = get_path("data/processed/reviews_augmented.csv")
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"增强后总评论数：{len(combined)} → {out}")
    return combined


if __name__ == "__main__":
    real = pd.read_csv(get_path("data/processed/all_reviews.csv"))
    meta = pd.read_csv(get_path("data/raw/tmdb_metadata.csv"))
    cfg = load_config()
    augment_dataset(real, meta, per_genre=cfg["openai"]["synthetic_per_genre"])
