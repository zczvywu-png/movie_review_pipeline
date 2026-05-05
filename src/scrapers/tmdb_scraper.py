"""
TMDB 海报 / 元数据抓取（图像数据源 + 元数据补全）
================================

设计依据
- TMDB 提供官方免费 REST API，无需 Selenium，比 IMDb 抓图更稳定。
- 抓两类资源：
    1) 电影元数据（评分、概述、类型、时长）→ 给 ML 阶段做特征/标签
    2) 海报图片（按类型分目录保存）→ 给 CLIP/ResNet50 对比 + MobileNet 分类
- 海报存储约定：
    data/posters/{genre}/{tmdb_id}_{idx}.jpg
- 一并抓取多张海报（不同语言版本/横版背景图）以扩充数据集，向作业要求的 200-300 元素靠拢。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.utils import ensure_dir, env, get_logger, get_path, load_config

log = get_logger("tmdb")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"   # 500 像素宽，足够 224×224 训练


def _api_key() -> str:
    key = env("TMDB_API_KEY")
    if not key or key.startswith("xxxx"):
        raise RuntimeError(
            "未设置 TMDB_API_KEY。请把 .env.example 改名 .env 并填入真实 Key（免费注册 https://www.themoviedb.org/settings/api）"
        )
    return key


def fetch_movie_metadata(tmdb_id: int) -> dict:
    url = f"{TMDB_BASE}/movie/{tmdb_id}"
    resp = requests.get(url, params={"api_key": _api_key(), "language": "en-US"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        "tmdb_id": tmdb_id,
        "title": data.get("title"),
        "original_title": data.get("original_title"),
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        "overview": data.get("overview"),
        "tagline": data.get("tagline"),
        "tmdb_genres": [g["name"] for g in data.get("genres", [])],
        "poster_path": data.get("poster_path"),
        "backdrop_path": data.get("backdrop_path"),
    }


def fetch_movie_images(tmdb_id: int, max_posters: int = 8, max_backdrops: int = 4) -> dict:
    """获取一部电影的所有海报和背景图列表（不下载文件，只返回 URL）"""
    url = f"{TMDB_BASE}/movie/{tmdb_id}/images"
    resp = requests.get(url, params={"api_key": _api_key()}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    posters = [TMDB_IMG_BASE + p["file_path"] for p in data.get("posters", [])[:max_posters]]
    backdrops = [
        TMDB_IMG_BASE + b["file_path"] for b in data.get("backdrops", [])[:max_backdrops]
    ]
    return {"posters": posters, "backdrops": backdrops}


def download_image(url: str, save_to: Path) -> bool:
    if save_to.exists() and save_to.stat().st_size > 1024:
        return True
    try:
        resp = requests.get(url, timeout=20, stream=True)
        if resp.status_code != 200:
            return False
        save_to.write_bytes(resp.content)
        return True
    except Exception as e:
        log.warning(f"下载失败 {url}: {e}")
        return False


def scrape_all_tmdb(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (metadata_df, posters_df)
    metadata_df : 每部电影一行
    posters_df  : 每张图一行，含本地文件路径，可直接喂给 CLIP/ResNet50
    """
    meta_rows: list[dict] = []
    img_rows: list[dict] = []
    posters_root = ensure_dir(get_path("data/posters"))
    delay = cfg["scrape"]["request_delay_sec"]

    # 为达到 200-300 张图像数据集要求，每部电影抓 8 张 poster + 4 张 backdrop
    max_posters = 8
    max_backdrops = 4

    for genre, movies in cfg["movies"].items():
        genre_dir = ensure_dir(posters_root / genre)
        log.info(f"========== Genre: {genre} ==========")
        for m in tqdm(movies, desc=f"  {genre}"):
            tmdb_id = m["tmdb_id"]
            try:
                meta = fetch_movie_metadata(tmdb_id)
                meta["genre"] = genre
                meta["movie_title_cfg"] = m["title"]
                meta["imdb_id"] = m["imdb_id"]
                meta_rows.append(meta)
            except Exception as e:
                log.warning(f"[{tmdb_id}] 元数据失败：{e}")
                continue

            try:
                imgs = fetch_movie_images(tmdb_id, max_posters, max_backdrops)
            except Exception as e:
                log.warning(f"[{tmdb_id}] 图片列表失败：{e}")
                continue

            for idx, url in enumerate(imgs["posters"]):
                fname = f"{tmdb_id}_p{idx}.jpg"
                fpath = genre_dir / fname
                ok = download_image(url, fpath)
                if ok:
                    img_rows.append(
                        {
                            "tmdb_id": tmdb_id,
                            "movie_title": meta["title"],
                            "genre": genre,
                            "image_type": "poster",
                            "image_idx": idx,
                            "url": url,
                            "local_path": str(fpath.relative_to(get_path("."))),
                        }
                    )
            for idx, url in enumerate(imgs["backdrops"]):
                fname = f"{tmdb_id}_b{idx}.jpg"
                fpath = genre_dir / fname
                ok = download_image(url, fpath)
                if ok:
                    img_rows.append(
                        {
                            "tmdb_id": tmdb_id,
                            "movie_title": meta["title"],
                            "genre": genre,
                            "image_type": "backdrop",
                            "image_idx": idx,
                            "url": url,
                            "local_path": str(fpath.relative_to(get_path("."))),
                        }
                    )
            time.sleep(delay)

    meta_df = pd.DataFrame(meta_rows)
    img_df = pd.DataFrame(img_rows)
    log.info(f"TMDB 元数据：{len(meta_df)} 部电影")
    log.info(f"TMDB 图片：{len(img_df)} 张（按类型分目录保存到 data/posters/）")

    meta_df.to_csv(get_path("data/raw/tmdb_metadata.csv"), index=False, encoding="utf-8-sig")
    img_df.to_csv(get_path("data/raw/tmdb_images.csv"), index=False, encoding="utf-8-sig")
    return meta_df, img_df


if __name__ == "__main__":
    cfg = load_config()
    scrape_all_tmdb(cfg)
