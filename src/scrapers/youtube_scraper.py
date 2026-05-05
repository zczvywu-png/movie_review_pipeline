"""
YouTube 电影预告片评论抓取
================================

设计依据
- 老师 WS1 用 Selenium 抓 Amazon 的"无限滚动 + 翻页"流程，YouTube 评论同样是无限滚动，
  但 YouTube DOM 在 Shadow DOM 内，Selenium 选择器极易失效。
- 改用 `youtube-comment-downloader`：
    - 它直接命中 YouTube 内部 InnerTube API，比 Selenium 稳得多；
    - 对评论翻页/排序的处理已经由社区维护好；
    - 不需要 YouTube Data API Key（避免每天 10000 quota 限制）。
- 仍然保留"按视频抓 -> 限速 -> 清洗 -> 落库"的整体管线，与 WS1 思路一致。
"""
from __future__ import annotations

import json
import time
from itertools import islice
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from youtube_comment_downloader import (
    SORT_BY_POPULAR,
    YoutubeCommentDownloader,
)

from src.utils import clean_text, get_logger, get_path, is_meaningful, load_config

log = get_logger("youtube")


def fetch_comments_for_video(
    video_id: str,
    max_comments: int = 250,
    sort_by: int = SORT_BY_POPULAR,
    language: str = "en",
) -> list[dict]:
    """
    抓单个视频的评论。

    参数
    ----
    video_id : YouTube 视频 ID（11 位）
    max_comments : 最多抓多少条
    sort_by : 排序方式 0=top, 1=newest
    language : 语言，影响 YouTube 返回的本地化字段
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    downloader = YoutubeCommentDownloader()

    try:
        gen = downloader.get_comments_from_url(url, sort_by=sort_by, language=language)
    except Exception as e:
        log.warning(f"[{video_id}] 抓取失败：{e}")
        return []

    comments: list[dict] = []
    for c in tqdm(islice(gen, max_comments * 2), total=max_comments, desc=f"  {video_id}"):
        # downloader 返回字段：cid, text, time, author, channel, votes, replies, photo, heart, reply, time_parsed
        text = clean_text(c.get("text", ""), keep_emoji=False)
        if not is_meaningful(text):
            continue
        comments.append(
            {
                "comment_id": c.get("cid"),
                "text": text,
                "raw_text": c.get("text", ""),
                "author": c.get("author"),
                "votes": _to_int(c.get("votes", 0)),
                "replies": _to_int(c.get("replies", 0)),
                "time": c.get("time"),
                "is_reply": bool(c.get("reply", False)),
                "is_hearted": bool(c.get("heart", False)),
            }
        )
        if len(comments) >= max_comments:
            break

    return comments


def _trailer_ids(movie: dict) -> list[str]:
    """从 movie 配置里取出预告片 ID 列表，兼容字符串和列表两种写法。"""
    raw = (
        movie.get("youtube_trailer_ids")
        or movie.get("youtube_trailer_id")
        or []
    )
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    return []


def _to_int(v) -> int:
    """评论里点赞数会出现 '1.2K' / '3M' 这种字符串"""
    if isinstance(v, int):
        return v
    if not v:
        return 0
    s = str(v).strip().upper().replace(",", "")
    try:
        if s.endswith("K"):
            return int(float(s[:-1]) * 1000)
        if s.endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        return int(float(s))
    except Exception:
        return 0


def scrape_all_movies(
    cfg: dict,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    遍历 config 中所有电影，按类型抓评论，统一落到一张大表。
    """
    rows: list[dict] = []
    n_per_video = cfg["scrape"]["comments_per_video"]
    delay = cfg["scrape"]["request_delay_sec"]
    movies_cfg = cfg["movies"]

    for genre, movies in movies_cfg.items():
        log.info(f"========== Genre: {genre} ==========")
        for m in movies:
            tids = _trailer_ids(m)
            comments: list[dict] = []
            used_tid: str | None = None
            for tid in tids:
                log.info(f"  -> {m['title']} ({m['year']}) | trying trailer={tid}")
                comments = fetch_comments_for_video(tid, max_comments=n_per_video)
                log.info(f"     抓到有效评论 {len(comments)} 条")
                if comments:
                    used_tid = tid
                    break
                if len(tids) > 1:
                    log.warning(f"     0 条，尝试下一个备用预告片...")
                time.sleep(delay)

            if not comments:
                log.warning(f"     {m['title']} 全部预告片均为空，跳过")
                continue

            for c in comments:
                c.update(
                    {
                        "genre": genre,
                        "movie_title": m["title"],
                        "movie_year": m["year"],
                        "tmdb_id": m["tmdb_id"],
                        "imdb_id": m["imdb_id"],
                        "trailer_id": used_tid,
                        "source": "youtube",
                    }
                )
                rows.append(c)
            time.sleep(delay)

    df = pd.DataFrame(rows)
    log.info(f"YouTube 评论汇总：{len(df)} 条，覆盖 {df['movie_title'].nunique()} 部电影")

    if output_csv is None:
        output_csv = get_path("data/raw/youtube_comments.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log.info(f"已写入 {output_csv}")
    return df


def scrape_smoke_test(cfg: dict, n: int = 5) -> pd.DataFrame:
    """冒烟测试：每个视频只抓 n 条"""
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["scrape"]["comments_per_video"] = n
    out = get_path("data/raw/youtube_comments_smoke.csv")
    return scrape_all_movies(cfg, output_csv=out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="冒烟测试，每视频抓 5 条")
    args = parser.parse_args()

    cfg = load_config()
    if args.smoke:
        scrape_smoke_test(cfg, n=5)
    else:
        scrape_all_movies(cfg)
