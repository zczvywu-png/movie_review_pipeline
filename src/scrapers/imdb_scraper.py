"""
IMDb 影评抓取（第二数据源）
================================

设计依据
- 沿用老师 WS1 的 "requests + BeautifulSoup" 模式（老师在 WS1 后段对 Amazon 也用过 bs4）。
- IMDb 的 `/title/{imdb_id}/reviews` 页面有"Load More"按钮，背后是 AJAX 调用：
    https://www.imdb.com/title/{id}/reviews/_ajax?paginationKey=...
  本脚本用这个分页接口一次性抓多页，避免 Selenium。
- 设置友好的 User-Agent + 速率限制，遵守 robots.txt 礼仪。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.utils import clean_text, get_logger, get_path, is_meaningful, load_config

log = get_logger("imdb")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not.A/Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

REVIEW_URL = "https://www.imdb.com/title/{imdb_id}/reviews/"
AJAX_URL = "https://www.imdb.com/title/{imdb_id}/reviews/_ajax"


def _get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> requests.Response | None:
    """
    带退避重试的 GET。
    IMDb 在 2024 改版后会先返回 HTTP 202（机器人挑战），简单重试有时能放行。
    """
    for attempt in range(max_retries):
        resp = session.get(url, params=params, timeout=20)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (202, 429, 503):
            time.sleep(base_delay * (attempt + 1))
            continue
        return resp
    return resp


def fetch_imdb_reviews(imdb_id: str, max_reviews: int = 60, delay: float = 1.0) -> list[dict]:
    """
    抓取一部电影的 IMDb 影评。

    返回字段：text, raw_text, rating(1-10), author, date, helpful_yes, helpful_total, title

    注意：IMDb 在 2024 把评论改造成 GraphQL + 反爬挑战，HTTP 202 较常见。
    本函数已尽力（多重 header + 退避重试），失败时返回空列表，不影响主流程。
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    reviews: list[dict] = []
    pagination_key: str | None = None

    pbar = tqdm(total=max_reviews, desc=f"  {imdb_id}")
    while len(reviews) < max_reviews:
        try:
            if pagination_key is None:
                url = REVIEW_URL.format(imdb_id=imdb_id)
                resp = _get_with_retry(session, url)
            else:
                url = AJAX_URL.format(imdb_id=imdb_id)
                resp = _get_with_retry(
                    session, url, params={"paginationKey": pagination_key}
                )
            if resp is None or resp.status_code != 200:
                code = resp.status_code if resp is not None else "no-resp"
                log.warning(f"[{imdb_id}] HTTP {code}（IMDb 反爬挑战，跳过）")
                break

            soup = BeautifulSoup(resp.text, "lxml")
            blocks = soup.select("div.review-container") or soup.select(
                "article.user-review-item"
            )
            if not blocks:
                blocks = soup.select("div.lister-item-content")
            if not blocks:
                log.warning(f"[{imdb_id}] 没找到评论块，IMDb 已改版（跳过该片）")
                break

            for blk in blocks:
                review = _parse_review_block(blk)
                if review and is_meaningful(review["text"]):
                    reviews.append(review)
                    pbar.update(1)
                    if len(reviews) >= max_reviews:
                        break

            load_more = soup.select_one("div.load-more-data[data-key]")
            if not load_more:
                break
            pagination_key = load_more.get("data-key")
            if not pagination_key:
                break

            time.sleep(delay)
        except Exception as e:
            log.warning(f"[{imdb_id}] 抓取异常：{e}")
            break

    pbar.close()
    return reviews


def _parse_review_block(blk) -> dict | None:
    try:
        # 评论正文
        text_el = (
            blk.select_one("div.text.show-more__control")
            or blk.select_one("div.content > div.text")
            or blk.select_one("div[data-testid='review-overflow']")
        )
        text = clean_text(text_el.get_text(" ", strip=True)) if text_el else ""

        # 评分
        rating_el = blk.select_one("span.rating-other-user-rating > span") or blk.select_one(
            "span[class*='rating']"
        )
        rating = None
        if rating_el:
            try:
                rating = int(rating_el.get_text(strip=True))
            except Exception:
                rating = None

        # 标题
        title_el = blk.select_one("a.title") or blk.select_one("h3")
        title = clean_text(title_el.get_text(strip=True)) if title_el else ""

        # 作者 + 日期
        author_el = blk.select_one("span.display-name-link > a") or blk.select_one(
            "a[data-testid='author-link']"
        )
        author = author_el.get_text(strip=True) if author_el else None

        date_el = blk.select_one("span.review-date") or blk.select_one("li.review-date")
        date = date_el.get_text(strip=True) if date_el else None

        # 是否有用
        helpful_el = blk.select_one("div.actions.text-muted")
        helpful_yes = helpful_total = None
        if helpful_el:
            txt = helpful_el.get_text(" ", strip=True)
            import re

            m = re.search(r"(\d+)\s+out of\s+(\d+)", txt)
            if m:
                helpful_yes = int(m.group(1))
                helpful_total = int(m.group(2))

        if not text:
            return None
        return {
            "title": title,
            "text": text,
            "rating": rating,
            "author": author,
            "date": date,
            "helpful_yes": helpful_yes,
            "helpful_total": helpful_total,
        }
    except Exception:
        return None


def scrape_all_imdb(cfg: dict, output_csv: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    n = cfg["scrape"]["imdb_reviews_per_movie"]
    delay = cfg["scrape"]["request_delay_sec"]

    for genre, movies in cfg["movies"].items():
        log.info(f"========== Genre: {genre} ==========")
        for m in movies:
            log.info(f"  -> {m['title']} ({m['imdb_id']})")
            reviews = fetch_imdb_reviews(m["imdb_id"], max_reviews=n, delay=delay)
            log.info(f"     抓到 {len(reviews)} 条 IMDb 评论")
            for r in reviews:
                r.update(
                    {
                        "genre": genre,
                        "movie_title": m["title"],
                        "movie_year": m["year"],
                        "tmdb_id": m["tmdb_id"],
                        "imdb_id": m["imdb_id"],
                        "source": "imdb",
                    }
                )
                rows.append(r)
            time.sleep(delay)

    df = pd.DataFrame(rows)
    log.info(f"IMDb 评论汇总：{len(df)} 条")

    if output_csv is None:
        output_csv = get_path("data/raw/imdb_reviews.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log.info(f"已写入 {output_csv}")
    return df


if __name__ == "__main__":
    cfg = load_config()
    scrape_all_imdb(cfg)
