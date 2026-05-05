"""
Movie Review Pipeline 主入口
================================

支持单步执行或一键全跑：

    python main.py scrape          # 1. 抓 YouTube + IMDb + TMDB
    python main.py merge           # 2. 合并评论数据源
    python main.py vectorize       # 3. 文本 + 图像向量化对比
    python main.py annotate        # 4. GPT 情感打标
    python main.py augment         # 5. GPT 合成评论增强
    python main.py ml              # 6. 聚类 + 情感分类 + 海报分类
    python main.py viz             # 7. 全部可视化
    python main.py agent "你的提问" # 8. 调用 Agent
    python main.py all             # 9. 一键全跑
    python main.py smoke           # 10. 冒烟测试（小规模，验证管线）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# 让 `python main.py` 能直接 import src.*
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, get_path, load_config

log = get_logger("main")


def step_scrape(smoke: bool = False):
    from src.scrapers.imdb_scraper import scrape_all_imdb
    from src.scrapers.tmdb_scraper import scrape_all_tmdb
    from src.scrapers.youtube_scraper import scrape_all_movies, scrape_smoke_test

    cfg = load_config()
    log.info("====== STEP 1/8: 抓取 YouTube 评论 ======")
    if smoke:
        scrape_smoke_test(cfg, n=5)
    else:
        scrape_all_movies(cfg)

    log.info("====== STEP 2/8: 抓取 IMDb 影评 ======")
    if smoke:
        cfg_smoke = {**cfg, "scrape": {**cfg["scrape"], "imdb_reviews_per_movie": 5}}
        scrape_all_imdb(cfg_smoke)
    else:
        scrape_all_imdb(cfg)

    log.info("====== STEP 3/8: 抓取 TMDB 海报 + 元数据 ======")
    try:
        scrape_all_tmdb(cfg)
    except RuntimeError as e:
        log.warning(f"TMDB 抓取跳过：{e}")


def step_merge():
    from src.scrapers.merge_data import merge_reviews
    log.info("====== 合并评论数据 ======")
    merge_reviews()


def step_vectorize():
    log.info("====== 文本向量化对比 ======")
    from src.vectorize.text_vectorize import compare_vectorizers
    df = pd.read_csv(get_path("data/processed/all_reviews.csv"))
    compare_vectorizers(df, label_col="genre", text_col="text")

    log.info("====== 图像向量化对比 ======")
    img_csv = get_path("data/raw/tmdb_images.csv")
    if img_csv.exists():
        from src.vectorize.image_vectorize import compare_image_vectorizers
        compare_image_vectorizers()
    else:
        log.warning("无 TMDB 图像，跳过图像向量化")


def step_annotate():
    log.info("====== GPT 情感打标 ======")
    from src.api.sentiment_analysis import annotate_dataframe
    cfg = load_config()
    df = pd.read_csv(get_path("data/processed/all_reviews.csv"))
    annotate_dataframe(df, sample_size=cfg["openai"]["sentiment_sample_size"])


def step_augment():
    log.info("====== GPT 合成评论增强 ======")
    from src.api.synthetic_reviews import augment_dataset
    cfg = load_config()
    real = pd.read_csv(get_path("data/processed/all_reviews.csv"))
    meta_csv = get_path("data/raw/tmdb_metadata.csv")
    meta = pd.read_csv(meta_csv) if meta_csv.exists() else None
    augment_dataset(real, meta, per_genre=cfg["openai"]["synthetic_per_genre"])


def step_ml():
    log.info("====== ML 聚类 ======")
    from src.ml.text_clustering import run_full_clustering
    df = pd.read_csv(get_path("data/processed/all_reviews.csv"))
    run_full_clustering(df)

    log.info("====== ML 情感分类 ======")
    sent_csv = get_path("data/processed/reviews_with_sentiment.csv")
    if sent_csv.exists():
        from src.ml.sentiment_classifier import benchmark_sentiment
        df_s = pd.read_csv(sent_csv)
        benchmark_sentiment(df_s)
    else:
        log.warning("缺情感标签 CSV，跳过情感分类")

    log.info("====== ML 海报分类 ======")
    posters_root = get_path("data/posters")
    if any(posters_root.iterdir()):
        from src.ml.poster_classifier import train_poster_classifier
        train_poster_classifier()
    else:
        log.warning("无海报数据，跳过海报分类")


def step_viz():
    log.info("====== 可视化 ======")
    from src.viz.visualize import plot_all
    plot_all()


def step_agent(question: str):
    log.info(f"====== Agent 提问：{question} ======")
    from src.api.movie_agent import run_agent
    print("\n=== Agent Answer ===\n", run_agent(question))


def step_all():
    t0 = time.time()
    step_scrape()
    step_merge()
    step_vectorize()
    step_annotate()
    step_augment()
    step_ml()
    step_viz()
    log.info(f"全流程总耗时：{(time.time()-t0)/60:.1f} 分钟")


def step_smoke():
    """冒烟测试：每视频抓 5 条，跑通整条链"""
    log.info("====== 冒烟测试 ======")
    step_scrape(smoke=True)

    # 把 youtube_comments_smoke.csv 当主源
    smoke_yt = get_path("data/raw/youtube_comments_smoke.csv")
    main_yt = get_path("data/raw/youtube_comments.csv")
    if smoke_yt.exists() and not main_yt.exists():
        import shutil
        shutil.copy(smoke_yt, main_yt)

    step_merge()
    step_vectorize()
    log.info("冒烟测试完成（GPT/ML/可视化在小数据上意义不大，跳过）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    actions = {
        "scrape": step_scrape,
        "merge": step_merge,
        "vectorize": step_vectorize,
        "annotate": step_annotate,
        "augment": step_augment,
        "ml": step_ml,
        "viz": step_viz,
        "all": step_all,
        "smoke": step_smoke,
    }

    if cmd == "agent":
        q = " ".join(sys.argv[2:]) or "Compare audience sentiment between action and horror movies."
        step_agent(q)
    elif cmd in actions:
        actions[cmd]()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
        sys.exit(1)
