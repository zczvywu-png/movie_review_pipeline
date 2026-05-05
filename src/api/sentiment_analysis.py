"""
GPT 情感分析（批量打标）
================================

设计依据
- 给抓回来的评论批量打"情感 + 主题 + 情绪强度"标签，用于：
    1) 后续 ML 模块的"情感分类监督学习"训练标签来源；
    2) 可视化里"不同电影类型的情感分布"分析。
- 一次给 GPT 喂 10 条评论批量打分，比逐条调省 10 倍 API 费。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.api.gpt_client import chat_json, make_client
from src.utils import get_logger, get_path, load_config

log = get_logger("sentiment")

SYSTEM_PROMPT = """You are an expert movie review analyst. For each review I give you,
return a JSON object with these fields:
- sentiment: one of "positive", "negative", "neutral"
- intensity: integer 1-5 (1 = mild, 5 = very strong)
- topic: one of "acting", "plot", "visuals", "music", "general", "off_topic"
- contains_spoiler: boolean
Return ONLY a JSON object, no commentary."""

BATCH_SYSTEM_PROMPT = """You are an expert movie review analyst. I will give you a JSON list of reviews.
For each review, return a JSON object with the same index containing:
- sentiment: "positive" | "negative" | "neutral"
- intensity: integer 1-5
- topic: "acting" | "plot" | "visuals" | "music" | "general" | "off_topic"
- contains_spoiler: true | false

Return ONLY a JSON object of the form:
{"results": [{"index": 0, "sentiment": "...", ...}, {"index": 1, ...}, ...]}
Do not include any other text."""


def annotate_batch(
    client, batch: list[str], model: str, batch_idx: int = 0
) -> list[dict]:
    payload = [{"index": i, "review": t[:600]} for i, t in enumerate(batch)]
    user_msg = "Reviews:\n" + str(payload)
    try:
        result = chat_json(
            client,
            messages=[
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.0,
            max_tokens=900,
        )
        items = result.get("results", []) if isinstance(result, dict) else []
        # 用 index 排回原顺序，缺失的填默认值
        out = [_default_label() for _ in batch]
        for it in items:
            i = it.get("index")
            if isinstance(i, int) and 0 <= i < len(batch):
                out[i] = {
                    "sentiment": it.get("sentiment", "neutral"),
                    "intensity": int(it.get("intensity", 3) or 3),
                    "topic": it.get("topic", "general"),
                    "contains_spoiler": bool(it.get("contains_spoiler", False)),
                }
        return out
    except Exception as e:
        log.warning(f"[batch {batch_idx}] GPT 失败：{e}，回退默认标签")
        return [_default_label() for _ in batch]


def _default_label() -> dict:
    return {"sentiment": "neutral", "intensity": 3, "topic": "general", "contains_spoiler": False}


def annotate_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    sample_size: int | None = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    给整张评论表打标签。

    sample_size 控制抽样规模（控制成本，None 表示全部）。
    返回值在原 df 基础上新增 sentiment / intensity / topic / contains_spoiler 四列。
    """
    cfg = load_config()["openai"]
    model = cfg["model"]
    batch_size = cfg["batch_size"]

    if sample_size and len(df) > sample_size:
        # 按 genre 等比例抽样，避免某类型缺失
        df_work = (
            df.groupby("genre", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), sample_size // df["genre"].nunique()), random_state=42))
            .reset_index(drop=True)
        )
        log.info(f"按类型分层抽样 {len(df_work)} / {len(df)} 条评论送给 GPT")
    else:
        df_work = df.reset_index(drop=True).copy()

    client = make_client()
    texts = df_work[text_col].astype(str).tolist()
    results: list[dict | None] = [None] * len(texts)

    batches = [(i, texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]

    def _worker(args):
        idx, batch = args
        out = annotate_batch(client, batch, model=model, batch_idx=idx)
        return idx, out

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="GPT 打标"):
            idx, labels = fut.result()
            for j, lab in enumerate(labels):
                results[idx + j] = lab

    out_df = df_work.copy()
    out_df["sentiment"] = [r["sentiment"] for r in results]
    out_df["intensity"] = [r["intensity"] for r in results]
    out_df["topic"] = [r["topic"] for r in results]
    out_df["contains_spoiler"] = [r["contains_spoiler"] for r in results]

    save_to = get_path("data/processed/reviews_with_sentiment.csv")
    out_df.to_csv(save_to, index=False, encoding="utf-8-sig")
    log.info(f"打标完成 → {save_to}")
    log.info(
        "情感分布：\n" + out_df["sentiment"].value_counts().to_string()
    )
    return out_df


if __name__ == "__main__":
    df = pd.read_csv(get_path("data/processed/all_reviews.csv"))
    cfg = load_config()
    annotate_dataframe(df, sample_size=cfg["openai"]["sentiment_sample_size"])
