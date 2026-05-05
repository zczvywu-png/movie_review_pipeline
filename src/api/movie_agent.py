"""
简易电影评论 Agent（ReAct 风格）
================================

设计依据
- 作业要求"您可以根据各种书籍开发人工智能代理"。本项目主题是电影，对应改成
  "电影评论分析 Agent"。
- Agent 暴露 4 个工具：
    1) search_movies(query)        : 在已抓数据里按标题模糊搜索
    2) get_movie_summary(tmdb_id)  : 返回该电影的评论统计 + GPT 总结
    3) compare_genres(g1, g2)      : 对比两个类型的情感分布
    4) recommend_similar(tmdb_id)  : 用 CLIP/Doc2Vec 做相似检索给出推荐
- 没用 LangChain 等重框架，手写 ReAct 循环，老师能直接看懂代码。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.gpt_client import chat, make_client
from src.utils import get_logger, get_path, load_config

log = get_logger("agent")


# ---------- 工具实现 ----------
class MovieAgentTools:
    def __init__(self):
        self.reviews_df: pd.DataFrame | None = None
        self.meta_df: pd.DataFrame | None = None
        self.text_matrix: np.ndarray | None = None
        self.image_matrix: np.ndarray | None = None
        self._load()

    def _load(self):
        # 评论 + 情感标签（如果已生成）
        sentiment_csv = get_path("data/processed/reviews_with_sentiment.csv")
        all_csv = get_path("data/processed/all_reviews.csv")
        if sentiment_csv.exists():
            self.reviews_df = pd.read_csv(sentiment_csv)
        elif all_csv.exists():
            self.reviews_df = pd.read_csv(all_csv)
        else:
            log.warning("尚无评论 CSV，部分工具不可用")

        meta_csv = get_path("data/raw/tmdb_metadata.csv")
        if meta_csv.exists():
            self.meta_df = pd.read_csv(meta_csv)

        # Doc2Vec 矩阵（用于推荐）
        d2v_path = get_path("outputs/models/text_vec/doc2vec_matrix.npy")
        if d2v_path.exists():
            self.text_matrix = np.load(d2v_path)

    # ---- tool 1 ----
    def search_movies(self, query: str) -> list[dict]:
        if self.meta_df is None:
            return []
        q = query.lower()
        hits = self.meta_df[self.meta_df["title"].str.lower().str.contains(q, na=False)]
        return hits[["tmdb_id", "title", "release_date", "vote_average", "genre"]].head(5).to_dict("records")

    # ---- tool 2 ----
    def get_movie_summary(self, tmdb_id: int) -> dict:
        if self.reviews_df is None:
            return {"error": "no reviews loaded"}
        sub = self.reviews_df[self.reviews_df["tmdb_id"] == int(tmdb_id)]
        if sub.empty:
            return {"error": f"no reviews for tmdb_id={tmdb_id}"}
        out = {
            "tmdb_id": int(tmdb_id),
            "title": sub["movie_title"].iloc[0],
            "n_reviews": int(len(sub)),
        }
        if "sentiment" in sub.columns:
            out["sentiment_counts"] = sub["sentiment"].value_counts().to_dict()
        if "topic" in sub.columns:
            out["topic_counts"] = sub["topic"].value_counts().to_dict()
        if "votes" in sub.columns:
            out["avg_votes"] = float(sub["votes"].fillna(0).mean())
        out["sample_reviews"] = sub["text"].head(3).tolist()
        return out

    # ---- tool 3 ----
    def compare_genres(self, g1: str, g2: str) -> dict:
        if self.reviews_df is None or "sentiment" not in self.reviews_df.columns:
            return {"error": "need sentiment-labeled reviews"}
        df = self.reviews_df
        a = df[df["genre"] == g1]["sentiment"].value_counts(normalize=True).to_dict()
        b = df[df["genre"] == g2]["sentiment"].value_counts(normalize=True).to_dict()
        return {g1: a, g2: b}

    # ---- tool 4 ----
    def recommend_similar(self, tmdb_id: int, k: int = 3) -> list[dict]:
        if self.reviews_df is None or self.text_matrix is None:
            return []
        df = self.reviews_df
        mask = (df["tmdb_id"] == int(tmdb_id)).values
        if not mask.any():
            return []
        # 该电影评论的中心向量
        centroid = self.text_matrix[mask].mean(axis=0, keepdims=True)
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(centroid, self.text_matrix)[0]
        # 把每部电影的相似度求平均
        df_sim = df.copy()
        df_sim["sim"] = sims
        agg = (
            df_sim.groupby(["tmdb_id", "movie_title", "genre"])["sim"]
            .mean()
            .reset_index()
            .sort_values("sim", ascending=False)
        )
        agg = agg[agg["tmdb_id"] != int(tmdb_id)].head(k)
        return agg.to_dict("records")


# ---------- ReAct Agent ----------
TOOL_SCHEMA = """You have these tools:

1) search_movies(query: str) -> list of movies matching the title
2) get_movie_summary(tmdb_id: int) -> stats and sample reviews of one movie
3) compare_genres(g1: str, g2: str) -> sentiment distribution of two genres
4) recommend_similar(tmdb_id: int) -> 3 similar movies based on review embeddings

Respond in this format ONLY:
THOUGHT: <your reasoning>
ACTION: <tool_name>
ACTION_INPUT: <JSON dict of arguments, e.g. {"query": "John Wick"}>

When you have enough info, respond with:
THOUGHT: <final reasoning>
FINAL_ANSWER: <natural-language answer to the user>"""


def run_agent(question: str, max_steps: int = 5) -> str:
    """单轮 ReAct 循环"""
    tools = MovieAgentTools()
    client = make_client()
    model = load_config()["openai"]["model"]

    system = "You are a helpful movie analyst. Use the tools to answer the user.\n\n" + TOOL_SCHEMA
    history = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        log.info(f"--- step {step + 1} ---")
        out = chat(client, history, model=model, temperature=0.0, max_tokens=500)
        log.info(out)

        if "FINAL_ANSWER:" in out:
            answer = out.split("FINAL_ANSWER:", 1)[1].strip()
            return answer

        # 解析 ACTION
        m = re.search(r"ACTION:\s*(\w+)\s*\nACTION_INPUT:\s*(\{.*?\})", out, re.DOTALL)
        if not m:
            return "(agent stopped: cannot parse action)"
        tool_name = m.group(1).strip()
        try:
            args = json.loads(m.group(2))
        except Exception:
            args = {}

        if not hasattr(tools, tool_name):
            obs = f"Unknown tool: {tool_name}"
        else:
            try:
                obs = getattr(tools, tool_name)(**args)
            except Exception as e:
                obs = f"Tool error: {e}"

        history.append({"role": "assistant", "content": out})
        history.append({"role": "user", "content": f"OBSERVATION: {json.dumps(obs, default=str)[:1500]}"})

    return "(agent reached max steps without final answer)"


if __name__ == "__main__":
    q = "Compare the audience sentiment of action movies and horror movies, then recommend a movie similar to John Wick."
    print(run_agent(q))
