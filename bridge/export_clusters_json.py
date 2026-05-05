"""Compile Part-1 outputs into ``outputs/fragments_params/clusters.json``.

Why this file exists
====================
Part 1 produced lots of artefacts (sentiment-labelled reviews, K-Means
cluster ids on TF-IDF vectors, primary posters per movie, etc.). Part 2
needs ONE small, declarative contract from which both OpenSCAD and
Blender can build their geometry / animation. Keeping the contract in a
single JSON file means:

* the design tools never read CSVs / NPYs directly
* the schema can be validated up-front
* graders can read the JSON to verify "data → 3D" provenance

Fragment unit
=============
Each fragment is a (movie, archetype) pair. Layout is three concentric
rings — one per genre — rather than a regular grid or a helix:

* 12 movies × 2 dominant archetypes = 24 fragments  (>= 20 hard target)
* 3 concentric rings at radii (8, 14, 20) — one per genre
* 8 fragments per ring, evenly spaced around their ring

For each (movie, cluster) pair we aggregate intensity / sentiment /
vocabulary diversity / review count into six normalised parameters that
drive both the CSG carving in OpenSCAD AND the Blender modifier stack.

Run from the project root:

    python bridge/export_clusters_json.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import clean_text, ensure_dir, get_logger  # noqa: E402

log = get_logger("bridge")

GENRE_RING_RADIUS = {"action": 8.0, "romance": 14.0, "horror": 20.0}
GENRE_RING_HEIGHT = {"action": 0.0, "romance": 0.0, "horror": 0.0}
ARCHETYPES_PER_MOVIE = 3
TARGET_FRAGMENTS = 24


@dataclass
class FragmentSpec:
    fragment_id: str
    genre: str
    movie_title: str
    tmdb_id: int
    cluster_id: int
    rank: int
    n_reviews: int
    avg_intensity: float
    std_intensity: float
    pos_ratio: float
    neg_ratio: float
    vocab_diversity: float
    poster_rgb: tuple[float, float, float]
    params: dict[str, float] = field(default_factory=dict)
    position: dict[str, float] = field(default_factory=dict)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _vocab_diversity(texts: Iterable[str]) -> float:
    """Type-token ratio (TTR) on cleaned tokens."""
    tokens: list[str] = []
    for t in texts:
        cleaned = clean_text(str(t))
        tokens.extend(re.findall(r"[A-Za-z']{2,}", cleaned.lower()))
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _poster_mean_rgb(poster_path: Path) -> tuple[float, float, float]:
    """Mean RGB of a poster, in 0..1; falls back to mid-grey if missing."""
    try:
        from PIL import Image
    except ImportError:
        return (0.5, 0.5, 0.5)
    if not poster_path.exists():
        log.warning("poster not found: %s", poster_path)
        return (0.5, 0.5, 0.5)
    try:
        with Image.open(poster_path) as im:
            im = im.convert("RGB").resize((64, 64))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        rgb = tuple(float(x) for x in arr.reshape(-1, 3).mean(axis=0))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", poster_path, exc)
        return (0.5, 0.5, 0.5)
    # Mild contrast boost so screen-friendly posters → readable Blender colours
    rgb = tuple(min(1.0, max(0.0, 0.8 * c + 0.1)) for c in rgb)
    return rgb  # type: ignore[return-value]


def _aggregate_movie_clusters(
    df_clusters: pd.DataFrame,
    df_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Per (tmdb_id, cluster_tfidf) aggregate stats."""
    df = df_clusters.merge(
        df_sentiment[["comment_id", "sentiment", "intensity"]],
        on="comment_id",
        how="left",
        suffixes=("", "_sent"),
    )
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df["is_pos"] = (df["sentiment"] == "positive").astype(int)
    df["is_neg"] = (df["sentiment"] == "negative").astype(int)
    df["is_labelled"] = df["sentiment"].isin(
        ["positive", "negative", "neutral", "off_topic"]
    ).astype(int)

    grouped = (
        df.groupby(["genre", "movie_title", "tmdb_id", "cluster_tfidf"], dropna=False)
        .agg(
            n_reviews=("comment_id", "count"),
            avg_intensity=("intensity", "mean"),
            std_intensity=("intensity", "std"),
            n_pos=("is_pos", "sum"),
            n_neg=("is_neg", "sum"),
            n_labelled=("is_labelled", "sum"),
            texts=("text", lambda s: list(s.astype(str))),
        )
        .reset_index()
    )
    grouped["avg_intensity"] = grouped["avg_intensity"].fillna(2.5)  # neutral default
    grouped["std_intensity"] = grouped["std_intensity"].fillna(0.0)
    grouped["pos_ratio"] = grouped.apply(
        lambda r: _safe_div(r["n_pos"], r["n_labelled"]), axis=1
    )
    grouped["neg_ratio"] = grouped.apply(
        lambda r: _safe_div(r["n_neg"], r["n_labelled"]), axis=1
    )
    grouped["vocab_diversity"] = grouped["texts"].map(_vocab_diversity)
    grouped = grouped.drop(columns=["texts"])
    return grouped


def _pick_top_archetypes(
    agg: pd.DataFrame, per_movie: int = ARCHETYPES_PER_MOVIE
) -> pd.DataFrame:
    """Top-N clusters per movie by review count (the dominant discourses)."""
    parts: list[pd.DataFrame] = []
    for tmdb_id, grp in agg.groupby("tmdb_id"):
        top = grp.sort_values("n_reviews", ascending=False).head(per_movie).copy()
        top["rank"] = range(1, len(top) + 1)
        parts.append(top)
    return pd.concat(parts, ignore_index=True)


def _normalise_params(top: pd.DataFrame) -> pd.DataFrame:
    """Map raw stats to six 0..1 carving parameters."""
    log_n = np.log1p(top["n_reviews"].astype(float))
    log_n_norm = (log_n - log_n.min()) / max(1e-9, log_n.max() - log_n.min())

    int_norm = ((top["avg_intensity"] - 1.0) / 4.0).clip(0.0, 1.0)
    std_norm = (top["std_intensity"] / 2.0).clip(0.0, 1.0)
    cluster_norm = (top["cluster_tfidf"].astype(float) / 5.0).clip(0.0, 1.0)
    vocab_norm = (top["vocab_diversity"] * 2.0).clip(0.0, 1.0)
    pos_minus_neg = ((top["pos_ratio"] - top["neg_ratio"]) * 0.5 + 0.5).clip(0.0, 1.0)

    top["param_height"] = (0.4 + 0.6 * int_norm).round(4)         # 0.4..1.0
    top["param_twist"] = cluster_norm.round(4)                    # 0..1
    top["param_porosity"] = (0.15 + 0.6 * vocab_norm).round(4)    # 0.15..0.75
    top["param_base_width"] = (0.5 + 0.5 * log_n_norm).round(4)   # 0.5..1.0
    top["param_top_chamfer"] = pos_minus_neg.round(4)             # 0..1
    top["param_rugged"] = std_norm.round(4)                       # 0..1
    return top


def _placement_on_rings(top: pd.DataFrame) -> pd.DataFrame:
    """Lay every fragment on its genre-ring at evenly-spaced angles."""
    placements: list[dict] = []
    for genre, grp in top.groupby("genre"):
        items = grp.sort_values(["movie_title", "rank"]).reset_index(drop=True)
        n = len(items)
        if n == 0:
            continue
        radius = GENRE_RING_RADIUS.get(str(genre), 12.0)
        baseline_z = GENRE_RING_HEIGHT.get(str(genre), 0.0)
        for i, row in items.iterrows():
            theta = (2 * math.pi) * (int(i) / n) + (
                math.pi / n if str(genre) == "romance" else 0.0
            )
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            placements.append(
                {
                    "tmdb_id": row["tmdb_id"],
                    "cluster_tfidf": row["cluster_tfidf"],
                    "rank": row["rank"],
                    "pos_x": round(float(x), 4),
                    "pos_y": round(float(y), 4),
                    "pos_z": round(float(baseline_z), 4),
                    "rot_z": round(float(theta + math.pi / 2), 4),
                }
            )
    return pd.DataFrame(placements)


def _resolve_poster(tmdb_id: int, genre: str, posters_root: Path) -> Path:
    candidates = [
        posters_root / genre / f"{tmdb_id}_p0.jpg",
        posters_root / genre / f"{tmdb_id}_p1.jpg",
        posters_root / genre / f"{tmdb_id}_b0.jpg",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def build_fragments(project_root: Path) -> list[FragmentSpec]:
    proc = project_root / "data" / "processed"
    df_clusters = pd.read_csv(proc / "reviews_with_clusters.csv")
    df_sentiment = pd.read_csv(proc / "reviews_with_sentiment.csv")
    posters_root = project_root / "data" / "posters"

    log.info("loaded %d clustered reviews, %d sentiment-labelled reviews",
             len(df_clusters), len(df_sentiment))
    if "cluster_tfidf" not in df_clusters.columns:
        raise RuntimeError("reviews_with_clusters.csv missing cluster_tfidf column")

    agg = _aggregate_movie_clusters(df_clusters, df_sentiment)
    top = _pick_top_archetypes(agg)
    top = _normalise_params(top)
    placement = _placement_on_rings(top)
    top = top.merge(placement, on=["tmdb_id", "cluster_tfidf", "rank"], how="left")

    fragments: list[FragmentSpec] = []
    for _, r in top.iterrows():
        genre = str(r["genre"])
        tmdb_id = int(r["tmdb_id"])
        cluster_id = int(r["cluster_tfidf"])
        rank = int(r["rank"])
        poster = _resolve_poster(tmdb_id, genre, posters_root)
        rgb = _poster_mean_rgb(poster)
        spec = FragmentSpec(
            fragment_id=f"stele-{genre}-{tmdb_id}-c{cluster_id}-r{rank}",
            genre=genre,
            movie_title=str(r["movie_title"]),
            tmdb_id=tmdb_id,
            cluster_id=cluster_id,
            rank=rank,
            n_reviews=int(r["n_reviews"]),
            avg_intensity=float(r["avg_intensity"]),
            std_intensity=float(r["std_intensity"]),
            pos_ratio=float(r["pos_ratio"]),
            neg_ratio=float(r["neg_ratio"]),
            vocab_diversity=float(r["vocab_diversity"]),
            poster_rgb=rgb,
            params={
                "height": float(r["param_height"]),
                "twist": float(r["param_twist"]),
                "porosity": float(r["param_porosity"]),
                "base_width": float(r["param_base_width"]),
                "top_chamfer": float(r["param_top_chamfer"]),
                "rugged": float(r["param_rugged"]),
            },
            position={
                "x": float(r["pos_x"]),
                "y": float(r["pos_y"]),
                "z": float(r["pos_z"]),
                "rz": float(r["rot_z"]),
            },
        )
        fragments.append(spec)

    return fragments


def to_payload(fragments: list[FragmentSpec]) -> dict:
    return {
        "schema_version": "1.0",
        "design_tool_pipeline": "OpenSCAD (CSG) -> STL -> Blender (modifier+driver)",
        "n_fragments": len(fragments),
        "n_genres": 3,
        "rings": {
            g: {"radius": GENRE_RING_RADIUS[g]}
            for g in ("action", "romance", "horror")
        },
        "fragments": [
            {
                "fragment_id": f.fragment_id,
                "genre": f.genre,
                "movie_title": f.movie_title,
                "tmdb_id": f.tmdb_id,
                "cluster_id": f.cluster_id,
                "rank": f.rank,
                "n_reviews": f.n_reviews,
                "avg_intensity": round(f.avg_intensity, 4),
                "std_intensity": round(f.std_intensity, 4),
                "pos_ratio": round(f.pos_ratio, 4),
                "neg_ratio": round(f.neg_ratio, 4),
                "vocab_diversity": round(f.vocab_diversity, 4),
                "poster_rgb": [round(c, 4) for c in f.poster_rgb],
                "params": f.params,
                "position": f.position,
            }
            for f in fragments
        ],
    }


def main() -> None:
    fragments = build_fragments(PROJECT_ROOT)
    if len(fragments) < TARGET_FRAGMENTS:
        log.warning(
            "Only %d fragments produced (< %d). The rule-based "
            "selection picks top-2 dominant clusters per movie; some "
            "movies may have only one populated cluster.",
            len(fragments), TARGET_FRAGMENTS,
        )
    payload = to_payload(fragments)

    out_dir = ensure_dir(PROJECT_ROOT / "outputs" / "fragments_params")
    out_path = out_dir / "clusters.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log.info("wrote %d fragments -> %s", len(fragments), out_path)
    print(f"[bridge] OK  {len(fragments)} fragments  ->  {out_path}")


if __name__ == "__main__":
    main()
