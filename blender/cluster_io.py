"""Shared loader for the Part-2 contract JSON. Used by every Blender script.

Lives outside ``src/`` so it can be loaded from inside Blender's Python
runtime without importing the rest of the Part-1 codebase (which pulls
Pandas / scikit-learn etc. that are not bundled with Blender).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the ``movie_review_pipeline/`` folder."""
    return Path(__file__).resolve().parent.parent


def clusters_json_path() -> Path:
    return project_root() / "outputs" / "fragments_params" / "clusters.json"


def stl_dir() -> Path:
    return project_root() / "outputs" / "stl"


def render_dir() -> Path:
    p = project_root() / "outputs" / "renders"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_payload(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else clusters_json_path()
    if not p.exists():
        raise FileNotFoundError(
            f"clusters.json not found: {p}\n"
            "Run `python bridge/export_clusters_json.py` from the project "
            "root first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def remap(value: float, lo: float, hi: float) -> float:
    """Map a normalised 0..1 value onto [lo, hi]."""
    v = max(0.0, min(1.0, float(value)))
    return lo + (hi - lo) * v


GENRE_BASE_HUE = {
    "action":  (0.95, 0.20, 0.18),  # red-iron
    "romance": (0.90, 0.55, 0.62),  # warm rose
    "horror":  (0.18, 0.20, 0.32),  # near-black indigo
}


def genre_tint(genre: str) -> tuple[float, float, float]:
    return GENRE_BASE_HUE.get(str(genre), (0.55, 0.55, 0.55))
