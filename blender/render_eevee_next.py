"""Step 5 — configure render output (EEVEE Next preferred, EEVEE legacy
fallback) and write to ``outputs/renders/movie_review.mp4``.

Run *last* in the Scripting workspace; afterwards click
**Render → Render Animation** (or call ``bpy.ops.render.render(animation=True)``
from the script if you want a fully-headless run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy


def _script_dir() -> Path:
    try:
        text = bpy.context.space_data.text
        if text and text.filepath:
            return Path(bpy.path.abspath(text.filepath)).resolve().parent
    except (AttributeError, RuntimeError, ReferenceError):
        pass
    p = Path(__file__).resolve()
    if p.exists():
        return p.parent
    raise RuntimeError("Run from Blender's Scripting workspace.")


_HERE = _script_dir()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

sys.modules.pop("cluster_io", None)
from cluster_io import render_dir  # noqa: E402


# ----------------------- engine pick (EEVEE Next > EEVEE > Workbench) ---- #


def _pick_engine() -> str:
    """Return the most modern EEVEE engine available in this Blender."""
    candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"]
    items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    available = {it.identifier for it in items}
    for c in candidates:
        if c in available:
            return c
    return "BLENDER_EEVEE"


# ----------------------- main ------------------------------------------- #


def main() -> None:
    scene = bpy.context.scene
    engine = _pick_engine()
    scene.render.engine = engine
    print(f"[render] engine = {engine}")

    rd = scene.render
    rd.resolution_x = 1920
    rd.resolution_y = 1080
    rd.resolution_percentage = 100
    rd.fps = 24
    rd.image_settings.file_format = "FFMPEG"
    rd.ffmpeg.format = "MPEG4"
    rd.ffmpeg.codec = "H264"
    rd.ffmpeg.constant_rate_factor = "MEDIUM"
    rd.ffmpeg.audio_codec = "AAC"
    rd.filepath = str(render_dir() / "movie_review.mp4")

    # Filmic tone-mapping — softer highlights, retains the lantern bloom
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"

    # EEVEE-specific quality knobs
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        # Sampling — EEVEE-Next uses 'taa_render_samples', legacy EEVEE
        # uses 'taa_render_samples' too but exposes Bloom toggles.
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 24  # Was 64; 24 keeps banding-free shading at ~1/3 the cost.
        if hasattr(eevee, "use_bloom"):
            eevee.use_bloom = True
        if hasattr(eevee, "bloom_intensity"):
            eevee.bloom_intensity = 0.06
        if hasattr(eevee, "use_ssr"):
            eevee.use_ssr = True
        if hasattr(eevee, "use_gtao"):
            eevee.use_gtao = True
        if hasattr(eevee, "gtao_distance"):
            eevee.gtao_distance = 0.6

    print(f"[render] {rd.resolution_x}x{rd.resolution_y} "
          f"@ {rd.fps}fps, {scene.frame_start}..{scene.frame_end} -> "
          f"{rd.filepath}")
    print("[render] When ready: Render -> Render Animation  "
          "(or call bpy.ops.render.render(animation=True) from a script).")


if __name__ == "__main__":
    main()
