"""Step 2 — attach a per-cluster modifier stack to every imported stele.

Why a modifier stack
====================
Blender's **modifier stack** is an ordered list of parametric operators
applied on top of the base mesh:

    [Array]  →  [Solidify]  →  [Bevel]  →  [Decimate]  →  [Wireframe]

None of these touch vertex coordinates directly; they expose float
parameters that are perfectly suited to being driven by F-Curve
drivers (next file). All five modifiers are documented operators with
stable names, so cross-Blender-version compatibility is good.

Per-cluster preset
==================
Each TF-IDF cluster (0..5) gets a slightly different modifier mix —
clusters representing "praise" discourse get lots of Solidify (thick,
solid), clusters for "critique" get heavy Decimate (eroded), etc. The
preset is applied based on each fragment's stored ``cluster_id``
custom-property (set by ``import_steles.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

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
from cluster_io import remap  # noqa: E402

COLLECTION_NAME = "Sentiment_Steles"


# ---------- per-cluster modifier-stack presets --------------------------- #
#
# Each preset is a list of (modifier-type, attribute_dict) tuples applied
# top-to-bottom. Numeric values starting with `_drv` are placeholders that
# will be replaced by drivers in `driver_bindings.py`.

PRESET_LOVE = [   # cluster 0: "love / fan praise"
    ("BEVEL",     {"width": 0.012, "segments": 3, "limit_method": "ANGLE",
                   "angle_limit": 0.55}),
    # Solidify thickness reduced 0.04m -> 0.004m so the OpenSCAD-carved
    # 6..16 mm "review windows" remain visible instead of being filled in.
    ("SOLIDIFY",  {"thickness": 0.004, "offset": 1.0}),
]
PRESET_REPLAY = [   # cluster 1: "rewatch / nostalgia"
    ("ARRAY",     {"count": 3, "relative_offset_displace": (0.0, 0.0, 1.05),
                   "use_merge_vertices": True}),
    ("BEVEL",     {"width": 0.010, "segments": 2}),
    ("SOLIDIFY",  {"thickness": 0.003}),
]
PRESET_GLORY = [   # cluster 2: "best / masterpiece"
    ("BEVEL",     {"width": 0.018, "segments": 4, "profile": 0.7}),
    ("SOLIDIFY",  {"thickness": 0.005}),
]
PRESET_INTENT = [  # cluster 3: "want / promotion"
    ("ARRAY",     {"count": 2, "relative_offset_displace": (1.06, 0.0, 0.0),
                   "use_merge_vertices": True}),
    ("BEVEL",     {"width": 0.010, "segments": 2}),
    ("DECIMATE",  {"ratio": 0.85, "decimate_type": "COLLAPSE"}),
]
PRESET_TIMECAP = [  # cluster 4: "years later / time capsule"
    ("DECIMATE",  {"ratio": 0.55, "decimate_type": "COLLAPSE"}),
    ("BEVEL",     {"width": 0.014, "segments": 2}),
    ("WIREFRAME", {"thickness": 0.008, "use_replace": False, "use_even_offset": True}),
]
PRESET_DISCONT = [  # cluster 5: "didn't / criticism"
    ("DECIMATE",  {"ratio": 0.40, "decimate_type": "COLLAPSE"}),
    ("BEVEL",     {"width": 0.006, "segments": 1}),
    ("SOLIDIFY",  {"thickness": 0.003, "use_rim_only": True}),
]

CLUSTER_PRESETS: dict[int, list[tuple[str, dict]]] = {
    0: PRESET_LOVE,
    1: PRESET_REPLAY,
    2: PRESET_GLORY,
    3: PRESET_INTENT,
    4: PRESET_TIMECAP,
    5: PRESET_DISCONT,
}


def _set_attrs(modifier: bpy.types.Modifier, attrs: dict) -> None:
    for k, v in attrs.items():
        try:
            setattr(modifier, k, v)
        except AttributeError:
            # Some modifiers expose limit_method only when angle_limit is set, etc.
            # Silently skip unknown attrs so different Blender minor versions don't
            # break the script.
            pass


def _strip_existing_modifiers(obj: bpy.types.Object) -> None:
    while obj.modifiers:
        obj.modifiers.remove(obj.modifiers[0])


def _attach_preset(obj: bpy.types.Object,
                   preset: list[tuple[str, dict]]) -> None:
    for idx, (mod_type, attrs) in enumerate(preset):
        name = f"P{idx:02d}_{mod_type.title()}"
        mod = obj.modifiers.new(name=name, type=mod_type)
        _set_attrs(mod, attrs)


def _attach_universal_finishers(obj: bpy.types.Object) -> None:
    """Two modifiers every stele gets, regardless of cluster preset.

    `Bevel_Finisher` smooths terminal edges so the OpenSCAD CSG output
    reads cleanly under EEVEE; `DriverHook_Empty` is an unused
    EmptyAxis modifier we co-opt as a parameter carrier — drivers in
    the next script bind to its `strength` value to keep the rest of
    the stack stable across Blender versions.
    """
    finisher = obj.modifiers.new(name="Bevel_Finisher", type="BEVEL")
    finisher.width = 0.006
    finisher.segments = 1
    finisher.limit_method = "ANGLE"
    finisher.angle_limit = 0.78

    # SimpleDeform is one of the few modifiers whose `factor` field is
    # available across all Blender 3.x/4.x versions — perfect anchor for
    # the data-driven 'controversy twist' (set by drivers later).
    deform = obj.modifiers.new(name="Driven_Twist", type="SIMPLE_DEFORM")
    deform.deform_method = "TWIST"
    deform.deform_axis = "Z"
    deform.factor = 0.0


def main() -> None:
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if not coll:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' missing. "
            "Run import_steles.py first."
        )

    n = 0
    for obj in list(coll.objects):
        if obj.type != "MESH":
            continue
        cluster_id = int(obj.get("cluster_id", 0))
        preset = CLUSTER_PRESETS.get(cluster_id, PRESET_LOVE)

        _strip_existing_modifiers(obj)
        _attach_preset(obj, preset)
        _attach_universal_finishers(obj)

        # gentle shading: smooth normals, but leave the carved planar
        # faces flat so OpenSCAD's window cuts still read as architecture
        if obj.data is not None and hasattr(obj.data, "polygons"):
            for poly in obj.data.polygons:
                poly.use_smooth = False

        n += 1

    print(f"[modifiers] attached preset to {n} steles "
          f"({len(CLUSTER_PRESETS)} cluster-presets registered)")


if __name__ == "__main__":
    main()
