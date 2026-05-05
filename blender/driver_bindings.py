"""Step 3 — bind F-Curve drivers from per-fragment data to modifier
parameters.

Why use drivers
===============
Blender drivers are formulas attached to a property: every frame Blender
reads a "driver expression" (Python-evaluated arithmetic over named
variables) and writes the result to a target socket. Compared with
keyframes (sampled at discrete frames) or per-vertex node-graph
evaluation (no native time component), drivers let us treat the
**timeline itself** as a free variable: a single global "mood" empty's
location.x is the time-driven proxy for sentiment polarity, and every
modifier amount across all 24 steles is a closed-form function of
`mood.location.x` plus that stele's own data (avg_intensity,
std_intensity, etc.).

What gets driven
================
For each stele we add drivers to:
  * SimpleDeform 'Driven_Twist'.factor  — std_intensity * mood
  * Solidify (any in stack).thickness   — base value * (1 + 0.4*mood)
  * Subsurf (any in stack).levels       — clamped 0..2 by avg_intensity
  * Object scale Z                      — 0.85 + 0.3 * pos_ratio_at_t

Plus a single global ``Mood_Driver`` empty whose X location goes from
-1.0 (cynical) → +1.0 (euphoric) over the 65-second timeline. The
animation script will animate this single empty; everything else
follows automatically.
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

COLLECTION_NAME = "Sentiment_Steles"
MOOD_EMPTY_NAME = "Mood_Driver"


def _ensure_mood_empty() -> bpy.types.Object:
    """Create or fetch the global mood-driving empty.

    location.x ∈ [-1, +1] becomes the universal "mood" the animation
    script keyframes; every per-stele driver references it via Python
    expression `mood`.
    """
    obj = bpy.data.objects.get(MOOD_EMPTY_NAME)
    if obj is None:
        bpy.ops.object.empty_add(type="ARROWS", location=(0, 0, 0.5))
        obj = bpy.context.active_object
        obj.name = MOOD_EMPTY_NAME
        obj.show_in_front = True
        # park it well outside the camera frame so it never appears in renders
        obj.location.z = -8.0
    return obj


def _add_driver(target,
                data_path: str,
                expr: str,
                variables: dict[str, tuple[bpy.types.ID, str]],
                index: int = -1) -> None:
    """Helper that wraps Blender's verbose driver API.

    `variables` maps a driver-variable name to (id_block, data_path).
    Most of our drivers only need one variable: the mood empty's X.
    """
    if index >= 0:
        fcurve = target.driver_add(data_path, index)
    else:
        fcurve = target.driver_add(data_path)
    drv = fcurve.driver
    drv.type = "SCRIPTED"
    # remove pre-existing variables (idempotent re-runs)
    for var in list(drv.variables):
        drv.variables.remove(var)
    for var_name, (id_block, dp) in variables.items():
        var = drv.variables.new()
        var.name = var_name
        var.type = "SINGLE_PROP"
        var.targets[0].id = id_block
        var.targets[0].data_path = dp
    drv.expression = expr


def _drive_simple_deform_twist(obj: bpy.types.Object,
                               mood: bpy.types.Object) -> None:
    mod = obj.modifiers.get("Driven_Twist")
    if mod is None:
        return
    # std_intensity baked as a custom prop on the object itself
    _add_driver(
        target=mod,
        data_path="factor",
        expr="(std * 0.6) * mood",  # radians; Blender treats SimpleDeform.factor as radians for TWIST
        variables={
            "mood": (mood, "location.x"),
            "std":  (obj, '["std_intensity"]'),
        },
    )


def _drive_solidify_thickness(obj: bpy.types.Object,
                              mood: bpy.types.Object) -> None:
    for mod in obj.modifiers:
        if mod.type == "SOLIDIFY":
            base = float(mod.thickness) or 0.03
            _add_driver(
                target=mod,
                data_path="thickness",
                expr=f"{base:.5f} * (1.0 + 0.45 * pos * (1.0 + mood))",
                variables={
                    "mood": (mood, "location.x"),
                    "pos":  (obj, '["pos_ratio"]'),
                },
            )


def _drive_subsurf_level(obj: bpy.types.Object,
                         mood: bpy.types.Object) -> None:
    for mod in obj.modifiers:
        if mod.type == "SUBSURF":
            _add_driver(
                target=mod,
                data_path="levels",
                # avg_intensity is 1..5; we map ((avg-1)/4 + mood/2) clamped to 0..2
                expr=("max(0, min(2, "
                      "round((avg - 1.0) / 4.0 + max(mood, 0.0) * 0.5)"
                      "))"),
                variables={
                    "mood": (mood, "location.x"),
                    "avg":  (obj, '["avg_intensity"]'),
                },
            )


def _drive_decimate_ratio(obj: bpy.types.Object,
                          mood: bpy.types.Object) -> None:
    for mod in obj.modifiers:
        if mod.type == "DECIMATE":
            base = float(mod.ratio) or 0.5
            _add_driver(
                target=mod,
                data_path="ratio",
                # negative-leaning fragments erode harder when mood goes south
                expr=f"max(0.05, min(1.0, {base:.5f} - 0.35 * neg * (1.0 - mood) * 0.5))",
                variables={
                    "mood": (mood, "location.x"),
                    "neg":  (obj, '["neg_ratio"]'),
                },
            )


def _drive_bevel_width(obj: bpy.types.Object,
                       mood: bpy.types.Object) -> None:
    fin = obj.modifiers.get("Bevel_Finisher")
    if fin is None:
        return
    base = float(fin.width) or 0.006
    _add_driver(
        target=fin,
        data_path="width",
        expr=f"{base:.5f} * (1.0 + 0.4 * (1.0 - abs(mood)))",
        variables={"mood": (mood, "location.x")},
    )


def _drive_object_scale_z(obj: bpy.types.Object,
                          mood: bpy.types.Object) -> None:
    """A subtle whole-object stretch: positive mood → towers grow taller.

    Multiplier is clamped to >= 1.0 so we never shrink (which used to make
    the tower's pivoted base lift off the ground when scale < 1).
    """
    _add_driver(
        target=obj,
        data_path="scale",
        index=2,
        expr=("base * (1.0 + 0.45 * pos * max(mood, 0.0))"),
        variables={
            "mood": (mood, "location.x"),
            "pos":  (obj, '["pos_ratio"]'),
            "base": (obj, '["p_height"]'),
        },
    )


def _drive_emission_strength(obj: bpy.types.Object,
                             mood: bpy.types.Object) -> None:
    """Bind the Principled BSDF emission strength to (pos_ratio * mood)."""
    if not obj.data or not obj.data.materials:
        return
    mat = obj.data.materials[0]
    if not mat or not mat.use_nodes:
        return
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf or "Emission Strength" not in bsdf.inputs:
        return
    socket = bsdf.inputs["Emission Strength"]
    _add_driver(
        target=socket,
        data_path="default_value",
        expr="max(0.0, pos * (1.0 + mood))",
        variables={
            "mood": (mood, "location.x"),
            "pos":  (obj, '["pos_ratio"]'),
        },
    )


def main() -> None:
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if not coll:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' missing. "
                           "Run import_steles.py first.")

    mood = _ensure_mood_empty()

    n_drivers = 0
    n_objs = 0
    for obj in coll.objects:
        if obj.type != "MESH":
            continue
        n_objs += 1
        for fn in (
            _drive_simple_deform_twist,
            _drive_solidify_thickness,
            _drive_subsurf_level,
            _drive_decimate_ratio,
            _drive_bevel_width,
            _drive_object_scale_z,
            _drive_emission_strength,
        ):
            try:
                fn(obj, mood)
                n_drivers += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[drivers] WARN {obj.name}.{fn.__name__}: {exc}")

    print(f"[drivers] bound {n_drivers} drivers across {n_objs} steles, "
          f"global mood empty = '{MOOD_EMPTY_NAME}'.")


if __name__ == "__main__":
    main()
