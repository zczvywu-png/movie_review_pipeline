"""Step 1 of the Blender pipeline — import 24+ STL fragments and place
them on three concentric rings.

Design choice
=============
This file deliberately uses **only the high-level ``bpy.ops`` /
``bpy.data`` API** and the **modifier stack** — no low-level bmesh
vertex edits, no Geometry-Nodes node-trees. The geometry comes from
OpenSCAD's CSG output (``outputs/stl/*.stl``); Blender only stages,
materialises and animates it.

Run order (in Blender 4.x's Scripting workspace):
    1. import_steles.py        ← THIS file (resets scene, imports STLs,
                                  puts them on rings, makes materials)
    2. modifier_stack.py
    3. driver_bindings.py
    4. camera_and_animation.py
    5. render_eevee_next.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _script_dir() -> Path:
    """Find this script on disk, even when Blender's Run Script gives a
    relative ``__file__`` (a known quirk on Windows)."""
    try:
        text = bpy.context.space_data.text
        if text and text.filepath:
            return Path(bpy.path.abspath(text.filepath)).resolve().parent
    except (AttributeError, RuntimeError, ReferenceError):
        pass
    try:
        p = Path(__file__).resolve()
        if p.is_absolute() and p.exists():
            return p.parent
    except (NameError, OSError):
        pass
    raise RuntimeError(
        "Cannot locate this script on disk. "
        "Open the .py file in the Scripting workspace and Run Script from there."
    )


_HERE = _script_dir()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Blender keeps `sys.modules` populated across Run Script calls, so a sibling
# module edited on disk would not be picked up unless we evict it first.
sys.modules.pop("cluster_io", None)
from cluster_io import (  # noqa: E402
    genre_tint,
    load_payload,
    remap,
    stl_dir,
)


COLLECTION_NAME = "Sentiment_Steles"
GROUND_PLANE_SIZE = 80.0
STL_UNIT_SCALE = 0.012  # OpenSCAD outputs in millimetres; Blender world is metres


# ---------- scene reset --------------------------------------------------- #


def _reset_scene() -> None:
    """Wipe the default cube/light/camera and start with a clean ground."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block in (
        bpy.data.meshes, bpy.data.materials, bpy.data.lights,
        bpy.data.cameras, bpy.data.images, bpy.data.curves,
        bpy.data.collections,
    ):
        for item in list(block):
            if getattr(item, "users", 0) == 0 or block is bpy.data.collections:
                try:
                    block.remove(item)
                except RuntimeError:
                    pass

    # ground plane
    bpy.ops.mesh.primitive_plane_add(size=GROUND_PLANE_SIZE, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Ground"
    mat = bpy.data.materials.new("Ground_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.04, 0.04, 0.05, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.92
    plane.data.materials.append(mat)

    # central lantern (so the carved windows are readable)
    bpy.ops.object.light_add(type="POINT", location=(0, 0, 6.5))
    lantern = bpy.context.active_object
    lantern.name = "Center_Lantern"
    lantern.data.energy = 1500
    lantern.data.color = (1.0, 0.86, 0.62)
    lantern.data.shadow_soft_size = 1.0

    # cool fill from above (key+rim later from the camera dolly)
    bpy.ops.object.light_add(type="SUN", location=(10, -10, 18))
    sun = bpy.context.active_object
    sun.name = "Sky_Fill"
    sun.data.energy = 1.5
    sun.data.color = (0.55, 0.62, 0.85)
    sun.rotation_euler = (math.radians(50), math.radians(15), math.radians(45))


def _ensure_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


# ---------- material per fragment ---------------------------------------- #


def _make_material(name: str,
                   base_rgb: tuple[float, float, float],
                   genre: str,
                   pos_ratio: float) -> bpy.types.Material:
    """Build a Principled BSDF that mixes poster-mean colour with the
    genre tint. Emission strength tracks `pos_ratio` — positive-leaning
    discourse is illuminated."""
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        return mat

    g_r, g_g, g_b = genre_tint(genre)
    p_r, p_g, p_b = base_rgb
    mix = lambda a, b, t: a * (1 - t) + b * t  # noqa: E731
    r = mix(p_r, g_r, 0.4)
    g = mix(p_g, g_g, 0.4)
    b = mix(p_b, g_b, 0.4)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.78
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.30
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.30

    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (
            mix(0.05, 1.0, pos_ratio),
            mix(0.05, 0.85, pos_ratio),
            mix(0.05, 0.6, pos_ratio),
            1.0,
        )
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = remap(pos_ratio, 0.0, 0.6)
    return mat


# ---------- STL import or primitive fallback ----------------------------- #


def _import_or_fallback(stl_path: Path,
                        fragment: dict,
                        coll: bpy.types.Collection) -> bpy.types.Object:
    """Import an STL when available, else build a primitive that follows
    the same six parameters so the animation still renders end-to-end
    even if the OpenSCAD step has not been run yet."""
    name = fragment["fragment_id"]
    if stl_path.exists():
        before = set(bpy.data.objects)
        try:
            bpy.ops.wm.stl_import(filepath=str(stl_path))
        except (AttributeError, RuntimeError):
            bpy.ops.import_mesh.stl(filepath=str(stl_path))
        new = [o for o in bpy.data.objects if o not in before]
        if not new:
            raise RuntimeError(f"STL import produced no new object: {stl_path}")
        obj = new[0]
        obj.name = name
        # Move the object into our collection
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
        # Scale millimetres → metres so the layout numbers in clusters.json
        # (which are in metres) make sense visually.
        obj.scale = (STL_UNIT_SCALE,) * 3
        return obj

    # ---- primitive fallback ---- #
    p = fragment["params"]
    h = remap(p["height"], 1.5, 3.5)
    bw = remap(p["base_width"], 0.4, 1.0)
    bpy.ops.mesh.primitive_cone_add(
        radius1=bw, radius2=bw * 0.92, depth=h, vertices=12,
        location=(0, 0, h / 2),
    )
    obj = bpy.context.active_object
    obj.name = name
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def _place_on_ring(obj: bpy.types.Object, fragment: dict) -> None:
    pos = fragment["position"]
    obj.location = (pos["x"], pos["y"], pos["z"])
    obj.rotation_euler = (0.0, 0.0, pos["rz"])
    # lift the object so its base sits on z=0 (compute world bbox)
    bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_z = min(v.z for v in bbox)
    obj.location.z -= min_z


# ---------- main --------------------------------------------------------- #


def main() -> None:
    payload = load_payload()
    print(f"[import] loaded {len(payload['fragments'])} fragments from JSON")

    _reset_scene()
    coll = _ensure_collection(COLLECTION_NAME)
    stl_root = stl_dir()
    stl_root.mkdir(parents=True, exist_ok=True)

    n_imported = 0
    n_fallback = 0
    for f in payload["fragments"]:
        stl_path = stl_root / f"{f['fragment_id']}.stl"
        obj = _import_or_fallback(stl_path, f, coll)
        if stl_path.exists():
            n_imported += 1
        else:
            n_fallback += 1

        _place_on_ring(obj, f)

        mat = _make_material(
            name=f"Mat_{f['fragment_id']}",
            base_rgb=tuple(f["poster_rgb"]),
            genre=f["genre"],
            pos_ratio=float(f["pos_ratio"]),
        )
        if obj.data is not None and hasattr(obj.data, "materials"):
            obj.data.materials.clear()
            obj.data.materials.append(mat)

        # Stash all per-fragment parameters as Blender custom properties so
        # later scripts (driver_bindings.py, camera_and_animation.py) can
        # read them without re-parsing the JSON.
        for key, value in f["params"].items():
            obj[f"p_{key}"] = float(value)
        obj["genre"] = f["genre"]
        obj["cluster_id"] = int(f["cluster_id"])
        obj["pos_ratio"] = float(f["pos_ratio"])
        obj["neg_ratio"] = float(f["neg_ratio"])
        obj["avg_intensity"] = float(f["avg_intensity"])
        obj["std_intensity"] = float(f["std_intensity"])

    print(f"[import] {n_imported} STL imports, {n_fallback} primitive fallbacks; "
          f"all placed on collection '{COLLECTION_NAME}'.")
    if n_fallback:
        print("[import] NOTE: run `python openscad/batch_generate.py` to "
              "replace primitive fallbacks with the OpenSCAD CSG meshes.")


if __name__ == "__main__":
    main()
