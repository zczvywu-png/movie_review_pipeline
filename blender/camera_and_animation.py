"""Step 4 — animate the global mood empty + camera flight, all via NLA.

Animation paradigm
==================
Use the **NLA editor** + **F-Curve modifiers** rather than dense
keyframing:
  - one short *Action* describes the mood arc once (positive → cynical
    → euphoric); the NLA editor *replays* it across the timeline.
  - The camera follows a *Bezier curve* via the Follow-Path constraint,
    so we never type a single (x,y,z) keyframe for it.
  - F-Curve modifiers (CYCLES) keep the mood looping subtly without
    drawing a single extra keyframe.

Result: 65 seconds of mood-driven animation that is described
declaratively by ONE 4-keyframe Action plus ONE Bezier curve. The 24
steles all react via the drivers from `driver_bindings.py`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


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


FPS = 24
DURATION_S = 65          # > 60 s, satisfies brief
TOTAL_FRAMES = FPS * DURATION_S  # 1560

CAMERA_NAME = "Hero_Cam"
CAMERA_PATH_NAME = "Cam_Path"
CAMERA_TARGET_NAME = "Cam_Target"
MOOD_EMPTY_NAME = "Mood_Driver"


# ---------- mood arc as a tiny Action ----------------------------------- #


def _build_mood_action() -> bpy.types.Action:
    """A 4-keyframe arc: cynical → praise → backlash → catharsis."""
    name = "Mood_Arc"
    if name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[name])
    action = bpy.data.actions.new(name)

    # Each row: (frame, mood_value)
    keys = [
        (1,                          -0.8),
        (TOTAL_FRAMES * 0.30,         0.7),
        (TOTAL_FRAMES * 0.55,        -0.55),
        (TOTAL_FRAMES * 0.80,         0.95),
        (TOTAL_FRAMES,                0.10),
    ]

    fc = action.fcurves.new(data_path="location", index=0)
    for frame, value in keys:
        kp = fc.keyframe_points.insert(frame=frame, value=value, options={"FAST"})
        kp.interpolation = "BEZIER"
        kp.handle_left_type = "AUTO_CLAMPED"
        kp.handle_right_type = "AUTO_CLAMPED"
    return action


def _attach_action_via_nla(obj: bpy.types.Object,
                           action: bpy.types.Action) -> None:
    """Bind the Action to the object using an NLA strip — that way the
    curve is reusable, can be muted/repeated, and we never modify
    ``obj.animation_data.action`` directly."""
    if obj.animation_data is None:
        obj.animation_data_create()
    ad = obj.animation_data
    ad.action = None  # we drive everything from the NLA
    # remove old tracks (idempotent)
    for tr in list(ad.nla_tracks):
        ad.nla_tracks.remove(tr)
    track = ad.nla_tracks.new()
    track.name = "Mood_Track"
    strip = track.strips.new(name=action.name, start=1, action=action)
    strip.action_frame_start = 1
    strip.action_frame_end = TOTAL_FRAMES
    strip.frame_start = 1
    strip.frame_end = TOTAL_FRAMES


# ---------- camera path (Bezier curve + Follow Path constraint) -------- #


def _build_camera_path() -> bpy.types.Object:
    """Create a Bezier curve that orbits the three rings.

    The shape is a flattened heart-shape, NOT a circle (so the camera's
    distance from origin varies subtly across the loop).
    """
    if CAMERA_PATH_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[CAMERA_PATH_NAME], do_unlink=True)
    if CAMERA_PATH_NAME in bpy.data.curves:
        bpy.data.curves.remove(bpy.data.curves[CAMERA_PATH_NAME])

    curve_data = bpy.data.curves.new(CAMERA_PATH_NAME, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.path_duration = TOTAL_FRAMES

    spline = curve_data.splines.new("BEZIER")
    n = 8
    spline.bezier_points.add(n - 1)
    for i, bp in enumerate(spline.bezier_points):
        t = (i / n) * 2 * math.pi
        # flattened heart-ish path; pulled out (was 26+/-6 -> 34+/-5) so the
        # camera no longer clips into the horror ring (radius 20).
        r = 34.0 + 5.0 * math.cos(t * 2.0)
        z = 8.0 + 4.0 * math.sin(t * 1.0 + 0.6)
        bp.co = Vector((r * math.cos(t), r * math.sin(t), z))
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    spline.use_cyclic_u = True

    obj = bpy.data.objects.new(CAMERA_PATH_NAME, curve_data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _build_camera_target() -> bpy.types.Object:
    """A small empty the camera always tracks — moves slowly so we get
    a smooth gaze without painting any keyframes for the camera."""
    if CAMERA_TARGET_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[CAMERA_TARGET_NAME],
                                do_unlink=True)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 3))
    obj = bpy.context.active_object
    obj.name = CAMERA_TARGET_NAME

    # Two Action keyframes, then loop with an F-Curve CYCLES modifier so
    # the target gently sways without painting a curve per frame.
    if obj.animation_data is None:
        obj.animation_data_create()
    act = bpy.data.actions.new("Cam_Target_Sway")
    obj.animation_data.action = act
    for axis_idx in (0, 1):
        fc = act.fcurves.new(data_path="location", index=axis_idx)
        fc.keyframe_points.insert(frame=1, value=0.0)
        fc.keyframe_points.insert(frame=120,
                                  value=2.5 if axis_idx == 0 else -1.8)
        # CYCLES modifier: re-runs the 1..120 segment forever
        m = fc.modifiers.new("CYCLES")
        m.mode_before = "REPEAT_OFFSET"
        m.mode_after = "REPEAT_OFFSET"
    return obj


def _build_camera(path_obj: bpy.types.Object,
                  target_obj: bpy.types.Object) -> bpy.types.Object:
    """The hero camera follows the curve and tracks the target.

    Net effect: a continuous orbit that we never had to keyframe.
    """
    cam_data = bpy.data.cameras.get(CAMERA_NAME) or bpy.data.cameras.new(CAMERA_NAME)
    cam_data.lens = 38.0

    if CAMERA_NAME in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[CAMERA_NAME], do_unlink=True)
    cam = bpy.data.objects.new(CAMERA_NAME, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    follow = cam.constraints.new("FOLLOW_PATH")
    follow.target = path_obj
    follow.use_curve_follow = True
    follow.forward_axis = "FORWARD_Y"
    follow.up_axis = "UP_Z"

    # animate the path's eval_time so the camera traverses the loop in
    # exactly TOTAL_FRAMES — only TWO keyframes for the entire flight.
    curve_data = path_obj.data
    curve_data.use_path = True
    if curve_data.animation_data is None:
        curve_data.animation_data_create()
    act = bpy.data.actions.new("Cam_Path_Drive")
    curve_data.animation_data.action = act
    fc = act.fcurves.new(data_path="eval_time")
    fc.keyframe_points.insert(frame=1, value=0.0).interpolation = "LINEAR"
    fc.keyframe_points.insert(frame=TOTAL_FRAMES,
                              value=float(TOTAL_FRAMES)).interpolation = "LINEAR"

    track = cam.constraints.new("TRACK_TO")
    track.target = target_obj
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return cam


# ---------- main --------------------------------------------------------- #


def _set_scene_timing() -> None:
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = TOTAL_FRAMES


def main() -> None:
    mood = bpy.data.objects.get(MOOD_EMPTY_NAME)
    if mood is None:
        raise RuntimeError(
            f"'{MOOD_EMPTY_NAME}' empty missing. Run driver_bindings.py first."
        )

    _set_scene_timing()
    action = _build_mood_action()
    _attach_action_via_nla(mood, action)
    print(f"[anim] mood action '{action.name}' bound via NLA "
          f"({TOTAL_FRAMES} frames @ {FPS}fps)")

    target = _build_camera_target()
    path = _build_camera_path()
    _build_camera(path, target)
    print("[anim] camera + Bezier path + target empty configured")


if __name__ == "__main__":
    main()
