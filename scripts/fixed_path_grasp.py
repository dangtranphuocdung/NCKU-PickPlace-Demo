"""
Fixed-orientation visual grasping with customizable retreat waypoints.

Workflow:
  1. Move to Ready Pose (Approach Waypoint 0) and pre-open gripper.
  2. Live YOLO object detection & 3D position estimation at Ready Pose.
  3. Press G:
     - Dynamically compute object's (X, Y, Z) in base frame from YOLO.
     - Keep gripper orientation locked to FIXED_GRASP_RPY.
     - Traverse optional APPROACH_WAYPOINTS -> Pre-Grasp -> Grasp (X, Y, Z).
     - Close gripper (distance-based, matching main.py).
     - Execute customizable RETREAT_WAYPOINTS sequence.
     - Return to Ready Pose and pre-open gripper.

Keys:
  G: capture current frame and execute fixed-orientation grasp + waypoint retreat.
  C: macro centering shot over object.
  R: resume live preview.
  SPACE (during waypoint pause): resume executing remaining retreat waypoints.
  Q/Esc: release, home, and exit.

Usage:
    python scripts/fixed_path_grasp.py
    python scripts/fixed_path_grasp.py --dry-run
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import make_camera
from drivers.robot.grasp_driver import (
    GraspDriver,
    ensure_rebot_sdk_in_syspath,
    selected_arm_config,
)

ensure_rebot_sdk_in_syspath()

try:
    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose
except ImportError:
    RebotArm = None  # type: ignore
    RebotArmEndPose = None  # type: ignore
from utils.camera_utils import (
    compose_cam_to_base_transform,
    hand_eye_compensation_matrix,
    load_config,
    load_hand_eye,
)
from utils.ordinary_grasp import GraspPose, draw_grasp, estimate_grasps, select_best_grasp
from utils.transforms import (
    canonicalize_parallel_gripper_tcp_rotation,
    pose6d_to_mat4,
    rotation_matrix_to_euler_zyx,
)
from utils.yolo_utils import DetectorMemoryError, load_yolo, warmup_yolo


# ==============================================================================
# LOCAL GRASP & WAYPOINT CONFIGURATION (Customize your path here)
# ==============================================================================

# 1. Ready / Scan Pose (Approach Waypoint 0)
#    Set to None to use robot.ready_pose from config/default.yaml,
#    or customize the dictionary below directly.
LOCAL_READY_POSE: Optional[dict[str, float]] = {
    "x": 0.3,
    "y": 0.0,
    "z": 0.3,
    "roll": 0.0,
    "pitch": 0.7,
    "yaw": 0.0,
    "duration": 3.0,
}

# 2. Fixed Grasp & Transport Orientation (in radians)
#    The gripper will lock to this orientation during approach, grasp, and retreat
#    (unless a specific waypoint overrides roll/pitch/yaw).
FIXED_GRASP_RPY: dict[str, float] = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
}

# 3. Gripper Width Configuration (in meters, max hardware stroke is 0.09m)
#    - GRIPPER_READY_OPEN_M: How wide the gripper opens at the Ready Pose (e.g. 0.045 = 4.5cm).
#    - GRIPPER_CLOSE_OFFSET_M: Subtracted from YOLO jaw_width_m when closing (matching main.py).
#    - GRIPPER_FIXED_CLOSE_M: Set to a float (e.g. 0.015) to close to a fixed width,
#      or leave as None to use max(0.0, jaw_width_m - GRIPPER_CLOSE_OFFSET_M) like main.py.
GRIPPER_READY_OPEN_M: float = 0.055
GRIPPER_CLOSE_OFFSET_M: float = 0.01
GRIPPER_FIXED_CLOSE_M: Optional[float] = 0.03

# 4. Optional Intermediate Approach Waypoints (between Ready Pose and Object)
#    Ready Pose is already Waypoint 0. Add dicts here if you want extra via-points before grasping.
APPROACH_WAYPOINTS: list[dict[str, Any]] = [
    # Example:
    # {"x": 0.28, "y": 0.00, "z": 0.25, "duration": 2.0},
]

# 5. Pre-Grasp Offset relative to the detected object's (xg, yg, zg) in Base Frame
#    Set ENABLE_PREGRASP = False to move directly from Ready Pose to the grasp point.
ENABLE_PREGRASP: bool = True
PREGRASP_OFFSET_BASE: dict[str, float] = {
    "dx": 0.00,   # meters in base X (e.g. -5cm back before approaching)
    "dy": 0.00,    # meters in base Y
    "dz": 0.1,    # meters in base Z (e.g. +5cm above object)
    "duration": 5.0,
}

# Duration (seconds) for the final move into the grasp point (xg, yg, zg)
GRASP_MOVE_DURATION: float = 2.0

# 6. Retreat Waypoints (executed sequentially AFTER the gripper closes on the object)
#    Each waypoint dictionary supports:
#      - Relative to grasp point: {"dx": ..., "dy": ..., "dz": ..., "duration": ...}
#      - Absolute in base frame:  {"x": ...,  "y": ...,  "z": ...,  "duration": ...}
#      - Optional orientation override: "roll", "pitch", "yaw" (defaults to FIXED_GRASP_RPY)
#      - Optional gripper command after reaching waypoint:
#          "gripper": "open"  -> opens gripper to GRIPPER_READY_OPEN_M
#          "gripper": "close" -> closes gripper to 0.0m
#          "gripper": 0.04    -> opens gripper to 0.04m (4.0cm)
#      - Optional pause trigger to keep arm steady at this pose:
#          "pause": True      -> holds pose indefinitely; press [SPACE] to resume
#
#    How to add a pause:
#      Option A (within a waypoint dict):
#          {"x": 0.4, "y": -0.05, "z": 0.15, "roll": 0.7, "pitch": 0.0, "yaw": -0.2, "duration": 2.0, "pause": True}
#      Option B (as a dedicated entry between waypoints):
#          pause_waypoint()   # or simply: pause()
def pause_waypoint(prompt: Optional[str] = None) -> dict[str, Any]:
    """Helper to insert a pause between waypoints. Arm holds pose until [SPACE] is pressed."""
    return {"pause": True, "prompt": prompt or "Arm paused. Press [SPACE] to resume motion..."}


pause = pause_waypoint  # Alias so both pause() and pause_waypoint() work


RETREAT_WAYPOINTS: list[dict[str, Any]] = [
    # WP 1: Lift straight up +10cm in Z from wherever the object was grasped
    {"dx": 0.00, "dy": 0.00, "dz": 0.10, "duration": 2.0},


    {"x": 0.35, "y": 0.30, "z": 0.18, "roll": -0.8, "pitch": 0.0, "yaw": 1.2, "duration": 3.0},
    #pause("Hold pose until SPACE to continue retreat sequence..."),

    #{"x": 0.4, "y": -0.05, "z": 0.25, "roll": 0.7, "pitch": 0.0, "yaw": -0.3, "duration": 2.0},

    # WP 3: Lower to placement height and release the object
    #{"x": 0.25, "y": -0.15, "z": 0.15, "duration": 1.5, "gripper": "open"},

    # WP 4: Lift back up to clear the placed object before returning to Ready Pose
    #{"x": 0.25, "y": -0.15, "z": 0.25, "duration": 2.0},
]

# ==============================================================================


# def _wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
#     thread = getattr(controller, "_send_thread", None)
#     if thread is not None and thread.is_alive():
#         thread.join(timeout=duration + extra + 2.0)
#     else:
#         time.sleep(duration + extra)

def _wait_motion(controller: RebotArmEndPose, duration: float, settle_time: float = 0.4) -> None:
    """Wait for trajectory execution and guarantee settle time for gravity lag."""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + 2.0)
    else:
        time.sleep(duration)
    # Crucial: Sleep AFTER the trajectory thread finishes to let gravity lag settle to 0
    if settle_time > 0:
        time.sleep(settle_time)


def _move_ready(
    controller: RebotArmEndPose,
    ready_cfg: dict[str, Any],
    grasp_driver: Optional[GraspDriver] = None,
    open_at_ready: bool = True,
) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.35)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.28)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 0.6)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)
    if open_at_ready and grasp_driver is not None:
        print(f"[Ready] Open gripper to {GRIPPER_READY_OPEN_M * 100:.1f}cm at Ready Pose")
        grasp_driver.open_gripper(distance_m=GRIPPER_READY_OPEN_M)


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


def _compute_camera_centering_tcp_pose(
    obj_xy: tuple[float, float],
    table_z_m: float,
    scan_z: float,
    scan_rpy: dict[str, float],
    T_hand_eye: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[float, float, float]:
    """Calculate the exact robot TCP (x, y, z) so that the eye-in-hand camera's
    optical axis intersects the table plane exactly at obj_xy (image dead-center)."""
    rot = pose6d_to_mat4(0, 0, 0, scan_rpy["roll"], scan_rpy["pitch"], scan_rpy["yaw"])[:3, :3]
    T_comp = hand_eye_compensation_matrix(cfg)

    # Camera optical axis unit vector in robot base frame
    R_cam = rot @ T_hand_eye[:3, :3]
    dir_b = R_cam[:, 2]

    # Camera position offset relative to TCP in robot base frame
    p_cam_rel = rot @ T_hand_eye[:3, 3] + T_comp[:3, 3]

    z_cam = scan_z + p_cam_rel[2]
    if abs(dir_b[2]) < 1e-6:
        return float(obj_xy[0]), float(obj_xy[1]), float(scan_z)
    t = (table_z_m - z_cam) / dir_b[2]

    tcp_x = obj_xy[0] - p_cam_rel[0] - t * dir_b[0]
    tcp_y = obj_xy[1] - p_cam_rel[1] - t * dir_b[1]
    return float(tcp_x), float(tcp_y), float(scan_z)


def _resolve_waypoint_pose(
    wp: dict[str, Any],
    grasp_xyz: tuple[float, float, float],
    default_rpy: dict[str, float],
) -> tuple[float, float, float, float, float, float, float]:
    """Resolve a waypoint dict (relative dx/dy/dz or absolute x/y/z) to (x, y, z, r, p, y, duration)."""
    xg, yg, zg = grasp_xyz
    if any(k in wp for k in ("dx", "dy", "dz")):
        x = xg + float(wp.get("dx", 0.0))
        y = yg + float(wp.get("dy", 0.0))
        z = zg + float(wp.get("dz", 0.0))
    else:
        x = float(wp.get("x", xg))
        y = float(wp.get("y", yg))
        z = float(wp.get("z", zg))

    roll = float(wp.get("roll", default_rpy["roll"]))
    pitch = float(wp.get("pitch", default_rpy["pitch"]))
    yaw = float(wp.get("yaw", default_rpy["yaw"]))
    duration = float(wp.get("duration", 2.0))
    return x, y, z, roll, pitch, yaw, duration


def _handle_waypoint_gripper(grasp_driver: GraspDriver, gripper_cmd: Any) -> None:
    if gripper_cmd is None:
        return
    if isinstance(gripper_cmd, (int, float)):
        dist = float(gripper_cmd)
        print(f"  [Gripper] Move to {dist * 100:.1f}cm")
        grasp_driver.open_gripper(distance_m=dist)
    elif isinstance(gripper_cmd, str):
        cmd = gripper_cmd.strip().lower()
        if cmd == "open":
            print(f"  [Gripper] Open to {GRIPPER_READY_OPEN_M * 100:.1f}cm")
            grasp_driver.open_gripper(distance_m=GRIPPER_READY_OPEN_M)
        elif cmd == "close":
            print("  [Gripper] Close to 0.0cm")
            grasp_driver.open_gripper(distance_m=0.0)


def _is_pure_pause(wp: Any) -> bool:
    """Check if waypoint is a dedicated pause action rather than a motion waypoint."""
    if callable(wp):
        return True
    if isinstance(wp, dict):
        has_coords = any(k in wp for k in ("x", "y", "z", "dx", "dy", "dz"))
        return bool(wp.get("pause", False)) and not has_coords
    return False


def _wait_for_user_resume(
    prompt: str = "Arm paused. Press [SPACE] to resume motion...",
    cam: Optional[Any] = None,
    window_name: Optional[str] = None,
) -> bool:
    """Pause execution and hold current robot pose steady until user presses SPACE.

    Listens to both OpenCV camera window (if active) and the terminal keyboard input.
    Returns True to resume remaining waypoints, or False if aborted (e.g. Q or ESC).
    """
    print(f"\n[PAUSE] {prompt}")
    print("        [SPACE]  -> Resume and execute remaining waypoints")
    print("        [Q / ESC]-> Abort retreat and return safely to Ready Pose\n")

    old_termios = None
    is_tty = sys.stdin.isatty()
    if is_tty:
        try:
            import termios
            import tty

            old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            old_termios = None

    try:
        while True:
            # 1. Update camera preview in OpenCV window if available
            if cam is not None and window_name is not None:
                try:
                    frame, _ = cam.get_frame()
                    if frame is not None:
                        display = frame.copy()
                        h, w = display.shape[:2]
                        # Top semi-transparent banner
                        overlay = display.copy()
                        cv2.rectangle(overlay, (0, 0), (w, 54), (25, 25, 25), -1)
                        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
                        cv2.putText(
                            display,
                            "[PAUSED] ARM STEADY AT WAYPOINT",
                            (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 215, 255),
                            2,
                        )
                        cv2.putText(
                            display,
                            "Press [SPACE] to resume  |  [Q] abort to ready pose",
                            (12, 46),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (255, 255, 255),
                            1,
                        )
                        cv2.imshow(window_name, display)
                except Exception:
                    pass

            # 2. Check OpenCV window keypress
            if window_name is not None:
                try:
                    key = cv2.waitKey(25) & 0xFF
                    if key in (32, 10, 13):  # Space, Enter, Return
                        print("[PAUSE] SPACE received via camera window. Resuming motion...")
                        return True
                    if key in (ord("q"), ord("Q"), 27):  # 'q', 'Q', ESC
                        print("[PAUSE] Abort received via camera window. Returning to Ready Pose...")
                        return False
                except Exception:
                    pass

            # 3. Check terminal keyboard input
            if is_tty:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in (" ", "\n", "\r"):
                        print("[PAUSE] SPACE received via terminal. Resuming motion...")
                        return True
                    if ch in ("q", "Q"):
                        print("[PAUSE] Abort received via terminal. Returning to Ready Pose...")
                        return False
                    if ch == "\x1b":
                        # Differentiate standalone ESC from escape sequence (like arrow keys)
                        r_esc, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if r_esc:
                            sys.stdin.read(2)  # consume rest of arrow key escape sequence
                        else:
                            print("[PAUSE] ESC received via terminal. Returning to Ready Pose...")
                            return False
            else:
                time.sleep(0.05)
    finally:
        if is_tty and old_termios is not None:
            try:
                import termios

                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)
            except Exception:
                pass


def _compute_fixed_orientation_grasp(
    position_cam: np.ndarray,
    T_cam2base: np.ndarray,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, float, float, float, float, float], tuple[float, float, float, float, float, float]]:
    """Transform YOLO 3D camera position to Base Frame and pair with FIXED_GRASP_RPY."""
    p_cam_h = np.array([position_cam[0], position_cam[1], position_cam[2], 1.0], dtype=np.float64)
    p_base = (np.asarray(T_cam2base, dtype=np.float64) @ p_cam_h)[:3]

    xg = float(p_base[0]) + float(insertion_depth_m)
    yg = float(p_base[1])
    zg = float(p_base[2]) #- 0.01

    rxg = float(FIXED_GRASP_RPY["roll"])
    ryg = float(FIXED_GRASP_RPY["pitch"])
    rzg = float(FIXED_GRASP_RPY["yaw"])
    
    xp = xg + float(PREGRASP_OFFSET_BASE.get("dx", 0.0))
    yp = yg + float(PREGRASP_OFFSET_BASE.get("dy", 0.0))
    zp = zg + float(PREGRASP_OFFSET_BASE.get("dz", 0.0))

    grasp6d = (xg, yg, zg, rxg, ryg, rzg)
    pre6d = (xp, yp, zp, rxg, ryg, rzg)
    return grasp6d, pre6d


def _execute_fixed_path_grasp(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, float, float, float, float, float],
    pre6d: tuple[float, float, float, float, float, float],
    ready_cfg: dict[str, Any],
    dry_run: bool = False,
    jaw_width_m: float = 0.02,
    cam: Optional[Any] = None,
    window_name: Optional[str] = None,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d
    grasp_xyz = (xg, yg, zg)

    print("\n================ GRASP & WAYPOINT PLAN ================")
    for i, wp in enumerate(APPROACH_WAYPOINTS):
        wp_item = wp() if callable(wp) else wp
        if _is_pure_pause(wp_item):
            prompt_str = wp_item.get("prompt", "Hold pose until SPACE") if isinstance(wp_item, dict) else "Pause"
            print(f"  [Approach WP #{i+1}] *** PAUSE: {prompt_str} ***")
            continue
        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp_item, grasp_xyz, FIXED_GRASP_RPY)
        p_tag = " [PAUSE AFTER]" if isinstance(wp_item, dict) and wp_item.get("pause") else ""
        print(f"  [Approach WP #{i+1}] xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f}) rpy=({wr:+.3f},{wpitch:+.3f},{wyaw:+.3f}) dur={wdur:.1f}s{p_tag}")
    if ENABLE_PREGRASP:
        print(f"  [Pre-Grasp]      xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f}) dur={PREGRASP_OFFSET_BASE.get('duration', 2.0):.1f}s")
    print(f"  [Grasp Target]   xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f}) dur={GRASP_MOVE_DURATION:.1f}s")
    for i, wp in enumerate(RETREAT_WAYPOINTS):
        wp_item = wp() if callable(wp) else wp
        if _is_pure_pause(wp_item):
            prompt_str = wp_item.get("prompt", "Hold pose until SPACE") if isinstance(wp_item, dict) else "Pause"
            print(f"  [Retreat WP #{i+1}]  *** PAUSE: {prompt_str} ***")
            continue
        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp_item, grasp_xyz, FIXED_GRASP_RPY)
        g_tag = f" gripper={wp_item['gripper']}" if "gripper" in wp_item else ""
        p_tag = " [PAUSE AFTER]" if isinstance(wp_item, dict) and wp_item.get("pause") else ""
        print(f"  [Retreat WP #{i+1}]  xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f}) rpy=({wr:+.3f},{wpitch:+.3f},{wyaw:+.3f}) dur={wdur:.1f}s{g_tag}{p_tag}")
    print("=======================================================")

    if dry_run:
        print("[Grasp] dry run; skip physical motion")
        return False

    # 1. Optional extra approach waypoints (Ready Pose is already Waypoint 0 with gripper open)
    for i, wp in enumerate(APPROACH_WAYPOINTS):
        wp_item = wp() if callable(wp) else wp
        if _is_pure_pause(wp_item):
            prompt = wp_item.get("prompt") if isinstance(wp_item, dict) else None
            msg = prompt or f"Holding pose steady at Approach WP #{i+1}."
            if not _wait_for_user_resume(msg, cam=cam, window_name=window_name):
                print(f"[Approach] User aborted at Approach WP #{i+1}")
                return False
            continue

        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp_item, grasp_xyz, FIXED_GRASP_RPY)
        print(f"[Approach] Moving to Approach WP #{i+1}: xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f})")
        if not controller.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur):
            print(f"[Approach] Approach WP #{i+1} IK failed")
            return False
        _wait_motion(controller, wdur)
        _handle_waypoint_gripper(grasp_driver, wp_item.get("gripper"))

        if isinstance(wp_item, dict) and wp_item.get("pause"):
            prompt = wp_item.get("prompt") or f"Holding pose steady at Approach WP #{i+1}."
            if not _wait_for_user_resume(prompt, cam=cam, window_name=window_name):
                print(f"[Approach] User aborted at Approach WP #{i+1}")
                return False

    # 2. Pre-Grasp
    if ENABLE_PREGRASP:
        pre_dur = float(PREGRASP_OFFSET_BASE.get("duration", 2.0))
        print(f"[Grasp] Move to pregrasp: xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})")
        if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=pre_dur):
            print("[Grasp] Pregrasp IK failed")
            return False
        _wait_motion(controller, pre_dur)

    # 3. Move to Grasp
    print(f"[Grasp] Move to grasp: xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=GRASP_MOVE_DURATION):
        print("[Grasp] Grasp IK failed")
        return False
    _wait_motion(controller, GRASP_MOVE_DURATION)

    # 4. Close Gripper (matching main.py technique)
    if GRIPPER_FIXED_CLOSE_M is not None:
        target_jaw_width = max(0.0, float(GRIPPER_FIXED_CLOSE_M))
    else:
        target_jaw_width = max(0.0, jaw_width_m - GRIPPER_CLOSE_OFFSET_M)
    print(f"[Grasp] Closing gripper to {target_jaw_width * 100:.1f}cm")
    #grasp_driver.open_gripper(distance_m=target_jaw_width)
    ok = grasp_driver.grasp(force=1.5)  # Clamps with 0.40 N·m torque, stops when touching object
    print("[Grasp] Object secured" if ok else "[Grasp] Warning: Empty grasp")

    # 5. Execute Retreat Waypoints
    print("[Retreat] Starting retreat waypoint sequence...")
    for i, wp in enumerate(RETREAT_WAYPOINTS):
        wp_item = wp() if callable(wp) else wp
        if _is_pure_pause(wp_item):
            prompt = wp_item.get("prompt") if isinstance(wp_item, dict) else None
            msg = prompt or f"Holding pose steady at Retreat WP #{i+1}."
            if not _wait_for_user_resume(msg, cam=cam, window_name=window_name):
                print(f"  [Retreat] User aborted at WP #{i+1}")
                break
            continue

        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp_item, grasp_xyz, FIXED_GRASP_RPY)
        p_str = " [PAUSE]" if isinstance(wp_item, dict) and wp_item.get("pause") else ""
        print(f"  -> [Retreat WP #{i+1}/{len(RETREAT_WAYPOINTS)}] xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f}) rpy=({wr:+.2f},{wpitch:+.2f},{wyaw:+.2f}) ({wdur:.1f}s){p_str}")
        if not controller.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur):
            print(f"  [Retreat] Warning: IK failed at Retreat WP #{i+1}, stopping retreat early")
            break
        _wait_motion(controller, wdur)
        _handle_waypoint_gripper(grasp_driver, wp_item.get("gripper"))

        if isinstance(wp_item, dict) and wp_item.get("pause"):
            prompt = wp_item.get("prompt") or f"Holding pose steady at Retreat WP #{i+1}."
            if not _wait_for_user_resume(prompt, cam=cam, window_name=window_name):
                print(f"  [Retreat] User aborted at WP #{i+1}")
                break

    # 6. Return to Ready Pose (Approach Waypoint 0) and open gripper for next cycle
    print("[Grasp] Return to Ready Pose (WP 0)")
    _move_ready(controller, ready_cfg, grasp_driver=grasp_driver, open_at_ready=True)
    return True


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        draw_grasp(display, grasp)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if best is not None:
        x_m, y_m, z_m = best.position.tolist()
        best_text = (
            f"best={best.class_name} conf={best.conf:.2f} "
            f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={best.jaw_width_m * 100:.1f}cm"
        )
        cv2.putText(
            display,
            best_text,
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 140),
            2,
        )
    return display


def _print_best_grasp(grasp: GraspPose) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print("\n[G] Best grasp:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy (vision)={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy (vision)={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")
    print(f"  FIXED_GRASP_RPY (used)={[FIXED_GRASP_RPY['roll'], FIXED_GRASP_RPY['pitch'], FIXED_GRASP_RPY['yaw']]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-orientation grasp with customizable retreat waypoints")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; do not move the arm")
    parser.add_argument('--device', default=None, help='Override YOLO device, e.g. 0 or cpu')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = (
        LOCAL_READY_POSE
        if LOCAL_READY_POSE is not None
        else robot_cfg.get(
            "ready_pose",
            {"x": 0.25, "y": 0.0, "z": 0.18, "roll": 0.0, "pitch": 0.7, "yaw": 0.0, "duration": 3.0},
        )
    )
    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    cam = make_camera(cfg)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Fixed Path Grasp - Waypoint Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=grasp  C=center  R=resume  Q/ESC=quit\n")

    controller: Optional[RebotArmEndPose] = None
    rebotarm: Optional[RebotArm] = None
    grasp_driver: Optional[GraspDriver] = None
    T_hand_eye: Optional[np.ndarray] = None
    yolo_opts: dict[str, Any] = {}
    robot_ready = False

    try:
        cam.open()
        cam.warm_up(15)
        K = cam.K.astype(np.float32)

        cam_type = str(cam_cfg.get("type", "")).lower()
        hand_eye_filename = cfg.get("calibration", {}).get("hand_eye_file", "hand_eye.npz")
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type, hand_eye_filename)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        yolo_cfg = cfg.get("yolo", {})
        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})

        model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
        insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
        depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
        table_z_m = float(grasp_cfg.get("table_z_m", 0.0))
        infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

        print(f"=== Load YOLO: {model_name} ===")
        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT, device_override=args.device)

        # Check full-resolution inference before enabling motors or moving to
        # ready; Jetson CUDA allocation errors must not occur first after motion.
        warmup_frame, _ = cam.get_frame()
        if warmup_frame is None:
            raise RuntimeError("Detector startup check failed: no camera frame; robot was not started")
        try:
            warmup_yolo(model, warmup_frame, yolo_opts)
        except DetectorMemoryError as exc:
            print(f'[Detector] {exc}')
            return 1
        del warmup_frame

        print("=== Init robot ===")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        try:
            grasp_driver.start()
        except RuntimeError as exc:
            print(f"[Robot] {exc}")
            return 1
        robot_ready = True
        print(f"[Robot] mode: {selected.controller_mode}")

        print("[Robot] Move to Ready Pose (Approach Waypoint 0) & pre-open gripper")
        _move_ready(controller, ready_cfg, grasp_driver=grasp_driver, open_at_ready=not args.dry_run)

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and (frame_index % infer_every == 0 or not last_results):
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )

                T_cam2base_live = None
                if T_hand_eye is not None and robot_ready:
                    T_cam2base_live = _cam_to_base(T_hand_eye, grasp_driver, cfg)

                last_grasps = estimate_grasps(
                    last_results,
                    depth_mm,
                    K,
                    depth_quantile=depth_quantile,
                    T_cam2base=T_cam2base_live,
                    table_z_m=table_z_m,
                )

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | G=grasp R=resume Q=quit"
            best_live = select_best_grasp(last_grasps)
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_live, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue

            if key in (ord("c"), ord("C")):
                print("\n[C] MACRO SHOT: Centering over object...")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None:
                    continue
                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                if T_hand_eye is None or not robot_ready:
                    print("[C] Hand-eye calibration unavailable")
                    continue
                T_cam2base_snap = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                snap_grasps = estimate_grasps(
                    snap_results,
                    snap_depth,
                    K,
                    depth_quantile=depth_quantile,
                    T_cam2base=T_cam2base_snap,
                    table_z_m=table_z_m,
                )
                best_macro = select_best_grasp(snap_grasps)
                if best_macro is None:
                    print("[C] No object found to center on!")
                    continue

                # Extract pure detected object position on table in Base frame (no grasp offsets)
                p_cam_h = np.array([best_macro.position[0], best_macro.position[1], best_macro.position[2], 1.0], dtype=np.float64)
                p_base_obj = (np.asarray(T_cam2base_snap, dtype=np.float64) @ p_cam_h)[:3]

                scan_z = float(ready_cfg.get("z", 0.28))
                scan_rpy = {
                    "roll": float(ready_cfg.get("roll", 0.0)),
                    "pitch": float(ready_cfg.get("pitch", 0.6)),
                    "yaw": float(ready_cfg.get("yaw", 0.0)),
                }
                cx, cy, cz = _compute_camera_centering_tcp_pose(
                    (p_base_obj[0], p_base_obj[1]),
                    table_z_m,
                    scan_z,
                    scan_rpy,
                    T_hand_eye,
                    cfg,
                )
                print(f"[C] Centering camera directly over object: TCP xyz=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})")
                if not args.dry_run:
                    controller.move_to_traj(
                        x=cx,
                        y=cy,
                        z=cz,
                        roll=scan_rpy["roll"],
                        pitch=scan_rpy["pitch"],
                        yaw=scan_rpy["yaw"],
                        duration=3.0,
                    )
                    _wait_motion(controller, 3.0)
                    print("[C] Centering complete! Camera live preview active. Press G to grasp.")
                frozen = False
                continue

            if key in (ord("g"), ord("G")):
                print("\n[G] Capture and estimate fixed-orientation grasp")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None:
                    print("[G] Frame capture failed")
                    continue

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )

                T_cam2base_snap = None
                if T_hand_eye is not None and robot_ready:
                    T_cam2base_snap = _cam_to_base(T_hand_eye, grasp_driver, cfg)

                snap_grasps = estimate_grasps(
                    snap_results,
                    snap_depth,
                    K,
                    depth_quantile=depth_quantile,
                    T_cam2base=T_cam2base_snap,
                    table_z_m=table_z_m,
                )
                best = select_best_grasp(snap_grasps)
                if best is None:
                    print("[G] No valid grasp")
                    continue

                _print_best_grasp(best)

                snap_display = _render_display(snap_color, snap_grasps, best, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                grasp6d, pre6d = _compute_fixed_orientation_grasp(
                    best.position,
                    T_cam2base,
                    insertion_depth_m,
                )
                _execute_fixed_path_grasp(
                    controller,
                    grasp_driver,
                    grasp6d,
                    pre6d,
                    ready_cfg,
                    dry_run=args.dry_run,
                    jaw_width_m=best.jaw_width_m,
                    cam=cam,
                    window_name=window_name,
                )

    finally:
        print("\n[Exit] " + ("Release gripper and home" if robot_ready else "Close connections"))
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)

