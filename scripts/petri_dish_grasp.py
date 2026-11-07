"""
Top-Down Fixed-Orientation Visual Grasping for Petri Dish with Lid.

Designed for flat labware (Petri dishes, microplates, well plates, container lids)
that require approaching vertically from above (top-down) rather than from the side.

Key Features:
  - Top-Down Vertical Descent: Gripper pitch locked to 1.5708 rad (~90 deg pointing straight down).
  - Maintained Level Orientation: The Petri dish remains perfectly horizontal from pre-grasp,
    grasp, lift, transit, to place (prevents agar/liquid sloshing or lid separation).
  - Pre-Grasp Hover: Hovers directly above the dish center (+Z) before vertical descent.
  - Force-Controlled Clamping: Squeezes gently with configurable holding torque from config/default.yaml
    (prevents cracking the plastic lid or popping it off).
  - Vertical Lift & Level Placement: Lifts straight up (+Z) before translating to the place target.

Keys:
  G: Capture frame, compute top-down 3D pose, and execute grasp + waypoint retreat.
  C: Top-down macro centering shot directly over the detected Petri dish.
  R: Resume live preview.
  Q/Esc: Release gripper, return home, and exit.

Usage:
    python scripts/petri_dish_grasp.py
    python scripts/petri_dish_grasp.py --dry-run
    python scripts/petri_dish_grasp.py --arm-mode posvel
"""

from __future__ import annotations

import argparse
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

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
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
from utils.yolo_utils import load_yolo


# ==============================================================================
# LOCAL TOP-DOWN GRASP & WAYPOINT CONFIGURATION (Petri Dish)
# ==============================================================================

# 1. Ready / Scan Pose (Approach Waypoint 0)
#    Camera scans the table from an elevated vantage point.
LOCAL_READY_POSE: Optional[dict[str, float]] = {
    "x": 0.30,
    "y": 0.00,
    "z": 0.27,
    "roll": 0.0,
    "pitch": 0.7,
    "yaw": 0.0,
    "duration": 3.0,
}

# 2. Fixed Top-Down Grasp & Transport Orientation (in radians)
#    pitch = 1.5708 rad (~90 degrees) points the gripper tool-forward axis straight DOWN (-Z in base).
#    This orientation is strictly maintained across Pre-Grasp, Grasp, Lift, Transit, and Place
#    so the Petri dish stays completely flat and level throughout!
FIXED_GRASP_RPY: dict[str, float] = {
    "roll": 0.0,
    "pitch": 1.4,  # Exactly 90 degrees down (tool pointing down towards table)
    "yaw": 0.0,
}

READY_GRASP_RPY: dict[str, float] = {
    "roll": 0.0,
    "pitch": 0.7,  # Slightly angled down for camera scan (not fully vertical)
    "yaw": 0.0,
}

# 3. Gripper Configuration
#    - GRIPPER_READY_OPEN_M: Pre-open jaw width in meters (max stroke is 0.09m = 9cm).
#      Set wide enough to clear the Petri dish lid diameter (e.g. 0.085m for standard 60-80mm dishes).
#    - Clamping Force: Automatically loaded from config/default.yaml (robot.gripper.<type>.default_force).
GRIPPER_READY_OPEN_M: float = 0.09

# 4. Optional Intermediate Approach Waypoints (between Ready Pose and Object)
#    Ready Pose is already Waypoint 0.
APPROACH_WAYPOINTS: list[dict[str, Any]] = [
    # Example:
    # {"x": 0.30, "y": 0.00, "z": 0.30, "duration": 2.0},
]

# 5. Pre-Grasp Vertical Hover Offset (relative to Petri dish center in Base Frame)
#    Directly ABOVE the detected Petri dish (+Z). The robot hovers here before descending.
ENABLE_PREGRASP: bool = True
PREGRASP_OFFSET_BASE: dict[str, float] = {
    "dx": 0.00,    # Perfectly centered in X
    "dy": 0.00,    # Perfectly centered in Y
    "dz": 0.05,    # +5 cm directly above the Petri dish in Z
    "duration": 2.5,
}

# 6. Grasp Height Adjustment (in meters)
#    Offset relative to the detected center/table height (p_base[2]) to target the Petri dish lid/rim.
#    E.g. +0.012 = 1.2 cm above table surface (clamps onto the side walls of the dish and lid).
PETRI_GRASP_Z_OFFSET_M: float = -0.022

# Duration (seconds) for the vertical descent from pre-grasp into the grasp position
GRASP_MOVE_DURATION: float = 2.0

# 7. Retreat & Placement Waypoints (executed sequentially AFTER grasping the Petri dish)
#    All waypoints inherit FIXED_GRASP_RPY (pitch = 1.5708) to keep the dish level!
RETREAT_WAYPOINTS: list[dict[str, Any]] = [
    # WP 1: Vertical lift straight UP (+6cm) to clear surface without exceeding arm kinematic reach
    {"dx": 0.00, "dy": 0.00, "dz": 0.06, "duration": 2.0},

    # WP 2: Horizontal transit above the placement target (dish remains level)
    {"x": 0.25, "y": -0.15, "z": 0.22, "duration": 2.0},

    # WP 3: Lower gently to the placement surface and open gripper
    {"x": 0.25, "y": -0.15, "z": 0.12, "duration": 1.5, "gripper": "open"},

    # WP 4: Lift straight UP to clear the placed Petri dish before returning home
    {"x": 0.25, "y": -0.15, "z": 0.25, "duration": 1.5},
]

# ==============================================================================


def _wait_motion(controller: RebotArmEndPose, duration: float, settle_time: float = 0.4) -> None:
    """Wait for trajectory execution and guarantee settle time for gravity lag."""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + 2.0)
    else:
        time.sleep(duration)
    # Sleep AFTER the trajectory thread finishes to let gravity lag settle to 0
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
        x=float(ready_cfg.get("x", 0.32)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.28)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 0.6)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)
    if open_at_ready and grasp_driver is not None:
        print(f"[Ready] Pre-open gripper to {GRIPPER_READY_OPEN_M * 100:.1f}cm at Ready Pose")
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
        elif cmd in ("grasp", "hold"):
            print("  [Gripper] Grasp with force control (from config)")
            grasp_driver.grasp()


def _compute_top_down_grasp(
    position_cam: np.ndarray,
    T_cam2base: np.ndarray,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, float, float, float, float, float], tuple[float, float, float, float, float, float]]:
    """Transform YOLO 3D camera position to Base Frame for top-down vertical grasp."""
    p_cam_h = np.array([position_cam[0], position_cam[1], position_cam[2], 1.0], dtype=np.float64)
    p_base = (np.asarray(T_cam2base, dtype=np.float64) @ p_cam_h)[:3]

    # For top-down grasp:
    # X and Y align with the Petri dish center
    # Z targets the dish rim/lid height, adjusting for insertion depth downwards
    xg = float(p_base[0])
    yg = float(p_base[1])
    zg = float(p_base[2]) + float(PETRI_GRASP_Z_OFFSET_M) - float(insertion_depth_m)

    # Locked top-down orientation (pitch = 1.5708, straight down)
    rxg = float(FIXED_GRASP_RPY["roll"])
    ryg = float(FIXED_GRASP_RPY["pitch"])
    rzg = float(FIXED_GRASP_RPY["yaw"])

    # Pre-grasp hovers vertically above the dish (+Z)
    xp = xg + float(PREGRASP_OFFSET_BASE.get("dx", 0.0))
    yp = yg + float(PREGRASP_OFFSET_BASE.get("dy", 0.0))
    zp = zg + float(PREGRASP_OFFSET_BASE.get("dz", 0.08))

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
) -> bool:
    del jaw_width_m  # Width is automatically handled by force-controlled stall contact
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d
    grasp_xyz = (xg, yg, zg)

    print("\n================ TOP-DOWN PETRI DISH GRASP PLAN ================")
    print(f"  [Top-Down RPY]   roll={rxg:+.3f} pitch={ryg:+.3f} (downward) yaw={rzg:+.3f}")
    for i, wp in enumerate(APPROACH_WAYPOINTS):
        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, FIXED_GRASP_RPY)
        print(f"  [Approach WP #{i+1}] xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f}) rpy=({wr:+.3f},{wpitch:+.3f},{wyaw:+.3f}) dur={wdur:.1f}s")
    if ENABLE_PREGRASP:
        print(f"  [Pre-Grasp Hover] xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f}) dur={PREGRASP_OFFSET_BASE.get('duration', 2.5):.1f}s")
    print(f"  [Grasp Descent]  xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f}) dur={GRASP_MOVE_DURATION:.1f}s")
    for i, wp in enumerate(RETREAT_WAYPOINTS):
        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, FIXED_GRASP_RPY)
        g_tag = f" gripper={wp['gripper']}" if "gripper" in wp else ""
        print(f"  [Retreat WP #{i+1}]  xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f}) rpy=({wr:+.3f},{wpitch:+.3f},{wyaw:+.3f}) dur={wdur:.1f}s{g_tag}")
    print("================================================================")

    if dry_run:
        print("[Grasp] dry run; skip physical motion")
        return False

    # 1. Optional extra approach waypoints
    for i, wp in enumerate(APPROACH_WAYPOINTS):
        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, FIXED_GRASP_RPY)
        print(f"[Approach] Moving to Approach WP #{i+1}: xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f})")
        if not controller.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur):
            print(f"[Approach] Approach WP #{i+1} IK failed")
            return False
        _wait_motion(controller, wdur)
        _handle_waypoint_gripper(grasp_driver, wp.get("gripper"))

    # 2. Pre-Grasp Hover (directly above dish)
    if ENABLE_PREGRASP:
        pre_dur = float(PREGRASP_OFFSET_BASE.get("duration", 2.5))
        print(f"[Grasp] Move to pre-grasp hover: xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})")
        if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=pre_dur):
            print("[Grasp] Pre-grasp hover IK failed")
            return False
        _wait_motion(controller, pre_dur)

    # 3. Vertical Descent to Grasp Position
    print(f"[Grasp] Vertical descent to grasp: xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=GRASP_MOVE_DURATION):
        print("[Grasp] Grasp descent IK failed")
        return False
    _wait_motion(controller, GRASP_MOVE_DURATION)

    # 4. Close Gripper with Force Control (reads default_force from config/default.yaml)
    #holding_torque = abs(getattr(grasp_driver, "_default_force", 0.30))
    #print(f"[Grasp] Closing gripper with force control (holding torque: {holding_torque:.2f} N·m from config)...")
    ok = grasp_driver.grasp(force=0.05)  # Clamps with gentle torque, stops on physical contact
    print("[Grasp] Petri dish secured" if ok else "[Grasp] Warning: Empty grasp / contact not reached")

    # 5. Execute Retreat Waypoints (Vertical lift & level transport)
    print("[Retreat] Starting retreat waypoint sequence...")
    for i, wp in enumerate(RETREAT_WAYPOINTS):
        wx, wy, wz, wr, wpitch, wyaw, wdur = _resolve_waypoint_pose(wp, grasp_xyz, FIXED_GRASP_RPY)
        print(f"  -> [Retreat WP #{i+1}/{len(RETREAT_WAYPOINTS)}] xyz=({wx:+.3f},{wy:+.3f},{wz:+.3f}) rpy=({wr:+.2f},{wpitch:+.2f},{wyaw:+.2f}) ({wdur:.1f}s)")
        if not controller.move_to_traj(wx, wy, wz, wr, wpitch, wyaw, duration=wdur):
            print(f"  [Retreat] Warning: IK failed at Retreat WP #{i+1}, stopping retreat early")
            break
        _wait_motion(controller, wdur)
        _handle_waypoint_gripper(grasp_driver, wp.get("gripper"))

    # 6. Return to Ready Pose and pre-open gripper for next cycle
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
    print("\n[G] Detected object:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy (vision)={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy (vision)={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")
    print(f"  FIXED_TOP_DOWN_RPY={[FIXED_GRASP_RPY['roll'], FIXED_GRASP_RPY['pitch'], FIXED_GRASP_RPY['yaw']]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-down fixed-orientation Petri dish grasp pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; do not move the arm")
    parser.add_argument(
        "--arm-mode",
        choices=["posvel", "mit"],
        default=None,
        help="Arm control mode override: 'posvel' (recommended, high holding precision) or 'mit'",
    )
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
            {"x": 0.32, "y": 0.0, "z": 0.28, "roll": 0.0, "pitch": 0.6, "yaw": 0.0, "duration": 3.0},
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

    window_name = "Petri Dish Top-Down Grasp (Level Fixed RPY)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=grasp (top-down)  C=center (top-down view)  R=resume  Q/ESC=quit\n")

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
        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

        print("=== Init robot ===")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        arm_mode = args.arm_mode if args.arm_mode is not None else selected.controller_mode

        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=arm_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        print(f"[Robot] mode: {arm_mode}")

        print("[Robot] Move to Ready Pose (WP 0) & pre-open gripper")
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

            best_live = select_best_grasp(last_grasps)
            status = f"LIVE | FPS: {fps_value:.1f} | Top-Down Mode" if not frozen else "PAUSED [R: resume | G: grasp]"
            display = _render_display(color_bgr, last_grasps, best_live, status) if not frozen else last_display
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

            # Centering shot directly above Petri dish
            if key in (ord("c"), ord("C")):
                print("\n[C] MACRO SHOT: Centering top-down over Petri dish...")
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
                    "roll": float(READY_GRASP_RPY["roll"]),
                    "pitch": float(READY_GRASP_RPY["pitch"]),
                    "yaw": float(READY_GRASP_RPY["yaw"]),
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
                        x=cx - 0.03,
                        y=cy,
                        z=cz,
                        roll=scan_rpy["roll"],
                        pitch=scan_rpy["pitch"],
                        yaw=scan_rpy["yaw"],
                        duration=3.0,
                    )
                    _wait_motion(controller, 3.0)
                    print("[C] Camera centered! Live preview active. Press G for top-down precision grasp.")
                frozen = False
                continue

            # Top-down grasp execution
            if key in (ord("g"), ord("G")):
                print("\n[G] Capture and estimate top-down Petri dish grasp")
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

                snap_display = _render_display(snap_color, snap_grasps, best, "SNAPSHOT - Top-Down Grasp")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                grasp6d, pre6d = _compute_top_down_grasp(
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
                )

    finally:
        print("\n[Exit] Release gripper and home")
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] Error releasing gripper: {exc}")

        try:
            if robot_ready and controller is not None and getattr(controller, "_running", False):
                controller.safe_home()
        except Exception as exc:
            print(f"[Exit] Error during safe home: {exc}")

        try:
            if robot_ready and controller is not None:
                controller.end()
        except Exception as exc:
            print(f"[Exit] Error shutting down controller: {exc}")

        try:
            cam.close()
        except Exception as exc:
            print(f"[Exit] Error closing camera: {exc}")

        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

