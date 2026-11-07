#!/usr/bin/env python3
"""
move_arm_ik.py — Simple Robot Arm IK Trajectory Mover.

Moves the arm using Cartesian coordinates: x, y, z, roll, pitch, yaw.

Usage:
    # 1. Single motion via CLI arguments:
    python scripts/move_arm_ik.py 0.30 0.00 0.30 0.00 0.70 0.00
    python scripts/move_arm_ik.py --x 0.30 --y 0.00 --z 0.30 --roll 0.00 --pitch 0.70 --yaw 0.00

    # 2. Interactive prompt (enter poses repeatedly to take dataset photos):
    python scripts/move_arm_ik.py

    # 3. Dry-run simulation (without hardware):
    python scripts/move_arm_ik.py 0.30 0.00 0.30 0.00 0.70 0.00 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.camera_utils import load_config

try:
    from drivers.robot.grasp_driver import (
        GraspDriver,
        ensure_rebot_sdk_in_syspath,
        selected_arm_config,
    )

    ensure_rebot_sdk_in_syspath()

    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose
except ImportError:
    GraspDriver = None
    ensure_rebot_sdk_in_syspath = None
    selected_arm_config = None
    RebotArm = None
    RebotArmEndPose = None


def wait_motion(controller: Any, duration: float, extra: float = 0.4) -> None:
    """Wait for trajectory execution to complete."""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def move_arm(
    controller: Any,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    duration: float = 2.0,
    dry_run: bool = False,
) -> bool:
    """Move arm end-effector via IK trajectory."""
    print(
        f"\n[Move] Target: x={x:.3f}, y={y:.3f}, z={z:.3f}, "
        f"roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f} ({duration:.1f}s)"
    )
    if dry_run or controller is None:
        print("[Dry-run] Simulated motion completed.")
        return True

    if not getattr(controller, "_running", False):
        print("[Error] Controller control loop is not running.")
        return False

    success = controller.move_to_traj(
        x=float(x),
        y=float(y),
        z=float(z),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        duration=float(duration),
    )
    if success is False:
        print("[Error] IK solver failed or motion trajectory could not be generated.")
        return False

    wait_motion(controller, duration)
    print("[Move] Done.")
    return True


def parse_pose_str(line: str) -> Optional[list[float]]:
    try:
        parts = [float(val) for val in line.replace(",", " ").split() if val]
        if len(parts) == 6:
            return parts
    except ValueError:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Move robot arm using IK (x, y, z, roll, pitch, yaw)")
    parser.add_argument("coords", nargs="*", type=float, help="Optional 6 numbers: x y z roll pitch yaw")
    parser.add_argument("--x", type=float, default=None, help="Target X in meters")
    parser.add_argument("--y", type=float, default=None, help="Target Y in meters")
    parser.add_argument("--z", type=float, default=None, help="Target Z in meters")
    parser.add_argument("--roll", type=float, default=None, help="Target Roll in radians")
    parser.add_argument("--pitch", type=float, default=None, help="Target Pitch in radians")
    parser.add_argument("--yaw", type=float, default=None, help="Target Yaw in radians")
    parser.add_argument("--duration", type=float, default=2.0, help="Trajectory duration in seconds (default: 2.0)")
    parser.add_argument("--no-hold", action="store_true", help="Do not pause/hold at target pose on exit")
    parser.add_argument("--dry-run", action="store_true", help="Simulate motion without hardware connection")
    parser.add_argument("--config", default="config/default.yaml", help="Path to default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get("ready_pose", {})

    controller = None
    rebotarm = None
    grasp_driver = None

    if not args.dry_run:
        if RebotArm is None or RebotArmEndPose is None or GraspDriver is None:
            print("[Error] reBotArm SDK or GraspDriver not found. Use --dry-run for testing.")
            return 1
        print("[Robot] Connecting to arm hardware...")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        print(f"[Robot] Connected and control loop active (mode: {selected.controller_mode}).")

    try:
        # Determine target pose from CLI if provided
        target_pose = None
        if len(args.coords) == 6:
            target_pose = args.coords
        elif any(v is not None for v in [args.x, args.y, args.z, args.roll, args.pitch, args.yaw]):
            target_pose = [
                args.x if args.x is not None else float(ready_cfg.get("x", 0.30)),
                args.y if args.y is not None else float(ready_cfg.get("y", 0.00)),
                args.z if args.z is not None else float(ready_cfg.get("z", 0.30)),
                args.roll if args.roll is not None else float(ready_cfg.get("roll", 0.00)),
                args.pitch if args.pitch is not None else float(ready_cfg.get("pitch", 0.70)),
                args.yaw if args.yaw is not None else float(ready_cfg.get("yaw", 0.00)),
            ]

        # Case 1: Pose passed via CLI -> move once
        if target_pose is not None:
            x, y, z, roll, pitch, yaw = target_pose
            success = move_arm(controller, x, y, z, roll, pitch, yaw, duration=args.duration, dry_run=args.dry_run)
            if success and not args.dry_run and not args.no_hold:
                try:
                    input("\n[Hold] Arm reached target pose. Take your photos, then press Enter to exit/home (or Ctrl+C)... ")
                except (EOFError, KeyboardInterrupt):
                    pass
            return 0

        # Case 2: No pose on CLI -> interactive prompt for taking dataset photos
        print("\n=== Robot Arm IK Mover ===")
        print("Enter: x y z roll pitch yaw (or 'q' to quit)")
        print(f"Example Ready Pose: 0.30 0.00 0.30 0.00 0.70 0.00")
        print("=" * 45)

        while True:
            try:
                line = input("\nPose [x y z r p y] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break

            pose = parse_pose_str(line)
            if pose is None:
                print("Invalid input. Please enter 6 numbers: x y z roll pitch yaw")
                print("Example: 0.30 0.00 0.30 0.00 0.70 0.00")
                continue

            x, y, z, roll, pitch, yaw = pose
            move_arm(controller, x, y, z, roll, pitch, yaw, duration=args.duration, dry_run=args.dry_run)

    finally:
        if grasp_driver is not None:
            try:
                grasp_driver.release_gripper()
            except Exception:
                pass
        if controller is not None and getattr(controller, "_running", False):
            try:
                controller.end()
            except Exception:
                pass
        elif rebotarm is not None:
            try:
                rebotarm.disconnect()
            except Exception:
                pass
        print("[Exit] Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
