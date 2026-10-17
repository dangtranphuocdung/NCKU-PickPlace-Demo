#!/usr/bin/env python3
"""
reBot Arm B601 — Vibration & Dynamic Tracking Analysis Tool.

Records real motor feedback (actual joint angles, velocities, torques) directly
from the Damiao/Robstride encoders during stationary holding and trajectory movement.
Performs offline analysis to calculate tracking errors, Cartesian XYZ deviations
via forward kinematics, windowed FFT vibration spectra, and statistical variations
(RMS, Peak-to-Peak, Max Deviation).

Usage:
    # 1. Stationary holding test at current physical pose (safest baseline, 5s)
    python scripts/vibration_analysis.py --mode stationary --duration 5.0

    # 2. Stationary holding test at verified Ready Pose
    python scripts/vibration_analysis.py --mode stationary --target-pose ready --duration 5.0

    # 3. Dynamic motion test between current pose and Ready Pose (3s)
    python scripts/vibration_analysis.py --mode motion --duration 3.0

    # 4. Combined test (motion -> stationary hold -> return)
    python scripts/vibration_analysis.py --mode both --duration 5.0

    # 5. Offline re-analysis on existing CSV data without robot hardware
    python scripts/vibration_analysis.py --analyze-only data/vibration_analysis/<timestamp>/vibration_data.csv

    # 6. Dry run verification (tests data pipeline and plotting without hardware)
    python scripts/vibration_analysis.py --dry-run --mode both
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.robot.grasp_driver import ensure_rebot_sdk_in_syspath, selected_arm_config
from utils.camera_utils import load_config
from utils.vibration_utils import (
    VibrationDataset,
    calculate_metrics,
    compute_cartesian_trajectories,
    compute_fft,
    generate_summary_report,
    load_vibration_csv,
    plot_cartesian_tracking,
    plot_fft_spectra,
    plot_joint_tracking,
    plot_overview_dashboard,
    save_summary_report,
    save_vibration_csv,
)

# Ensure SDK is in sys.path
ensure_rebot_sdk_in_syspath()


# ---------------------------------------------------------------------------
# Synchronous Real-Time Data Recorder
# ---------------------------------------------------------------------------

class SynchronousVibrationRecorder:
    """Thread-safe recorder hooked directly into the robot control loop callback.
    
    Logs commanded and actual motor feedback synchronously on every control tick
    with high-resolution monotonic timestamps.
    """

    def __init__(self, num_joints: int = 6) -> None:
        self.num_joints = num_joints
        self._active = False
        self._phase = "idle"
        self._t_start: float = 0.0

        self.time_records: List[float] = []
        self.q_cmd_records: List[np.ndarray] = []
        self.q_act_records: List[np.ndarray] = []
        self.v_act_records: List[np.ndarray] = []
        self.tau_act_records: List[np.ndarray] = []
        self.phase_records: List[str] = []
        self._lock = threading.Lock()

    def start_recording(self, initial_phase: str = "stationary") -> None:
        with self._lock:
            self._phase = initial_phase
            self._t_start = time.perf_counter()
            self._active = True

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def stop_recording(self) -> None:
        with self._lock:
            self._active = False

    def on_control_tick(self, arm: Any, q_cmd: np.ndarray) -> None:
        """Called inside the control loop callback tick."""
        if not self._active:
            return

        now = time.perf_counter() - self._t_start
        # Read latest motor feedback from CAN RX queue (fast, non-blocking)
        try:
            state = arm.get_state(request_feedback=False)
            q_act = state[0][:self.num_joints].copy()
            v_act = state[1][:self.num_joints].copy() if len(state) > 1 else np.zeros(self.num_joints)
            tau_act = state[2][:self.num_joints].copy() if len(state) > 2 else np.zeros(self.num_joints)
        except Exception:
            return

        cmd = np.asarray(q_cmd[:self.num_joints], dtype=np.float64).copy()

        with self._lock:
            self.time_records.append(now)
            self.q_cmd_records.append(cmd)
            self.q_act_records.append(q_act)
            self.v_act_records.append(v_act)
            self.tau_act_records.append(tau_act)
            self.phase_records.append(self._phase)

    def to_dataset(self) -> VibrationDataset:
        with self._lock:
            time_arr = np.array(self.time_records, dtype=np.float64)
            q_cmd_arr = np.array(self.q_cmd_records, dtype=np.float64)
            q_act_arr = np.array(self.q_act_records, dtype=np.float64)
            v_act_arr = np.array(self.v_act_records, dtype=np.float64)
            tau_act_arr = np.array(self.tau_act_records, dtype=np.float64)
            phases = list(self.phase_records)

        return VibrationDataset(
            time_s=time_arr,
            q_cmd=q_cmd_arr,
            q_act=q_act_arr,
            phases=phases,
            v_act=v_act_arr,
            tau_act=tau_act_arr,
        )


# ---------------------------------------------------------------------------
# Terminal Reporting
# ---------------------------------------------------------------------------

def print_terminal_summary(dataset: VibrationDataset) -> None:
    """Print a clean, structured summary table directly in the terminal."""
    metrics = generate_summary_report(dataset)
    
    print("\n" + "=" * 80)
    print("  reBot Arm B601 — VIBRATION & TRACKING ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  Duration:      {metrics['duration_s']:.2f} s")
    print(f"  Total Samples: {metrics['sample_count']}")
    print(f"  Sampling Rate: {metrics['sampling_rate_hz']:.1f} Hz")
    print("-" * 80)
    print(f"  {'Joint':<8} | {'RMS (mrad)':<12} | {'P2P (mrad)':<12} | {'MaxDev (mrad)':<14} | {'Dominant Frequencies':<20}")
    print("-" * 80)

    for j_name, j_data in metrics["joints"].items():
        freqs_str = ", ".join(f"{f:.1f}Hz" for f in j_data["dominant_frequencies_hz"][:3]) or "None"
        print(
            f"  {j_name:<8} | {j_data['rms_mrad']:<12.3f} | {j_data['p2p_mrad']:<12.3f} | "
            f"{j_data['max_dev_mrad']:<14.3f} | {freqs_str:<20}"
        )

    if "cartesian" in metrics:
        print("-" * 80)
        print("  Cartesian Tool-Tip Deviations (Forward Kinematics):")
        print("-" * 80)
        print(f"  {'Axis':<8} | {'RMS (mm)':<12} | {'P2P (mm)':<12} | {'MaxDev (mm)':<14}")
        print("-" * 80)
        for ax_name, ax_data in metrics["cartesian"].items():
            print(
                f"  {ax_name:<8} | {ax_data['rms_mm']:<12.3f} | {ax_data['p2p_mm']:<12.3f} | {ax_data['max_dev_mm']:<14.3f}"
            )

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Offline Post-Processing Pipeline
# ---------------------------------------------------------------------------

def process_and_save_dataset(
    dataset: VibrationDataset,
    output_dir: Path,
) -> None:
    """Run forward kinematics, generate CSV, summary reports, and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Analysis] Processing {dataset.num_samples} samples collected over {dataset.duration_s:.2f}s...")

    # 1. Compute Forward Kinematics if not already populated
    if dataset.cart_cmd is None or dataset.cart_act is None:
        print("[Analysis] Calculating Cartesian forward kinematics via Pinocchio...")
        cart_cmd, cart_act = compute_cartesian_trajectories(dataset.q_cmd, dataset.q_act)
        dataset.cart_cmd = cart_cmd
        dataset.cart_act = cart_act

    # 2. Save CSV
    csv_path = output_dir / "vibration_data.csv"
    print(f"[Analysis] Saving full time-series data to: {csv_path}")
    save_vibration_csv(dataset, csv_path)

    # 3. Save Summary Reports (Text and JSON)
    txt_path = output_dir / "vibration_summary.txt"
    json_path = output_dir / "vibration_metrics.json"
    print(f"[Analysis] Saving summary report to: {txt_path}")
    save_summary_report(dataset, txt_path, json_path)

    # 4. Generate Visualizations
    plot1 = output_dir / "joint_tracking_errors.png"
    print(f"[Analysis] Generating joint tracking plot: {plot1.name}")
    plot_joint_tracking(dataset, plot1)

    if dataset.cart_cmd is not None:
        plot2 = output_dir / "cartesian_xyz_tracking.png"
        print(f"[Analysis] Generating Cartesian tracking plot: {plot2.name}")
        plot_cartesian_tracking(dataset, plot2)

    plot3 = output_dir / "joint_vibration_fft.png"
    print(f"[Analysis] Generating FFT spectral analysis plot: {plot3.name}")
    plot_fft_spectra(dataset, plot3)

    plot4 = output_dir / "vibration_dashboard.png"
    print(f"[Analysis] Generating consolidated dashboard: {plot4.name}")
    plot_overview_dashboard(dataset, plot4)

    # 5. Display terminal summary
    print_terminal_summary(dataset)
    print(f"[Analysis] All analysis artifacts successfully saved to:\n  -> {output_dir.resolve()}\n")


# ---------------------------------------------------------------------------
# Dry-Run Simulation (Testing Pipeline Without Hardware)
# ---------------------------------------------------------------------------

def execute_dry_run(mode: str, duration: float, output_dir: Path) -> None:
    """Run deterministic dry-run verification without robot hardware."""
    print(f"\n[Dry-Run] Simulating '{mode}' test (duration={duration:.1f}s) without physical hardware...")
    
    fs = 200.0
    n = int(fs * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    n_joints = 6

    # Nominal safe pose
    q_nominal = np.array([0.0, -0.4, 0.8, 0.0, 0.6, 0.0], dtype=np.float64)

    q_cmd = np.zeros((n, n_joints), dtype=np.float64)
    q_act = np.zeros((n, n_joints), dtype=np.float64)
    phases: List[str] = []

    if mode == "stationary":
        for i in range(n):
            q_cmd[i] = q_nominal
            # Add small deterministic multi-tone vibration to actual encoder (5 Hz and 18 Hz jitter)
            jitter = (
                0.0015 * np.sin(2.0 * np.pi * 5.0 * t[i]) +
                0.0008 * np.sin(2.0 * np.pi * 18.0 * t[i])
            )
            q_act[i] = q_nominal + jitter
            phases.append("stationary")

    elif mode == "motion":
        # Smooth minimum-jerk motion from q_nominal to target
        q_target = q_nominal + np.array([0.15, 0.10, -0.15, 0.05, -0.10, 0.05])
        for i in range(n):
            s = t[i] / duration
            # Minimum-jerk polynomial: 10s^3 - 15s^4 + 6s^5
            poly = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
            q_c = q_nominal + (q_target - q_nominal) * poly
            q_cmd[i] = q_c
            # Add dynamic tracking error with 12 Hz vibration
            dyn_err = 0.0020 * np.sin(2.0 * np.pi * 12.0 * t[i])
            q_act[i] = q_c + dyn_err
            phases.append("moving")

    else:  # both
        n_half = n // 2
        q_target = q_nominal + np.array([0.15, 0.10, -0.15, 0.05, -0.10, 0.05])
        for i in range(n):
            if i < n_half:
                s = t[i] / (duration / 2.0)
                poly = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
                q_c = q_nominal + (q_target - q_nominal) * poly
                q_cmd[i] = q_c
                q_act[i] = q_c + 0.0020 * np.sin(2.0 * np.pi * 12.0 * t[i])
                phases.append("moving")
            else:
                q_cmd[i] = q_target
                jitter = 0.0012 * np.sin(2.0 * np.pi * 5.0 * t[i])
                q_act[i] = q_target + jitter
                phases.append("stationary")

    # Generate synthetic Cartesian coordinates for dry-run
    cart_cmd = np.zeros((n, 3), dtype=np.float64)
    cart_act = np.zeros((n, 3), dtype=np.float64)
    p_base_nom = np.array([0.25, 0.0, 0.35], dtype=np.float64)

    for i in range(n):
        if mode == "stationary":
            cart_cmd[i] = p_base_nom
            cart_act[i] = p_base_nom + 0.0004 * np.array([
                np.sin(2.0 * np.pi * 5.0 * t[i]),
                np.cos(2.0 * np.pi * 5.0 * t[i]),
                np.sin(2.0 * np.pi * 18.0 * t[i]),
            ])
        else:
            s = t[i] / duration
            poly = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
            p_target = p_base_nom + np.array([0.05, 0.04, -0.06])
            p_c = p_base_nom + (p_target - p_base_nom) * poly
            cart_cmd[i] = p_c
            cart_act[i] = p_c + 0.0006 * np.array([
                np.sin(2.0 * np.pi * 12.0 * t[i]),
                np.cos(2.0 * np.pi * 12.0 * t[i]),
                np.sin(2.0 * np.pi * 12.0 * t[i]),
            ])

    dataset = VibrationDataset(
        time_s=t,
        q_cmd=q_cmd,
        q_act=q_act,
        phases=phases,
        v_act=np.zeros_like(q_act),
        tau_act=np.zeros_like(q_act),
        cart_cmd=cart_cmd,
        cart_act=cart_act,
    )

    process_and_save_dataset(dataset, output_dir)


# ---------------------------------------------------------------------------
# Physical Robot Hardware Execution
# ---------------------------------------------------------------------------

def execute_robot_test(
    mode: str,
    duration: float,
    target_pose_mode: str,
    arm_mode: Optional[str],
    output_dir: Path,
) -> int:
    """Connect to reBot Arm B601, execute safe test, record data, and analyze."""
    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose

    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get("ready_pose", {})

    selected = selected_arm_config(robot_cfg.get("repo_root"))
    effective_arm_mode = arm_mode or selected.controller_mode

    print("\n" + "=" * 60)
    print("  reBot Arm B601 — VIBRATION RECORDING SESSION")
    print("=" * 60)
    print(f"  Mode:            {mode}")
    print(f"  Duration:        {duration:.1f} s")
    print(f"  Target Pose:     {target_pose_mode}")
    print(f"  Arm Type:        {selected.arm_type} ({effective_arm_mode})")
    print("=" * 60 + "\n")

    rebotarm: Optional[RebotArm] = None
    controller: Optional[RebotArmEndPose] = None
    recorder = SynchronousVibrationRecorder(num_joints=6)

    try:
        print("[Robot] Connecting to arm...")
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=effective_arm_mode)

        # Hook the control loop callback to record data synchronously
        original_loop_cb = controller._loop_cb

        def instrumented_loop_cb(arm_obj: RebotArm, dt: float) -> None:
            original_loop_cb(arm_obj, dt)
            recorder.on_control_tick(arm_obj, controller._q_target)

        controller._loop_cb = instrumented_loop_cb

        # Start controller and control loop
        controller.start()
        time.sleep(0.5)

        # ----------------------------------------------------------------------
        # SAFETY: Read Initial Encoder Pose — NO Random Values
        # ----------------------------------------------------------------------
        q_init, _, _ = rebotarm.get_state(request_feedback=True)
        q_current = q_init[:6].copy()
        print(f"[Robot] Initial Encoder Angles (rad): {[f'{v:+.3f}' for v in q_current]}")

        # Set initial controller target to current position (prevents jumps)
        controller._q_target[:6] = q_current

        # ----------------------------------------------------------------------
        # Execute Selected Test Routine
        # ----------------------------------------------------------------------
        if mode == "stationary":
            if target_pose_mode == "ready":
                print("[Robot] Moving to Ready Pose before stationary recording...")
                controller.move_to_traj(
                    x=float(ready_cfg.get("x", 0.25)),
                    y=float(ready_cfg.get("y", 0.0)),
                    z=float(ready_cfg.get("z", 0.35)),
                    roll=float(ready_cfg.get("roll", 0.0)),
                    pitch=float(ready_cfg.get("pitch", 1.2)),
                    yaw=float(ready_cfg.get("yaw", 0.0)),
                    duration=3.0,
                )
                time.sleep(3.5)

            print(f"\n>>> RECORDING STARTED: Stationary holding for {duration:.1f}s... <<<")
            recorder.start_recording(initial_phase="stationary")
            
            t_end = time.monotonic() + duration
            while time.monotonic() < t_end:
                time.sleep(0.05)
                
            recorder.stop_recording()
            print(">>> RECORDING COMPLETE <<<\n")

        elif mode == "motion":
            print("[Robot] Preparing smooth motion test...")
            # Move between current pose and Ready Pose
            recorder.start_recording(initial_phase="moving")
            print(f"\n>>> RECORDING STARTED: Motion execution for {duration:.1f}s... <<<")
            
            controller.move_to_traj(
                x=float(ready_cfg.get("x", 0.25)),
                y=float(ready_cfg.get("y", 0.0)),
                z=float(ready_cfg.get("z", 0.35)),
                roll=float(ready_cfg.get("roll", 0.0)),
                pitch=float(ready_cfg.get("pitch", 1.2)),
                yaw=float(ready_cfg.get("yaw", 0.0)),
                duration=duration,
            )
            time.sleep(duration + 0.5)
            recorder.stop_recording()
            print(">>> RECORDING COMPLETE <<<\n")

        elif mode == "both":
            print("[Robot] Preparing combined motion & hold test...")
            recorder.start_recording(initial_phase="moving_to_target")
            print(f"\n>>> RECORDING STARTED: Moving to Ready Pose... <<<")
            
            controller.move_to_traj(
                x=float(ready_cfg.get("x", 0.25)),
                y=float(ready_cfg.get("y", 0.0)),
                z=float(ready_cfg.get("z", 0.35)),
                roll=float(ready_cfg.get("roll", 0.0)),
                pitch=float(ready_cfg.get("pitch", 1.2)),
                yaw=float(ready_cfg.get("yaw", 0.0)),
                duration=3.0,
            )
            time.sleep(3.5)

            recorder.set_phase("stationary_hold")
            print(f">>> Stationary holding for {duration:.1f}s... <<<")
            time.sleep(duration)

            recorder.set_phase("returning_home")
            print(">>> Returning smoothly to initial pose... <<<")
            # Smoothly ramp back to initial joint target
            t_ramp = 3.0
            dt_step = 0.02
            steps = int(t_ramp / dt_step)
            q_target_now = controller._q_target[:6].copy()
            for s in range(steps):
                alpha = (s + 1) / steps
                controller._q_target[:6] = q_target_now + (q_current - q_target_now) * alpha
                time.sleep(dt_step)

            recorder.stop_recording()
            print(">>> RECORDING COMPLETE <<<\n")

    except KeyboardInterrupt:
        print("\n[Robot] Interrupted by user.")
        recorder.stop_recording()
    finally:
        print("[Robot] Safely disengaging control loop...")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None and getattr(rebotarm, "_connected", False):
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Robot/Exit] Clean shutdown note: {exc}")

    # Process and save collected data offline
    dataset = recorder.to_dataset()
    if dataset.num_samples < 10:
        print(f"[Warn] Insufficient samples collected ({dataset.num_samples}). Aborting analysis.")
        return 1

    process_and_save_dataset(dataset, output_dir)
    return 0


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="reBot Arm B601 Vibration & Dynamic Tracking Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["stationary", "motion", "both"],
        default="stationary",
        help="Test mode: 'stationary' (safest holding test), 'motion' (trajectory tracking), or 'both'",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds",
    )
    parser.add_argument(
        "--target-pose",
        choices=["current", "ready"],
        default="current",
        help="Starting pose for stationary test ('current' holds wherever arm is; 'ready' moves to Ready Pose)",
    )
    parser.add_argument(
        "--arm-mode",
        choices=["posvel", "mit"],
        default=None,
        help="Arm control mode (defaults to configured hardware mode)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save CSV and generated plots (defaults to data/vibration_analysis/<timestamp>)",
    )
    parser.add_argument(
        "--analyze-only",
        type=str,
        default=None,
        help="Path to an existing CSV file to re-run analysis and plot generation without robot hardware",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run simulated test pipeline and generate plots without connecting to robot hardware",
    )

    args = parser.parse_args()

    # Mode 1: Offline re-analysis on existing CSV
    if args.analyze_only:
        raw_target = str(args.analyze_only).strip()
        vib_root = PROJECT_ROOT / "data" / "vibration_analysis"

        if raw_target.lower() in ("latest", "last"):
            # Find the most recently created session directory containing vibration_data.csv
            candidates = sorted(
                [d for d in vib_root.iterdir() if d.is_dir() and (d / "vibration_data.csv").is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                print(f"[Error] No previous vibration recording sessions found in: {vib_root}")
                return 1
            csv_file = candidates[0] / "vibration_data.csv"
            print(f"[Offline] Auto-resolved latest session: {candidates[0].name}")
        else:
            target_path = Path(raw_target)
            if target_path.is_dir() and (target_path / "vibration_data.csv").is_file():
                csv_file = target_path / "vibration_data.csv"
            else:
                csv_file = target_path

        if not csv_file.is_file():
            print(f"[Error] Specified CSV file does not exist: {csv_file}")
            print("\nAvailable recorded sessions in data/vibration_analysis/:")
            if vib_root.is_dir():
                for d in sorted(vib_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                    if d.is_dir() and (d / "vibration_data.csv").is_file():
                        print(f"  - {d.name} -> python scripts/vibration_analysis.py --analyze-only {d / 'vibration_data.csv'}")
            print("\nTip: You can also simply run: python scripts/vibration_analysis.py --analyze-only latest")
            return 1

        print(f"[Offline] Loading dataset from: {csv_file}")
        dataset = load_vibration_csv(csv_file)
        out_dir = Path(args.out_dir) if args.out_dir else csv_file.parent / "reanalysis"
        process_and_save_dataset(dataset, out_dir)
        return 0

    # Prepare timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "data" / "vibration_analysis" / timestamp

    # Mode 2: Dry Run (verification without hardware)
    if args.dry_run:
        execute_dry_run(args.mode, args.duration, out_dir)
        return 0

    # Mode 3: Physical Robot Hardware Test
    return execute_robot_test(
        mode=args.mode,
        duration=args.duration,
        target_pose_mode=args.target_pose,
        arm_mode=args.arm_mode,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
