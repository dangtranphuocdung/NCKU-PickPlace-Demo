"""Vibration and dynamic tracking analysis utilities for reBot Arm B601.

This module provides signal processing, forward kinematics trajectory computation,
statistical metrics calculation, frequency-domain FFT analysis, CSV logging,
and publication-quality plotting routines.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure headless matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class VibrationDataset:
    """Container holding recorded joint, Cartesian, and timing data."""
    time_s: np.ndarray             # (N,) monotonic timestamps (seconds from start)
    q_cmd: np.ndarray              # (N, n_joints) commanded joint angles (rad)
    q_act: np.ndarray              # (N, n_joints) actual encoder joint angles (rad)
    phases: List[str]              # (N,) execution phase tags ('stationary', 'moving', etc.)
    v_act: Optional[np.ndarray] = None    # (N, n_joints) actual joint velocities (rad/s)
    tau_act: Optional[np.ndarray] = None  # (N, n_joints) actual motor torques (Nm)
    
    # Cartesian trajectories computed via Forward Kinematics (Pinocchio)
    cart_cmd: Optional[np.ndarray] = None  # (N, 3) [x, y, z] in meters
    cart_act: Optional[np.ndarray] = None  # (N, 3) [x, y, z] in meters
    
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.time_s)

    @property
    def num_joints(self) -> int:
        return self.q_cmd.shape[1] if self.q_cmd.ndim > 1 else 0

    @property
    def duration_s(self) -> float:
        if len(self.time_s) < 2:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def sampling_rate_hz(self) -> float:
        if len(self.time_s) < 2:
            return 0.0
        dt = np.diff(self.time_s)
        mean_dt = float(np.mean(dt))
        return 1.0 / mean_dt if mean_dt > 0 else 0.0

    @property
    def joint_errors(self) -> np.ndarray:
        """Tracking error e(t) = q_act - q_cmd in radians."""
        return self.q_act - self.q_cmd

    @property
    def cartesian_errors_m(self) -> Optional[np.ndarray]:
        """Cartesian error [dx, dy, dz] = cart_act - cart_cmd in meters."""
        if self.cart_act is not None and self.cart_cmd is not None:
            return self.cart_act - self.cart_cmd
        return None

    @property
    def cartesian_norm_errors_mm(self) -> Optional[np.ndarray]:
        """Cartesian 3D Euclidean error ||p_act - p_cmd|| in millimeters."""
        err_m = self.cartesian_errors_m
        if err_m is not None:
            return np.linalg.norm(err_m, axis=1) * 1000.0
        return None


# ---------------------------------------------------------------------------
# Forward Kinematics Calculation
# ---------------------------------------------------------------------------

def compute_cartesian_trajectories(
    q_cmd: np.ndarray,
    q_act: np.ndarray,
    repo_root: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute end-effector Cartesian XYZ trajectories for commanded and actual joints.
    
    Returns:
        (cart_cmd, cart_act) where each is an (N, 3) array in meters,
        or (None, None) if Pinocchio / robot model is unavailable.
    """
    try:
        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model
        model = load_robot_model()
    except Exception as exc:
        print(f"[VibrationUtils] Forward kinematics unavailable: {exc}")
        return None, None

    n_samples, n_joints = q_cmd.shape
    cart_cmd = np.zeros((n_samples, 3), dtype=np.float64)
    cart_act = np.zeros((n_samples, 3), dtype=np.float64)

    for i in range(n_samples):
        q_c_padded = pad_q_for_model(model, q_cmd[i], n_joints)
        q_a_padded = pad_q_for_model(model, q_act[i], n_joints)
        
        pos_c, _, _ = compute_fk(model, q_c_padded)
        pos_a, _, _ = compute_fk(model, q_a_padded)
        
        cart_cmd[i] = pos_c
        cart_act[i] = pos_a

    return cart_cmd, cart_act


# ---------------------------------------------------------------------------
# Statistical Metrics Calculation
# ---------------------------------------------------------------------------

@dataclass
class MetricSummary:
    rms: float
    p2p: float
    max_dev: float
    mean: float
    std: float


def calculate_metrics(
    actual: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> MetricSummary:
    """Calculate RMS, Peak-to-Peak (P2P), Maximum Deviation, Mean, and Std.
    
    If `reference` is provided, deviation and error are evaluated relative to reference:
        error = actual - reference
    Otherwise, deviation is evaluated relative to the signal's mean (steady-state jitter):
        residual = actual - mean(actual)
    """
    if len(actual) == 0:
        return MetricSummary(rms=0.0, p2p=0.0, max_dev=0.0, mean=0.0, std=0.0)

    actual = np.asarray(actual, dtype=np.float64)
    mean_val = float(np.mean(actual))
    std_val = float(np.std(actual))
    p2p_val = float(np.ptp(actual))

    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        error = actual - reference
        rms_val = float(np.sqrt(np.mean(error ** 2)))
        max_dev = float(np.max(np.abs(error)))
    else:
        residual = actual - mean_val
        rms_val = float(np.sqrt(np.mean(residual ** 2)))
        max_dev = float(np.max(np.abs(residual)))

    return MetricSummary(
        rms=rms_val,
        p2p=p2p_val,
        max_dev=max_dev,
        mean=mean_val,
        std=std_val,
    )


# ---------------------------------------------------------------------------
# FFT & Spectral Analysis
# ---------------------------------------------------------------------------

@dataclass
class FFTSpectrum:
    frequencies_hz: np.ndarray  # (K,) frequency bins (0 to Nyquist)
    amplitudes: np.ndarray      # (K,) single-sided amplitude spectrum
    peak_frequencies_hz: List[float]  # top dominant vibration frequencies
    peak_amplitudes: List[float]      # amplitudes at peak frequencies


def compute_fft(
    time_s: np.ndarray,
    signal: np.ndarray,
    top_k: int = 3,
    min_freq_hz: float = 0.5,
) -> FFTSpectrum:
    """Compute one-sided Fast Fourier Transform (FFT) with Hann windowing and peak detection.
    
    Args:
        time_s: Time vector in seconds.
        signal: 1D vibration signal array.
        top_k: Number of dominant peaks to detect.
        min_freq_hz: Ignore frequencies below this threshold (e.g. DC drift).
    """
    n = len(signal)
    if n < 4:
        empty = np.array([], dtype=np.float64)
        return FFTSpectrum(frequencies_hz=empty, amplitudes=empty, peak_frequencies_hz=[], peak_amplitudes=[])

    signal = np.asarray(signal, dtype=np.float64)
    # Remove mean / DC component
    detrended = signal - np.mean(signal)

    # Calculate average sampling rate
    dt = float(np.mean(np.diff(time_s)))
    if dt <= 0:
        dt = 0.005  # fallback to 200 Hz
    fs = 1.0 / dt

    # Apply Hann window to eliminate spectral leakage
    window = np.hanning(n)
    windowed_signal = detrended * window
    # Window energy correction factor (for Hann window, mean(window) = 0.5)
    win_scale = np.mean(window) if np.mean(window) > 0 else 1.0

    # Real FFT
    rfft_vals = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    # Single-sided amplitude spectrum
    amplitudes = (2.0 / (n * win_scale)) * np.abs(rfft_vals)
    if len(amplitudes) > 0:
        amplitudes[0] /= 2.0  # DC bin is not doubled

    # Peak detection: find local maxima above min_freq_hz
    valid_idx = np.where(freqs >= min_freq_hz)[0]
    peak_freqs: List[float] = []
    peak_amps: List[float] = []

    if len(valid_idx) > 2:
        valid_amps = amplitudes[valid_idx]
        valid_freqs = freqs[valid_idx]
        
        # Simple local maxima detection: amplitude greater than both neighbors
        local_peaks = []
        for j in range(1, len(valid_amps) - 1):
            if valid_amps[j] > valid_amps[j - 1] and valid_amps[j] > valid_amps[j + 1]:
                local_peaks.append((valid_freqs[j], valid_amps[j]))

        # Sort peaks by amplitude descending
        local_peaks.sort(key=lambda item: item[1], reverse=True)
        for f, a in local_peaks[:top_k]:
            peak_freqs.append(round(float(f), 2))
            peak_amps.append(float(a))

    return FFTSpectrum(
        frequencies_hz=freqs,
        amplitudes=amplitudes,
        peak_frequencies_hz=peak_freqs,
        peak_amplitudes=peak_amps,
    )


# ---------------------------------------------------------------------------
# CSV Persistence
# ---------------------------------------------------------------------------

def save_vibration_csv(dataset: VibrationDataset, file_path: Path | str) -> None:
    """Save full time-series dataset to CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = dataset.num_samples
    n_joints = dataset.num_joints

    headers = ["time_s", "phase"]
    for j in range(n_joints):
        headers.extend([f"q_cmd_{j+1}_rad", f"q_act_{j+1}_rad", f"q_err_{j+1}_mrad"])
    
    has_cart = dataset.cart_cmd is not None and dataset.cart_act is not None
    if has_cart:
        headers.extend([
            "x_cmd_m", "y_cmd_m", "z_cmd_m",
            "x_act_m", "y_act_m", "z_act_m",
            "x_err_mm", "y_err_mm", "z_err_mm",
            "norm_err_mm"
        ])
    
    has_vel = dataset.v_act is not None
    if has_vel:
        for j in range(n_joints):
            headers.append(f"v_act_{j+1}_rad_s")

    has_tau = dataset.tau_act is not None
    if has_tau:
        for j in range(n_joints):
            headers.append(f"tau_act_{j+1}_Nm")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(n_samples):
            row: List[Any] = [f"{dataset.time_s[i]:.4f}", dataset.phases[i]]
            
            for j in range(n_joints):
                qc = dataset.q_cmd[i, j]
                qa = dataset.q_act[i, j]
                err_mrad = (qa - qc) * 1000.0
                row.extend([f"{qc:.6f}", f"{qa:.6f}", f"{err_mrad:.4f}"])
            
            if has_cart:
                xc, yc, zc = dataset.cart_cmd[i]
                xa, ya, za = dataset.cart_act[i]
                xe_mm = (xa - xc) * 1000.0
                ye_mm = (ya - yc) * 1000.0
                ze_mm = (za - zc) * 1000.0
                norm_mm = np.sqrt(xe_mm**2 + ye_mm**2 + ze_mm**2)
                row.extend([
                    f"{xc:.5f}", f"{yc:.5f}", f"{zc:.5f}",
                    f"{xa:.5f}", f"{ya:.5f}", f"{za:.5f}",
                    f"{xe_mm:.3f}", f"{ye_mm:.3f}", f"{ze_mm:.3f}",
                    f"{norm_mm:.3f}"
                ])
                
            if has_vel:
                for j in range(n_joints):
                    row.append(f"{dataset.v_act[i, j]:.4f}")
                    
            if has_tau:
                for j in range(n_joints):
                    row.append(f"{dataset.tau_act[i, j]:.4f}")

            writer.writerow(row)


def load_vibration_csv(file_path: Path | str) -> VibrationDataset:
    """Load VibrationDataset from a previously exported CSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty dataset in {path}")

    # Detect joint columns
    q_cmd_cols = [idx for idx, h in enumerate(headers) if h.startswith("q_cmd_")]
    q_act_cols = [idx for idx, h in enumerate(headers) if h.startswith("q_act_")]
    n_joints = len(q_cmd_cols)

    has_cart = "x_cmd_m" in headers and "x_act_m" in headers
    has_vel = any(h.startswith("v_act_") for h in headers)
    has_tau = any(h.startswith("tau_act_") for h in headers)

    time_list = []
    phases_list = []
    q_cmd_list = []
    q_act_list = []
    cart_cmd_list = []
    cart_act_list = []
    v_act_list = []
    tau_act_list = []

    for row in rows:
        time_list.append(float(row[0]))
        phases_list.append(row[1])
        
        q_c = [float(row[c]) for c in q_cmd_cols]
        q_a = [float(row[c]) for c in q_act_cols]
        q_cmd_list.append(q_c)
        q_act_list.append(q_a)

        if has_cart:
            xc = float(row[headers.index("x_cmd_m")])
            yc = float(row[headers.index("y_cmd_m")])
            zc = float(row[headers.index("z_cmd_m")])
            xa = float(row[headers.index("x_act_m")])
            ya = float(row[headers.index("y_act_m")])
            za = float(row[headers.index("z_act_m")])
            cart_cmd_list.append([xc, yc, zc])
            cart_act_list.append([xa, ya, za])

        if has_vel:
            v_cols = [headers.index(f"v_act_{j+1}_rad_s") for j in range(n_joints)]
            v_act_list.append([float(row[c]) for c in v_cols])

        if has_tau:
            tau_cols = [headers.index(f"tau_act_{j+1}_Nm") for j in range(n_joints)]
            tau_act_list.append([float(row[c]) for c in tau_cols])

    return VibrationDataset(
        time_s=np.array(time_list, dtype=np.float64),
        q_cmd=np.array(q_cmd_list, dtype=np.float64),
        q_act=np.array(q_act_list, dtype=np.float64),
        phases=phases_list,
        v_act=np.array(v_act_list, dtype=np.float64) if has_vel else None,
        tau_act=np.array(tau_act_list, dtype=np.float64) if has_tau else None,
        cart_cmd=np.array(cart_cmd_list, dtype=np.float64) if has_cart else None,
        cart_act=np.array(cart_act_list, dtype=np.float64) if has_cart else None,
    )


# ---------------------------------------------------------------------------
# Visualization Plotting Functions
# ---------------------------------------------------------------------------

def plot_joint_tracking(dataset: VibrationDataset, save_path: Path | str) -> None:
    """Plot commanded vs actual joint angles and tracking errors over time."""
    n_joints = dataset.num_joints
    t = dataset.time_s - dataset.time_s[0]

    fig, axes = plt.subplots(n_joints, 2, figsize=(14, 2.3 * n_joints), sharex=True)
    if n_joints == 1:
        axes = np.array([axes])

    for j in range(n_joints):
        q_c = dataset.q_cmd[:, j]
        q_a = dataset.q_act[:, j]
        err_mrad = (q_a - q_c) * 1000.0  # milliradians

        # Left column: Angles (rad)
        ax_ang = axes[j, 0]
        ax_ang.plot(t, q_c, "k--", label="Cmd", linewidth=1.5, alpha=0.8)
        ax_ang.plot(t, q_a, "tab:blue", label="Actual", linewidth=1.2)
        ax_ang.set_ylabel(f"Joint {j+1} (rad)", fontsize=10, fontweight="bold")
        ax_ang.grid(True, linestyle=":", alpha=0.6)
        if j == 0:
            ax_ang.set_title("Commanded vs. Actual Angle", fontsize=11, fontweight="bold")
            ax_ang.legend(loc="upper right", framealpha=0.8)

        # Right column: Tracking Error (mrad)
        ax_err = axes[j, 1]
        ax_err.plot(t, err_mrad, "tab:red", linewidth=1.0)
        ax_err.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax_err.set_ylabel("Error (mrad)", fontsize=10, fontweight="bold")
        ax_err.grid(True, linestyle=":", alpha=0.6)
        
        # Annotate RMS error
        rms_val = np.sqrt(np.mean(err_mrad ** 2))
        p2p_val = np.ptp(err_mrad)
        ax_err.text(
            0.98, 0.88,
            f"RMS: {rms_val:.2f} mrad\nP2P: {p2p_val:.2f} mrad",
            transform=ax_err.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )
        if j == 0:
            ax_err.set_title("Tracking Error: e(t) = q_act - q_cmd", fontsize=11, fontweight="bold")

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=10)

    fig.suptitle(f"reBot Arm B601 Joint Tracking Performance (Duration: {dataset.duration_s:.1f}s)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_cartesian_tracking(dataset: VibrationDataset, save_path: Path | str) -> None:
    """Plot Cartesian XYZ position variation and 3D Euclidean error."""
    if dataset.cart_cmd is None or dataset.cart_act is None:
        return

    t = dataset.time_s - dataset.time_s[0]
    axes_labels = ["X", "Y", "Z"]
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    for idx, axis_name in enumerate(axes_labels):
        cmd_mm = dataset.cart_cmd[:, idx] * 1000.0
        act_mm = dataset.cart_act[:, idx] * 1000.0
        err_mm = act_mm - cmd_mm

        ax = axes[idx]
        ax.plot(t, err_mm, color=f"tab:{['red', 'green', 'blue'][idx]}", linewidth=1.2, label=f"Δ{axis_name}")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"Δ{axis_name} Error (mm)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        rms_err = np.sqrt(np.mean(err_mm ** 2))
        p2p_err = np.ptp(err_mm)
        ax.text(
            0.98, 0.85,
            f"RMS: {rms_err:.2f} mm | P2P: {p2p_err:.2f} mm",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

    # 4th subplot: 3D Euclidean Norm Error
    norm_err_mm = dataset.cartesian_norm_errors_mm
    if norm_err_mm is not None:
        ax_norm = axes[3]
        ax_norm.plot(t, norm_err_mm, "tab:purple", linewidth=1.3, label="||ΔP||")
        ax_norm.set_ylabel("3D Error (mm)", fontsize=10, fontweight="bold")
        ax_norm.set_xlabel("Time (s)", fontsize=10)
        ax_norm.grid(True, linestyle=":", alpha=0.6)
        
        rms_norm = np.sqrt(np.mean(norm_err_mm ** 2))
        max_norm = np.max(norm_err_mm)
        ax_norm.text(
            0.98, 0.85,
            f"RMS 3D: {rms_norm:.2f} mm | Max: {max_norm:.2f} mm",
            transform=ax_norm.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

    fig.suptitle("Cartesian Tool-Tip Position Deviation (Forward Kinematics)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_fft_spectra(dataset: VibrationDataset, save_path: Path | str) -> None:
    """Plot single-sided FFT amplitude spectra for all joints with marked peak frequencies."""
    n_joints = dataset.num_joints
    t = dataset.time_s

    # For stationary tests, use residual (q_act - mean); for motion tests, use error (q_act - q_cmd)
    is_motion = np.max(np.abs(dataset.q_cmd - dataset.q_cmd[0])) > 1e-4

    fig, axes = plt.subplots(n_joints, 1, figsize=(11, 2.2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = np.array([axes])

    for j in range(n_joints):
        signal = dataset.joint_errors[:, j] if is_motion else (dataset.q_act[:, j] - np.mean(dataset.q_act[:, j]))
        # Convert to milliradians for readability
        signal_mrad = signal * 1000.0
        
        spec = compute_fft(t, signal_mrad, top_k=3, min_freq_hz=0.5)

        ax = axes[j]
        ax.plot(spec.frequencies_hz, spec.amplitudes, "tab:blue", linewidth=1.2)
        ax.set_ylabel(f"J{j+1} Amp (mrad)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Annotate top peaks
        peak_str = []
        for pk_f, pk_a in zip(spec.peak_frequencies_hz, spec.peak_amplitudes):
            ax.plot(pk_f, pk_a, "ro", markersize=4)
            peak_str.append(f"{pk_f:.1f}Hz ({pk_a:.2f})")
            
        annotation = f"Dominant: {', '.join(peak_str)}" if peak_str else "No prominent peaks"
        ax.text(
            0.98, 0.85,
            annotation,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=10)
    # Limit x axis to Nyquist or max 100 Hz
    max_f = min(100.0, float(np.max(spec.frequencies_hz))) if len(spec.frequencies_hz) > 0 else 50.0
    axes[-1].set_xlim(0.0, max_f)

    title_mode = "Tracking Error FFT" if is_motion else "Stationary Jitter FFT"
    fig.suptitle(f"Joint Vibration Frequency Spectra ({title_mode})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_overview_dashboard(dataset: VibrationDataset, save_path: Path | str) -> None:
    """Generate consolidated overview dashboard figure."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2)

    t = dataset.time_s - dataset.time_s[0]
    n_joints = dataset.num_joints

    # Panel 1: Joint Tracking Overview
    ax_joints = fig.add_subplot(gs[0, 0])
    for j in range(n_joints):
        ax_joints.plot(t, dataset.q_act[:, j], label=f"J{j+1}", linewidth=1.0)
    ax_joints.set_title("Joint Positions (rad)", fontsize=10, fontweight="bold")
    ax_joints.set_ylabel("Angle (rad)")
    ax_joints.grid(True, linestyle=":", alpha=0.5)
    ax_joints.legend(ncol=3, loc="upper right", fontsize=8)

    # Panel 2: Joint Tracking Errors
    ax_errs = fig.add_subplot(gs[1, 0])
    for j in range(n_joints):
        err_mrad = (dataset.q_act[:, j] - dataset.q_cmd[:, j]) * 1000.0
        ax_errs.plot(t, err_mrad, label=f"J{j+1}", linewidth=1.0)
    ax_errs.set_title("Tracking Errors: q_act - q_cmd (mrad)", fontsize=10, fontweight="bold")
    ax_errs.set_ylabel("Error (mrad)")
    ax_errs.set_xlabel("Time (s)")
    ax_errs.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: Cartesian Tip Deviation
    ax_cart = fig.add_subplot(gs[0, 1])
    if dataset.cartesian_errors_m is not None:
        err_mm = dataset.cartesian_errors_m * 1000.0
        ax_cart.plot(t, err_mm[:, 0], "r-", label="ΔX", linewidth=1.1)
        ax_cart.plot(t, err_mm[:, 1], "g-", label="ΔY", linewidth=1.1)
        ax_cart.plot(t, err_mm[:, 2], "b-", label="ΔZ", linewidth=1.1)
        norm_mm = dataset.cartesian_norm_errors_mm
        if norm_mm is not None:
            ax_cart.plot(t, norm_mm, "m--", label="3D Norm", linewidth=1.3)
        ax_cart.set_title("Cartesian Tool Tip Deviations (mm)", fontsize=10, fontweight="bold")
        ax_cart.set_ylabel("Deviation (mm)")
        ax_cart.grid(True, linestyle=":", alpha=0.5)
        ax_cart.legend(loc="upper right", fontsize=8)
    else:
        ax_cart.text(0.5, 0.5, "Cartesian FK Unavailable", ha="center", va="center")

    # Panel 4: FFT Spectrum for Joint 1 & Joint 2 (major base joints)
    ax_fft1 = fig.add_subplot(gs[1, 1])
    spec1 = compute_fft(dataset.time_s, (dataset.q_act[:, 0] - dataset.q_cmd[:, 0]) * 1000.0)
    spec2 = compute_fft(dataset.time_s, (dataset.q_act[:, 1] - dataset.q_cmd[:, 1]) * 1000.0)
    ax_fft1.plot(spec1.frequencies_hz, spec1.amplitudes, "tab:blue", label="Joint 1", linewidth=1.2)
    ax_fft1.plot(spec2.frequencies_hz, spec2.amplitudes, "tab:orange", label="Joint 2", linewidth=1.2)
    ax_fft1.set_title("Vibration Spectrum: Base Joints J1 & J2", fontsize=10, fontweight="bold")
    ax_fft1.set_ylabel("Amplitude (mrad)")
    ax_fft1.set_xlabel("Frequency (Hz)")
    ax_fft1.set_xlim(0.0, 50.0)
    ax_fft1.grid(True, linestyle=":", alpha=0.5)
    ax_fft1.legend(loc="upper right", fontsize=8)

    # Panel 5: Summary Metric Table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    
    col_labels = ["Joint / Metric", "RMS (mrad)", "P2P (mrad)", "Max Dev (mrad)", "Dominant Peak (Hz)"]
    table_data = []
    for j in range(n_joints):
        err_mrad = (dataset.q_act[:, j] - dataset.q_cmd[:, j]) * 1000.0
        m = calculate_metrics(err_mrad)
        sp = compute_fft(dataset.time_s, err_mrad)
        top_f = f"{sp.peak_frequencies_hz[0]:.1f} Hz" if sp.peak_frequencies_hz else "None"
        table_data.append([f"Joint {j+1}", f"{m.rms:.2f}", f"{m.p2p:.2f}", f"{m.max_dev:.2f}", top_f])

    if dataset.cartesian_norm_errors_mm is not None:
        m_cart = calculate_metrics(dataset.cartesian_norm_errors_mm)
        table_data.append(["Cartesian 3D (mm)", f"{m_cart.rms:.2f} mm", f"{m_cart.p2p:.2f} mm", f"{m_cart.max_dev:.2f} mm", "N/A"])

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.scale(1.0, 1.4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    fig.suptitle(
        f"reBot Arm B601 Vibration & Tracking Dashboard | Duration: {dataset.duration_s:.1f}s @ {dataset.sampling_rate_hz:.1f}Hz",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary Report Export
# ---------------------------------------------------------------------------

def generate_summary_report(dataset: VibrationDataset) -> Dict[str, Any]:
    """Generate structured summary dictionary of all vibration and tracking metrics."""
    n_joints = dataset.num_joints
    report: Dict[str, Any] = {
        "duration_s": dataset.duration_s,
        "sample_count": dataset.num_samples,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "joints": {},
    }

    is_motion = np.max(np.abs(dataset.q_cmd - dataset.q_cmd[0])) > 1e-4

    for j in range(n_joints):
        q_c = dataset.q_cmd[:, j]
        q_a = dataset.q_act[:, j]
        err_rad = q_a - q_c
        err_mrad = err_rad * 1000.0

        m_rad = calculate_metrics(q_a, q_c)
        fft_res = compute_fft(dataset.time_s, err_mrad if is_motion else (q_a - np.mean(q_a)) * 1000.0)

        report["joints"][f"joint_{j+1}"] = {
            "rms_rad": m_rad.rms,
            "rms_mrad": m_rad.rms * 1000.0,
            "p2p_rad": m_rad.p2p,
            "p2p_mrad": m_rad.p2p * 1000.0,
            "max_dev_rad": m_rad.max_dev,
            "max_dev_mrad": m_rad.max_dev * 1000.0,
            "dominant_frequencies_hz": fft_res.peak_frequencies_hz,
            "dominant_amplitudes_mrad": fft_res.peak_amplitudes,
        }

    if dataset.cartesian_errors_m is not None:
        cart_err_mm = dataset.cartesian_errors_m * 1000.0
        report["cartesian"] = {}
        for idx, axis in enumerate(["x", "y", "z"]):
            m_axis = calculate_metrics(cart_err_mm[:, idx])
            report["cartesian"][axis] = {
                "rms_mm": m_axis.rms,
                "p2p_mm": m_axis.p2p,
                "max_dev_mm": m_axis.max_dev,
            }
        
        norm_mm = dataset.cartesian_norm_errors_mm
        if norm_mm is not None:
            m_norm = calculate_metrics(norm_mm)
            report["cartesian"]["3d_norm"] = {
                "rms_mm": m_norm.rms,
                "p2p_mm": m_norm.p2p,
                "max_dev_mm": m_norm.max_dev,
            }

    return report


def save_summary_report(dataset: VibrationDataset, text_path: Path | str, json_path: Optional[Path | str] = None) -> None:
    """Save text summary and JSON metrics file."""
    metrics = generate_summary_report(dataset)
    t_path = Path(text_path)
    t_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 78,
        "reBot Arm B601 — Vibration & Dynamic Tracking Analysis Report",
        "=" * 78,
        f"Test Duration:   {metrics['duration_s']:.2f} s",
        f"Total Samples:   {metrics['sample_count']}",
        f"Sampling Rate:   {metrics['sampling_rate_hz']:.1f} Hz",
        "-" * 78,
        f"{'Joint':<8} | {'RMS (mrad)':<12} | {'P2P (mrad)':<12} | {'MaxDev (mrad)':<14} | {'Dominant Freqs (Hz)':<20}",
        "-" * 78,
    ]

    for j_name, j_data in metrics["joints"].items():
        freqs_str = ", ".join(f"{f:.1f}" for f in j_data["dominant_frequencies_hz"][:3]) or "None"
        lines.append(
            f"{j_name:<8} | {j_data['rms_mrad']:<12.3f} | {j_data['p2p_mrad']:<12.3f} | "
            f"{j_data['max_dev_mrad']:<14.3f} | {freqs_str:<20}"
        )

    if "cartesian" in metrics:
        lines.extend([
            "-" * 78,
            "Cartesian Tool-Tip Deviations (Forward Kinematics):",
            "-" * 78,
            f"{'Axis':<8} | {'RMS (mm)':<12} | {'P2P (mm)':<12} | {'MaxDev (mm)':<14}",
            "-" * 78,
        ])
        for ax_name, ax_data in metrics["cartesian"].items():
            lines.append(
                f"{ax_name:<8} | {ax_data['rms_mm']:<12.3f} | {ax_data['p2p_mm']:<12.3f} | {ax_data['max_dev_mm']:<14.3f}"
            )

    lines.append("=" * 78)

    t_path.write_text("\n".join(lines), encoding="utf-8")

    if json_path:
        j_path = Path(json_path)
        j_path.parent.mkdir(parents=True, exist_ok=True)
        j_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Deterministic Mathematical Unit Verification (No Random Values)
# ---------------------------------------------------------------------------

def run_deterministic_math_test() -> bool:
    """Verify FFT peak extraction and metric calculation using known analytical multi-tone signals."""
    print("[Self-Test] Running deterministic mathematical verification...")
    
    # 1. Generate known multi-tone signal: 5.0 Hz and 18.0 Hz sampled at 200 Hz for 4.0s
    fs = 200.0
    duration = 4.0
    n = int(fs * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    
    # Pure analytical signal with known amplitudes
    f1, a1 = 5.0, 1.50   # 5 Hz, amplitude 1.50
    f2, a2 = 18.0, 0.75  # 18 Hz, amplitude 0.75
    signal = a1 * np.sin(2.0 * np.pi * f1 * t) + a2 * np.sin(2.0 * np.pi * f2 * t)

    # 2. Test FFT Peak Extraction
    spectrum = compute_fft(t, signal, top_k=2, min_freq_hz=1.0)
    assert len(spectrum.peak_frequencies_hz) >= 2, f"Expected at least 2 peaks, got {spectrum.peak_frequencies_hz}"
    
    peak1 = spectrum.peak_frequencies_hz[0]
    peak2 = spectrum.peak_frequencies_hz[1]
    
    # Check frequency accuracy (within 0.3 Hz resolution)
    assert abs(peak1 - f1) < 0.3 or abs(peak2 - f1) < 0.3, f"5 Hz peak missed: peaks={spectrum.peak_frequencies_hz}"
    assert abs(peak1 - f2) < 0.3 or abs(peak2 - f2) < 0.3, f"18 Hz peak missed: peaks={spectrum.peak_frequencies_hz}"
    print(f"  ✓ FFT Peak Detection Passed: Detected {spectrum.peak_frequencies_hz} (ground truth: [{f1}, {f2}])")

    # 3. Test Statistical Metrics
    # For a sum of two orthogonal sinusoids: RMS = sqrt(a1^2 / 2 + a2^2 / 2)
    expected_rms = np.sqrt((a1**2 + a2**2) / 2.0)
    metrics = calculate_metrics(signal)
    
    assert abs(metrics.rms - expected_rms) < 0.05, f"RMS mismatch: {metrics.rms} vs expected {expected_rms}"
    print(f"  ✓ RMS Calculation Passed: {metrics.rms:.4f} vs expected {expected_rms:.4f}")

    # Check P2P
    assert abs(metrics.p2p - (2.0 * (a1 + a2))) < 0.25, f"P2P mismatch: {metrics.p2p}"
    print(f"  ✓ P2P Calculation Passed: {metrics.p2p:.4f}")

    print("[Self-Test] All deterministic mathematical tests PASSED!\n")
    return True


if __name__ == "__main__":
    run_deterministic_math_test()

