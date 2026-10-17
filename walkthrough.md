# Walkthrough: reBot Arm B601 Vibration & Dynamic Tracking Analysis Tool

We have implemented and verified a Python-based vibration and dynamic tracking analysis tool for the reBot Arm B601 robot manipulator.

---

## 1. Changes Made

### A. Core Mathematical & Signal Processing Engine
**[`utils/vibration_utils.py`](file:///home/dung-dang/rebot_grasp/utils/vibration_utils.py)**
- **Data Model**: Implemented `VibrationDataset` holding timestamps, commanded joint angles $q_{\text{cmd}}$, actual encoder feedback $q_{\text{act}}$, velocities $v_{\text{act}}$, torques $\tau_{\text{act}}$, and Cartesian trajectories.
- **Forward Kinematics (FK)**: Leveraged Pinocchio (`compute_fk`) to compute real-time Cartesian $(X, Y, Z)$ tool-tip coordinates and 3D Euclidean error $\Delta R = \sqrt{\Delta X^2 + \Delta Y^2 + \Delta Z^2}$.
- **Statistical Variations**: Evaluates RMS, Peak-to-Peak (P2P), Maximum Deviation, Mean, and Standard Deviation for all joints and Cartesian axes.
- **FFT Spectral Analysis**: Implemented windowed Fast Fourier Transform using a Hann window to suppress spectral leakage and automatic peak detection for dominant vibration frequencies (Hz) and amplitudes.
- **Deterministic Math Self-Test**: Added `run_deterministic_math_test()` using known analytical multi-tone signals ($f_1 = 5.0\text{ Hz}, f_2 = 18.0\text{ Hz}$) to mathematically verify FFT and metric extraction.
- **Visualization Suite**: Generates 4 publication-quality figures:
  1. `joint_tracking_errors.png`: Commanded vs. actual angles and error over time for all 6 joints.
  2. `cartesian_xyz_tracking.png`: End-effector tip displacement $(\Delta X, \Delta Y, \Delta Z)$ and 3D Euclidean norm in mm.
  3. `joint_vibration_fft.png`: Frequency spectra for all joints with marked resonance peaks.
  4. `vibration_dashboard.png`: Consolidated executive dashboard.
- **Persistence**: Exports full data to CSV (`vibration_data.csv`), structured JSON (`vibration_metrics.json`), and formatted text summary (`vibration_summary.txt`).

### B. Robot Integration & CLI Runner
**[`scripts/vibration_analysis.py`](file:///home/dung-dang/rebot_grasp/scripts/vibration_analysis.py)**
- **Real Motor Feedback Integration**: Hooks into `RebotArmEndPose`'s periodic control loop callback (`_loop_cb`) to log encoder feedback synchronously at the control rate (100–500 Hz).
- **Strict Safety Policy (NO Random Joint Values)**:
  - `stationary`: Holds at the robot's **current actual pose** (read from encoders at startup) or verified `ready_pose`.
  - `motion`: Moves along a smooth, deterministic minimum-jerk trajectory between verified safe poses.
  - `both`: Moves to target pose $\to$ holds stationary $\to$ returns home.
- **Offline Re-Analysis**: `--analyze-only <csv_path>` allows loading previously saved CSV datasets and regenerating plots and metrics without robot hardware.
- **Dry-Run Mode**: `--dry-run` verifies the data logging and plotting pipeline offline.
- **Terminal Summary**: Displays an immediate ASCII summary table in the terminal upon completion.

---

## 2. Validation & Test Results

### A. Mathematical Unit Self-Test
Executed `python -c "from utils.vibration_utils import run_deterministic_math_test; run_deterministic_math_test()"`:
- **FFT Peak Accuracy**: Correctly identified $[5.0\text{ Hz}, 18.0\text{ Hz}]$ peaks matching ground truth $[5.0\text{ Hz}, 18.0\text{ Hz}]$.
- **RMS Accuracy**: Calculated $1.1859$ matching theoretical $1.1859$.
- **Status**: PASSED.

### B. Dry-Run Pipeline Verification
Executed `python scripts/vibration_analysis.py --dry-run --mode both --duration 4.0`:
- Generated 800 synchronized samples at 200 Hz.
- Successfully produced all output artifacts in `data/vibration_analysis/test_dry_run/`.
- Offline re-analysis via `--analyze-only` successfully loaded the CSV and reproduced all metrics and plots.

---

## 3. Visualizations Generated

### Consolidated Overview Dashboard
![Vibration Dashboard](/home/dung-dang/.gemini/antigravity/brain/10d27355-8fd9-4286-9f80-e44a2b28d8ab/vibration_dashboard.png)

### Joint Tracking & Error Curves
![Joint Tracking Errors](/home/dung-dang/.gemini/antigravity/brain/10d27355-8fd9-4286-9f80-e44a2b28d8ab/joint_tracking_errors.png)

### Cartesian Tool-Tip Deviations (mm)
![Cartesian Tracking](/home/dung-dang/.gemini/antigravity/brain/10d27355-8fd9-4286-9f80-e44a2b28d8ab/cartesian_xyz_tracking.png)

### Joint Vibration Frequency Spectra (FFT)
![Joint Vibration FFT](/home/dung-dang/.gemini/antigravity/brain/10d27355-8fd9-4286-9f80-e44a2b28d8ab/joint_vibration_fft.png)

---

## 4. How to Use on the Robot

Make sure your conda environment is active:
```bash
conda activate rebotarm
```

### 1. Safest Baseline: Stationary Holding at Current Physical Pose
Records 5.0 seconds of encoder feedback while the robot holds wherever it currently is:
```bash
python scripts/vibration_analysis.py --mode stationary --duration 5.0
```

### 2. Stationary Holding at Ready Pose
Moves the robot to the configured Ready Pose from `config/default.yaml` and records for 5.0 seconds:
```bash
python scripts/vibration_analysis.py --mode stationary --target-pose ready --duration 5.0
```

### 3. Dynamic Motion Tracking Test
Records tracking errors and dynamic vibrations while moving smoothly to Ready Pose:
```bash
python scripts/vibration_analysis.py --mode motion --duration 3.0
```

### 4. Combined Test (Motion + Hold + Return)
```bash
python scripts/vibration_analysis.py --mode both --duration 5.0
```

### 5. Offline Re-Analysis on Previously Recorded Data
Re-generates plots and computes statistics on an existing CSV file without connecting to the robot:
```bash
python scripts/vibration_analysis.py --analyze-only data/vibration_analysis/<timestamp>/vibration_data.csv
```
