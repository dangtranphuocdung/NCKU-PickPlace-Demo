# 2D Planar Grasping Migration Guide

This document details the exact modifications made to transition the original 3D-depth-dependent YOLO/OBB grasping pipeline to a 2D planar assumption using ray-plane intersection.

## 1. What Was Implemented (New Files)

### `drivers/camera/webcam.py`
A brand new driver was created to support standard 2D USB webcams.
- **Function**: Uses OpenCV (`cv2.VideoCapture`) to stream RGB frames. It explicitly returns `None` for the depth map to signify the absence of hardware depth data.
- **Calibration Support**: Automatically attempts to load pre-calculated camera intrinsics (`K` and `D`) from `config/calibration/webcam/intrinsics.npz`. If absent, it generates a generic camera matrix based on the image resolution.

### `scripts/calibrate_camera.py`
A new utility script for camera calibration.
- **Function**: Allows you to hold a printed checkerboard in front of the 2D webcam to capture frames.
- **Output**: Uses OpenCV's `calibrateCamera` to compute the focal lengths, optical centers, and distortion coefficients, saving them natively to the `.npz` format expected by the rest of the project.

---

## 2. What Was Modified (Existing Files)

### `config/default.yaml`
- **Camera Type**: Changed `camera.type` from `orbbec_gemini2` (3D camera) to `webcam`. Added `camera.id: 0`.
- **New Parameter**: Added `table_z_m: 0.0` under the `grasp` pipeline configuration to establish the mathematical plane for ray intersection.

### `drivers/camera/__init__.py`
- Imported and registered `WebcamDriver` so `make_camera()` correctly instantiates it when the config specifies the `webcam` type.

### `utils/ordinary_grasp.py`
This file contains the core math for determining the 3D position of the object.
- **Ray-Plane Intersection**: Modified `estimate_grasp()` and `estimate_grasps()` to accept the camera-to-base transformation matrix (`T_cam2base`) and `table_z_m`.
- **Fallback Logic**: When `depth_mm` is `None` (as passed by the webcam driver), the code now computes a 3D ray originating from the camera lens and passing through the object's pixel center. It transforms this ray into the robot's base frame, finds where it intersects the `Z = table_z_m` plane, and reverse-projects that intersection point back into the camera frame.
- **Seamless Integration**: The resulting depth from the intersection (`z_m`) perfectly replaces the physical depth sensor value, allowing all subsequent physical size (`jaw_width_m`) and rotation calculations to remain entirely unchanged.

### `scripts/main.py` (Live Robot Execution)
- **Live Kinematics**: Modified the live execution loop to query the robot's current TCP pose and compute `T_cam2base` on every frame.
- **Data Passing**: Passed this live transformation matrix down into `estimate_grasps()` so the ray-plane intersection can dynamically adjust as the robot arm moves the camera.

### `scripts/ordinary_grasp_pipeline.py` (Standalone Testing)
- **Mock Transform**: Since this script runs without the robot arm, we added a mock `T_cam2base` matrix (simulating a camera hovering 0.5 meters directly above the table). 
- **Infinite Loop Fix**: Removed the strict `depth_mm is None` check which previously threw away all 2D frames, freezing the UI. Added `import numpy as np`.
- **UI Responsiveness & Frame Skipping**: Added an `infer_every` logic gate (driven by `infer_every_live` in config) to prevent the heavy YOLO neural network from running on every single frame. This prevents CPU hogging and allows OpenCV to instantly register `G` and `Q` keypresses on a CPU without lagging the video feed.
- **Interactive UI**: Updated the mouse-click callback so clicking anywhere on the 2D image performs the ray-plane intersection and prints the estimated `(X, Y, Z)` real-world coordinates.

### Calibration & Configuration Overrides
- **`config/default.yaml`**: 
  - Lowered `conf_threshold` from `0.50` to `0.25` to stabilize the Open-Vocabulary tracking boxes.
  - Fixed a hidden YAML syntax bug (`- #"comment"`) which injected a `None` type into the PyTorch class list and caused sorting crashes.
  - Added `intrinsics_file: "config/calibration/webcam/intrinsics.npz"` so the camera calibration matrix can be swapped out globally without terminal flags.
- **`drivers/camera/__init__.py` & `webcam.py`**: Added support for reading `intrinsics_file` from the YAML config. Resolved the path dynamically against `PROJECT_ROOT` to ensure the project runs flawlessly regardless of what directory the user executes the python command from.
- **`scripts/verify_calibration.py`**: Built a visualizer with side-by-side (`s`) and grid-overlay (`g`) modes to physically verify the camera distortion model.

---

## 3. Critical Parameters for Real Execution

Before executing `python scripts/main.py` on the physical robot, you **must** configure the following parameters in `config/default.yaml`.

### `camera.id`
- **Current Value**: `0`
- **Explanation**: This defines the USB port or device ID for your webcam (e.g., `/dev/video0`). If you have multiple cameras (like a laptop webcam and an external USB cam), you may need to change this to `1` or `2`.

### `grasp_pipeline.grasp.table_z_m`
- **Current Value**: `0.0`
- **Explanation**: This is the Z-height (in meters, relative to the robot's base frame) of the mathematical plane the camera ray intersects. 
- **⚠️ CRITICAL CHANGE REQUIRED**: Because the original 3D pipeline applied an `insertion_depth_m` (default: `0.015m`) to push the gripper *into* the object from its top surface, you **must set `table_z_m` to the average height of the objects you are grasping, NOT the table height**. 
  - If you leave it at `0.0`, the pipeline will intersect the table surface, apply the insertion depth, and attempt to drive the gripper to `Z = -0.015`, crashing into your table.
  - If your objects are 4cm tall, change this to `0.04`. The robot will then target `Z = 0.04` and insert the gripper to `Z = 0.025`, resulting in a safe and secure grasp.

### `grasp_pipeline.grasp.insertion_depth_m`
- **Current Value**: `0.015`
- **Explanation**: How far the TCP is pushed along the approach axis into the object. You can tune this alongside `table_z_m` to adjust how deep the gripper fingers go before closing.

