"""Standard 2D USB Webcam driver.

Uses cv2.VideoCapture. Returns a color image and None for depth.
Loads intrinsics from config/calibration/webcam/intrinsics.yaml if available.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CameraDriver, CameraFrameError


class WebcamDriver(CameraDriver):
    def __init__(
        self,
        camera_id: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: str = "",
        calib_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.camera_id = int(camera_id) if str(camera_id).isdigit() else camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.calib_dir = Path(calib_dir)
        self.calib_file = Path(calib_file) if calib_file else None

        self._cap: Optional[cv2.VideoCapture] = None
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None

        self._load_calibration()

    def _load_calibration(self) -> None:
        calib_file = self.calib_file if self.calib_file else self.calib_dir / "intrinsics.npz"
        if calib_file.exists():
            try:
                data = np.load(str(calib_file), allow_pickle=False)
                self._K = data["camera_matrix"].astype(np.float64)
                self._D = data["distortion_coefficients"].astype(np.float64)
                print(f"[Webcam] Loaded calibration from {calib_file}")
            except Exception as e:
                print(f"[Webcam] Failed to load calibration: {e}")
        
        if self._K is None:
            print("[Webcam] No calibration found. Using generic intrinsics based on resolution.")
            fx = fy = max(self.width, self.height)
            cx, cy = self.width / 2.0, self.height / 2.0
            self._K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ], dtype=np.float64)
            self._D = np.zeros((1, 5), dtype=np.float64)

    def open(self) -> None:
        if self._cap is not None:
            return
        self._reset_frame_failures()
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise CameraFrameError(f"Failed to open webcam {self.camera_id}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Give the camera a moment to initialize
        for _ in range(5):
            self._cap.read()
            
        print(f"[Webcam] Opened camera {self.camera_id} at {self.width}x{self.height}@{self.fps}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._cap is None:
            return None, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._record_frame_failure("cv2.VideoCapture read failed")
            return None, None

        # [MODIFICATION]: Image Undistortion
        # In a 2D planar grasping setup, the pinhole camera geometry (ray-plane math) 
        # strictly assumes the image is mathematically flat. Standard webcams have curved 
        # lenses (fisheye/barrel distortion) which warps the image, causing YOLO bounding boxes 
        # to drift physically away from where the object really is.
        # We run cv2.undistort on every single frame so that all subsequent math is perfectly accurate.
        if self._K is not None and self._D is not None and np.any(self._D):
            frame = cv2.undistort(frame, self._K, self._D, None, self._K)

        self._reset_frame_failures()
        return frame, None

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def D(self) -> np.ndarray:
        return self._D

