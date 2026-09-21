from __future__ import annotations

from pathlib import Path

from .base import CameraDriver, CameraFrameError
from .orbbec_gemini2 import OrbbecGemini2
from .realsense import RealsenseCamera
from .webcam import WebcamDriver

__all__ = ["CameraDriver", "CameraFrameError", "OrbbecGemini2", "RealsenseCamera", "WebcamDriver", "make_camera"]


def make_camera(cfg: dict) -> CameraDriver:
    """Create a camera driver from config/default.yaml."""
    cam_cfg  = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "").lower()
    w   = cam_cfg.get("color_width",  1280)
    h   = cam_cfg.get("color_height", 720)
    fps = cam_cfg.get("fps", 30)

    _root     = Path(__file__).resolve().parent.parent.parent
    calib_dir = str(_root / "config" / "calibration" / cam_type)
    
    calib_file = cam_cfg.get("intrinsics_file", None)
    if calib_file is not None:
        calib_file = str(_root / calib_file)

    if "orbbec" in cam_type:
        return OrbbecGemini2(w, h, fps, calib_dir=calib_dir)
    elif "realsense" in cam_type:
        return RealsenseCamera(w, h, fps, calib_dir=calib_dir)
    elif "webcam" in cam_type:
        camera_id = cam_cfg.get("id", 0)
        return WebcamDriver(camera_id, w, h, fps, calib_dir=calib_dir, calib_file=calib_file)
    else:
        raise ValueError(
            f"Unsupported camera type: {cam_type!r}\n"
            f"Set camera.type in config/default.yaml to:\n"
            f"  orbbec_gemini2 | realsense_d435i | realsense_d405 | webcam"
        )
