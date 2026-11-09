"""Utility script to visually verify camera calibration.

Loads the calibration data (.npz) and applies undistortion to a live
camera feed. It draws a grid over the undistorted image so you can 
verify that straight lines in the real world (like doors or tables) 
appear perfectly straight in the camera view.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.webcam_capture import add_capture_arguments, capture_options, open_webcam
from utils.camera_utils import load_config


def draw_grid(img: np.ndarray, grid_size: int = 80) -> None:
    """Draws a green grid over the image to help verify rectilinear projection."""
    h, w = img.shape[:2]
    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Visually verify camera calibration")
    parser.add_argument("--file", default=None,
                        help="Path to the intrinsics.npz file")
    add_capture_arguments(parser)
    args = parser.parse_args()
    opts = capture_options(args)

    cfg = load_config(args.config)
    calib_file = Path(args.file) if args.file else PROJECT_ROOT / cfg['camera']['intrinsics_file']
    calib_file = calib_file.resolve()
    if not calib_file.exists():
        print(f"Error: Calibration file not found at {calib_file}")
        return

    # Load calibration parameters
    try:
        data = np.load(str(calib_file), allow_pickle=False)
        K = data["camera_matrix"].astype(np.float64)
        D = data["distortion_coefficients"].astype(np.float64)
        print(f"Loaded calibration from: {calib_file}")
    except Exception as e:
        print(f"Failed to load calibration file: {e}")
        return

    # Open the camera
    cap = open_webcam(**opts)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return

    print("Opening live feed in FULL SCREEN...")
    print("Point the camera at straight edges (doors, tables, floor tiles).")
    print("Look at the edges of the screen to ensure the straight lines do not curve.")
    print("Press 'g' to toggle the grid on/off.")
    print("Press 's' to toggle side-by-side comparison mode.")
    print("Press 'ESC' to exit.")

    window_name = "Calibration Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    show_grid = True
    side_by_side = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Apply the undistortion using K and D
            undistorted = cv2.undistort(frame, K, D, None, K)

            if show_grid:
                draw_grid(undistorted)
                draw_grid(frame)

            if side_by_side:
                # Stack the original and undistorted side-by-side
                half_orig = cv2.resize(frame, (opts['width'] // 2, opts['height'] // 2))
                half_undist = cv2.resize(undistorted, (opts['width'] // 2, opts['height'] // 2))
                cv2.putText(half_orig, "Original (Distorted)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(half_undist, "Calibrated (Undistorted)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                display_img = np.hstack((half_orig, half_undist))
            else:
                # Show the full-resolution calibrated image
                display_img = undistorted
                cv2.putText(display_img, "Calibrated (Full Screen)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('g'):
                show_grid = not show_grid
            elif key == ord('s'):
                side_by_side = not side_by_side

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

