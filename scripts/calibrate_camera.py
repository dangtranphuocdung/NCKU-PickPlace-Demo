"""Camera intrinsic calibration script.

Captures images of a checkerboard and calculates the camera matrix (K)
and distortion coefficients (D), then saves them to NPZ.

Usage:
  python scripts/calibrate_camera.py
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", default=0, help="Camera device ID or path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners per a column")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners per a row")
    parser.add_argument("--square-size", type=float, default=0.023, help="Size of a square in meters")
    parser.add_argument("--calib-dir", default=str(PROJECT_ROOT / "config" / "calibration" / "webcam"))
    args = parser.parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(int(args.camera_id) if str(args.camera_id).isdigit() else args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return

    checkerboard_size = (args.cols, args.rows)
    
    # termination criteria for subpixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    window_name = "Calibration - Press SPACE to capture, ESC to finish"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"Looking for a {args.cols}x{args.rows} checkerboard.")
    print("Hold it in view and press SPACE when corners are detected to save a frame.")
    print("Press ESC when you have collected enough frames (15+ recommended) to compute calibration.")
    # initialize sample counter
    samples_collected = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            display = frame.copy()
            if ret_cb:
                # Draw and display the corners
                cv2.drawChessboardCorners(display, checkerboard_size, corners, ret_cb)
                cv2.putText(display, "Corners Found! Press SPACE to capture", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No corners detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(display, f"Samples: {samples_collected}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                break
            elif key == 32 and ret_cb: # SPACE
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)
                samples_collected += 1
                print(f"Captured sample {samples_collected}")
                
                # Flash screen to indicate capture
                cv2.imshow(window_name, np.zeros_like(display))
                cv2.waitKey(100)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if samples_collected < 3:
        print("Not enough samples collected for calibration (minimum 3 required).")
        return
    
    # Main calibration step
    print("Calculating calibration parameters...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("Calibration failed!")
        return

    print("\n=== Calibration Results ===")
    print("Camera matrix (K):")
    print(mtx)
    print("Distortion coefficients (D):")
    print(dist)
    print(f"RMS Reprojection Error: {ret:.4f}")

    # Save to npz
    calib_dir = Path(args.calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    out_file = calib_dir / "intrinsics_test_1.npz"

    np.savez(
        str(out_file),
        camera_matrix=mtx,
        distortion_coefficients=dist,
        rms_error=np.array([ret]),
        image_width=np.array([args.width]),
        image_height=np.array([args.height])
    )
    
    print(f"\nSaved calibration to {out_file}")

if __name__ == "__main__":
    main()

