"""Utility script to read and print camera calibration NPZ files."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Print camera calibration parameters from an .npz file")
    parser.add_argument(
        "--file", 
        default="config/calibration/webcam/intrinsics_test_1.npz", #Run 3 is _test_1
        help="Path to the intrinsics.npz file"
    )
    args = parser.parse_args()

    calib_file = Path(args.file).resolve()
    if not calib_file.exists():
        print(f"Error: Calibration file not found at {calib_file}")
        return

    print(f"Reading calibration from: {calib_file}\n")
    
    try:
        data = np.load(str(calib_file), allow_pickle=False)
        
        if "camera_matrix" in data.files:
            print("=== Camera Matrix (K) ===")
            print(data["camera_matrix"])
            print()
            
        if "distortion_coefficients" in data.files:
            print("=== Distortion Coefficients (D) ===")
            print(data["distortion_coefficients"])
            print()
            
        # Print any other scalar/array values stored in the NPZ
        print("=== Additional Metadata ===")
        for key in data.files:
            if key not in ["camera_matrix", "distortion_coefficients"]:
                val = data[key]
                # If it's a 1D array with 1 element, extract the scalar for cleaner printing
                if isinstance(val, np.ndarray) and val.size == 1:
                    print(f"{key}: {val[0]}")
                else:
                    print(f"{key}: {val}")
                    
    except Exception as e:
        print(f"Failed to load file: {e}")


if __name__ == "__main__":
    main()

