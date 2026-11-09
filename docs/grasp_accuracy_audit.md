# Jetson workspace accuracy and connection audit

Reviewed on 2026-11-08 against commit `212eeabbcfd0d167b22ef2d9ec519e8d9ed29f21` (`delete graspnet related files`). No motors were enabled and no robot motion was performed during this audit.

## Changes present before this review

| Change from HEAD | Effect |
| --- | --- |
| Webcam index 5 to 4 | Necessary host-specific selection. Camera 4 returned actual 1280x720 frames. This does not prove it is the same physical camera/lens used on the laptop. |
| Live inference every 1 to every 3 frames | Less frequent preview updates. Fixed-path C/G commands capture a new frame and run fresh inference, so their estimates do not use the cached preview detection. |
| One retreat waypoint mirrored | Y changes from -0.1 to +0.1 m, roll from +0.75 to -0.75 rad, yaw from -0.4 to +0.4 rad. Changes transport/placement after closing the gripper, not the detected pick point. Preserved as a user customization. |
| Driver bridge discovery, mode-result checks and initial feedback validation | Connection/startup behavior changes; detection and grasp geometry remain unchanged. |
| Fixed-path startup error handling | Reports failed initialization and closes connections without attempting a homing trajectory. |

Before this review, detector loading/parsing, grasp estimation, coordinate transforms and camera drivers were byte-identical to HEAD. YOLO configuration, confidence/IoU thresholds, robot/gripper configuration, calibration paths/compensation, table plane and grasp offsets also matched HEAD.

The SDK is an ignored, separate Git repository. Its only tracked local change was build/package metadata in `pyproject.toml`; its motor-control implementation had no tracked changes. Installed dependencies can still differ from the laptop.

Correction to the earlier audit text: neither HEAD nor the initial workspace contained a 480-pixel or FP16 inference override. Claims that those were current uncommitted changes were unsupported. The active checkpoint reports segmentation, training image size 640, and classes `falcon tube` and `petri dish`.

## Verification and fixes made

### Detection and grasp parity

- Nine synthetic segmentation cases, covering three mask resolutions (480x288, 640x384, 1280x720) and three object locations, produced exactly equal values in every grasp-result field when comparing HEAD and workspace estimators. This verifies parity, not ground-truth geometric correctness.
- The actual Jetson camera returned a 1280x720 frame with the configured intrinsics loaded.
- Unrestricted, default GPU inference initially failed during CUDA allocation with NvMap allocation errors and a PyTorch allocator assertion.
- With `CUDNN_CONV_WSCAP_DBG=128` and `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64`, HEAD and workspace GPU inference both completed at `imgsz=640`, `half=False`. Each detected one object in the same captured frame; boxes, confidences, class IDs and segmentation masks were exactly equal.
- Repeated that comparison using the implemented Jetson helper, without launch-time environment overrides: passed again. Native NvMap allocation diagnostics still appeared, but inference completed; future memory availability is not guaranteed.

Added Jetson-only memory defaults in `utils/yolo_utils.py`, respecting explicit environment overrides. The shared loader and the standalone detection/ordinary-grasp previews apply them before model use. No input downsampling, FP16 conversion, threshold adjustment, mask correction or calibration adjustment was introduced. Added a real-frame inference check before robot initialization in `fixed_path_grasp.py`, so an initial detector allocation failure occurs before enabling motors or moving to ready.

### Motor connection and ACK handling

Two independent read-only hardware probes verified all seven motors on `/dev/ttyACM1`, at 921600 baud. The configured `/dev/ttyACM0` did not pass verification. Arm mode registers read 2 and the gripper mode register read 1. Successful reopening also verified that the probe released the adapter.

Preserved automatic bridge discovery and native probe-handle cleanup. Added up to three attempts for SDK mode setup on the verified bridge, before either group is enabled. Persistent mode failure aborts startup. Missing/nonfinite initial feedback prevents target priming and the control loop from starting. Failed startup disconnects and explicitly closes native handles, including partially established connections, without calling the controller's homing routine.

`python -m unittest discover -s tests -v` passed all 13 tests in the `rebotarm` environment. Coverage includes incorrect-port fallback, partial motor replies, probe handle release, transient/persistent ACK failures, gripper mode failure, invalid/missing feedback, partial-connect cleanup, cleanup errors and unchanged MIT gains. Python compilation and `git diff --check` also passed.

Register probes do not exercise motor enabling, mode writes or physical grasp execution. Timeout handling is tested with mocks; hardware power, wiring faults or competing adapter users can still cause ACK timeouts. Those conditions must fail startup rather than be ignored.

## Limits and inherited behavior

This review verifies software/output parity for tested inputs, not measured detection mAP or physical grasp success rate. No labeled laptop/Jetson comparison set or measured grasp targets were available. Existing mask scaling, planar-height assumptions, pose/frame timing and gripper behavior were deliberately retained to preserve the previously tested behavior. For example, the webcam supplies no depth and projects detections onto the configured plane; `--dry-run` in the existing fixed-path workflow still initializes the robot and moves to ready, so it was not used for this audit.

`best.pt`, the selected webcam intrinsics and hand-eye file are ignored by Git. Their equality to the laptop copies cannot be established from HEAD. Current SHA-256 values for comparison:

| File | SHA-256 |
| --- | --- |
| models/best.pt | f4ee293469426e9daa807ccdc3ca4d276f2277de41c1738d7fdf2060737bc275 |
| config/calibration/webcam/intrinsics_test_Oct21.npz | 3576d995a105ee9dcc22a9ca51da485a4ac17bbd8b0d368835845fd91510e713 |
| config/calibration/webcam/hand_eye.npz | baf96cc55aa563df2d9dbf33ccbcaee8fdd5e75e0560fa03a3514d5392b70448 |

Verified environment: Python 3.10 (`/home/jetson/miniforge3/envs/rebotarm/bin/python`), torch 2.8.0, ultralytics 8.4.35, motorbridge 0.5.6, numpy 1.26.4 and opencv-python 4.7.0.72. The shell's default Python 3.13 lacks numpy; run this project in the `rebotarm` environment.

Read-only connection recheck, without enabling motors:

```bash
conda run -n rebotarm python scripts/check_robot_connection.py
```
