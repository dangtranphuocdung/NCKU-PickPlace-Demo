# Calibration with fixed focus

The calibration collection workflow uses the original 50 preset poses, manual
mode, marker detection, TSAI solver, display, and output format. Local randomized
sampling and optional motor telemetry were removed. Existing serial adapter
verification and ACK checks remain to preserve working motor connections.

The camera settings in `config/default.yaml` are shared with hand-eye collection:

```yaml
camera:
  id: 2
  autofocus: false
  focus: 0
```

Intrinsic calibration retains its original capture format and full-resolution
corner search, with fixed-focus setup and readback verification added:

```bash
python scripts/calibrate_camera.py
```

Its current output filename is `intrinsics_Nov08_Focus_0.npz` under
`config/calibration/webcam/`. This matches the selected YAML intrinsics path.

Original preset hand-eye collection:

```bash
python scripts/collect_handeye_eih.py
```

Original manual collection:

```bash
python scripts/collect_handeye_eih.py --manual
```

These commands operate the robot. The original hand-eye output is
`config/calibration/webcam/hand_eye_test_manually.npz`. The restored CLI accepts
`--manual`; the added `--output`, `--overwrite`, and `--motor-log` flags were removed.
The original save behavior overwrites that output when calibration succeeds.
Select the desired result through `calibration.hand_eye_file` before grasping.
No existing calibration NPZ files were changed by the rollback.

The configured marker remains AprilTag 36h11, ID 0, with a 100 mm black-square
side length excluding the surrounding white margin. Use varied sharp views for
calibration and keep the same focus for grasping; fixed focus does not guarantee
sharpness at every distance.
