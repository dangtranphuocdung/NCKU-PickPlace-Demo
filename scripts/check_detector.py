"""Check model/device inference without a GUI or robot connection."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from drivers.camera import make_camera
from utils.camera_utils import load_config
from utils.yolo_utils import DetectorMemoryError, load_yolo, warmup_yolo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--device', default=None, help='Override configured detector device, e.g. 0 or cpu')
    parser.add_argument('--no-camera', action='store_true', help='Use a blank frame to isolate GPU/model startup')
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    cam = None
    try:
        if args.no_camera:
            camera_cfg = cfg.get('camera', {})
            frame = np.zeros((camera_cfg.get('color_height',720),camera_cfg.get('color_width',1280),3),np.uint8)
        else:
            cam = make_camera(cfg)
            cam.open()
            cam.warm_up(15)
            frame, _ = cam.get_frame()
            if frame is None:
                raise RuntimeError('No camera frame available for detector check')
        model, opts = load_yolo(cfg, project_root=ROOT, device_override=args.device)
        warmup_yolo(model, frame, opts)
        inference = model.predictor.args
        print(f'PASS: detector on {opts["device"]}, frame={frame.shape}, '
              f'imgsz={inference.imgsz}, half={inference.half}, '
              f'conf={opts["conf"]}, iou={opts["iou"]}. No robot connection was made.')
        return 0
    except DetectorMemoryError as exc:
        print(f'[Detector] {exc}')
        return 1
    finally:
        if cam is not None:
            cam.close()


if __name__ == '__main__':
    raise SystemExit(main())
