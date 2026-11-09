"""Compare USB lens focus settings without loading a model or connecting a robot."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from utils.webcam_capture import (add_capture_arguments, capture_metadata, capture_options,
                                  focus_range, open_webcam, set_focus)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_arguments(parser)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/focus_checks')
    parser.add_argument('--benchmark', type=int, default=0, metavar='FRAMES',
                        help='Measure capture FPS without opening a window')
    args = parser.parse_args()
    opts = capture_options(args)
    cap = open_webcam(**opts)
    window = 'Focus test: S=save/reference P=print setting Q=quit'
    try:
        if args.benchmark:
            for _ in range(15):
                cap.read()
            start = time.perf_counter()
            for _ in range(args.benchmark):
                ok, _ = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
            print(f'Measured capture: {args.benchmark / (time.perf_counter() - start):.2f} FPS')
            return 0
        low, high, step = focus_range(opts['camera_id'])
        current = int(round(cap.get(cv2.CAP_PROP_FOCUS)))
        current = low + round((current - low) / step) * step
        set_focus(cap, current)
        print(f'Focus range {low}..{high}, step {step}. Keep camera/target stationary for comparison.')
        print('Center ROI sharpness is a relative score; compare at fixed lighting and framing.')
        print('S saves a full-resolution frame and makes its center crop the reference. P prints YAML.')
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Focus', window, (current-low)//step, (high-low)//step, lambda _: None)
        reference = None
        reference_label = ''
        changed_at = time.monotonic()
        start = time.perf_counter()
        count, fps = 0, 0.
        while True:
            requested = low + step * cv2.getTrackbarPos('Focus', window)
            if requested != current:
                set_focus(cap, requested)
                current = requested
                changed_at = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            count += 1
            elapsed = time.perf_counter() - start
            if elapsed >= 1.:
                fps = count / elapsed
                count, start = 0, time.perf_counter()
            h, w = frame.shape[:2]
            roi = frame[max(0,h//2-160):h//2+160,max(0,w//2-160):w//2+160]
            score = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            settling = time.monotonic() - changed_at < .7
            display = frame.copy()
            cv2.rectangle(display, (w//2-160,h//2-160), (w//2+160,h//2+160), (0,255,0), 1)
            label = f'focus={current} sharpness={score:.1f} preview={fps:.1f} FPS'
            cv2.putText(display, label + (' SETTLING' if settling else ''), (10,28),
                        cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
            if reference is not None:
                rh, rw = reference.shape[:2]
                display[h-rh:h, :rw] = reference
                cv2.putText(display, reference_label, (8,h-rh-8), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('p'):
                print(f'camera:  # update these keys in {args.config}\n  id: {opts["camera_id"]}\n  autofocus: false\n  focus: {current}')
            if key == ord('s'):
                if settling:
                    print('Wait for lens settling before saving.')
                    continue
                args.output_dir.mkdir(parents=True, exist_ok=True)
                path = args.output_dir / f'{time.time_ns()}_focus_{current}'
                if not cv2.imwrite(str(path.with_suffix('.png')), frame):
                    raise RuntimeError('Could not save comparison image')
                meta = capture_metadata(cap, opts['camera_id'])
                meta['center_roi_sharpness'] = float(score)
                path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
                reference, reference_label = roi.copy(), f'Reference focus={current}, score={score:.1f}'
                print(f'Saved {path}.png; {label}')
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
