"""Shared USB capture settings; no calibration or robot control."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cv2
import yaml


def add_capture_arguments(parser) -> None:
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'config/default.yaml'))
    parser.add_argument('--camera-id', help='Video index or stable /dev/v4l/by-path device')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--fps', type=int)
    parser.add_argument('--focus', type=int, help='Absolute lens control value, not a distance')


def capture_options(args) -> dict:
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['camera']
    opts = {
        'camera_id': cfg.get('id', 0), 'width': cfg.get('color_width', 1280),
        'height': cfg.get('color_height', 720), 'fps': cfg.get('fps', 30),
        'fourcc': cfg.get('fourcc', 'MJPG'), 'buffer_size': cfg.get('buffer_size', 2),
        'focus': cfg.get('focus'), 'autofocus': cfg.get('autofocus'),
    }
    for name in ('camera_id', 'width', 'height', 'fps', 'focus'):
        value = getattr(args, name, None)
        if value is not None:
            opts[name] = value
    if opts['focus'] is not None:
        if cfg.get('autofocus') is True:
            raise ValueError('Set camera.autofocus: false when using fixed focus')
        opts['autofocus'] = False
    return opts


def capture_metadata(cap, camera_id) -> dict:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {
        'camera_id': str(camera_id),
        'image_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'image_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'fourcc': ''.join(chr((code >> (8 * i)) & 255) for i in range(4)),
        'focus_absolute': cap.get(cv2.CAP_PROP_FOCUS),
        'autofocus': cap.get(cv2.CAP_PROP_AUTOFOCUS),
    }


def set_focus(cap, focus=None, autofocus=None) -> None:
    if focus is not None and autofocus:
        raise ValueError('Fixed focus requires autofocus=False')
    if focus is not None:
        autofocus = False
    if autofocus is not None:
        if not cap.set(cv2.CAP_PROP_AUTOFOCUS, int(autofocus)):
            raise RuntimeError('Camera rejected autofocus control; inspect v4l2-ctl --list-ctrls')
        if bool(round(cap.get(cv2.CAP_PROP_AUTOFOCUS))) != bool(autofocus):
            raise RuntimeError('Camera autofocus readback does not match requested mode')
    if focus is not None:
        if not cap.set(cv2.CAP_PROP_FOCUS, float(focus)):
            raise RuntimeError(f'Camera rejected focus={focus}')
        actual = cap.get(cv2.CAP_PROP_FOCUS)
        if abs(actual - float(focus)) > .5:
            raise RuntimeError(f'Camera focus readback {actual} differs from requested {focus}')


def open_webcam(camera_id, width=1280, height=720, fps=30, *, fourcc='MJPG',
                buffer_size=2, focus=None, autofocus=None):
    device = int(camera_id) if str(camera_id).isdigit() else camera_id
    backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
    if sys.platform.startswith('linux') and isinstance(device, int):
        device = f'/dev/video{device}'
    cap = cv2.VideoCapture(device, backend)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open webcam {camera_id}')
        if len(fourcc) != 4:
            raise ValueError('Camera FOURCC must have four characters')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        set_focus(cap, focus, autofocus)
        meta = capture_metadata(cap, camera_id)
        if (meta['image_width'], meta['image_height']) != (width, height):
            raise RuntimeError(f'Camera negotiated {meta}; requested {width}x{height}')
        print(f'[Webcam] Negotiated {meta}')
        if meta['fourcc'] != fourcc or abs(meta['fps'] - fps) > .5:
            print('[Webcam] WARNING: requested format/FPS was not negotiated')
        return cap
    except BaseException:
        cap.release()
        raise


def focus_range(camera_id) -> tuple[int, int, int]:
    device = f'/dev/video{camera_id}' if str(camera_id).isdigit() else str(camera_id)
    result = subprocess.run(['v4l2-ctl', '-d', device, '--list-ctrls'],
                            capture_output=True, text=True, check=True, timeout=5)
    match = re.search(r'focus_absolute[^\n]*min=(-?\d+) max=(-?\d+) step=(\d+)', result.stdout)
    if not match:
        raise RuntimeError('Camera does not expose an absolute focus range')
    return tuple(map(int, match.groups()))
