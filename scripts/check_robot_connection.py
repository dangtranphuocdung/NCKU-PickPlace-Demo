"""Check the configured robot serial connection without enabling or moving motors."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drivers.robot.grasp_driver import ensure_rebot_sdk_in_syspath, resolve_dm_serial_channel


def main():
    ensure_rebot_sdk_in_syspath()
    from reBotArm_control_py.actuator import RebotArm
    arm = RebotArm()
    try:
        resolve_dm_serial_channel(arm)
    except RuntimeError as exc:
        print(f"[Robot] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
