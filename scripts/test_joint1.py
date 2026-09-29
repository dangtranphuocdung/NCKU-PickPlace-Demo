import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.robot.grasp_driver import find_rebot_repo_root, selected_arm_config, _read_yaml
from reBotArm_control_py.actuator import RebotArm

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_joint1.py <radian>")
        print("Example: python scripts/test_joint1.py 1.0")
        sys.exit(1)
        
    target_rad = float(sys.argv[1])
    print(f"Target Joint 1 Radian: {target_rad}")

    if target_rad < -2.8 or target_rad > 2.8:
        return "Warning: Target radian is outside the typical URDF limits for joint 1 (-2.8 to +2.8 radians)"
    
    # Check URDF limits (informational)
    print("Note: URDF limit for joint1 is usually -2.8 to +2.8 radians")

    arm = RebotArm()
    arm.connect()
    print("Connected to robot.")

    # Enable and read initial state
    arm.arm.mode_pos_vel()
    arm.enable_all()
    
    q_initial = arm.get_state(request_feedback=True)[0][:arm.arm.num_joints]
    print(f"Initial physical angles: {q_initial}")

    # Prepare target: keep all other joints at their current position, only move joint 1
    q_target = q_initial.copy()
    q_target[0] = target_rad

    # Create a simple control loop to command the position over a few seconds
    print("Moving joint 1...")
    arm.start_control_loop(lambda dt: None, rate=arm.rate)
    
    # We can use arm.set_positions since the control loop is running
    # but we just want to safely command joint 1
    steps = 50
    for i in range(steps):
        alpha = (i + 1) / steps
        q_now = q_initial * (1 - alpha) + q_target * alpha
        arm.arm.send_pos_vel(q_now)
        time.sleep(0.05)

    print("Waiting 1 second for joint to settle...")
    time.sleep(1.0)
    
    # Stop control loop so we can manually query feedback
    arm.stop_control_loop()
    
    # Query actual physical encoder
    q_actual = arm.get_state(request_feedback=True)[0][:arm.arm.num_joints]
    actual_rad = q_actual[0]
    
    print("\n=== RESULTS ===")
    print(f"Commanded Target : {target_rad:.4f} rad")
    print(f"Physical Encoder : {actual_rad:.4f} rad")
    print(f"Error (Diff)     : {abs(target_rad - actual_rad):.4f} rad")
    
    arm.disconnect()

if __name__ == "__main__":
    main()

