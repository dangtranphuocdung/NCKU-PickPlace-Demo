import os
import sys
import time
import threading
import queue
import argparse
import numpy as np
from pathlib import Path

# reBotArm SDK imports
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
import pinocchio as pin
from reBotArm_control_py.kinematics import load_robot_model, get_end_effector_frame_id
from reBotArm_control_py.kinematics.robot_model import pad_q_for_model
from reBotArm_control_py.dynamics import compute_generalized_gravity

# Project-specific imports
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))
from utils.camera_utils import load_config
from utils.transforms import rotation_matrix_to_euler_zyx
from drivers.robot.grasp_driver import GraspDriver

# ==========================================
# Input Thread helper
# ==========================================
def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

# ==========================================
# Gravity compensation controller for manual mode
# ==========================================
class GravityCompController:
    """MIT mode with end-effector velocity locking for hand-guided poses."""
    KP = 8.0
    KD = 1.5
    V_THRESH  = 0.04   # End-effector linear velocity threshold (m/s)
    W_THRESH  = 0.08   # End-effector angular velocity threshold (rad/s)

    def __init__(self, arm: RebotArm) -> None:
        self._arm = arm
        self._model = load_robot_model()
        self._data  = self._model.createData()
        self._ee_id = get_end_effector_frame_id(self._model)

        self._n = None
        self._q_target = None
        self._integral = None
        self._gc_running = threading.Event()
        self._io_lock = threading.RLock()

    def start(self) -> None:
        """Connect, enable, set MIT mode, and start gravity compensation."""
        if not self._arm._connected:
            self._arm.connect()
        self._arm.arm.mode_mit(
            kp=np.full(self._arm.arm.num_joints, self.KP),
            kd=np.full(self._arm.arm.num_joints, self.KD),
        )
        if self._arm.has_gripper:
            self._arm.gripper.mode_mit()
        self._arm.enable_all()

        n = self._arm.arm.num_joints
        self._n = n
        self._wait_state_valid()
        with self._io_lock:
            q0 = self._arm.get_state()[0][:n]
        self._q_target = q0.copy()
        self._integral = np.zeros(n)

        self._gc_running.set()
        self._arm.start_control_loop(self._worker, rate=self._arm.rate)
        
    def stop(self) -> None:
        """Stop the gravity compensation control loop."""
        self._gc_running.clear()
        self._arm.stop_control_loop()
        print("[GravityComp] Stopped.")

    def _wait_state_valid(self, timeout: float = 2.0) -> None:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._io_lock:
                self._arm.get_state()
                ok = all(
                    m.get_state() is not None
                    for m in self._arm._motor_map.values()
                )
            if ok:
                return
            time.sleep(0.02)
        raise RuntimeError("[GravityComp] Arm feedback is not ready")

    def _worker(self, r, dt: float) -> None:
        if not self._gc_running.is_set():
            return
        model, data, ee_id = self._model, self._data, self._ee_id
        n, KP, KD = self._n, self.KP, self.KD
        try:
            with self._io_lock:
                q_all, qd_all, _ = r.get_state()
            q = q_all[:n]
            qd = qd_all[:n]
            q_model = pad_q_for_model(model, q, n)
            tau_g = compute_generalized_gravity(model=model, q=q_model)[:n]

            q_err = self._q_target - q
            self._integral += q_err
            np.clip(self._integral, -0.5, 0.5, out=self._integral)

            qd_model = np.zeros(model.nv)
            qd_model[: min(model.nv, n)] = qd[: min(model.nv, n)]
            pin.computeJointJacobians(model, data, q_model)
            pin.updateFramePlacements(model, data)
            J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd_model

            if (np.linalg.norm(v[:3]) > self.V_THRESH or
                    np.linalg.norm(v[3:]) > self.W_THRESH):
                self._q_target = q.copy()
                self._integral *= 0.9

            with self._io_lock:
                r.arm.send_mit(
                    pos=self._q_target,
                    vel=np.zeros(n),
                    kp=np.full(n, KP),
                    kd=np.full(n, KD),
                    tau=tau_g + self._integral,
                )
                if r.has_gripper:
                    r.gripper.send_mit(r.gripper.get_positions())
        except Exception:
            pass

# ==========================================
# Main script logic
# ==========================================
def main():
    cfg = load_config(root / "config" / "default.yaml")
    robot_cfg = cfg.get("robot", {})
    
    print("\n=== Waypoint Teach & Playback ===")
    print("Connecting to robot...")
    rebotarm = RebotArm()
    
    # Initialize components but don't start the standard controller yet
    controller = RebotArmEndPose(rebotarm, arm_control_mode="mit", use_gravity_ff=True)
    grasp_driver = GraspDriver(
        rebotarm,
        controller,
        gripper_config=robot_cfg.get("gripper"),
        repo_root=robot_cfg.get("repo_root"),
    )
    gc_ctrl = GravityCompController(rebotarm)
    
    waypoints = []
    
    line_queue = queue.Queue()
    make_input_thread(line_queue)

    def print_menu():
        print("\n--- Menu ---")
        print("[Enter] Save current TCP pose as waypoint")
        print("[p]     Playback saved waypoints")
        print("[c]     Clear saved waypoints")
        print("[q]     Quit")
        print("-------------")

    print_menu()
    
    # Start in teach mode
    gc_ctrl.start()
    
    # Save the initial home pose before the user moves it
    time.sleep(0.5)
    initial_home_pose = grasp_driver.get_tcp_pose()
    hx, hy, hz = initial_home_pose[:3, 3]
    hroll, hpitch, hyaw = rotation_matrix_to_euler_zyx(initial_home_pose[:3, :3])
    print(f"\n-> Captured Initial Pose as Home: X={hx:.3f} Y={hy:.3f} Z={hz:.3f}")
    
    mode = "TEACH"

    try:
        while True:
            try:
                line = line_queue.get(timeout=0.1)
                if line is None: 
                    break
                
                line = line.strip().lower()
                
                if mode == "TEACH":
                    if line == "":
                        # Save current pose
                        T = grasp_driver.get_tcp_pose()
                        waypoints.append(T)
                        t = T[:3, 3]
                        R = T[:3, :3]
                        r, p, y = rotation_matrix_to_euler_zyx(R)
                        print(f"-> Saved WP #{len(waypoints)}: X={t[0]:.3f} Y={t[1]:.3f} Z={t[2]:.3f} | RPY=({r:.2f}, {p:.2f}, {y:.2f})")
                    
                    elif line == "c":
                        waypoints.clear()
                        print("-> Waypoints cleared.")
                    
                    elif line == "p":
                        if not waypoints:
                            print("-> No waypoints to playback!")
                            continue
                        
                        print("\n=== Starting Playback ===")
                        speed_str = input("Enter playback duration per waypoint in seconds (default 3.0): ")
                        try:
                            duration = float(speed_str) if speed_str.strip() else 3.0
                        except ValueError:
                            print("Invalid input, defaulting to 3.0s")
                            duration = 3.0
                            
                        # Switch to playback mode
                        mode = "PLAYBACK"
                        gc_ctrl.stop()
                        
                        # Wait a bit for the gc loop to stop completely
                        time.sleep(0.5)
                        
                        # Read current state to seed the controller safely
                        q_now = rebotarm.get_state()[0][: rebotarm.arm.num_joints]
                        controller._q_target[:] = q_now
                        controller.start()
                        
                        # Go to home state first
                        print("Moving to Initial Home Pose first...")
                        
                        controller.move_to_traj(hx, hy, hz, roll=hroll, pitch=hpitch, yaw=hyaw, duration=duration)
                        time.sleep(duration + 0.5)
                        
                        print("Moving through waypoints...")
                        for i, wp in enumerate(waypoints):
                            t = wp[:3, 3]
                            r, p, y = rotation_matrix_to_euler_zyx(wp[:3, :3])
                            print(f"  -> Moving to WP #{i+1}...")
                            
                            controller.move_to_traj(t[0], t[1], t[2], roll=r, pitch=p, yaw=y, duration=duration)
                            
                            # Wait for motion to finish
                            time.sleep(duration + 0.5)
                        
                        print("=== Playback Complete ===")
                        # Switch back to teach mode
                        rebotarm.stop_control_loop()
                        time.sleep(0.5)
                        gc_ctrl.start()
                        mode = "TEACH"
                        print("-> Returned to TEACH mode. You can move the arm manually again.")
                        print_menu()
                        
                    elif line == "q":
                        break
                    
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print("\nInterrupted.")
        
    finally:
        print("\nShutting down safely...")
        if mode == "TEACH":
            gc_ctrl.stop()
        else:
            rebotarm.stop_control_loop()
            
        # Send arm to ready pose before disconnecting
        try:
            print("Moving to Initial Home Pose before shutdown...")
            time.sleep(0.5)
            q_now = rebotarm.get_state()[0][: rebotarm.arm.num_joints]
            controller._q_target[:] = q_now
            controller.start()
            
            controller.move_to_traj(hx, hy, hz, roll=hroll, pitch=hpitch, yaw=hyaw, duration=3.0)
            time.sleep(3.5)
            rebotarm.stop_control_loop()
        except Exception as e:
            print(f"Could not return to home pose: {e}")
            
        try:
            rebotarm.disconnect()
        except Exception:
            pass
        print("Disconnected.")
        os._exit(0)

if __name__ == "__main__":
    main()
