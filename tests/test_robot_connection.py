import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from drivers.robot.grasp_driver import GraspDriver, resolve_dm_serial_channel


class SerialConnectionTests(unittest.TestCase):
    def arm(self):
        return SimpleNamespace(
            _channel='/dev/ttyACM0',
            _all_joints=[SimpleNamespace(name=f'joint{i}', vendor='damiao', motor_id=i,
                                        feedback_id=i + 16, model='4310')
                         for i in range(1, 8)],
        )

    @patch('drivers.robot.grasp_driver.Path.glob', return_value=[])
    def test_configured_port_verified_without_enabling(self, glob):
        ctrl = Mock()
        with patch('motorbridge.Controller.from_dm_serial', return_value=ctrl):
            self.assertEqual(resolve_dm_serial_channel(self.arm()), '/dev/ttyACM0')
        self.assertEqual(ctrl.add_damiao_motor.call_count, 7)
        self.assertEqual(ctrl.add_damiao_motor.return_value.get_register_u32.call_count, 7)
        ctrl.enable_all.assert_not_called()
        ctrl.close.assert_called_once()
        ctrl.close_bus.assert_called_once()

    @patch('drivers.robot.grasp_driver.Path.glob', return_value=[])
    def test_probe_releases_all_native_handles_before_reopen(self, glob):
        ctrl = Mock()
        motors = [Mock() for _ in range(7)]
        ctrl.add_damiao_motor.side_effect = motors
        events = []
        ctrl.close_bus.side_effect = lambda: events.append('bus')
        ctrl.close.side_effect = lambda: events.append('controller')
        for motor in motors:
            motor.close.side_effect = lambda: events.append('motor')
        with patch('motorbridge.Controller.from_dm_serial', return_value=ctrl):
            resolve_dm_serial_channel(self.arm())
        self.assertEqual(events, ['bus'] + ['motor'] * 7 + ['controller'])
        for motor in motors:
            motor.close.assert_called_once()

    @patch('drivers.robot.grasp_driver.Path.glob')
    def test_wrong_port_falls_back_to_verified_bridge(self, glob):
        from pathlib import Path
        glob.side_effect = [[Path('/dev/ttyACM1')], []]
        wrong, correct = Mock(), Mock()
        wrong.add_damiao_motor.return_value.get_register_u32.side_effect = RuntimeError('timeout')
        arm = self.arm()
        with patch('motorbridge.Controller.from_dm_serial', side_effect=[wrong, correct]):
            self.assertEqual(resolve_dm_serial_channel(arm), '/dev/ttyACM1')
        self.assertEqual(arm._channel, '/dev/ttyACM1')
        for ctrl in (wrong, correct):
            ctrl.close.assert_called_once()
            ctrl.close_bus.assert_called_once()
            ctrl.add_damiao_motor.return_value.close.assert_called()
            ctrl.enable_all.assert_not_called()

    @patch('drivers.robot.grasp_driver.Path.glob', return_value=[])
    def test_partial_replies_rejected(self, glob):
        ctrl = Mock()
        ctrl.add_damiao_motor.return_value.get_register_u32.side_effect = [1, RuntimeError('timeout')]
        arm = self.arm()
        with patch('motorbridge.Controller.from_dm_serial', return_value=ctrl):
            with self.assertRaisesRegex(RuntimeError, 'No Damiao serial bridge'):
                resolve_dm_serial_channel(arm)
        self.assertEqual(arm._channel, '/dev/ttyACM0')
        ctrl.enable_all.assert_not_called()
        ctrl.close.assert_called_once()


class StartupTests(unittest.TestCase):
    def setUp(self):
        # Bypass kinematics/model loading; exercise the real startup methods.
        self.driver = GraspDriver.__new__(GraspDriver)
        self.arm_group = Mock()
        self.gripper_group = Mock()
        self.motor = Mock()
        self.bus = Mock()
        self.arm = Mock(hardware_yaml='rebotarm_dm.yaml')
        self.arm._motor_map = {'gripper': self.motor}
        self.arm._ctrl_map = {'damiao': self.bus}
        self.arm._connected = True
        self.driver._arm = self.arm
        self.driver._controller = SimpleNamespace(_running=False, _arm_control_mode='posvel')
        self.driver._arm_group = self.arm_group
        self.driver._gripper_group = self.gripper_group
        self.driver._gripper_name = 'gripper'
        self.gripper_group._mm = self.arm._motor_map
        self.driver._wait_startup_feedback = Mock()
        self.driver._prime_arm_target = Mock()
        self.driver._prime_gripper_state = Mock()
        self.probe = patch('drivers.robot.grasp_driver.resolve_dm_serial_channel').start()
        patch('drivers.robot.grasp_driver.time.sleep').start()
        self.addCleanup(patch.stopall)

    def assert_aborted(self):
        self.assertFalse(self.driver._controller._running)
        self.arm.start_control_loop.assert_not_called()
        self.arm.disconnect.assert_called_once()
        self.motor.close.assert_called_once()
        self.bus.close.assert_called_once()
        self.assertEqual(self.arm._motor_map, {})
        self.assertEqual(self.arm._ctrl_map, {})

    def test_transient_ack_failure_retried_before_either_group_enabled(self):
        events = []
        replies = iter([False, True])
        def arm_mode():
            events.append('arm mode')
            return next(replies)
        self.arm_group.mode_pos_vel.side_effect = arm_mode
        self.gripper_group.mode_mit.side_effect = lambda: events.append('gripper mode') or True
        self.arm_group.enable.side_effect = lambda: events.append('arm enable')
        self.gripper_group.enable.side_effect = lambda: events.append('gripper enable')
        self.driver.start()
        self.assertEqual(events, ['arm mode', 'arm mode', 'gripper mode', 'arm enable', 'gripper enable'])
        self.assertTrue(self.driver._controller._running)
        self.arm.disconnect.assert_not_called()
        self.driver._prime_arm_target.assert_called_once()

    def test_persistent_arm_ack_failure_never_enables(self):
        self.arm_group.mode_pos_vel.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
            self.driver.start()
        self.assertEqual(self.arm_group.mode_pos_vel.call_count, 3)
        self.arm_group.enable.assert_not_called()
        self.gripper_group.enable.assert_not_called()
        self.assert_aborted()

    def test_gripper_ack_failure_never_enables_arm(self):
        self.gripper_group.mode_mit.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'gripper MIT'):
            self.driver.start()
        self.arm_group.enable.assert_not_called()
        self.gripper_group.enable.assert_not_called()
        self.assert_aborted()

    def test_missing_feedback_disconnects_without_priming_or_trajectory(self):
        self.driver._wait_startup_feedback.side_effect = RuntimeError('missing feedback')
        with self.assertRaisesRegex(RuntimeError, 'missing feedback'):
            self.driver.start()
        self.driver._prime_arm_target.assert_not_called()
        self.assert_aborted()

    def test_partial_connect_failure_releases_native_handles(self):
        self.arm._connected = False
        self.arm.connect.side_effect = RuntimeError('partial connect')
        with self.assertRaisesRegex(RuntimeError, 'partial connect'):
            self.driver.start()
        self.assert_aborted()

    def test_cleanup_failure_preserves_original_error_and_closes_handles(self):
        self.arm_group.mode_pos_vel.return_value = False
        self.arm.disconnect.side_effect = RuntimeError('disconnect failed')
        with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
            self.driver.start()
        self.assert_aborted()

    def test_failed_probe_never_connects_or_enables(self):
        self.probe.side_effect = RuntimeError('no bridge')
        with self.assertRaisesRegex(RuntimeError, 'no bridge'):
            self.driver.start()
        self.arm.connect.assert_not_called()
        self.arm_group.enable.assert_not_called()
        self.gripper_group.enable.assert_not_called()

    def test_mit_arm_keeps_configured_gains(self):
        self.driver._controller._arm_control_mode = 'mit'
        self.driver.start()
        self.arm_group.mode_mit.assert_called_once_with(
            kp=self.arm_group._mit_kp, kd=self.arm_group._mit_kd)
        self.arm_group.mode_pos_vel.assert_not_called()

    def test_feedback_rejects_missing_or_nonfinite_motor_state(self):
        for state in (None, SimpleNamespace(pos=float('nan'), vel=0., torq=0.)):
            with self.subTest(state=state):
                self.arm_group._jcfgs = [SimpleNamespace(name='joint1')]
                self.arm_group._mm = {'joint1': Mock(get_state=Mock(return_value=state))}
                self.gripper_group._jcfgs = []
                with self.assertRaisesRegex(RuntimeError, 'joint1'):
                    GraspDriver._wait_startup_feedback(self.driver, timeout=0.)


if __name__ == '__main__':
    unittest.main()
