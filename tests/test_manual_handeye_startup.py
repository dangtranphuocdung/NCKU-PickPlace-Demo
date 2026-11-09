import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.collect_handeye_eih import GravityCompController


class ManualStartupTests(unittest.TestCase):
    def setUp(self):
        # Exercise startup without loading dynamics or accessing hardware.
        self.gc = GravityCompController.__new__(GravityCompController)
        self.arm = Mock(rate=500)
        self.arm.arm.num_joints = 6
        self.motors = [Mock() for _ in range(7)]
        self.bus = Mock()
        self.arm._all_joints = [object() for _ in range(7)]
        self.arm._motor_map = dict(zip(range(7), self.motors))
        self.arm._ctrl_map = {'damiao': self.bus}
        self.arm.get_state.return_value = (np.arange(7) * .1, np.zeros(7), np.zeros(7))
        self.gc._arm = self.arm
        self.gc._gc_running = threading.Event()
        self.gc._io_lock = threading.RLock()
        self.gc._wait_state_valid = Mock()
        self.gc.safe_home = Mock()
        self.probe = patch('scripts.collect_handeye_eih.resolve_dm_serial_channel').start()
        patch('scripts.collect_handeye_eih.time.sleep').start()
        self.addCleanup(patch.stopall)

    def assert_aborted(self):
        self.arm.start_control_loop.assert_not_called()
        self.assertFalse(self.gc._gc_running.is_set())
        self.gc.safe_home.assert_not_called()
        self.arm.disconnect.assert_called_once()
        for motor in self.motors:
            motor.close.assert_called_once()
        self.bus.close.assert_called_once()
        self.assertEqual(self.arm._motor_map, {})

    def test_probe_and_both_modes_precede_enable_with_original_gains(self):
        events = []
        self.probe.side_effect = lambda arm: events.append('probe')
        self.arm.connect.side_effect = lambda: events.append('connect')
        self.arm.arm.mode_mit.side_effect = lambda **kwargs: events.append('arm mode') or True
        self.arm.gripper.mode_mit.side_effect = lambda: events.append('gripper mode') or True
        self.arm.enable_all.side_effect = lambda: events.append('enable')
        self.gc.start()
        self.assertEqual(events, ['probe', 'connect', 'arm mode', 'gripper mode', 'enable'])
        kwargs = self.arm.arm.mode_mit.call_args.kwargs
        np.testing.assert_array_equal(kwargs['kp'], np.full(6, 8.))
        np.testing.assert_array_equal(kwargs['kd'], np.full(6, 1.5))
        np.testing.assert_array_equal(self.gc._q_target, np.arange(6) * .1)
        self.arm.start_control_loop.assert_called_once_with(self.gc._worker, rate=500)

    def test_probe_failure_never_connects(self):
        self.probe.side_effect = RuntimeError('no verified bridge')
        with self.assertRaisesRegex(RuntimeError, 'verified bridge'):
            self.gc.start()
        self.arm.connect.assert_not_called()
        self.arm.enable_all.assert_not_called()

    def check_ack_failure(self, group):
        group.mode_mit.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'not acknowledged'):
            self.gc.start()
        self.assertEqual(group.mode_mit.call_count, 3)
        self.arm.enable_all.assert_not_called()
        self.assert_aborted()

    def test_persistent_arm_ack_failure_never_enables(self):
        self.check_ack_failure(self.arm.arm)

    def test_persistent_gripper_ack_failure_never_enables(self):
        self.check_ack_failure(self.arm.gripper)

    def test_transient_ack_timeout_can_recover(self):
        self.arm.arm.mode_mit.side_effect = [False, True]
        self.gc.start()
        self.assertEqual(self.arm.arm.mode_mit.call_count, 2)
        self.arm.enable_all.assert_called_once()
        self.assertTrue(self.gc._gc_running.is_set())

    def test_feedback_failure_disconnects_without_homing(self):
        self.gc._wait_state_valid.side_effect = RuntimeError('feedback is not ready')
        with self.assertRaisesRegex(RuntimeError, 'feedback is not ready'):
            self.gc.start()
        self.assert_aborted()

    def test_partial_connect_failure_closes_handles(self):
        self.arm.connect.side_effect = RuntimeError('partial connect')
        with self.assertRaisesRegex(RuntimeError, 'partial connect'):
            self.gc.start()
        self.arm.enable_all.assert_not_called()
        self.assert_aborted()

    def test_feedback_rejects_missing_nonfinite_or_partial_states(self):
        valid = SimpleNamespace(pos=0., vel=0., torq=0.)
        for state in (None, SimpleNamespace(pos=float('nan'), vel=0., torq=0.), valid):
            for motor in self.motors:
                motor.get_state.return_value = state
            if state is valid:
                self.arm._motor_map.pop(6)
            with patch('scripts.collect_handeye_eih.time.monotonic', side_effect=[0.,0.,1.]):
                with self.assertRaisesRegex(RuntimeError, 'feedback is not ready'):
                    GravityCompController._wait_state_valid(self.gc, timeout=.1)


if __name__ == '__main__':
    unittest.main()
