import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from drivers.camera.webcam import WebcamDriver
from utils.webcam_capture import open_webcam, set_focus
from scripts import calibrate_camera


class FocusControlTests(unittest.TestCase):
    def camera(self):
        state = {cv2.CAP_PROP_AUTOFOCUS: 1., cv2.CAP_PROP_FOCUS: 68.}
        cap = Mock()
        def write(key, value):
            state[key] = value
            return True
        cap.set.side_effect = write
        cap.get.side_effect = lambda key: state.get(key, 0.)
        return cap

    def test_focus_zero_disables_auto_before_lens_write(self):
        cap = self.camera()
        set_focus(cap, 0)
        self.assertEqual(cap.set.call_args_list[0].args, (cv2.CAP_PROP_AUTOFOCUS, 0))
        self.assertEqual(cap.set.call_args_list[1].args, (cv2.CAP_PROP_FOCUS, 0.))

    def test_unspecified_focus_preserves_camera_controls(self):
        cap = self.camera()
        set_focus(cap)
        cap.set.assert_not_called()

    def test_autofocus_cannot_override_fixed_focus(self):
        with self.assertRaises(ValueError):
            set_focus(self.camera(), 80, True)

    def test_rejected_control_fails_explicitly(self):
        cap = self.camera()
        cap.set.side_effect = lambda *args: False
        with self.assertRaisesRegex(RuntimeError, 'rejected autofocus'):
            set_focus(cap, 80)

    def test_silently_ignored_lens_write_is_detected(self):
        cap = self.camera()
        cap.set.side_effect = lambda *args: True
        cap.get.side_effect = lambda key: 0. if key == cv2.CAP_PROP_AUTOFOCUS else 68.
        with self.assertRaisesRegex(RuntimeError, 'readback'):
            set_focus(cap, 80)

    @patch('utils.webcam_capture.cv2.VideoCapture')
    def test_wrong_capture_resolution_closes_camera(self, factory):
        cap = self.camera()
        cap.get.side_effect = lambda key: 640. if key == cv2.CAP_PROP_FRAME_WIDTH else 480.
        factory.return_value = cap
        with self.assertRaisesRegex(RuntimeError, 'negotiated'):
            open_webcam(2)
        cap.release.assert_called_once()


class CalibrationTests(unittest.TestCase):
    def calibration(self, directory):
        path = Path(directory) / 'intrinsics.npz'
        np.savez(path, camera_matrix=np.array([[900.,0,640.],[0,900.,360.],[0,0,1.]]),
                 distortion_coefficients=np.array([[.1,-.2,.001,.002,.1]]),
                 image_width=1280, image_height=720, focus_absolute=68)
        return path

    def test_rectified_frames_expose_zero_distortion_without_losing_raw_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            cam = WebcamDriver(calib_file=str(self.calibration(directory)))
            np.testing.assert_array_equal(cam.D, np.zeros((1,5)))
            self.assertNotEqual(float(cam._D[0,0]), 0.)

    @patch('drivers.camera.webcam.open_webcam')
    def test_calibration_focus_mismatch_closes_camera(self, opener):
        cap = Mock()
        state = {cv2.CAP_PROP_FOCUS: 100., cv2.CAP_PROP_AUTOFOCUS: 0.}
        cap.get.side_effect = lambda key: state.get(key, 0.)
        opener.return_value = cap
        with tempfile.TemporaryDirectory() as directory:
            cam = WebcamDriver(calib_file=str(self.calibration(directory)))
            with self.assertRaisesRegex(RuntimeError, 'Calibration requires fixed focus'):
                cam.open()
            cap.release.assert_called_once()

    def test_full_resolution_sample_uses_original_corner_refinement(self):
        gray = np.full((720,1280), 127, np.uint8)
        square, ox, oy = 50, 350, 150
        for y in range(7):
            for x in range(10):
                gray[oy+y*square:oy+(y+1)*square,ox+x*square:ox+(x+1)*square] = 255 if (x+y)%2 else 0
        found, expected = cv2.findChessboardCorners(gray, (9,6), None)
        self.assertTrue(found)
        expected = cv2.cornerSubPix(gray,expected,(11,11),(-1,-1),
                                  (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001))
        cap = Mock()
        state = {cv2.CAP_PROP_AUTOFOCUS: 1., cv2.CAP_PROP_FOCUS: 68.}
        def write(key, value):
            state[key] = value
            return True
        cap.set.side_effect = write
        cap.get.side_effect = lambda key: state.get(key, 0.)
        cap.read.return_value = (True, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        keys = iter([32,32,32,27])
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch('sys.argv', ['calibrate_camera.py','--calib-dir',directory]))
            stack.enter_context(patch.object(calibrate_camera.yaml, 'safe_load', return_value={
                'camera': {'focus': 0, 'autofocus': False, 'fps': 30, 'fourcc': 'MJPG'}}))
            stack.enter_context(patch.object(cv2,'VideoCapture',return_value=cap))
            for name in ['namedWindow','imshow','destroyAllWindows']:
                stack.enter_context(patch.object(cv2,name))
            stack.enter_context(patch.object(cv2,'waitKey',side_effect=lambda delay: next(keys) if delay == 1 else -1))
            solver = stack.enter_context(patch.object(cv2,'calibrateCamera',return_value=(.2,np.eye(3),np.zeros((1,5)),[],[])))
            saved = stack.enter_context(patch.object(np,'savez'))
            calibrate_camera.main()
            observations = solver.call_args.args[1]
            self.assertEqual(solver.call_args.args[2], (1280,720))
            self.assertEqual(len(observations),3)
            for actual in observations:
                np.testing.assert_array_equal(actual,expected)
            self.assertNotIn('focus_absolute', saved.call_args.kwargs)
            self.assertEqual([call.args for call in cap.set.call_args_list], [
                (cv2.CAP_PROP_FRAME_WIDTH,1280), (cv2.CAP_PROP_FRAME_HEIGHT,720),
                (cv2.CAP_PROP_AUTOFOCUS,0), (cv2.CAP_PROP_FOCUS,0.),
            ])
            self.assertEqual(state[cv2.CAP_PROP_AUTOFOCUS],0)
            self.assertEqual(state[cv2.CAP_PROP_FOCUS],0)
            cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
