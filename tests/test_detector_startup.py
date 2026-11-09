import unittest
from unittest.mock import Mock

import numpy as np

from utils.yolo_utils import DetectorMemoryError, warmup_yolo


class DetectorStartupTests(unittest.TestCase):
    def test_original_prediction_settings_are_retained(self):
        model=Mock()
        frame=np.zeros((720,1280,3),np.uint8)
        warmup_yolo(model,frame,{'device':'0','conf':.5,'iou':.45})
        model.predict.assert_called_once_with(frame,verbose=False,device='0',conf=.5,iou=.45)

    def test_gpu_memory_errors_get_actionable_diagnostic_without_retry_or_precision_change(self):
        for message in ('NVML_SUCCESS == r INTERNAL ASSERT FAILED', 'CUDA out of memory', 'Allocation on device'):
            model=Mock()
            model.predict.side_effect=RuntimeError(message)
            with self.subTest(message=message), self.assertRaisesRegex(DetectorMemoryError,'No motors were started'):
                warmup_yolo(model,np.zeros((720,1280,3),np.uint8),{'device':'0'})
            self.assertEqual(model.predict.call_count,1)

    def test_unrelated_errors_are_not_misreported_as_gpu_memory_errors(self):
        model=Mock()
        error=RuntimeError('invalid model input')
        model.predict.side_effect=error
        with self.assertRaises(RuntimeError) as caught:
            warmup_yolo(model,np.zeros((720,1280,3),np.uint8),{'device':'0'})
        self.assertIs(caught.exception,error)

    def test_cpu_errors_are_preserved(self):
        model=Mock()
        error=RuntimeError('CUDA out of memory')
        model.predict.side_effect=error
        with self.assertRaises(RuntimeError) as caught:
            warmup_yolo(model,np.zeros((720,1280,3),np.uint8),{'device':'cpu'})
        self.assertIs(caught.exception,error)


if __name__=='__main__':
    unittest.main()
