import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from int8_models import JEWEL_MODEL, RUBBED_MODEL, OpenVINOYolo, RubbedMarkDetector
from rub_preparation import RubPhotoModels


class Int8ModelTests(unittest.TestCase):
    def test_detector_boxes_are_clipped_candidates_not_segmentation(self):
        with patch('int8_models.OpenVINOYolo') as backend:
            backend.return_value.predict.return_value = [Mock(boxes=Mock(
                xyxy=torch.tensor([[-5., 10., 20., 25.], [60., 45., 100., 70.]])))]
            detector = RubbedMarkDetector()
            frame = np.zeros((50, 80, 3), np.uint8)
            regions = detector.regions(frame)
            expected = np.zeros((50, 80), np.uint8)
            expected[10:25, :20] = 255
            expected[45:, 60:] = 255
            np.testing.assert_array_equal(expected, regions)
            self.assertEqual('DETECTION_BOX_WITH_BEFORE_AFTER_CHANGE', detector.evidence_kind)
            backend.assert_called_once_with(RUBBED_MODEL, task='detect', imgsz=640)

    def test_real_exports_run_on_cpu_and_return_source_coordinates(self):
        cap = cv2.VideoCapture('/home/embsys/Videos/Rude_Appraiser.mp4')
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
            ok, frame = cap.read()
        finally:
            cap.release()
        self.assertTrue(ok)
        jewel = OpenVINOYolo(JEWEL_MODEL)
        result = jewel.predict(frame, conf=.15)[0]
        self.assertTrue(jewel.quantized)
        self.assertIn('CPU', jewel.compiled.get_property('EXECUTION_DEVICES'))
        self.assertGreater(len(result.boxes), 0)
        self.assertIsNotNone(result.masks)
        h, w = frame.shape[:2]
        for polygon in result.masks.xy:
            self.assertTrue((polygon[:, 0] >= 0).all() and (polygon[:, 0] <= w).all())
            self.assertTrue((polygon[:, 1] >= 0).all() and (polygon[:, 1] <= h).all())
        empty = jewel.predict(frame, conf=1.)[0]
        self.assertEqual(0, len(empty.boxes))
        self.assertIsNone(empty.masks)
        with self.assertRaises(ValueError):
            jewel.predict(frame, imgsz=640)
        with self.assertRaises(ValueError):
            OpenVINOYolo(RUBBED_MODEL, task='segment', imgsz=640)
        models = RubPhotoModels()
        self.addCleanup(models.close)
        regions = models.marks(frame)
        session = models.rub_session
        self.assertIsInstance(session, RubbedMarkDetector)
        self.assertEqual(frame.shape[:2], regions.shape)
        self.assertTrue(set(np.unique(regions)).issubset({0, 255}))
        models.marks(frame)
        self.assertIs(session, models.rub_session)


if __name__ == '__main__':
    unittest.main()
