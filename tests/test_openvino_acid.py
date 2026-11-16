from pathlib import Path
import unittest
import numpy as np

from acid_test.openvino_detector import OpenVINOAcidDetector
from acid_test.onnx_detector import OnnxAcidDetector
from acid_test.runtime import DETECTOR_PATH


class OpenVINOAcidTests(unittest.TestCase):
    def test_class_filter_and_camera_coordinates(self):
        raw = np.zeros((1, 38, 3), np.float32)
        # Acid, stronger rubbed-mask class, and low-confidence acid.
        raw[0, :6, :] = np.array([[104, 200, 300], [156, 156, 156],
                                 [52, 52, 52], [52, 52, 52],
                                 [.7, .8, .2], [.1, .9, .1]])
        found = OpenVINOAcidDetector.postprocess(raw, (400, 800, 3), .52, 0, 104, .30)
        self.assertEqual(1, len(found))
        self.assertEqual((150, 50, 250, 150), found[0].bbox)
        self.assertAlmostEqual(.7, found[0].confidence)

    def test_active_onnx_session_runs_and_reuses_cpu_model(self):
        self.assertEqual('acid_detector.onnx', DETECTOR_PATH.name)
        detector = OnnxAcidDetector(DETECTOR_PATH)
        session = detector.session
        for _ in range(2):
            found, elapsed = detector.predict(np.zeros((360, 640, 3), np.uint8), .30)
            self.assertEqual([], found)
            self.assertGreaterEqual(elapsed, 0)
        self.assertIs(session, detector.session)
        self.assertEqual({0: 'acid'}, detector.names)
        self.assertEqual(64, len(detector.model_sha256))
        self.assertEqual(['CPUExecutionProvider'], session.get_providers())


if __name__ == '__main__':
    unittest.main()
