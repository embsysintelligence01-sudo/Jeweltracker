from pathlib import Path
import unittest
from unittest.mock import patch,Mock
import numpy as np

from acid_test.openvino_detector import OpenVINOAcidDetector
DETECTOR_PATH=Path(__file__).resolve().parents[1]/'int_8_acid_model/acidph_int8.xml'


class OpenVINOAcidTests(unittest.TestCase):
    def test_rotated_detector_output_is_adapted_without_coordinate_offsets(self):
        backend=Mock(height=640,width=640,names={0:'acidph'})
        backend.detect.return_value=[dict(bbox=(101,205,144,299),score=.91)]
        with patch('acid_test.openvino_detector.OpenVINOOBB',return_value=backend):
            detector=OpenVINOAcidDetector(DETECTOR_PATH)
            frame=np.zeros((400,600,3),np.uint8)
            found,elapsed=detector.predict(frame,.6)
        self.assertEqual((101,205,144,299),found[0].bbox)
        self.assertEqual(.91,found[0].confidence)
        self.assertGreaterEqual(elapsed,0)
        self.assertIs(frame,backend.detect.call_args.args[0])
        self.assertEqual(dict(class_id=0,confidence=.6),backend.detect.call_args.kwargs)
        self.assertTrue(detector.quantized)
        self.assertEqual(64,len(detector.model_sha256))

    def test_actual_int8_session_runs_and_reuses_the_compiled_cpu_model(self):
        detector=OpenVINOAcidDetector(DETECTOR_PATH)
        model=detector.model.compiled
        for _ in range(2):
            found,elapsed=detector.predict(np.zeros((360,640,3),np.uint8),.6)
            self.assertEqual([],found)
            self.assertGreaterEqual(elapsed,0)
        self.assertIs(model,detector.model.compiled)
        self.assertTrue(detector.quantized)
        self.assertTrue(any('CPU' in x for x in model.get_property('EXECUTION_DEVICES')))


if __name__=='__main__':unittest.main()
