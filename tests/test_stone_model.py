import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import jewel_tracker_5 as tracker
from stone_onnx import StoneOnnxModel


class StoneModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((300, 400, 3), 230, np.uint8)
        self.model = Mock()
        self.roi = tracker.to_poly([(50, 50), (350, 50), (350, 280), (50, 280)])
        self.watcher = tracker.StoneWatcher(None, self.model, lambda: self.roi)

    def result(self, boxes=(), confidence=()):
        # Real Ultralytics Boxes supports len() and the two tensor properties.
        self.model.predict.return_value = [SimpleNamespace(boxes=StubBoxes(boxes, confidence))]

    def test_dark_shape_without_model_detection_is_never_a_stone(self):
        self.frame[70:250, 80:280] = 20
        self.result()
        self.watcher.update(self.frame)
        self.assertIsNone(self.watcher.bbox)
        self.assertFalse(self.watcher.present)
        self.model.predict.assert_called_once()

    def test_model_runs_on_every_check_and_misses_clear_previous_box(self):
        self.result([(60, 60, 220, 220)], [.9])
        self.watcher.update(self.frame)
        self.assertEqual((60, 60, 220, 220), self.watcher.bbox)
        self.assertTrue(self.watcher.present)
        self.result()
        self.watcher.update(self.frame)
        self.watcher.update(self.frame)
        self.assertEqual(3, self.model.predict.call_count)
        self.assertIsNone(self.watcher.bbox)
        self.assertFalse(self.watcher.present)

    def test_only_confidence_and_roi_select_model_boxes(self):
        self.result([(0, 0, 30, 30), (70, 80, 130, 100), (100, 100, 300, 250)], [.99, .9, .8])
        self.watcher.update(self.frame)
        # Bright, small, non-square model prediction is still accepted.
        self.assertEqual((70, 80, 130, 100), self.watcher.bbox)
        args = self.model.predict.call_args.kwargs
        self.assertEqual(640, args['imgsz'])
        self.assertEqual(tracker.STONE_CONF, args['conf'])

    def test_new_model_box_is_used_without_old_box_shrink_override(self):
        self.watcher.bbox = (50, 50, 300, 280)
        self.watcher.found_t = time.time()
        self.result([(100, 100, 220, 230)], [.9])
        self.watcher.update(self.frame)
        self.assertEqual((100, 100, 220, 230), self.watcher.bbox)

    def test_late_results_and_roi_changes_are_discarded(self):
        self.result([(60, 60, 220, 220)], [.9])
        self.watcher.update(self.frame, time.time() - tracker.STONE_RESULT_MAX_AGE_S - .1)
        self.assertIsNone(self.watcher.bbox)
        self.model.predict.side_effect = lambda *a, **k: self.change_roi()
        self.watcher.update(self.frame)
        self.assertIsNone(self.watcher.bbox)

    def test_cpu_stone_result_has_its_own_age_budget(self):
        self.result([(60, 60, 220, 220)], [.9])
        observed_at = time.time() - .6
        self.watcher.update(self.frame, observed_at)
        self.assertEqual((60, 60, 220, 220), self.watcher.bbox)
        self.assertEqual(observed_at, self.watcher.found_t)

    def change_roi(self):
        self.roi = None
        return [SimpleNamespace(boxes=StubBoxes([(60, 60, 220, 220)], [.9]))]

    def test_unavailable_model_cannot_invent_a_dark_shape_detection(self):
        self.watcher.model = None
        self.frame[70:250, 80:280] = 20
        self.watcher.update(self.frame)
        self.assertIsNone(self.watcher.bbox)

    def test_loop_uses_latest_frame_and_inference_counts_toward_interval(self):
        self.result([(60, 60, 220, 220)], [.9])
        self.watcher.grabber = Mock()
        self.watcher.grabber.latest.return_value = (self.frame, time.time())
        with patch.object(tracker.time, 'monotonic', side_effect=[10., 10.35]), \
             patch.object(tracker.time, 'sleep', side_effect=lambda _: setattr(self.watcher, 'running', False)) as sleep:
            self.watcher.run()
        self.watcher.grabber.latest.assert_called_once_with(copy_frame=False)
        self.assertAlmostEqual(tracker.STONE_CHECK_S - .35, sleep.call_args.args[0])

    def test_stale_camera_frame_is_not_sent_to_model(self):
        self.watcher.bbox = (60, 60, 220, 220)
        self.watcher.grabber = Mock()
        self.watcher.grabber.latest.return_value = (self.frame, time.time() - 2)
        with patch.object(tracker.time, 'sleep', side_effect=lambda _: setattr(self.watcher, 'running', False)):
            self.watcher.run()
        self.model.predict.assert_not_called()
        self.assertIsNone(self.watcher.bbox)

    def test_failed_prediction_clears_old_box_and_is_logged(self):
        self.watcher.bbox = (60, 60, 220, 220)
        self.watcher.grabber = Mock()
        self.watcher.grabber.latest.return_value = (self.frame, time.time())
        self.model.predict.side_effect = RuntimeError('failed inference')
        with patch.object(tracker, 'write_log') as log, \
             patch.object(tracker.time, 'sleep', side_effect=lambda _: setattr(self.watcher, 'running', False)):
            self.watcher.run()
        self.assertIsNone(self.watcher.bbox)
        log.assert_called_once()
        self.assertIn('failed inference', log.call_args.args[0])


class StubBoxes:
    def __init__(self, boxes, confidence):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(confidence, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class StoneBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = StoneOnnxModel(tracker.STONE_MODEL_PATH)
        cls.reference = tracker.YOLO(tracker.STONE_MODEL_PATH, task='segment')

    def test_original_onnx_boxes_and_confidence_are_preserved(self):
        root = Path(tracker.SCRIPT_DIR)
        paths = [root / 'diagnostics/antiflicker_20261021/final_verified_00.jpg',
                 root / 'captures/20261017_145247_capture.jpg',
                 root / 'captures/20261021_124914_capture.jpg']
        for path in paths:
            with self.subTest(path=path.name):
                frame = tracker.cv2.imread(str(path))
                self.assertIsNotNone(frame)
                expected = self.reference.predict(frame, imgsz=640, conf=.2, device='cpu', verbose=False)[0]
                actual = self.model.predict(frame)[0]
                np.testing.assert_allclose(actual.boxes.data.numpy(), expected.boxes.data.numpy(), atol=.01, rtol=0)
                self.assertEqual({0: 'St'}, actual.names)

    def test_execution_is_bounded_and_keeps_float32_precision(self):
        import openvino as ov

        self.assertEqual(1, self.model.compiled.get_property('NUM_STREAMS'))
        self.assertEqual(2, self.model.compiled.get_property('INFERENCE_NUM_THREADS'))
        self.assertEqual(ov.Type.f32, self.model.compiled.get_property('INFERENCE_PRECISION_HINT'))


if __name__ == '__main__':
    unittest.main()
