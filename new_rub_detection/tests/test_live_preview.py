"""Regression checks for preview latency, freshness, and photo geometry."""

from pathlib import Path
import queue
import threading
from time import monotonic
import unittest
from unittest.mock import Mock, patch

import numpy as np

from new_rub_detection import App
from new_rub_detection.live_camera import LatestCamera


class ControlledCapture:
    def __init__(self):
        self.frames = queue.Queue()
        self.released = threading.Event()

    def isOpened(self):
        return True

    def read(self):
        frame = self.frames.get(timeout=3)
        return frame is not None, frame

    def release(self):
        self.released.set()


def frame(value):
    return np.full((200, 240, 3), value, np.uint8)


class LatestCameraTests(unittest.TestCase):
    def setUp(self):
        self.cap = ControlledCapture()
        self.reader = LatestCamera(0, 240, 200, lambda *args: {}, lambda *args: self.cap)
        self.reader.start()

    def tearDown(self):
        self.reader.stopping.set()
        self.cap.frames.put(None)
        self.reader.stop()
        self.assertFalse(self.reader.is_alive())
        self.assertTrue(self.cap.released.is_set())

    def test_slow_consumer_gets_newest_frame_without_backlog(self):
        for value in (1, 2, 3, 4):
            self.cap.frames.put(frame(value))
        with self.reader.condition:
            self.assertTrue(self.reader.condition.wait_for(
                lambda: self.reader.packet is not None and self.reader.packet[0] == 4, 2))
        sequence, _, latest = self.reader.wait_after(1)
        self.assertEqual(sequence, 4)
        self.assertEqual(int(latest[0, 0, 0]), 4)

    def test_camera_read_error_reaches_inference_and_releases_device(self):
        self.cap.frames.put(None)
        with self.assertRaisesRegex(RuntimeError, "Camera frame read failed"):
            self.reader.wait_after(0)
        self.assertTrue(self.cap.released.wait(2))


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.worker = App.CameraWorker(0, 240, 200, Path("unused-test-output"))
        self.raw = frame(17)
        self.worker.reader.packet = (1, monotonic(), self.raw)
        self.bbox = (20, 20, 150, 150)
        self.quad = np.float32([[20, 20], [150, 20], [150, 150], [20, 150]])
        self.mask = np.zeros(self.raw.shape[:2], np.uint8)
        self.mask[20:150, 20:150] = 255

    def detect(self, bbox=None, quad=None):
        bbox = bbox if bbox is not None else self.bbox
        quad = quad if quad is not None else self.quad
        with patch.object(App, "infer_stone", return_value=[(bbox, 0.9, self.mask)]), \
                patch.object(App, "stone_quad", return_value=quad):
            return self.worker.live_frame(self.raw, None, object())

    def test_live_grid_is_optional_and_preview_does_not_modify_camera_frame(self):
        _, info, candidate = self.detect()
        self.worker.publish(monotonic(), candidate, info)
        with patch.object(App, "draw_stone_grid", wraps=App.draw_stone_grid) as draw:
            shown, _ = self.worker.preview()
            draw.assert_not_called()
            self.assertFalse(np.array_equal(shown, self.raw))
            self.worker.show_grid = True
            self.worker.preview()
            draw.assert_called_once()
        self.assertTrue(np.all(self.raw == 17))

    def test_expired_overlay_disappears_without_waiting_for_inference(self):
        _, info, candidate = self.detect()
        self.worker.publish(monotonic() - App.OVERLAY_MAX_AGE - 1, candidate, info)
        shown, info = self.worker.preview()
        np.testing.assert_array_equal(shown, self.raw)
        self.assertIn("fresh", info["stone"])

    def test_lost_detection_clears_geometry_and_cannot_be_captured(self):
        self.detect()
        with patch.object(App, "infer_stone", return_value=[]):
            _, info, candidate = self.worker.live_frame(self.raw, None, object())
        self.worker.publish(monotonic(), candidate, info)
        self.assertIsNone(candidate)
        self.assertIsNone(self.worker.smoothed_quad)
        shown, _ = self.worker.preview()
        np.testing.assert_array_equal(shown, self.raw)

    def test_photo_geometry_uses_current_prediction_instead_of_smoothed_position(self):
        self.detect()
        moved_quad = self.quad + (30, 0)
        moved_bbox = (50, 20, 180, 150)
        _, _, candidate = self.detect(moved_bbox, moved_quad)
        np.testing.assert_array_equal(candidate["quad"], moved_quad)
        self.assertEqual(candidate["bbox"], moved_bbox)
        self.assertFalse(np.array_equal(self.worker.smoothed_quad, moved_quad))

    def test_slow_inference_does_not_block_preview_and_skips_superseded_frames(self):
        cap = ControlledCapture()
        self.worker.reader = LatestCamera(0, 240, 200, lambda *args: {}, lambda *args: cap)
        entered = threading.Event()
        second_entered = threading.Event()
        release = threading.Event()
        second_release = threading.Event()
        inferred_values = []

        def slow_infer(model, image):
            inferred_values.append(int(image[0, 0, 0]))
            if len(inferred_values) == 1:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Test inference was not released")
            else:
                second_entered.set()
                second_release.wait(3)
            return []

        source = Path(App.__file__)
        with patch.object(App, "OnnxStoneModel", return_value=Mock()), \
                patch.object(App, "RUB_ONNX", source), \
                patch.object(App, "PHOTO_WORKER", source), \
                patch.object(App, "infer_stone", side_effect=slow_infer):
            self.worker.stone_model_path = source
            self.worker.start()
            try:
                cap.frames.put(frame(1))
                self.assertTrue(entered.wait(2))
                for value in (2, 3, 4, 5):
                    cap.frames.put(frame(value))
                with self.worker.reader.condition:
                    self.assertTrue(self.worker.reader.condition.wait_for(
                        lambda: self.worker.reader.packet[0] == 5, 2))
                # Inference is still blocked on frame 1, but preview shows frame 5.
                shown, _ = self.worker.preview()
                self.assertTrue(np.all(shown == 5))
                self.assertEqual(inferred_values, [1])
                release.set()
                self.assertTrue(second_entered.wait(2))
                self.assertEqual(inferred_values, [1, 5])
            finally:
                self.worker.requestInterruption()
                release.set()
                second_release.set()
                cap.frames.put(None)
                self.assertTrue(self.worker.wait(4000))
                self.assertFalse(self.worker.reader.is_alive())


if __name__ == "__main__":
    unittest.main()
