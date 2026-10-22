"""Bounded live inference scheduling and a preview with no frame queue."""
import threading
import time
from concurrent.futures import Future

import cv2


class StaleInference(RuntimeError):
    """A frame expired before its prediction could start."""


class InferenceDispatcher:
    """One model owner, two bounded pending slots, and priority for guarding.

    Callers wait for a result before submitting another frame. Mask decoding
    and tracking can overlap with the next prediction without sharing the
    mutable Ultralytics predictor across threads.
    """

    def __init__(self, predict):
        self.predict = predict
        self.condition = threading.Condition()
        self.pending = []
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='gold-inference', daemon=True)
        self.thread.start()

    def submit(self, kind, source, kwargs, expires_at=None):
        future = Future()
        with self.condition:
            while len(self.pending) >= 2 and not self.closed:
                self.condition.wait()
            if self.closed:
                raise RuntimeError('Gold inference is stopped')
            self.pending.append((kind, source, kwargs, expires_at, future))
            self.condition.notify_all()
        return future

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                index = next((i for i, item in enumerate(self.pending) if item[0] == 'bench'), 0)
                _, source, kwargs, expires_at, future = self.pending.pop(index)
                self.condition.notify_all()
            if not future.set_running_or_notify_cancel():
                continue
            if expires_at is not None and time.monotonic() > expires_at:
                future.set_exception(StaleInference('Camera frame expired before inference'))
                continue
            try:
                future.set_result(self.predict(source, **kwargs))
            except Exception as exc:
                future.set_exception(exc)

    def close(self):
        with self.condition:
            self.closed = True
            pending, self.pending = self.pending, []
            self.condition.notify_all()
        for *_, future in pending:
            if not future.done():
                future.set_exception(RuntimeError('Gold inference is stopped'))
        self.thread.join()


class LatestPreview:
    """Share only the latest immutable overlay, never captured video frames.

    The GUI applies it to the newest camera frame without acquiring the
    tracking/model lock. Expired overlays cannot present old boxes as live.
    """

    def __init__(self, max_age=0.3):
        self.max_age = max_age
        self.lock = threading.Lock()
        self.overlay = None

    def publish(self, image, captured_at):
        mask = cv2.bitwise_not(cv2.inRange(image, (0, 0, 0), (0, 0, 0)))
        with self.lock:
            self.overlay = (image, mask, captured_at)

    def clear(self):
        with self.lock:
            self.overlay = None

    def render(self, frame, captured_at):
        with self.lock:
            overlay = self.overlay
        if overlay is not None:
            image, mask, ts = overlay
            if image.shape == frame.shape and 0 <= captured_at - ts <= self.max_age:
                cv2.copyTo(image, mask, frame)
        return frame
