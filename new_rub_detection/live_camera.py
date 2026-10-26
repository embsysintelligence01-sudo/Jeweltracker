"""Continuously drain the camera, keeping only the newest frame."""

import sys
import threading
from time import monotonic

import cv2


class LatestCamera(threading.Thread):
    def __init__(self, camera, width, height, configure, capture_factory=None):
        super().__init__(name="camera-reader", daemon=True)
        self.camera, self.width, self.height = camera, width, height
        self.configure = configure
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.packet = None
        self.error = None

    def snapshot(self):
        # Frames are immutable after publication; callers copy before drawing.
        with self.condition:
            return self.packet

    def wait_after(self, sequence):
        with self.condition:
            self.condition.wait_for(
                lambda: self.error is not None or self.stopping.is_set()
                or (self.packet is not None and self.packet[0] > sequence),
                timeout=0.1,
            )
            if self.error is not None:
                raise RuntimeError(self.error)
            if self.stopping.is_set():
                return None
            if self.packet is not None and self.packet[0] > sequence:
                return self.packet
            return None

    def stop(self):
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        # A driver may be blocked in read(); it still owns and releases the device.
        if self.ident is not None:
            self.join(timeout=2)

    def run(self):
        cap = None
        try:
            backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
                        if sys.platform == "win32" else [cv2.CAP_V4L2, None])
            for backend in backends:
                if self.stopping.is_set():
                    return
                cap = (self.capture_factory(self.camera, backend) if backend is not None
                       else self.capture_factory(self.camera))
                if cap.isOpened():
                    break
                cap.release()
                cap = None
            if cap is None:
                raise RuntimeError(f"Cannot open USB camera {self.camera}")
            settings = self.configure(cap, self.width, self.height)
            print("Camera settings requested/applied:", settings, flush=True)
            sequence = 0
            while not self.stopping.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed")
                sequence += 1
                with self.condition:
                    self.packet = (sequence, monotonic(), frame)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
        finally:
            if cap is not None:
                cap.release()
