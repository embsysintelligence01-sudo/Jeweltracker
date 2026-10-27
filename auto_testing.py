"""Image/position gate for automatic test handoffs, without marker tracking."""
import cv2
import numpy as np


class StoneSettling:
    """Count observed quiet time in a fixed stone crop, ignoring live box shapes.

    An optional centre is the same tracked jewel's already smoothed position.
    Brief missing observations pause credit. A completed attempt stays latched
    until movement or a longer absence, preventing repeated automatic launches.
    """
    def __init__(self, stone, seconds=2., center_tolerance=8., gap_grace=.75):
        self.stone = tuple(map(float, stone))
        self.seconds = seconds
        self.center_tolerance = center_tolerance
        self.gap_grace = gap_grace
        self.elapsed = 0.
        self.anchor = self.center_anchor = None
        self.last_ts = self.missing_since = None
        self.paused = False
        self.latched = False

    def image(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.stone
        # Include the boundary so movement of a plain stone is visible too.
        pad = .08*max(x2-x1, y2-y1)
        x1, y1 = max(0, int(x1-pad)), max(0, int(y1-pad))
        x2, y2 = min(w, int(x2+pad)), min(h, int(y2+pad))
        if x2 <= x1 or y2 <= y1:
            return None
        gray = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (128, 128)), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)

    def update(self, frame, timestamp, *, visible=True, center=None):
        if self.last_ts is not None and timestamp <= self.last_ts:
            return False
        gap = 0. if self.last_ts is None else timestamp-self.last_ts
        self.last_ts = timestamp
        if not visible:
            if self.missing_since is None:
                self.missing_since = timestamp
            self.paused = True
            if timestamp-self.missing_since > self.gap_grace:
                self.anchor = self.center_anchor = None
                self.elapsed, self.latched = 0., False
            return False
        if ((gap > self.gap_grace and not self.latched)
                or (self.missing_since is not None
                    and timestamp-self.missing_since > self.gap_grace)):
            self.anchor = self.center_anchor = None
            self.elapsed, self.latched = 0., False
        delta = 0. if self.paused else gap
        self.paused, self.missing_since = False, None
        image = self.image(frame)
        if image is None:
            self.anchor, self.elapsed = None, 0.
            return False
        center = None if center is None else np.asarray(center, float)
        moved = False
        if self.anchor is not None:
            difference = image-self.anchor
            difference -= np.median(difference)  # tolerate uniform light flicker
            moved = np.mean(np.abs(difference) > 12) > .03
            if center is not None and self.center_anchor is not None:
                moved = moved or np.linalg.norm(center-self.center_anchor) > self.center_tolerance
        if self.anchor is None or moved:
            self.anchor = image
            self.center_anchor = None if center is None else center.copy()
            self.elapsed, self.latched = 0., False
            return False
        if self.latched:
            return False
        self.elapsed += delta
        if self.elapsed+1e-6 >= self.seconds:
            self.latched = True
            return True
        return False
