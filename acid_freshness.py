"""Conservative, stone-relative evidence of a *new* acid application.

Boxes are detector footprints, not chemical segmentation. Visual clearing is
not proof that a surface is chemically clean. Occlusion never clears history.
"""
import cv2
import numpy as np

SIZE = 256
DEST = np.float32([[0, 0], [255, 0], [255, 255], [0, 255]])


def normalized(frame, quad):
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(
        np.asarray(quad, np.float32), DEST), (SIZE, SIZE))


def footprint(box, quad):
    x1, y1, x2, y2 = box
    points = np.float32([[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]])
    points = cv2.perspectiveTransform(points, cv2.getPerspectiveTransform(
        np.asarray(quad, np.float32), DEST))[0]
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(np.clip(points, -SIZE, 2*SIZE)).astype(np.int32), 255)
    return mask


def changed_pixels(before, after):
    # Remove a uniform exposure/colour shift; only localized changes count.
    before = cv2.GaussianBlur(before, (3,3), 0)
    after = cv2.GaussianBlur(after, (3,3), 0)
    delta = after.astype(np.float32)-before.astype(np.float32)
    delta -= np.median(delta[16:-16, 16:-16], axis=(0,1))
    return np.max(np.abs(delta), axis=2) > 15


class AcidHistory:
    def __init__(self):
        self.mask = np.zeros((SIZE, SIZE), np.uint8)
        self.dry_reference = None
        self.clear_since = None
        self.clear_count = 0
        self.events = []

    def remember(self, found, quad):
        if quad is not None:
            for detection in found:
                self.mask |= footprint(detection['bbox'], quad)

    def observe_idle(self, frame, quad, found, timestamp, clear):
        if not clear or quad is None:
            self.clear_since, self.clear_count = None, 0
            return
        image = normalized(frame, quad)
        if found:
            self.remember(found, quad)
            self.clear_since, self.clear_count = None, 0
            return
        if not np.any(self.mask):
            if self.dry_reference is None:
                self.dry_reference = image.copy()
            return
        # Several negatives alone are insufficient. The former patch must also
        # resemble an observed dry reference on a clear, stable stone.
        if self.dry_reference is None or np.mean(changed_pixels(self.dry_reference, image)[self.mask > 0]) > .15:
            self.clear_since, self.clear_count = None, 0
            return
        if self.clear_since is None:
            self.clear_since = timestamp
        self.clear_count += 1
        if self.clear_count >= 3 and timestamp-self.clear_since >= .6:
            self.events.append(dict(timestamp=timestamp, status='VISUALLY_CLEARED',
                                    note='Visual evidence only; chemical cleanliness is not measured.'))
            self.mask[:] = 0
            self.clear_since, self.clear_count = None, 0


class FreshAcidGate:
    def __init__(self, history, frame=None, quad=None):
        self.old_mask = history.copy()
        self.baseline = normalized(frame, quad) if frame is not None and quad is not None else None
        self.rejections = []

    def check(self, frame, quad, found, timestamp, geometry_known=True):
        if not found:
            return None, 'ACID_NOT_DETECTED'
        if not geometry_known or quad is None or self.baseline is None:
            return self._reject(timestamp, 'FRESH_APPLICATION_NOT_CONFIRMED', found)
        image = normalized(frame, quad)
        changed = changed_pixels(self.baseline, image)
        old = cv2.dilate(self.old_mask, np.ones((9,9), np.uint8)) > 0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skin = (hsv[:,:,0] <= 14) & (hsv[:,:,1] >= 110) & (hsv[:,:,2] > 60)
        reasons = []
        for detection in sorted(found, key=lambda d: d['confidence'], reverse=True):
            region = footprint(detection['bbox'], quad) > 0
            area = np.count_nonzero(region)
            if area < 4:
                reasons.append('FRESH_APPLICATION_NOT_CONFIRMED')
                continue
            overlap = np.count_nonzero(region & old)/max(1, min(area, np.count_nonzero(old)))
            if overlap >= .25:
                # Same-patch brightness changes, spreading and a tool appearing
                # cannot establish a fresh chemical application with this model.
                reasons.append('PREVIOUS_ACID_IGNORED')
                continue
            if np.mean(skin[region]) > .3:
                reasons.append('FRESH_APPLICATION_NOT_CONFIRMED')
                continue
            fraction = float(np.mean(changed[region]))
            outside = changed[~region]
            if fraction < .18 or (outside.size and fraction < float(outside.mean())+.10):
                # A patch already in the baseline must not become a new test.
                reasons.append('FRESH_APPEARANCE_TOO_WEAK')
                continue
            return dict(detection, freshness='NEW_APPEARANCE_OUTSIDE_PREVIOUS_ACID',
                        fresh_changed_fraction=fraction), 'FRESH_APPLICATION_DETECTED'
        reason = 'PREVIOUS_ACID_IGNORED' if 'PREVIOUS_ACID_IGNORED' in reasons else reasons[0]
        return self._reject(timestamp, reason, found)

    def _reject(self, timestamp, reason, found):
        self.rejections.append(dict(timestamp=timestamp, reason=reason,
                                    bboxes=[list(d['bbox']) for d in found]))
        self.rejections = self.rejections[-100:]
        return None, reason

    def fork(self):
        clone = FreshAcidGate(self.old_mask)
        clone.baseline = None if self.baseline is None else self.baseline.copy()
        return clone
