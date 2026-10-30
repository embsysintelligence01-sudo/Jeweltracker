"""Stone-relative acid history and advisory baseline-change measurements.

Historical overlap and weak baseline change never block detector candidates.
Current visibility/tool/patch checks live in acid_application.
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
        self.status = 'NO_PREVIOUS_ACID'
        self.checked_at = None
        self.last_sample = -float('inf')
        self.reference_mask = np.zeros((SIZE,SIZE), bool)

    def remember(self, found, quad, baseline=None):
        if quad is not None:
            region=np.zeros_like(self.mask)
            for detection in found:
                region |= footprint(detection['bbox'], quad)
            self.remember_region(region,baseline)

    def remember_region(self, region, baseline=None):
        new=(region>0) & (self.mask==0)
        if baseline is not None:
            if self.dry_reference is None:self.dry_reference=baseline.copy()
            self.dry_reference[new]=baseline[new]
            self.reference_mask[new]=True
        self.mask |= region
        if np.any(region):
            self.status='PREVIOUS_ACID_UNCERTAIN'

    def snapshot(self, timestamp):
        status=self.status
        if np.any(self.mask) and (self.checked_at is None or timestamp-self.checked_at>2):
            status='PREVIOUS_ACID_UNCERTAIN'
        return dict(status=status,checked_at=self.checked_at,prior_location_known=bool(np.any(self.mask)))

    def observe_idle(self, frame, quad, found, timestamp, clear):
        if timestamp<=self.last_sample:return
        if timestamp-self.last_sample>1.5:
            self.clear_since,self.clear_count=None,0
        self.last_sample=timestamp
        if not clear or quad is None:
            self.clear_since, self.clear_count = None, 0
            self.status='PREVIOUS_ACID_UNCERTAIN' if np.any(self.mask) else 'NO_PREVIOUS_ACID'
            return
        image = normalized(frame, quad)
        self.checked_at=timestamp
        if found:
            self.remember(found, quad)
            self.status='PREVIOUS_ACID_VISIBLE'
            self.clear_since, self.clear_count = None, 0
            return
        if not np.any(self.mask):
            self.dry_reference = image.copy()
            self.reference_mask[:]=True
            if self.status!='VISUALLY_CLEARED':self.status='NO_PREVIOUS_ACID'
            return
        # Several negatives alone are insufficient. The former patch must also
        # resemble an observed dry reference on a clear, stable stone.
        if (self.dry_reference is None or not np.all(self.reference_mask[self.mask>0])
                or np.mean(changed_pixels(self.dry_reference, image)[self.mask > 0]) > .15):
            self.clear_since, self.clear_count = None, 0
            self.status='PREVIOUS_ACID_UNCERTAIN'
            return
        if self.clear_since is None:
            self.clear_since = timestamp
        self.status='CHECKING_VISUAL_CLEARANCE'
        self.clear_count += 1
        if self.clear_count >= 3 and timestamp-self.clear_since >= .6:
            self.events.append(dict(timestamp=timestamp, status='VISUALLY_CLEARED',
                                    note='Visual evidence only; chemical cleanliness is not measured.'))
            self.mask[:] = 0
            self.status='VISUALLY_CLEARED'
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
            return self._reject(timestamp, 'STONE_GEOMETRY_UNCERTAIN', found)
        image = normalized(frame, quad)
        changed = changed_pixels(self.baseline, image)
        old = cv2.dilate(self.old_mask, np.ones((9,9), np.uint8)) > 0
        reasons = []
        for detection in sorted(found, key=lambda d: d['confidence'], reverse=True):
            region = footprint(detection['bbox'], quad) > 0
            area = np.count_nonzero(region)
            if area < 4:
                reasons.append('INVALID_ACID_REGION')
                continue
            overlap = np.count_nonzero(region & old)/max(1, min(area, np.count_nonzero(old)))
            fraction = float(np.mean(changed[region]))
            outside = changed[~region]
            weak = fraction < .18 or (outside.size and fraction < float(outside.mean())+.10)
            # Baseline change/history are audit observations, never blockers.
            # The independent verifier checks current visibility and persistence.
            return dict(detection, freshness='FRESHNESS_NOT_REQUIRED',
                        freshness_warning=True,
                        weak_change_warning=bool(weak),
                        previous_acid_overlap=float(overlap),
                        previous_acid_overlap_warning=bool(overlap>=.25),
                        fresh_changed_fraction=fraction), 'ACID_DETECTOR_CANDIDATE'
        reason = reasons[0]
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
