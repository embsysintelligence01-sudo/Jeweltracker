"""Measure persistent model-supported areas independently of application timing.

These are visual estimates, not chemical segmentation. This class
cannot confirm an application or change a reaction timer.
"""
import copy
import cv2
import numpy as np

from acid_freshness import normalized, footprint, changed_pixels, DEST
from acid_application import AcidApplicationVerifier


def region_box(mask, quad, shape):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    points = np.float32([[[xs.min(), ys.min()], [xs.max(), ys.min()],
                         [xs.max(), ys.max()], [xs.min(), ys.max()]]])
    points = cv2.perspectiveTransform(points, cv2.getPerspectiveTransform(
        DEST, np.asarray(quad, np.float32)))[0]
    lo = np.maximum(0, np.floor(points.min(0)).astype(int))
    hi = np.minimum([shape[1], shape[0]], np.ceil(points.max(0)).astype(int))
    return tuple(map(int, [*lo, *hi]))


class AcidAreaMeasurement:
    def __init__(self, baseline, old_mask, onset, detection, quad):
        self.baseline = baseline.copy()
        self.old_mask = old_mask.copy()
        self.old = cv2.dilate(old_mask, np.ones((9, 9), np.uint8)) > 0
        self.onset = onset
        self.anchor = footprint(detection['bbox'], quad) > 0
        self.mask = np.zeros((256, 256), np.uint8)
        self.previous = None
        self.last_ts = -float('inf')
        self.accepted_frames = 0
        self.samples = 0
        self.updated_at = None
        self.source_frame = None
        self.source_quad = None
        self.rejections = {}

    def _reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0)+1
        return False

    def fork(self):
        # Images, geometry and prior observations are immutable. Copy only
        # mutable accumulation state before handing work to another thread.
        result=copy.copy(self)
        result.mask=self.mask.copy()
        result.rejections=dict(self.rejections)
        return result

    def observe(self, frame, quad, found, timestamp, clear=True, geometry_known=True):
        if timestamp <= self.last_ts or not self.onset <= timestamp <= self.onset+5:
            return False
        self.last_ts = timestamp
        self.samples += 1
        if not clear or not geometry_known or quad is None:
            self.previous = None
            return self._reject('VISIBILITY_OR_GEOMETRY_UNCERTAIN')
        image = normalized(frame, quad)
        changed = changed_pixels(self.baseline, image)
        if changed.mean() > .3:
            self.previous = None
            return self._reject('BROAD_SURFACE_CHANGE')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Timing has already been confirmed. A dark, elongated wet streak must
        # not be rejected as a silver needle just because both are narrow.
        brighter = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)
                    - cv2.cvtColor(self.baseline, cv2.COLOR_BGR2GRAY).astype(np.int16)) > 20
        skin = (hsv[:, :, 0] <= 14) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] > 80)
        light = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 150)
        nearby = cv2.dilate((self.anchor | (self.mask > 0)).astype(np.uint8),
                            np.ones((31, 31), np.uint8)) > 0
        # Application timing/freshness was already verified. History belongs
        # to the pre-rubbing stone check, not to visible-area measurement.
        new_pixels = changed & ~skin & ~light
        new_pixels[:5]=False;new_pixels[-5:]=False
        new_pixels[:,:5]=False;new_pixels[:,-5:]=False
        _,components,stats,_=cv2.connectedComponentsWithStats(new_pixels.astype(np.uint8),8)
        region = np.zeros((256, 256), bool)
        for detection in sorted(found, key=lambda d: d.get('confidence', 0), reverse=True)[:8]:
            candidate = footprint(detection['bbox'], quad) > 0
            candidate[:5] = False; candidate[-5:] = False
            candidate[:, :5] = False; candidate[:, -5:] = False
            fresh = candidate
            reason=None
            if fresh.sum() < 5:reason='AREA_TOO_SMALL'
            elif not np.any(fresh & nearby):reason='UNRELATED_LOCATION'
            elif skin[fresh].mean() > .2 or light[fresh].mean() > .8:reason='HAND_OR_GLARE'
            elif AcidApplicationVerifier._tool_like(changed & brighter, fresh):reason='BRIGHT_APPLICATOR_LIKE_CHANGE'
            if reason:
                self._reject(reason)
                continue
            region |= fresh
            # Grow beyond a partial detector box only through genuinely changed
            # pixels connected to its supported patch. Unconnected changes and
            # unchanged rubbed marks cannot enlarge the acid estimate.
            for component in np.unique(components[fresh & new_pixels]):
                if component and stats[component,cv2.CC_STAT_AREA] <= .15*256*256:
                    region |= components == component
        if region.mean() > .25:
            self.previous=None
            return self._reject('AREA_TOO_BROAD_TO_LOCALIZE')
        if not np.any(region):
            self.previous=None
            return self._reject('NO_CLEAR_DETECTOR_REGION_NEAR_APPLICATION')
        previous, self.previous = self.previous, (timestamp, image, region)
        if previous is None or not .03 <= timestamp-previous[0] <= 1.25:
            return self._reject('WAITING_FOR_SUCCESSIVE_CLEAR_AREA_FRAME')
        union = region | previous[2]
        intersection = region & previous[2]
        if (intersection.sum()/max(1, union.sum()) < .65
                or not np.any(union)
                or changed_pixels(previous[1], image)[union].mean() > .12):
            return self._reject('AREA_NOT_PERSISTENT_OR_STILL_MOVING')
        self.mask[union] = 255
        self.updated_at = timestamp
        self.source_frame = frame
        self.source_quad = np.asarray(quad,np.float32).copy()
        self.accepted_frames += 1
        return True


def compare_area(area, rubbed):
    """Compare supported footprints; never interpret missing area as a miss."""
    if area is None or not np.any(area) or rubbed is None or not np.any(rubbed):
        return dict(acid_target_status='COMPARISON_UNAVAILABLE', acid_new_mask_overlap=None)
    acid, rub = area > 0, rubbed > 0
    hit = np.count_nonzero(acid & rub)
    overlap = hit/max(1, min(acid.sum(), rub.sum()))
    status = ('ACID_AT_DIFFERENT_LOCATION' if hit == 0 else
              'ACID_OVERLAPS_NEW_RUB' if hit/rub.sum() >= .8 else 'ACID_PARTIALLY_OVERLAPS_NEW_RUB')
    return dict(acid_target_status=status, acid_new_mask_overlap=float(overlap),
                acid_estimated_rub_coverage=float(hit/rub.sum()),
                acid_estimated_area_on_rub=float(hit/acid.sum()))
