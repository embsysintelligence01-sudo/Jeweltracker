"""Still-photo rub guidance and acid targets for the existing tracker camera."""
from dataclasses import asdict
from pathlib import Path
import threading

import cv2
import numpy as np

from new_rub_detection.core import OnnxStoneModel, infer_stone
from new_rub_detection.example_pair import infer_photo
from new_rub_detection.flow import RubCheck
from new_rub_detection.geometry import align_to_reference, project_mask, stone_quad
from new_rub_detection.photo_worker import create_session


def mask_box(mask):
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    if not w or not h:
        raise RuntimeError('RETAKE - no confirmed new-rub area')
    return x, y, x+w, y+h


def clear_surface(frame, stone_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    region = stone_mask > 0
    skin = (hsv[:, :, 0] <= 14) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] > 60)
    if np.count_nonzero(skin & region) > .08 * max(1, np.count_nonzero(region)):
        raise RuntimeError('RETAKE - remove hands and jewellery from the stone, then capture again')


class RubPhotoModels:
    """Lazy window-owned models, serialized across sparse snapshot jobs."""
    def __init__(self, model_dir=None):
        self.directory = Path(model_dir or Path(__file__).parent / 'new_rub_detection/models')
        self.lock = threading.RLock()
        self.stone_model = self.rub_session = None

    def _stone(self):
        if self.stone_model is None:
            self.stone_model = OnnxStoneModel(self.directory / 'newstone1.onnx')
        return self.stone_model

    def photo(self, frame):
        with self.lock:
            stone = self._stone()
            if self.rub_session is None:
                self.rub_session = create_session(self.directory / 'rubbedmask.onnx')
            stone_mask, rubbed_mask, quad = infer_photo(frame, stone, self.rub_session)
            clear_surface(frame, stone_mask)
            return stone_mask, rubbed_mask, quad

    def surface(self, frame):
        with self.lock:
            found = [d for d in infer_stone(self._stone(), frame)
                     if (d[0][2]-d[0][0]) * (d[0][3]-d[0][1]) >= .02*frame.shape[0]*frame.shape[1]]
            area = lambda d: (d[0][2]-d[0][0]) * (d[0][3]-d[0][1])
            found.sort(key=lambda d: area(d)*d[1], reverse=True)
            if not found or (len(found) > 1 and area(found[1]) > .6*area(found[0])):
                raise RuntimeError('RETAKE - expose one complete stone')
            bbox, _, mask = found[0]
            quad = stone_quad(mask, bbox)
            if quad is None:
                raise RuntimeError('RETAKE - stone corners unclear')
            clear_surface(frame, mask)
            return mask, quad

    def close(self):
        with self.lock:
            if self.stone_model is not None:
                self.stone_model.close()
            self.stone_model = self.rub_session = None


class RubCycle:
    """One immutable-before / retryable-after comparison per jewel cycle."""
    def __init__(self, models):
        self.models = models
        self.check = RubCheck()
        self.before_result = None
        self.attempted = False

    def before(self, frame):
        self.before_result = None
        result = self.check.before(frame, *self.models.photo(frame))
        if result['status'] == 'READY':
            self.before_result = result
        return result

    def after(self, frame):
        if self.before_result is None:
            raise RuntimeError('Capture a valid before-rubbing reference first')
        stone_mask, rubbed_mask, quad = self.models.photo(frame)
        result = self.check.after(frame, stone_mask, rubbed_mask, quad)
        if result['status'] != 'NEW_RUB':
            raise RuntimeError(result['message'])
        projected = project_mask(result['new_mask'], frame.shape, result['frame_to_reference'])
        result['photo'] = draw_acid_roi(frame.copy(), projected)
        target = RubTarget(self.models, self.check.reference, self.check.reference_region,
                           result['new_mask'], frame.shape,
                           projected,
                           quad, self.before_result, result)
        return result, target


class RubTarget:
    """Confirmed mark in before-photo coordinates; relocate before acid capture."""
    def __init__(self, models, reference, region, new_mask, shape, roi_mask, quad, before, after):
        self.models = models
        self.reference, self.region = reference.copy(), region.copy()
        self.new_mask = new_mask.copy()
        self.shape = tuple(shape)
        self.roi_mask = (roi_mask > 0).astype(np.uint8)*255
        self.stone = mask_box(cv2.fillConvexPoly(np.zeros(shape[:2], np.uint8), quad.astype(np.int32), 255))
        self.streak = mask_box(self.roi_mask)
        selected = before['recommendation']
        self.metadata = dict(selected_zone=selected.name if selected else None,
                             zones=[asdict(zone) for zone in before['zones']],
                             old_mark_count=before['old_mark_count'],
                             new_mark_count=after['new_mark_count'],
                             new_coverage=after['coverage'],
                             rubbed_model='rubbedmask.onnx', stone_model='newstone1.onnx')
        self.evidence = dict(before_rubbing=before['photo'].copy(), after_rubbing=after['photo'].copy(),
                             old_rub_mask=before['old_mask'].copy(), new_rub_mask=self.new_mask,
                             after_new_rub_roi=self.roi_mask)

    def locate(self, frame):
        if tuple(frame.shape) != self.shape:
            raise RuntimeError('Camera resolution changed; capture a new rubbing reference')
        stone_mask, quad = self.models.surface(frame)
        aligned, transform, detail = align_to_reference(frame, quad, self.reference, self.region, stone_mask)
        if aligned is None:
            raise RuntimeError(f'RETAKE - {detail}; uncover the stone and click Ready again')
        roi = project_mask(self.new_mask, frame.shape, transform)
        roi[stone_mask == 0] = 0
        if np.count_nonzero(roi) < 20:
            raise RuntimeError('RETAKE - new-rub ROI cannot be located on the visible stone')
        stone = mask_box(stone_mask)
        return dict(roi_mask=roi, stone=stone, streak=mask_box(roi))


def draw_acid_roi(image, mask):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 255), 2)
    if contours:
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.putText(image, 'NEW RUB / ACID ROI', (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 0, 255), 2)
    return image
