"""Still-photo rub guidance and acid targets for the existing tracker camera."""
from pathlib import Path
import threading

import cv2
import numpy as np

from new_rub_detection.flow import RubCheck
from new_rub_detection.geometry import project_mask, stone_surface
from new_rub_detection.photo_worker import create_session, detect_mask
from stone_onnx import StoneOnnxModel
from auto_testing import StoneSettling
from int8_models import RUBBED_MODEL, RubbedMarkDetector


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
    """Shared INT8 stone and lazy INT8 mark detector for sparse captures."""
    mark_evidence_kind = RubbedMarkDetector.evidence_kind

    def __init__(self, model_dir=None):
        self.directory = Path(model_dir or Path(__file__).parent / 'new_rub_detection/models')
        self.rub_model_path = self.directory / 'rubbedmask.onnx' if model_dir is not None else RUBBED_MODEL
        if model_dir is not None:
            self.mark_evidence_kind = 'SEGMENTATION'
        self.lock = threading.RLock()
        self.stone_model = self.rub_session = None
        self.roi = None

    def use_stone_model(self, model, roi=None):
        with self.lock:
            self.stone_model = model
            self.roi = None if roi is None else roi.copy()

    def _stone(self):
        if self.stone_model is None:
            self.stone_model = StoneOnnxModel(Path(__file__).parent / 'INT8_openvino_model/best.xml')
        return self.stone_model

    def photo(self, frame):
        with self.lock:
            stone_mask, quad = self.surface(frame)
            return stone_mask, self.marks(frame), quad

    def marks(self, frame):
        return self.mark_observation(frame)[0]

    def mark_observation(self, frame):
        """Return candidate rectangles with their frame, using one inference."""
        with self.lock:
            if self.rub_session is None:
                self.rub_session = (RubbedMarkDetector(self.rub_model_path) if self.rub_model_path.suffix == '.xml'
                                    else create_session(self.rub_model_path))
            if isinstance(self.rub_session, RubbedMarkDetector):
                return self.rub_session.regions_with_boxes(frame)
            mask=detect_mask(frame, self.rub_session)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            boxes=[]
            for contour in contours:
                x,y,w,h=cv2.boundingRect(contour)
                boxes.append((x,y,x+w,y+h))
            return mask, boxes

    def surface(self, frame):
        with self.lock:
            result = self._stone().predict(frame, conf=.20, return_masks=True)[0]
            found = []
            if result.boxes is not None and result.masks is not None:
                for bbox, confidence, mask in zip(result.boxes.xyxy.cpu().numpy(),
                                                  result.boxes.conf.cpu().numpy(),
                                                  result.masks.data.cpu().numpy()):
                    if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) < .02*frame.shape[0]*frame.shape[1]:
                        continue
                    center = tuple(map(float, (bbox[:2]+bbox[2:])/2))
                    if self.roi is not None and cv2.pointPolygonTest(self.roi, center, False) < 0:
                        continue
                    found.append((float(confidence), bbox, mask))
            if not found:
                raise RuntimeError('RETAKE - expose one complete stone')
            _, bbox, mask = max(found, key=lambda item: item[0])
            mask, quad = stone_surface(mask, bbox)
            clear_surface(frame, mask)
            return mask, quad

    def close(self):
        with self.lock:
            # The live worker can still own the shared compiled stone model.
            self.stone_model = self.rub_session = None


class RubCycle:
    """One before/after image comparison per jewel, without marker detection."""
    def __init__(self, models):
        self.models = models
        self.check = RubCheck()
        self.before_result = None
        self.attempted = False
        self.stone = None

    def before(self, frame, surface=None):
        if surface is None:
            stone, marks, quad = self.models.photo(frame)
        else:
            stone, quad = surface
            marks = self.models.marks(frame)
        check = RubCheck()
        result = check.before(frame, stone, marks, quad)
        # Keep a validated reference through a failed capture/model call.
        self.check = check
        self.before_result = result if result['status'] == 'READY' else None
        self.stone = mask_box(stone) if self.before_result is not None else None
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
        target = RubTarget(result['new_mask'], frame.shape, projected, quad,
                           self.before_result, result)
        target.metadata['rubbed_model'] = str(vars(self.models).get('rub_model_path', RUBBED_MODEL))
        target.metadata['rubbed_region_kind'] = vars(self.models).get('mark_evidence_kind',
            getattr(type(self.models), 'mark_evidence_kind', 'SEGMENTATION'))
        return result, target


class StoneCleanup:
    """Wait for repeated clear, quiet photos before saving the next reference.

    Runs on the photo executor, only between completed acid tests and the next
    before-rubbing reference. Missing surface observations never mean clean.
    """
    def __init__(self, models, acid_models):
        self.models, self.acid_models = models, acid_models
        self.gate = None
        self.clear_frames = 0
        self.last_ts = None
        self.reference_message = None

    def waiting(self, message):
        self.gate = None
        self.clear_frames = 0
        return dict(status='WAIT_FOR_CLEANUP', message=message)

    def before(self, cycle, frame, timestamp):
        if self.last_ts is not None and timestamp <= self.last_ts:
            return dict(status='WAIT_FOR_CLEANUP', message='Waiting for fresh camera frames after cleaning.')
        gap = 0. if self.last_ts is None else timestamp-self.last_ts
        self.last_ts = timestamp
        if gap > 1.5:
            self.gate, self.clear_frames = None, 0
        try:
            stone, quad = self.models.surface(frame)
            # Check the full oriented surface too: a segmentation can leave a
            # hole around a hand or cleaning cloth instead of hiding the stone.
            whole = cv2.fillConvexPoly(np.zeros_like(stone), np.rint(quad).astype(np.int32), 255)
            clear_surface(frame, whole)
            if np.count_nonzero(stone) < .90*np.count_nonzero(whole):
                return self.waiting('Finish cleaning and uncover the complete stone. The previous result is saved.')
        except RuntimeError:
            return self.waiting('Finish cleaning and uncover the complete stone. The previous result is saved.')

        detector, _ = self.acid_models.get()
        detections, _ = detector.predict(frame, .60)
        h, w = stone.shape
        for detection in detections:
            if detection.confidence < .60:
                continue
            x1, y1, x2, y2 = np.clip(detection.bbox, (0, 0, 0, 0), (w, h, w, h)).astype(int)
            if np.any(whole[y1:y2, x1:x2]):
                return self.waiting('Acid is still detected on the stone. Clean and dry it before the next jewel.')

        bbox = mask_box(whole)
        if self.gate is not None:
            previous = np.asarray(self.gate.stone)
            # Large relocation starts a new fixed crop; normal box jitter does
            # not move the crop used for the two-second image comparison.
            shift = np.linalg.norm((np.asarray(bbox[:2])+bbox[2:]-previous[:2]-previous[2:])/2)
            if shift > .15*max(previous[2:]-previous[:2]):
                self.gate, self.clear_frames = None, 0
        if self.gate is None:
            self.gate = StoneSettling(bbox, gap_grace=1.5)
        self.clear_frames += 1
        settled = self.gate.update(frame, timestamp)
        if not settled or self.clear_frames < 3:
            remaining = max(0., 2.-self.gate.elapsed)
            return dict(status='WAIT_FOR_CLEANUP', message=self.reference_message or (
                f'No acid detected. Keep the uncovered, dry stone still: {remaining:.1f}s before its next reference.'))

        # Capture marks from the very same clear photo, avoiding a later frame
        # that could contain the next hand or jewel. Reuse its validated mask.
        try:
            cycle.attempted = True
            result = cycle.before(frame, surface=(stone, quad))
            self.reference_message = None if result['status'] == 'READY' else result['message']
            return result
        finally:
            self.gate, self.clear_frames = None, 0


class RubTarget:
    """Freeze the confirmed new-rub region through Ready and acid capture."""
    def __init__(self, new_mask, shape, roi_mask, quad, before, after):
        self.new_mask = new_mask.copy()
        self.shape = tuple(shape)
        self.roi_mask = (roi_mask > 0).astype(np.uint8)*255
        self.stone = mask_box(cv2.fillConvexPoly(np.zeros(shape[:2], np.uint8), quad.astype(np.int32), 255))
        self.streak = mask_box(self.roi_mask)
        self.metadata = dict(reference_mode='before_after', reject_old_mark_overlap=True,
                             old_mark_count=before['old_mark_count'],
                             new_mark_count=after['new_mark_count'],
                             new_coverage=after['coverage'],
                             rubbed_model=str(RUBBED_MODEL), stone_model='INT8_openvino_model/best.xml',
                             rubbed_region_kind=RubbedMarkDetector.evidence_kind,
                             stone_geometry='mask_derived_oriented_quad',
                             acid_region_mode='fixed_after_new_rub')
        self.evidence = dict(before_rubbing=before['photo'].copy(), after_rubbing=after['photo'].copy(),
                             old_rub_mask=before['old_mask'].copy(), new_rub_mask=self.new_mask,
                             after_new_rub_roi=self.roi_mask)

    def locate(self, frame):
        # The operator keeps the stone still after new-rub confirmation. A hand
        # hiding the live detector must not replace or invalidate this region.
        if tuple(frame.shape) != self.shape:
            raise RuntimeError('Camera resolution changed; capture a new rubbing reference')
        return dict(roi_mask=self.roi_mask.copy(), stone=self.stone, streak=self.streak)


def render_rub_preview(result, show_existing_marks=False):
    """Render optional old-mark annotations without changing detection or evidence."""
    image = result.get('preview')
    if image is None:
        return None
    image = (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy())
    old_mask = result.get('preview_old_mask')
    if show_existing_marks and old_mask is not None:
        old = old_mask > 0
        image[old] = (0.5 * image[old] + 0.5 * np.array((0, 255, 0))).astype(np.uint8)
    return image


def draw_acid_roi(image, mask):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        cv2.rectangle(image,(x,y),(x+w-1,y+h-1),(255,0,255),2)
    if contours:
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.putText(image, 'NEW RUB DETECTION', (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 0, 255), 2)
    return image
