from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

def bbox_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = bbox_area((max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)))
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union else 0.0


def inside_ratio(box: tuple[int, int, int, int], reference: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = reference
    intersection = bbox_area((max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)))
    return intersection / max(1, bbox_area(box))


def pad_bbox(
    box: tuple[int, int, int, int],
    ratio: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio
    return (
        int(np.clip(round(x1 - pad_x), 0, frame_width)),
        int(np.clip(round(y1 - pad_y), 0, frame_height)),
        int(np.clip(round(x2 + pad_x), 0, frame_width)),
        int(np.clip(round(y2 + pad_y), 0, frame_height)),
    )


@dataclass
class FlowOutput:
    state: str
    instruction: str
    display_frame_bgr: np.ndarray | None = None
    analysis_roi: tuple[int, int, int, int] | None = None
    stone_bbox: tuple[int, int, int, int] | None = None
    acid_bbox: tuple[int, int, int, int] | None = None
    reaction_bbox: tuple[int, int, int, int] | None = None
    reaction_crop_bgr: np.ndarray | None = None
    result_class: str = ""
    result_confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    classifier_comparison: dict[str, Any] = field(default_factory=dict)
    camera_processing_fps: float = 0.0
    stone_ms: float = 0.0
    acid_ms: float = 0.0
    classifier_ms: float = 0.0
    stone_counter: int = 0
    acid_counter: int = 0
    error: str = ""


class PurityFlow:
    STONE_CONFIDENCE = 0.20
    ACID_CONFIDENCE = 0.70
    STONE_MINIMUM_AREA_RATIO = 0.02
    STONE_MINIMUM_IOU = 0.80
    STONE_STABLE_FRAMES = 5
    ACID_INSIDE_STONE_RATIO = 0.50
    ACID_CONFIRM_FRAMES = 3
    REACTION_PADDING = 0.08
    CLASSIFIER_CONFIDENCE = 0.70

    def __init__(self, stone_model: Any, acid_model: Any, classifier: Any):
        self.stone_model = stone_model
        self.acid_model = acid_model
        self.classifier = classifier
        self._last_processed_at: float | None = None
        self.reset()

    def reset(self) -> None:
        self.state = "WAIT_FOR_STONE"
        self.stone_bbox = None
        self.previous_stone_bbox = None
        self.acid_bbox = None
        self.reaction_bbox = None
        self.reaction_crop_bgr = None
        self.stone_stable_counter = 0
        self.acid_counter = 0
        self.result_class = ""
        self.result_confidence = 0.0
        self.probabilities: dict[str, float] = {}
        self.classifier_comparison: dict[str, Any] = {}
        self.stone_ms = 0.0
        self.acid_ms = 0.0
        self.classifier_ms = 0.0
        self.error = ""

    def start_acid_check(self) -> bool:
        if self.state != "READY_FOR_ACID_CHECK" or self.stone_bbox is None:
            return False
        self.state = "CHECKING_ACID"
        self.acid_bbox = None
        self.reaction_bbox = None
        self.reaction_crop_bgr = None
        self.acid_counter = 0
        self.stone_ms = 0.0
        self.acid_ms = 0.0
        self.classifier_ms = 0.0
        return True

    def process(self, frame: np.ndarray) -> FlowOutput:
        started = time.monotonic()
        if self.state in {"PASS", "FAIL", "RETRY"}:
            return self.output()
        if self.state == "CHECKING_ACID":
            return self._process_acid(frame, started)
        tracking_ready_stone = self.state == "READY_FOR_ACID_CHECK"
        self.error = ""
        try:
            stone_detections, self.stone_ms = self.stone_model.predict(frame, self.STONE_CONFIDENCE)
        except Exception as exc:
            self.error = f"Stone inference failed: {exc}"
            return self.output()

        reference_area = getattr(self.stone_model, "reference_area", None)
        analysis_area = (
            reference_area(frame)
            if callable(reference_area)
            else frame.shape[0] * frame.shape[1]
        )
        minimum_area = analysis_area * self.STONE_MINIMUM_AREA_RATIO
        stones = [item for item in stone_detections if bbox_area(item.bbox) >= minimum_area]
        current_stone = max(stones, key=lambda item: bbox_area(item.bbox) * item.confidence) if stones else None
        self.stone_bbox = current_stone.bbox if current_stone else None

        if current_stone is None:
            self.state = "WAIT_FOR_STONE"
            self.stone_stable_counter = 0
            self.previous_stone_bbox = None
            self._update_fps(started)
            return self.output()

        if tracking_ready_stone:
            self.previous_stone_bbox = current_stone.bbox
            self.stone_stable_counter = self.STONE_STABLE_FRAMES
            self._update_fps(started)
            return self.output()

        self.state = "STONE_STABILIZING"
        if self.previous_stone_bbox is not None and bbox_iou(self.previous_stone_bbox, current_stone.bbox) >= self.STONE_MINIMUM_IOU:
            self.stone_stable_counter += 1
        else:
            self.stone_stable_counter = 1
        self.previous_stone_bbox = current_stone.bbox
        if self.stone_stable_counter >= self.STONE_STABLE_FRAMES:
            self.state = "READY_FOR_ACID_CHECK"
        self._update_fps(started)
        return self.output()

    def _process_acid(self, frame: np.ndarray, started: float) -> FlowOutput:
        self.error = ""
        try:
            acid_detections, self.acid_ms = self.acid_model.predict(frame, self.ACID_CONFIDENCE)
        except Exception as exc:
            self.error = f"Acid inference failed: {exc}"
            self._update_fps(started)
            return self.output()

        acids = [
            item for item in acid_detections
            if inside_ratio(item.bbox, self.stone_bbox) >= self.ACID_INSIDE_STONE_RATIO
        ]
        acid = max(acids, key=lambda item: bbox_area(item.bbox) * item.confidence) if acids else None
        if acid is None:
            self.acid_bbox = None
            self.acid_counter = 0
            self._update_fps(started)
            return self.output()

        self.acid_bbox = acid.bbox
        self.acid_counter += 1
        if self.acid_counter < self.ACID_CONFIRM_FRAMES:
            self._update_fps(started)
            return self.output()

        self.reaction_bbox = pad_bbox(
            acid.bbox,
            self.REACTION_PADDING,
            frame.shape[1],
            frame.shape[0],
        )
        x1, y1, x2, y2 = self.reaction_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self.state = "RETRY"
            self.error = "The padded acid reaction crop is empty"
            self._update_fps(started)
            return self.output()
        self.reaction_crop_bgr = crop.copy()
        try:
            name, confidence, probabilities, self.classifier_ms = self.classifier.predict(crop)
        except Exception as exc:
            self.state = "RETRY"
            self.error = f"Reaction classification failed: {exc}"
            self._update_fps(started)
            return self.output()
        self.result_class = name
        self.result_confidence = confidence
        self.probabilities = probabilities
        self.classifier_comparison = dict(
            getattr(self.classifier, "last_comparison", {})
        )
        if confidence < self.CLASSIFIER_CONFIDENCE or name == "glare":
            self.state = "RETRY"
        elif name == "gold":
            self.state = "PASS"
        else:
            self.state = "FAIL"
        self._update_fps(started)
        return self.output()

    def _update_fps(self, started: float) -> None:
        now = time.monotonic()
        if self._last_processed_at is not None and now > self._last_processed_at:
            self.processing_fps = 1.0 / (now - self._last_processed_at)
        else:
            self.processing_fps = 0.0
        self._last_processed_at = now

    def instruction(self) -> str:
        if self.error:
            return self.error
        return {
            "WAIT_FOR_STONE": "Place the touchstone fully inside the camera view.",
            "STONE_STABILIZING": "Keep the touchstone still.",
            "READY_FOR_ACID_CHECK": "Stone stable. Rub the jewellery, place acid, then press CHECK ACID.",
            "CHECKING_ACID": "Checking the acid reaction. Keep the stone still.",
            "PASS": "Result: GOLD / VALID.",
            "FAIL": "Result: NOT GOLD.",
            "RETRY": "Glare or low confidence. Repeat the test.",
        }[self.state]

    def output(self) -> FlowOutput:
        return FlowOutput(
            state=self.state,
            instruction=self.instruction(),
            stone_bbox=self.stone_bbox,
            acid_bbox=self.acid_bbox,
            reaction_bbox=self.reaction_bbox,
            reaction_crop_bgr=None if self.reaction_crop_bgr is None else self.reaction_crop_bgr.copy(),
            result_class=self.result_class,
            result_confidence=self.result_confidence,
            probabilities=dict(self.probabilities),
            classifier_comparison=dict(self.classifier_comparison),
            camera_processing_fps=getattr(self, "processing_fps", 0.0),
            stone_ms=self.stone_ms,
            acid_ms=self.acid_ms,
            classifier_ms=self.classifier_ms,
            stone_counter=self.stone_stable_counter,
            acid_counter=self.acid_counter,
            error=self.error,
        )
