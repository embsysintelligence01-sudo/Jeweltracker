from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


ANALYZER_VERSION = "temporal_contrast_v1"
ANALYSIS_SIZE = (224, 224)


@dataclass(frozen=True)
class TemporalMetrics:
    analyzer_version: str
    baseline_streak_pixels: int
    baseline_contrast: float
    final_contrast: float
    early_reaction_retention: float
    contrast_retention: float
    final_change: float
    maximum_glare_fraction: float
    quality: str
    suggestion: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Temporal analysis received an empty image")
    resized = cv2.resize(image_bgr, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)


def median_frame(frames_bgr: list[np.ndarray]) -> np.ndarray:
    if not frames_bgr:
        raise ValueError("At least one frame is required")
    resized = [
        cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        for frame in frames_bgr
    ]
    return np.median(np.stack(resized), axis=0).astype(np.uint8)


def _streak_mask(baseline_gray: np.ndarray) -> np.ndarray:
    background = cv2.GaussianBlur(baseline_gray, (0, 0), 7.0)
    local_contrast = baseline_gray - background
    threshold = max(7.0, float(np.percentile(local_contrast, 88)))
    mask = (local_contrast >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask.astype(bool)


def _masked_contrast(gray: np.ndarray, streak_mask: np.ndarray) -> float:
    if not np.any(streak_mask):
        return 0.0
    ring = cv2.dilate(streak_mask.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    ring &= ~streak_mask
    if not np.any(ring):
        return 0.0
    return float(abs(np.median(gray[streak_mask]) - np.median(gray[ring])))


def analyze_sequence(
    baseline_frames_bgr: list[np.ndarray],
    observation_frames_bgr: list[np.ndarray],
    transition_frames_bgr: list[np.ndarray] | None = None,
) -> TemporalMetrics:
    """Return development metrics, not a production gold verdict."""
    if not baseline_frames_bgr or not observation_frames_bgr:
        raise ValueError("Baseline and observation frames are required")

    baseline_bgr = median_frame(baseline_frames_bgr)
    baseline_gray = _gray(baseline_bgr)
    streak_mask = _streak_mask(baseline_gray)
    streak_pixels = int(np.count_nonzero(streak_mask))
    baseline_contrast = _masked_contrast(baseline_gray, streak_mask)

    transition_frames_bgr = transition_frames_bgr or []
    if transition_frames_bgr:
        transition_tail = median_frame(
            transition_frames_bgr[-min(3, len(transition_frames_bgr)):]
        )
        early_contrast = _masked_contrast(_gray(transition_tail), streak_mask)
    else:
        early_contrast = 0.0

    tail_count = max(1, min(5, len(observation_frames_bgr)))
    final_bgr = median_frame(observation_frames_bgr[-tail_count:])
    final_gray = _gray(final_bgr)
    final_contrast = _masked_contrast(final_gray, streak_mask)
    retention = final_contrast / max(baseline_contrast, 1.0)
    early_retention = (
        early_contrast / max(baseline_contrast, 1.0)
        if transition_frames_bgr
        else retention
    )
    final_change = float(np.mean(cv2.absdiff(baseline_gray, final_gray)) / 255.0)

    glare_fractions = []
    for frame in baseline_frames_bgr + observation_frames_bgr:
        resized = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        glare = (hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 55)
        glare_fractions.append(float(np.mean(glare)))
    maximum_glare = max(glare_fractions, default=0.0)

    if streak_pixels < 35 or baseline_contrast < 5.0:
        quality = "insufficient_baseline_streak"
        suggestion = "inconclusive"
    elif maximum_glare > 0.08:
        quality = "excessive_glare"
        suggestion = "inconclusive"
    elif min(early_retention, retention) >= 0.78:
        quality = "usable"
        suggestion = "gold_22k"
    elif min(early_retention, retention) <= 0.70:
        quality = "usable"
        suggestion = "non_gold"
    else:
        quality = "borderline"
        suggestion = "inconclusive"

    return TemporalMetrics(
        analyzer_version=ANALYZER_VERSION,
        baseline_streak_pixels=streak_pixels,
        baseline_contrast=baseline_contrast,
        final_contrast=final_contrast,
        early_reaction_retention=early_retention,
        contrast_retention=retention,
        final_change=final_change,
        maximum_glare_fraction=maximum_glare,
        quality=quality,
        suggestion=suggestion,
    )
