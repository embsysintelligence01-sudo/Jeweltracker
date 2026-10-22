from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

ANALYZER_VERSION = "temporal_feature_retention_v2"
ANALYSIS_SIZE = (448, 448)
REACTION_RADIUS = 150
MIN_BASELINE_FEATURES = 50
MAX_BASELINE_GLARE_COMPONENT_PIXELS = 50
MIN_SCENE_CHANGE = 0.008
GOLD_RETENTION = 0.10
NON_GOLD_RETENTION = 0.08


@dataclass(frozen=True)
class FeatureRetentionMetrics:
    analyzer_version: str
    baseline_feature_points: int
    early_feature_matches: int
    final_feature_matches: int
    early_feature_retention: float
    final_feature_retention: float
    final_change: float
    baseline_glare_fraction: float
    baseline_glare_max_component_pixels: int
    quality: str
    suggestion: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _analysis_frame(frames_bgr: list[np.ndarray]) -> np.ndarray:
    resized = [
        cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        for frame in frames_bgr
    ]
    return np.median(np.stack(resized), axis=0).astype(np.uint8)


def _reaction_mask() -> np.ndarray:
    mask = np.zeros(ANALYSIS_SIZE, dtype=np.uint8)
    center = (ANALYSIS_SIZE[0] // 2, ANALYSIS_SIZE[1] // 2)
    cv2.circle(mask, center, REACTION_RADIUS, 255, -1)
    return mask


def _baseline_glare(baseline_bgr: np.ndarray) -> tuple[float, int]:
    hsv = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2HSV)
    glare = ((hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 55)).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(glare, 8)
    maximum_component = (
        int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
    )
    return float(np.mean(glare)), maximum_component


def _feature_matches(
    baseline_keypoints,
    baseline_descriptors: np.ndarray | None,
    target_bgr: np.ndarray,
    mask: np.ndarray,
    sift,
) -> int:
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, mask)
    if (
        baseline_descriptors is None
        or target_descriptors is None
        or len(target_descriptors) < 2
    ):
        return 0

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        baseline_descriptors, target_descriptors, k=2
    )
    retained = 0
    for candidates in matches:
        if len(candidates) != 2:
            continue
        best, second = candidates
        if best.distance >= 0.75 * second.distance:
            continue
        baseline_point = baseline_keypoints[best.queryIdx].pt
        target_point = target_keypoints[best.trainIdx].pt
        displacement = np.hypot(
            baseline_point[0] - target_point[0],
            baseline_point[1] - target_point[1],
        )
        if displacement < 30.0:
            retained += 1
    return retained


def _decision(
    *,
    baseline_features: int,
    early_retention: float,
    final_retention: float,
    scene_change: float,
    glare_component_pixels: int,
) -> tuple[str, str]:
    if glare_component_pixels >= MAX_BASELINE_GLARE_COMPONENT_PIXELS:
        return "excessive_baseline_glare", "inconclusive"
    if baseline_features < MIN_BASELINE_FEATURES:
        return "insufficient_baseline_streak", "inconclusive"
    if scene_change < MIN_SCENE_CHANGE:
        return "insufficient_reaction_change", "inconclusive"

    retention = min(early_retention, final_retention)
    if retention >= GOLD_RETENTION:
        return "usable", "gold_22k"
    if retention <= NON_GOLD_RETENTION:
        return "usable", "non_gold"
    return "borderline", "inconclusive"


def analyze_sequence_v2(
    baseline_frames_bgr: list[np.ndarray],
    observation_frames_bgr: list[np.ndarray],
    transition_frames_bgr: list[np.ndarray] | None = None,
) -> FeatureRetentionMetrics:
    """Return conservative feature-retention metrics for development use."""
    if not baseline_frames_bgr or not observation_frames_bgr:
        raise ValueError("Baseline and observation frames are required")

    baseline_bgr = _analysis_frame(baseline_frames_bgr)
    final_bgr = _analysis_frame(observation_frames_bgr[-5:])
    transition_frames_bgr = transition_frames_bgr or []
    early_bgr = (
        _analysis_frame(transition_frames_bgr[-3:])
        if transition_frames_bgr
        else final_bgr
    )

    mask = _reaction_mask()
    sift = cv2.SIFT_create(
        nfeatures=300, contrastThreshold=0.02, edgeThreshold=15
    )
    baseline_gray = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    baseline_keypoints, baseline_descriptors = sift.detectAndCompute(
        baseline_gray, mask
    )
    baseline_features = len(baseline_keypoints)
    early_matches = _feature_matches(
        baseline_keypoints, baseline_descriptors, early_bgr, mask, sift
    )
    final_matches = _feature_matches(
        baseline_keypoints, baseline_descriptors, final_bgr, mask, sift
    )
    early_retention = early_matches / max(baseline_features, 1)
    final_retention = final_matches / max(baseline_features, 1)

    final_gray = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY)
    final_change = float(
        np.mean(cv2.absdiff(baseline_gray, final_gray)) / 255.0
    )
    glare_fraction, glare_component = _baseline_glare(baseline_bgr)
    quality, suggestion = _decision(
        baseline_features=baseline_features,
        early_retention=early_retention,
        final_retention=final_retention,
        scene_change=final_change,
        glare_component_pixels=glare_component,
    )

    return FeatureRetentionMetrics(
        analyzer_version=ANALYZER_VERSION,
        baseline_feature_points=baseline_features,
        early_feature_matches=early_matches,
        final_feature_matches=final_matches,
        early_feature_retention=early_retention,
        final_feature_retention=final_retention,
        final_change=final_change,
        baseline_glare_fraction=glare_fraction,
        baseline_glare_max_component_pixels=glare_component,
        quality=quality,
        suggestion=suggestion,
    )
