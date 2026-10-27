"""Stone-relative geometry for the separate live SOP viewer."""

import cv2
import numpy as np


SIZE = 512
DEST = np.float32([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]])


def stone_quad(mask, bbox):
    """Find the four visible surface corners inside the Hailo stone ROI."""
    x1, y1, x2, y2 = map(int, bbox)
    roi = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.45 * max(1, roi.size):
        return None
    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(corners) != 4 or not cv2.isContourConvex(corners):
        return None
    if not 0.8 < cv2.contourArea(corners) / cv2.contourArea(contour) < 1.2:
        return None
    quad = corners.reshape(4, 2).astype(np.float32)
    quad += (x1, y1)
    center = quad.mean(axis=0)
    angle = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angle)]
    quad = np.roll(quad, -np.argmin(quad.sum(axis=1)), axis=0)
    if min(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)) < 80:
        return None
    return quad


def normalize(frame, quad):
    matrix = cv2.getPerspectiveTransform(quad, DEST)
    crop = cv2.warpPerspective(frame, matrix, (SIZE, SIZE))
    return crop, matrix


def stone_surface(mask, bbox):
    """Return the oriented surface and its mask, never a filled detector box.

    Use the same four-corner contour as the photo app. A small marker notch may
    prevent four-corner approximation; a tightly filled rotated rectangle is a
    fallback for that case. Uncertain geometry requires a new capture.
    """
    h, w = mask.shape
    x1, y1, x2, y2 = np.clip(bbox, (0, 0, 0, 0), (w, h, w, h)).astype(int)
    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)*255
    contours, _ = cv2.findContours(clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('RETAKE - stone surface mask unavailable')
    contour = max(contours, key=cv2.contourArea)
    component = cv2.drawContours(np.zeros_like(clipped), [contour], -1, 255, -1)
    clipped = cv2.bitwise_and(clipped, component)
    quad = stone_quad(clipped, (x1, y1, x2, y2))
    if quad is None:
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) < 80 or cv2.contourArea(contour)/max(1., rect[1][0]*rect[1][1]) < .90:
            raise RuntimeError('RETAKE - stone corners unclear; expose the complete surface')
        quad = cv2.boxPoints(rect).astype(np.float32)
        center = quad.mean(axis=0)
        quad = quad[np.argsort(np.arctan2(quad[:, 1]-center[1], quad[:, 0]-center[0]))]
        quad = np.roll(quad, -np.argmin(quad.sum(axis=1)), axis=0)
    oriented = cv2.fillConvexPoly(np.zeros_like(clipped), np.rint(quad).astype(np.int32), 255)
    return cv2.bitwise_and(clipped, oriented), quad


def _align_by_stone_image(frame, quad, reference, reference_mask, stone_mask, failure):
    """Use a unique, bounded image match when sparse stone features fail."""
    crop, initial = normalize(frame, quad)
    current_mask = cv2.warpPerspective(stone_mask, initial, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    current_mask = cv2.erode(current_mask, np.ones((11, 11), np.uint8))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    interior = np.zeros((SIZE, SIZE), np.uint8)
    interior[40:-40, 40:-40] = 255
    candidates = []
    for turns in range(4):
        valid = cv2.bitwise_and(reference_mask, np.rot90(current_mask, turns).copy())
        valid = cv2.bitwise_and(valid, interior)
        if np.count_nonzero(valid) < 0.4 * SIZE * SIZE:
            continue
        ref_values = ref_gray[valid > 0].astype(np.float32)
        cur_values = np.rot90(cur_gray, turns)[valid > 0].astype(np.float32)
        if min(ref_values.std(), cur_values.std()) < 5:
            continue
        score = float(np.corrcoef(ref_values, cur_values)[0, 1])
        candidates.append((score, turns, valid))
    candidates.sort(reverse=True)
    if len(candidates) < 2 or candidates[0][0] < 0.55 or candidates[0][0] - candidates[1][0] < 0.15:
        return None, None, failure
    _, turns, valid = candidates[0]
    rotated = np.rot90(crop, turns).copy()
    try:
        score, warp = cv2.findTransformECC(
            ref_gray, cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY),
            np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5), valid, 5)
    except cv2.error:
        return None, None, failure
    scales = np.linalg.svd(warp[:, :2])[1]
    if (score < 0.70 or np.linalg.det(warp[:, :2]) <= 0 or
            min(scales) < 0.85 or max(scales) > 1.15 or
            np.linalg.norm(warp[:, 2]) > 40):
        return None, None, failure
    matched = cv2.warpAffine(rotated, warp, (SIZE, SIZE),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    difference = ref_gray.astype(np.float32) - cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY)
    residual = np.abs(difference[valid > 0] - np.median(difference[valid > 0]))
    if np.median(residual) > 6 or np.percentile(residual, 80) > 14:
        return None, None, failure
    rotations = (
        np.eye(3, dtype=np.float32),
        np.float32([[0, 1, 0], [-1, 0, SIZE - 1], [0, 0, 1]]),
        np.float32([[-1, 0, SIZE - 1], [0, -1, SIZE - 1], [0, 0, 1]]),
        np.float32([[0, -1, SIZE - 1], [1, 0, 0], [0, 0, 1]]),
    )
    correction = np.eye(3, dtype=np.float32)
    correction[:2] = warp
    transform = np.linalg.inv(correction) @ rotations[turns] @ initial
    aligned = cv2.warpPerspective(frame, transform, (SIZE, SIZE))
    return aligned, transform, f"aligned (stone image, {score:.2f} score)"


def align_to_reference(frame, quad, reference, reference_mask, stone_mask, orb=None):
    """Return current-frame to saved-stone transform only for a unique texture match."""
    crop, initial = normalize(frame, quad)
    orb = orb or cv2.ORB_create(nfeatures=1500)
    current_mask = cv2.warpPerspective(stone_mask, initial, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    current_mask = cv2.erode(current_mask, np.ones((11, 11), np.uint8))
    ref_points, ref_desc = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), reference_mask)
    cur_points, cur_desc = orb.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), current_mask)
    if ref_desc is None or cur_desc is None or min(len(ref_points), len(cur_points)) < 20:
        return _align_by_stone_image(frame, quad, reference, reference_mask,
                                     stone_mask, "stone texture insufficient")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(cur_desc, ref_desc, k=2)
    good = [a for a, b in pairs if a.distance < 0.70 * b.distance]
    if len(good) < 18:
        return _align_by_stone_image(frame, quad, reference, reference_mask,
                                     stone_mask, "stone orientation ambiguous")
    source = np.float32([cur_points[m.queryIdx].pt for m in good])
    target = np.float32([ref_points[m.trainIdx].pt for m in good])
    correction, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if correction is None or inliers is None or int(inliers.sum()) < 15 or inliers.mean() < 0.60:
        return _align_by_stone_image(frame, quad, reference, reference_mask,
                                     stone_mask, "stone alignment failed")
    projected = cv2.perspectiveTransform(DEST[None], correction)[0]
    if not np.isfinite(projected).all() or np.any(projected < -80) or np.any(projected > SIZE + 80):
        return _align_by_stone_image(frame, quad, reference, reference_mask,
                                     stone_mask, "stone alignment outside surface")
    area = abs(cv2.contourArea(projected))
    if not 0.55 * SIZE * SIZE < area < 1.50 * SIZE * SIZE:
        return _align_by_stone_image(frame, quad, reference, reference_mask,
                                     stone_mask, "stone alignment scale invalid")
    if cv2.contourArea(projected, oriented=True) * cv2.contourArea(DEST, oriented=True) <= 0:
        return _align_by_stone_image(frame, quad, reference, reference_mask,
                                     stone_mask, "stone alignment mirrored")
    transform = correction @ initial
    aligned = cv2.warpPerspective(frame, transform, (SIZE, SIZE))
    return aligned, transform, f"aligned ({int(inliers.sum())} matches)"


def tag_stable(corners, baseline, max_corner_motion=15.0):
    return float(np.max(np.linalg.norm(corners - baseline, axis=1))) <= max_corner_motion


def project_mask(mask, frame_shape, frame_to_stone):
    inverse = np.linalg.inv(frame_to_stone)
    return cv2.warpPerspective(mask, inverse, (frame_shape[1], frame_shape[0]),
                               flags=cv2.INTER_NEAREST)
