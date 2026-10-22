"""
Still-capture jewel crops and mask, ported from SOP-monitoring-rpi integrated_ui_app.py
(_threshold_jewelry_mask, _score_threshold_jewelry_mask, build_shared_jewelry_mask,
_component_items_from_mask, analyze_pledge_jewel_count_capture) and Weight-Scale-SOP
backend/app/analysis/vision.py (build_jewellery_mask, separate_jewellery_items).

The jewellery area is cropped, everything to ignore (outside the ROI, the touchstone) is painted white
and erased from the mask, and an Otsu-based mask is built. The jewels themselves are counted by YOLO;
each YOLO jewel is cut out from that mask onto pure white - the input the SigLIP classifier expects
(component_items_from_mask is the repos' Otsu count, kept for reference).
"""
import cv2
import numpy as np

COUNT_CAPTURE_MIN_AREA_RATIO = 0.00045  # as in integrated_ui_app.py
COUNT_CAPTURE_MIN_AREA_PX = 350
COUNT_CAPTURE_PADDING_PX = 14
ERASE_DILATE_PX = 25                    # erase regions are grown like the repo's ArUco erase mask
STONE_ERASE_PAD_PX = 40                 # the stone's edge and shadow must not become a "jewel"
RING_MIN_HOLE_RATIO = 0.5               # closed loop: interior hole / outline area
RING_MIN_ELLIPSE_IOU = 0.92             # closed loop: outline close to an ellipse


def threshold_jewelry_mask(image_bgr, bg_is_white, threshold_value=220):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if bg_is_white:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, fixed = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        thresholded = cv2.bitwise_or(otsu, fixed)
    else:
        thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 10)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
    thresholded = cv2.bitwise_or(thresholded, edges)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colored = ((saturation > 45) & (value > 28) & (((hue >= 5) & (hue <= 110)) | (hue >= 125))).astype(np.uint8) * 255
    colored = cv2.dilate(colored, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    thresholded = cv2.bitwise_or(thresholded, colored)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (thresholded > 0).astype(np.uint8)


def score_threshold_jewelry_mask(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    mask_ratio = float(mask_u8.mean()) if mask_u8.size else 1.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    image_h, image_w = mask_u8.shape[:2]
    min_component_area = max(80, int(mask_u8.size * 0.0005))
    useful, largest_border, largest_inner = 0, 0, 0
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        x, y, w, h = (int(stats[idx, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                    cv2.CC_STAT_HEIGHT))
        if x <= 1 or y <= 1 or x + w >= image_w - 1 or y + h >= image_h - 1:
            largest_border = max(largest_border, area)
        else:
            useful += 1
            largest_inner = max(largest_inner, area)
    if useful == 0:
        return -1_000_000.0
    score = float(largest_inner) + useful * 250.0
    if mask_ratio < 0.002:
        score -= 100_000.0
    if mask_ratio > 0.45:
        score -= mask_ratio * mask_u8.size
    if largest_border > largest_inner:
        score -= largest_border * 0.75
    return score


def build_shared_jewelry_mask(image_bgr, erase_mask=None):
    corners = [image_bgr[0, 0], image_bgr[0, -1], image_bgr[-1, 0], image_bgr[-1, -1]]
    preferred_white = float(np.mean([c.mean() for c in corners])) > 180.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidates = []
    for bg_is_white in (preferred_white, not preferred_white):
        candidate = threshold_jewelry_mask(image_bgr, bg_is_white)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidates.append((score_threshold_jewelry_mask(candidate), candidate))
    mask = max(candidates, key=lambda item: item[0])[1].astype(np.uint8)
    if erase_mask is not None and np.any(erase_mask):
        erase = cv2.dilate((erase_mask > 0).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERASE_DILATE_PX, ERASE_DILATE_PX)))
        mask[erase > 0] = 0
    prepared = np.full_like(image_bgr, 255)
    prepared[mask > 0] = image_bgr[mask > 0]
    return prepared, mask


def component_items_from_mask(image_bgr, mask, min_area_px, pad_px):
    binary = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape[:2]
    items = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x, y, w, h = (int(stats[idx, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                    cv2.CC_STAT_HEIGHT))
        if w < 10 or h < 10:
            continue
        if max(w, h) / max(1, min(w, h)) > 10.0 and area < min_area_px * 6:
            continue
        if area / max(1, w * h) < 0.01 and area < min_area_px * 10:
            continue
        x1, y1 = max(0, x - pad_px), max(0, y - pad_px)
        x2, y2 = min(width, x + w + pad_px), min(height, y + h + pad_px)
        component = labels[y1:y2, x1:x2] == idx
        crop = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
        crop[component] = image_bgr[y1:y2, x1:x2][component]
        items.append({"bbox": (x1, y1, x2, y2), "area_px": area, "crop_bgr": crop,
                      "mask": component.astype(np.uint8)})
    items.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))  # reading order, as in the repos
    return items


def jewellery_mask(frame_bgr, roi_poly, stone_box=None):
    """The repo's Otsu-based jewellery mask over the ROI (outside the ROI and the touchstone erased), as a
    full-frame mask. Used to cut each YOLO-counted jewel out on white, not for counting."""
    h, w = frame_bgr.shape[:2]
    if roi_poly is None:
        rx, ry, rw, rh = 0, 0, w, h
        inside = np.full((h, w), 255, np.uint8)
    else:
        poly = roi_poly.reshape(-1, 2).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)  # a ROI drawn on a larger image
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        rx, ry, rw, rh = cv2.boundingRect(poly)
        inside = np.zeros((rh, rw), np.uint8)
        cv2.fillPoly(inside, [poly - (rx, ry)], 255)
    working = frame_bgr[ry:ry + rh, rx:rx + rw].copy()
    erase = (inside == 0).astype(np.uint8)
    if stone_box is not None:
        x1, y1, x2, y2 = (int(v) for v in stone_box)
        p = STONE_ERASE_PAD_PX
        erase[max(0, y1 - p - ry):max(0, y2 + p - ry), max(0, x1 - p - rx):max(0, x2 + p - rx)] = 1
    working[erase > 0] = 255
    _, mask = build_shared_jewelry_mask(working, erase_mask=erase)
    full = np.zeros((h, w), np.uint8)
    full[ry:ry + rh, rx:rx + rw] = mask
    return full


def jewel_crop(frame_bgr, otsu_mask, det_mask, det_origin, pad=COUNT_CAPTURE_PADDING_PX, grow=15):
    """One YOLO jewel on pure white: the Otsu pixels inside its detection mask (grown by `grow` px for thin
    strands YOLO's mask misses), padded like the repo's item crops. Falls back to the YOLO mask.
    Returns (crop_bgr, crop_mask, (x1, y1, x2, y2))."""
    h, w = frame_bgr.shape[:2]
    region = np.zeros((h, w), np.uint8)
    ox, oy = (int(v) for v in det_origin)
    mh, mw = det_mask.shape
    region[oy:oy + mh, ox:ox + mw] = det_mask[:h - oy, :w - ox] > 0
    region = cv2.dilate(region, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    keep = (otsu_mask > 0) & (region > 0)
    if np.count_nonzero(keep) < 50:
        keep = region > 0
    ys, xs = np.nonzero(keep)
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = np.full((y2 - y1, x2 - x1, 3), 255, np.uint8)
    sub = keep[y1:y2, x1:x2]
    crop[sub] = frame_bgr[y1:y2, x1:x2][sub]
    return crop, sub.astype(np.uint8), (x1, y1, x2, y2)


def ring_shape(mask):
    """Is the jewel a closed loop (bangle / finger ring)? Measured on real pieces: bangles have one interior
    hole of 0.73-0.82 of their outline area and an outline that fits an ellipse (IoU 0.98); chains, necklaces
    and small pieces 0.00-0.16 and 0.55-0.87. Returns (is_ring, outer diameter in px)."""
    m = cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return False, 0.0
    outers = [i for i, hh in enumerate(hierarchy[0]) if hh[3] < 0]
    o = max(outers, key=lambda i: cv2.contourArea(contours[i]))
    outer_area = cv2.contourArea(contours[o])
    if len(contours[o]) < 5 or outer_area <= 0:
        return False, 0.0
    holes = [cv2.contourArea(contours[i]) for i, hh in enumerate(hierarchy[0]) if hh[3] == o]
    big_holes = [a for a in holes if a > 0.02 * outer_area]
    ellipse = cv2.fitEllipse(contours[o])
    fitted, filled = np.zeros_like(m), np.zeros_like(m)
    cv2.ellipse(fitted, ellipse, 255, -1)
    cv2.drawContours(filled, contours, o, 255, -1)
    iou = np.count_nonzero(fitted & filled) / max(1, np.count_nonzero(fitted | filled))
    is_ring = (len(big_holes) == 1 and big_holes[0] / outer_area >= RING_MIN_HOLE_RATIO
               and iou >= RING_MIN_ELLIPSE_IOU)
    return is_ring, float(max(ellipse[1]))
