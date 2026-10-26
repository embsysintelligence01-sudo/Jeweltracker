"""Choose a clear rubbing target in the saved 512-pixel stone reference."""

from dataclasses import dataclass

import cv2
import numpy as np

if __package__:
    from .geometry import SIZE
else:
    from geometry import SIZE


# Pixel distances in the normalized stone, independent of camera resolution.
EDGE_MARGIN = 12
OLD_MARK_MARGIN = 16
TARGET_RADIUS = 24
TARGET_COLOR = (255, 170, 0)  # BGR: blue target, green marks remain visible.


@dataclass(frozen=True)
class ZoneSpace:
    name: str
    bounds: tuple[int, int, int, int]
    old_marks: int
    coverage: float
    clear_pixels: int
    clearance: float
    target: tuple[int, int] | None


def zone_name(point):
    """Zone labels always refer to the saved before photo, including after rotation."""
    x, y = point
    return f"Z{1 + int(x >= SIZE // 2) + 2 * int(y >= SIZE // 2)}"


def analyze_zones(old_mask, stone_region):
    """Prefer a usable empty zone, then the largest clear circle in a marked zone.

    A target circle must fit inside one zone, clear of old masks and actual stone
    edges. Overall mark count and coverage do not prevent using remaining space.
    """
    stone = np.asarray(stone_region, dtype=bool)
    marks = (np.asarray(old_mask) > 0) & stone
    if stone.shape != (SIZE, SIZE) or marks.shape != (SIZE, SIZE):
        raise ValueError(f"Zone analysis requires {SIZE}x{SIZE} masks")
    _, labels = cv2.connectedComponents(marks.astype(np.uint8), connectivity=8)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * EDGE_MARGIN + 1, 2 * EDGE_MARGIN + 1))
    interior = cv2.erode(stone.astype(np.uint8), edge_kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    mark_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * OLD_MARK_MARGIN + 1, 2 * OLD_MARK_MARGIN + 1))
    near_marks = cv2.dilate(marks.astype(np.uint8), mark_kernel) > 0
    clear = interior & ~near_marks
    half = SIZE // 2
    bounds = [(0, 0, half, half), (half, 0, SIZE, half),
              (0, half, half, SIZE), (half, half, SIZE, SIZE)]
    zones = []
    for number, (x1, y1, x2, y2) in enumerate(bounds, 1):
        area = np.s_[y1:y2, x1:x2]
        patch = clear[area].astype(np.uint8)
        # Padding makes the zone boundary an obstacle, even for a full clear patch.
        distances = cv2.distanceTransform(np.pad(patch, 1), cv2.DIST_L2,
                                         cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        _, radius, _, center = cv2.minMaxLoc(distances)
        target = ((center[0] + x1, center[1] + y1) if radius >= TARGET_RADIUS + 1
                  else None)
        label_ids = np.unique(labels[area])
        zones.append(ZoneSpace(
            f"Z{number}", (x1, y1, x2, y2), int(np.count_nonzero(label_ids)),
            float(np.count_nonzero(marks[area]) / max(1, np.count_nonzero(stone[area]))),
            int(np.count_nonzero(patch)), float(radius), target,
        ))
    usable = [zone for zone in zones if zone.target is not None]
    selected = max(usable, key=lambda zone: (zone.old_marks == 0, zone.clearance,
                                           zone.clear_pixels, -zone.coverage), default=None)
    return zones, selected


def rubbing_instruction(zones, selected):
    if selected is None:
        return "CLEAN THE STONE - no zone has enough clear space. Clean and capture again."
    if all(zone.old_marks > 0 for zone in zones):
        prefix = f"All zones contain old marks. {selected.name} still has clear space."
    elif selected.old_marks:
        prefix = f"{selected.name} contains old marks but still has clear space."
    else:
        prefix = f"{selected.name} is clear."
    return f"{prefix} Rub inside the blue circle in {selected.name}, away from green marks."


def rubbing_result(locations, selected):
    """Report reference-zone locations without rejecting rubbing in another zone."""
    names = sorted({zone_name(stone_point) for stone_point, _ in locations})
    found = ", ".join(names)
    if selected is not None and names == [selected.name]:
        return f"New rubbing detected in the suggested zone {selected.name}."
    suffix = f" Suggested zone was {selected.name}; rubbing accepted." if selected else ""
    return f"New rubbing detected in {found} (before-photo zones).{suffix}"


def draw_recommendation(image, selected, frame_to_stone=None):
    """Project the selected zone and a round reference target onto a photo/crop."""
    if selected is None:
        return image
    inverse = np.eye(3, dtype=np.float32) if frame_to_stone is None else np.linalg.inv(frame_to_stone)

    def project(points):
        points = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        return np.rint(cv2.perspectiveTransform(points, inverse)[0]).astype(np.int32)

    x1, y1, x2, y2 = selected.bounds
    outline = project([(x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1)])
    cv2.polylines(image, [outline], True, TARGET_COLOR, 3, cv2.LINE_AA)
    cx, cy = selected.target
    angles = np.linspace(0, 2 * np.pi, 65)
    circle = project(np.column_stack((cx + TARGET_RADIUS * np.cos(angles),
                                      cy + TARGET_RADIUS * np.sin(angles))))
    cv2.polylines(image, [circle], True, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.polylines(image, [circle], True, TARGET_COLOR, 3, cv2.LINE_AA)
    px, py = project([(cx, cy)])[0]
    cv2.drawMarker(image, (int(px), int(py)), TARGET_COLOR, cv2.MARKER_CROSS, 12, 2)
    text = f"RUB {selected.name}"
    text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    position = (max(0, min(image.shape[1] - text_w, int(px - text_w // 2))),
                max(text_h + 2, min(image.shape[0] - 4, int(py - 35))))
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TARGET_COLOR, 2, cv2.LINE_AA)
    return image
