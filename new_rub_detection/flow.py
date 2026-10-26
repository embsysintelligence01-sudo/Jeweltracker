"""Stateful before/after comparison for any host GUI, camera, or web framework.

This adapter uses the copied core functions and the same thresholds as App.py.
It accepts full-frame masks and corners so the host can choose its inference and
capture lifecycle. It does not open a camera, start a GUI, or write any files.
"""

import cv2
import numpy as np

from .core import (marked_photo, merged_mark_regions, new_mark_locations,
                   new_mark_regions, stone_region)
from .geometry import DEST, SIZE, align_to_reference, normalize, project_mask
from .rubbing_zones import analyze_zones, draw_recommendation, rubbing_instruction, rubbing_result


class RubCheck:
    """Use one instance per stone test. Call reset() when starting a different stone."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.reference = self.reference_region = self.old_mask = None
        self.recommendation = None

    @staticmethod
    def _validate(frame, stone_mask, rubbed_mask, quad):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 BGR image with shape H x W x 3")
        if stone_mask.shape != frame.shape[:2] or rubbed_mask.shape != frame.shape[:2]:
            raise ValueError("Both masks must match the captured frame's H x W dimensions")
        quad = np.asarray(quad, dtype=np.float32)
        if quad.shape != (4, 2) or not np.isfinite(quad).all() or abs(cv2.contourArea(quad)) < 1:
            raise ValueError("quad must contain four valid, perimeter-ordered stone corners")
        bbox = (*np.floor(quad.min(axis=0)).astype(int), *np.ceil(quad.max(axis=0)).astype(int))
        return quad, bbox

    def before(self, frame, stone_mask, rubbed_mask, quad):
        """Save the reference and return READY, CLEAN_THE_STONE, or a validation error."""
        self.reset()
        quad, bbox = self._validate(frame, stone_mask, rubbed_mask, quad)
        reference, transform = normalize(frame, quad)
        inner = stone_region(stone_mask, transform)
        old = cv2.warpPerspective(rubbed_mask, transform, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        old, count, coverage = merged_mark_regions(old, inner)
        zones, selected = analyze_zones(old, inner)
        photo = marked_photo(frame, bbox, quad, None, rubbed_mask, stone_mask)
        preview = marked_photo(reference, (0, 0, SIZE - 1, SIZE - 1), DEST,
                               None, old, inner.astype(np.uint8) * 255)
        if selected is not None:
            self.reference = reference.copy()
            self.reference_region = inner.astype(np.uint8) * 255
            self.old_mask = old.copy()
            self.recommendation = selected
            draw_recommendation(photo, selected, transform)
            draw_recommendation(preview, selected)
        return {"status": "READY" if selected else "CLEAN_THE_STONE",
                "message": rubbing_instruction(zones, selected), "zones": zones,
                "recommendation": selected, "photo": photo, "preview": preview,
                "old_mask": old, "old_mark_count": count, "coverage": coverage}

    def after(self, frame, stone_mask, rubbed_mask, quad):
        """Return NEW_RUB or RETAKE, with reference/frame locations only on success."""
        result = {"status": "RETAKE", "message": "Capture a valid before photo first.",
                  "new_mask": None, "locations": [], "photo": None, "preview": None}
        if self.reference is None:
            return result
        quad, bbox = self._validate(frame, stone_mask, rubbed_mask, quad)
        result["photo"] = marked_photo(frame, bbox, quad, None, rubbed_mask, stone_mask,
                                       show_grid=False)
        aligned, transform, detail = align_to_reference(
            frame, quad, self.reference, self.reference_region, stone_mask)
        result["alignment"] = detail
        if aligned is None:
            result["message"] = f"RETAKE - {detail}; no rubbing zone confirmed."
            return result
        inner = stone_region(stone_mask, transform)
        marks = cv2.warpPerspective(rubbed_mask, transform, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        new = new_mark_regions(marks, self.old_mask, inner)
        change = cv2.absdiff(aligned, self.reference).mean(axis=2)
        new[change < 12] = 0
        new, count, fraction = merged_mark_regions(new, inner)
        reference_quad = cv2.perspectiveTransform(DEST[None], np.linalg.inv(transform))[0]
        photo = marked_photo(frame, bbox, quad, None, rubbed_mask, stone_mask,
                             grid_quad=reference_quad)
        result.update(photo=photo, preview=new, new_mark_count=count, coverage=fraction,
                      frame_to_reference=transform.copy())
        if not 0.001 <= fraction <= 0.15:
            result["message"] = "RETAKE - new rubbing unclear; no rubbing zone confirmed."
            return result
        locations = new_mark_locations(new, transform)
        projected = project_mask(new, frame.shape, transform) > 0
        magenta = np.full_like(photo, (255, 0, 255))
        photo[projected] = cv2.addWeighted(photo, 0.4, magenta, 0.6, 0)[projected]
        preview = aligned.copy()
        colored = np.full_like(preview, (255, 0, 255))
        preview[new > 0] = cv2.addWeighted(preview, 0.4, colored, 0.6, 0)[new > 0]
        for index, (_, point) in enumerate(locations, 1):
            cv2.circle(photo, point, 8, (255, 255, 255), 2)
            cv2.putText(photo, f"NEW {index}", (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        result.update(status="NEW_RUB", message=rubbing_result(locations, self.recommendation),
                      new_mask=new, locations=locations, preview=preview)
        return result
