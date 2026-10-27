"""Stateful before/after comparison for any host GUI, camera, or web framework.

This adapter compares new marks against a saved reference without recommending zones.
It accepts full-frame masks and corners so the host can choose its inference and
capture lifecycle. It does not open a camera, start a GUI, or write any files.
"""

import cv2
import numpy as np

from .core import (marked_photo, merged_mark_regions, new_mark_locations,
                   new_mark_regions, stone_region)
from .geometry import SIZE, align_to_reference, normalize, project_mask
from .rubbing_zones import analyze_zones


class RubCheck:
    """Use one instance per stone test. Call reset() when starting a different stone."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.reference = self.reference_region = self.old_mask = None

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
        """Require usable clear space, without recommending a rubbing zone."""
        self.reset()
        quad, bbox = self._validate(frame, stone_mask, rubbed_mask, quad)
        reference, transform = normalize(frame, quad)
        inner = stone_region(stone_mask, transform)
        old = cv2.warpPerspective(rubbed_mask, transform, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        old, count, coverage = merged_mark_regions(old, inner)
        if np.count_nonzero(inner) < 80:
            raise ValueError("RETAKE - expose the stone surface before capturing a reference")
        # Reuse the original clearance check, but never expose or draw its target.
        _, clear_space = analyze_zones(old, inner)
        ready = clear_space is not None
        if ready:
            self.reference = reference.copy()
            self.reference_region = inner.astype(np.uint8) * 255
            self.old_mask = old.copy()
        photo = marked_photo(frame, bbox, quad, None, rubbed_mask, stone_mask, show_grid=False)
        return {"status": "READY" if ready else "CLEAN_THE_STONE",
                "message": ("Reference captured. Rub a jewel on the stone, then remove hands and jewellery before checking."
                            if ready else "CLEAN THE STONE - not enough clear space for a new rub. Clean the stone, then capture a new reference."),
                "photo": photo, "preview": reference.copy(), "preview_old_mask": old.copy(),
                "old_mask": old, "old_mark_count": count, "coverage": coverage}

    def after(self, frame, stone_mask, rubbed_mask, quad, *, frame_to_reference=None,
              alignment_detail=None):
        """Return NEW_RUB or RETAKE, with reference/frame locations only on success."""
        result = {"status": "RETAKE", "message": "Capture a valid before photo first.",
                  "new_mask": None, "locations": [], "photo": None, "preview": None}
        if self.reference is None:
            return result
        quad, bbox = self._validate(frame, stone_mask, rubbed_mask, quad)
        result["photo"] = marked_photo(frame, bbox, quad, None, rubbed_mask, stone_mask,
                                       show_grid=False)
        if frame_to_reference is None:
            aligned, transform, detail = align_to_reference(
                frame, quad, self.reference, self.reference_region, stone_mask)
        else:
            transform = np.asarray(frame_to_reference, float)
            if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-9:
                raise ValueError('RETAKE - invalid marker alignment')
            aligned = cv2.warpPerspective(frame, transform, (SIZE, SIZE))
            detail = alignment_detail or 'aligned (marker snapshot)'
        result["alignment"] = detail
        if aligned is None:
            result["message"] = f"RETAKE - {detail}; no new rub confirmed."
            return result
        inner = stone_region(stone_mask, transform)
        marks = cv2.warpPerspective(rubbed_mask, transform, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
        new = new_mark_regions(marks, self.old_mask, inner, reject_old_overlap=True)
        change = cv2.absdiff(aligned, self.reference).mean(axis=2)
        new[change < 12] = 0
        new, count, fraction = merged_mark_regions(new, inner)
        photo = marked_photo(frame, bbox, quad, None, rubbed_mask, stone_mask,
                             show_grid=False)
        result.update(photo=photo, preview=new, new_mark_count=count, coverage=fraction,
                      preview_old_mask=np.where(inner, self.old_mask, 0).astype(np.uint8),
                      frame_to_reference=transform.copy())
        if fraction < 0.001:
            result["message"] = "RETAKE - No new rub detected. Please rub on a clear area and retry."
            return result
        if fraction > 0.15:
            result["message"] = "RETAKE - new rubbing unclear; too much of the stone has changed."
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
        result.update(status="NEW_RUB", message="New rub detected on the stone.",
                      new_mask=new, locations=locations, preview=preview)
        return result
