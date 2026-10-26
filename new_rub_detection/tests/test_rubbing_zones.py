import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from new_rub_detection.App import CameraWorker, draw_stone_grid
from new_rub_detection.geometry import DEST, SIZE
from new_rub_detection.rubbing_zones import (EDGE_MARGIN, OLD_MARK_MARGIN, TARGET_RADIUS, analyze_zones,
                           draw_recommendation, rubbing_instruction, rubbing_result, zone_name)


class ZoneSelectionTests(unittest.TestCase):
    def setUp(self):
        self.stone = np.ones((SIZE, SIZE), bool)
        self.old = np.zeros((SIZE, SIZE), np.uint8)

    def mark_every_zone(self):
        for x, y in ((64, 64), (320, 64), (64, 320), (320, 320)):
            self.old[y:y + 24, x:x + 24] = 255

    def test_empty_zone_preferred_over_marked_zone_with_more_room(self):
        # Three lightly marked zones have more space than the narrow empty Z2.
        self.mark_every_zone()
        self.old[:256, 256:] = 0
        self.stone[:256, 256:340] = False
        zones, selected = analyze_zones(self.old, self.stone)
        self.assertEqual(selected.name, "Z2")
        self.assertEqual(selected.old_marks, 0)
        self.assertTrue(any(zone.clearance > selected.clearance for zone in zones))

    def test_one_old_mark_per_zone_does_not_require_cleaning(self):
        self.mark_every_zone()
        zones, selected = analyze_zones(self.old, self.stone)
        self.assertEqual([zone.old_marks for zone in zones], [1, 1, 1, 1])
        self.assertIsNotNone(selected)
        instruction = rubbing_instruction(zones, selected)
        self.assertIn("All zones contain old marks", instruction)
        self.assertIn(f"blue circle in {selected.name}", instruction)

    def test_marked_zone_with_most_clear_room_is_selected(self):
        self.mark_every_zone()
        self.old[:190, :256] = 255
        self.old[:190, 256:] = 255
        self.old[256:446, :256] = 255
        _, selected = analyze_zones(self.old, self.stone)
        self.assertEqual(selected.name, "Z4")
        self.assertEqual(selected.old_marks, 1)

    def test_target_circle_keeps_clear_of_marks_edges_and_zone_boundaries(self):
        self.mark_every_zone()
        self.stone[160:240, 160:240] = False
        zones, selected = analyze_zones(self.old, self.stone)
        x, y = selected.target
        yy, xx = np.ogrid[:SIZE, :SIZE]
        target = (xx - x) ** 2 + (yy - y) ** 2 <= TARGET_RADIUS ** 2
        self.assertTrue(np.all(self.stone[target]))
        # An independent geometric check includes every pixel of the suggested circle.
        mark_y, mark_x = np.where(self.old > 0)
        nearest_mark = np.sqrt((mark_x - x) ** 2 + (mark_y - y) ** 2).min()
        self.assertGreater(nearest_mark, TARGET_RADIUS + OLD_MARK_MARGIN)
        x1, y1, x2, y2 = selected.bounds
        self.assertGreaterEqual(x - TARGET_RADIUS, x1)
        self.assertLess(x + TARGET_RADIUS, x2)
        self.assertGreaterEqual(y - TARGET_RADIUS, y1)
        self.assertLess(y + TARGET_RADIUS, y2)
        self.assertGreaterEqual(x - TARGET_RADIUS, EDGE_MARGIN)
        self.assertEqual(zone_name(selected.target), selected.name)

    def test_fully_marked_or_fragmented_space_requires_cleaning(self):
        for fragmented in (False, True):
            with self.subTest(fragmented=fragmented):
                self.old[:] = 255
                if fragmented:
                    # Plenty of free pixels, but no room for a whole rubbing target.
                    self.old[:, ::2] = 0
                zones, selected = analyze_zones(self.old, self.stone)
                self.assertIsNone(selected)
                self.assertIn("CLEAN THE STONE", rubbing_instruction(zones, selected))

    def test_no_stone_cannot_produce_a_target(self):
        _, selected = analyze_zones(self.old, np.zeros_like(self.stone))
        self.assertIsNone(selected)

    def test_mark_crossing_zone_boundary_is_counted_in_each_touched_zone(self):
        self.old[100:150, 245:270] = 255
        zones, _ = analyze_zones(self.old, self.stone)
        self.assertEqual([zone.old_marks for zone in zones], [1, 1, 0, 0])

    def test_reference_zone_boundaries_and_other_zone_acceptance(self):
        self.assertEqual([zone_name(p) for p in ((255, 255), (256, 255),
                                               (255, 256), (256, 256))],
                         ["Z1", "Z2", "Z3", "Z4"])
        _, selected = analyze_zones(self.old, self.stone)
        message = rubbing_result([((384, 384), (50, 60))], selected)
        self.assertIn("Z4", message)
        self.assertIn("rubbing accepted", message)

    def test_target_is_projected_onto_perspective_photo(self):
        _, selected = analyze_zones(self.old, self.stone)
        stone_to_frame = cv2.getPerspectiveTransform(
            DEST, np.float32([[90, 40], [680, 110], [610, 650], [30, 560]]))
        image = np.zeros((700, 750, 3), np.uint8)
        draw_recommendation(image, selected, np.linalg.inv(stone_to_frame))
        point = cv2.perspectiveTransform(np.float32([[selected.target]]), stone_to_frame)[0, 0]
        x, y = np.rint(point).astype(int)
        self.assertGreater(int(image[y, x].sum()), 0)


class PhotoRecommendationFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.worker = CameraWorker(0, SIZE, SIZE, Path(self.directory.name))
        self.worker.session_dir = Path(self.directory.name)
        self.frame = np.random.default_rng(8).integers(60, 130, (SIZE, SIZE, 3), dtype=np.uint8)
        self.old = np.zeros((SIZE, SIZE), np.uint8)
        for x, y in ((64, 64), (320, 64), (64, 320), (320, 320)):
            self.old[y:y + 24, x:x + 24] = 255
        self.emitted = []
        self.worker.photo_ready.connect(lambda *args: self.emitted.append(args))

    def complete_photo(self, kind, image, mask):
        mask_path = Path(self.directory.name) / f"{kind}_mask_full.png"
        cv2.imwrite(str(mask_path), mask)
        process = Mock(returncode=0)
        process.poll.return_value = 0
        process.communicate.return_value = (b"", b"")
        self.worker.photo_process = process
        self.worker.pending = {
            "kind": kind, "frame": image.copy(), "mask_path": mask_path,
            "bbox": (0, 0, SIZE - 1, SIZE - 1), "quad": DEST.copy(), "tag": None,
            "stone_mask": np.full((SIZE, SIZE), 255, np.uint8),
        }
        self.worker.finish_photo_if_ready()

    def test_before_photo_accepts_all_marked_zones_and_saves_guidance(self):
        self.complete_photo("before", self.frame, self.old)
        self.assertEqual(self.worker.stage, "AFTER")
        self.assertIsNotNone(self.worker.recommendation)
        self.assertIn("All zones contain old marks", self.worker.guidance)
        saved = json.loads((Path(self.directory.name) / "zone_recommendation.json").read_text())
        self.assertEqual(saved["selected_zone"], self.worker.recommendation.name)
        self.assertEqual([zone["old_marks"] for zone in saved["zones"]], [1, 1, 1, 1])
        self.assertTrue((Path(self.directory.name) / "before_zones.png").is_file())
        np.testing.assert_array_equal(self.worker.reference, self.frame)
        self.assertEqual(self.emitted[0][1].shape, (SIZE, SIZE, 3))

    def test_no_clear_patch_keeps_before_stage_and_disables_after_capture(self):
        self.complete_photo("before", self.frame, np.full_like(self.old, 255))
        self.assertEqual(self.worker.stage, "BEFORE")
        self.assertIsNone(self.worker.reference)
        self.assertIsNone(self.worker.recommendation)
        self.assertIn("CLEAN THE STONE", self.worker.result)

    def test_after_photo_reports_suggested_zone_and_accepts_another_zone(self):
        for use_suggested_zone in (True, False):
            with self.subTest(use_suggested_zone=use_suggested_zone):
                self.complete_photo("before", self.frame, self.old)
                selected = self.worker.recommendation
                point = selected.target if use_suggested_zone else (384, 384)
                after_mask = self.old.copy()
                after_frame = self.frame.copy()
                cv2.circle(after_mask, point, 12, 255, -1)
                cv2.circle(after_frame, point, 12, (220, 220, 220), -1)
                with patch("new_rub_detection.App.align_to_reference", return_value=(
                        after_frame, np.eye(3), "aligned (test)")):
                    self.complete_photo("after", after_frame, after_mask)
                self.assertEqual(self.worker.result, "New rubbed mask detected")
                if use_suggested_zone:
                    self.assertIn(f"suggested zone {selected.name}", self.worker.guidance)
                else:
                    self.assertIn("Z4", self.worker.guidance)
                    self.assertIn("rubbing accepted", self.worker.guidance)

    def test_uncertain_alignment_does_not_confirm_rubbing_zone(self):
        self.complete_photo("before", self.frame, self.old)
        with patch("new_rub_detection.App.align_to_reference", return_value=(None, None, "ambiguous")):
            self.complete_photo("after", self.frame, self.old)
        self.assertIn("RETAKE", self.worker.guidance)
        self.assertIn("no rubbing zone confirmed", self.worker.guidance)

    def test_rotated_after_photo_draws_and_reports_saved_reference_zones(self):
        self.complete_photo("before", self.frame, self.old)
        selected = self.worker.recommendation
        aligned_after = self.frame.copy()
        reference_mask = self.old.copy()
        cv2.circle(aligned_after, selected.target, 12, (220, 220, 220), -1)
        cv2.circle(reference_mask, selected.target, 12, 255, -1)
        after = np.rot90(aligned_after, 2).copy()
        after_mask = np.rot90(reference_mask, 2).copy()
        transform = np.float64([[-1, 0, SIZE - 1], [0, -1, SIZE - 1], [0, 0, 1]])
        with patch("new_rub_detection.App.align_to_reference", return_value=(aligned_after, transform, "aligned")), \
                patch("new_rub_detection.App.draw_stone_grid", wraps=draw_stone_grid) as grid:
            self.complete_photo("after", after, after_mask)
        self.assertIn(f"suggested zone {selected.name}", self.worker.guidance)
        grid.assert_called_once()
        np.testing.assert_allclose(grid.call_args.args[1], SIZE - 1 - DEST)

    def test_reset_clears_previous_target(self):
        self.complete_photo("before", self.frame, self.old)
        self.worker.reset()
        self.assertIsNone(self.worker.recommendation)
        self.assertEqual(self.worker.stage, "BEFORE")
        self.assertIn("Capture before", self.worker.guidance)


if __name__ == "__main__":
    unittest.main()
