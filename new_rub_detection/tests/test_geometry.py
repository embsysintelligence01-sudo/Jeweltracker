import unittest

import cv2
import numpy as np

from new_rub_detection.geometry import SIZE, align_to_reference, normalize, project_mask, stone_quad, tag_stable


class StoneGeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.frame = np.zeros((1200, 1200, 3), np.uint8)
        self.mask = np.zeros((1200, 1200), np.uint8)
        self.mask[344:856, 344:856] = 255
        self.frame[344:856, 344:856] = (85, 110, 135)
        for _ in range(550):
            x, y = rng.integers(360, 840, size=2)
            color = tuple(int(v) for v in rng.integers(20, 230, size=3))
            cv2.circle(self.frame, (int(x), int(y)), int(rng.integers(2, 7)), color, -1)
        self.reference, matrix = normalize(self.frame, stone_quad(self.mask, (344, 344, 856, 856)))
        self.reference_mask = cv2.warpPerspective(self.mask, matrix, (SIZE, SIZE))

    def test_translation_and_rotations(self):
        for degrees in (45, 90, 180):
            with self.subTest(degrees=degrees):
                matrix = cv2.getRotationMatrix2D((600, 600), degrees, 1)
                matrix[:, 2] += (24, -30)
                frame = cv2.warpAffine(self.frame, matrix, (1200, 1200))
                mask = cv2.warpAffine(self.mask, matrix, (1200, 1200),
                                      flags=cv2.INTER_NEAREST)
                ys, xs = np.where(mask)
                box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
                quad = stone_quad(mask, box)
                self.assertIsNotNone(quad)
                aligned, transform, detail = align_to_reference(
                    frame, quad, self.reference, self.reference_mask, mask)
                self.assertIsNotNone(aligned, detail)
                self.assertLess(float(cv2.absdiff(aligned, self.reference).mean()), 6)
                saved = np.zeros((SIZE, SIZE), np.uint8)
                saved[200:220, 300:320] = 255
                projected = project_mask(saved, frame.shape, transform)
                self.assertGreater(np.count_nonzero(projected), 0)

    def test_image_fallback_when_feature_matches_are_unavailable(self):
        class NoFeatures:
            def detectAndCompute(self, image, mask):
                return [], None

        for degrees in (0, 45, 90, 180):
            with self.subTest(degrees=degrees):
                matrix = cv2.getRotationMatrix2D((600, 600), degrees, 1)
                matrix[:, 2] += (24, -30)
                frame = cv2.warpAffine(self.frame, matrix, (1200, 1200))
                mask = cv2.warpAffine(self.mask, matrix, (1200, 1200),
                                      flags=cv2.INTER_NEAREST)
                ys, xs = np.where(mask)
                box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
                quad = stone_quad(mask, box)
                aligned, _, detail = align_to_reference(
                    frame, quad, self.reference, self.reference_mask, mask, orb=NoFeatures())
                self.assertIsNotNone(aligned, detail)
                self.assertIn("stone image", detail)
                self.assertLess(float(cv2.absdiff(aligned, self.reference).mean()), 6)

    def test_symmetric_surface_stays_ambiguous(self):
        class NoFeatures:
            def detectAndCompute(self, image, mask):
                return [], None

        plain = np.full((SIZE, SIZE, 3), 80, np.uint8)
        cv2.circle(plain, (130, 130), 45, (130, 130, 130), -1)
        cv2.circle(plain, (SIZE - 1 - 130, SIZE - 1 - 130), 45,
                   (130, 130, 130), -1)
        full = np.full((SIZE, SIZE), 255, np.uint8)
        aligned, _, detail = align_to_reference(
            plain, np.float32([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]]),
            plain, full, full, orb=NoFeatures())
        self.assertIsNone(aligned, detail)

    def test_ambiguous_surface_rejected(self):
        plain = np.full_like(self.reference, 110)
        aligned, transform, detail = align_to_reference(
            self.frame, stone_quad(self.mask, (344, 344, 856, 856)),
            plain, self.reference_mask, self.mask)
        self.assertIsNone(aligned)
        self.assertIsNone(transform)
        self.assertIn("insufficient", detail)

    def test_moved_fixed_tag_rejected(self):
        tag = np.float32([[10, 10], [30, 10], [30, 30], [10, 30]])
        self.assertTrue(tag_stable(tag + 5, tag))
        self.assertFalse(tag_stable(tag + 20, tag))


if __name__ == "__main__":
    unittest.main()
