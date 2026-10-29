import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jewel_identity as ji


class ColourHistogramExtractor:
    """Stand-in for the ResNet18 extractor: unit-length 4x4x4 colour histogram of the non-bench pixels."""

    def features(self, crops):
        out = []
        for c in crops:
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            keep = ~((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 170))
            px = c[keep] if keep.any() else c.reshape(-1, 3)
            h = np.histogramdd(px // 64, bins=(4, 4, 4), range=((0, 4),) * 3)[0].ravel().astype(np.float32)
            out.append(h / max(np.linalg.norm(h), 1e-6))
        return np.stack(out)


def jewel(colours, seed):
    """A 160 px blob made of patches in the given BGR colours, on white bench."""
    rng = np.random.default_rng(seed)
    img = np.full((160, 160, 3), 255, np.uint8)
    for i, col in enumerate(colours):
        cx, cy = 80 + int(rng.integers(-25, 25)), 80 + int(rng.integers(-25, 25))
        cv2.circle(img, (cx, cy), 34 - 5 * i, col, -1)
    return img


GOLD = jewel([(40, 160, 200), (30, 120, 170), (60, 190, 225)], 1)       # gold tones
RUBY = jewel([(150, 90, 20), (170, 140, 40), (110, 60, 10)], 2)          # blue-green stones
BENCH = np.full((160, 160, 3), 250, np.uint8)


class IdentityMatcherTests(unittest.TestCase):
    def matcher(self, crops):
        m = ji.IdentityMatcher(ColourHistogramExtractor())
        m.train(crops, [BENCH, BENCH.copy()], views=12)
        return m

    def test_different_jewels_keep_their_own_ids_when_turned_and_partly_hidden(self):
        m = self.matcher({1: GOLD, 2: RUBY})
        rng = np.random.default_rng(3)
        for jid, crop in ((1, GOLD), (2, RUBY)):
            answers = [m.predict(ji.augment_view(crop, rng))[0] for _ in range(12)]
            self.assertEqual(0, sum(a is not None and a != jid for a in answers), answers)
            self.assertGreaterEqual(sum(a == jid for a in answers), 8, answers)

    def test_look_alike_jewels_are_never_guessed(self):
        m = self.matcher({1: GOLD, 2: GOLD.copy()})
        rng = np.random.default_rng(4)
        for _ in range(12):
            self.assertEqual((None, 0.0), m.predict(ji.augment_view(GOLD, rng)))

    def test_bench_is_not_a_jewel(self):
        m = self.matcher({1: GOLD, 2: RUBY})
        jid, p = m.predict(BENCH.copy())
        self.assertIsNone(jid)
        self.assertGreaterEqual(p, 0.6)

    def test_not_trained_answers_nothing(self):
        m = ji.IdentityMatcher(ColourHistogramExtractor())
        self.assertEqual((None, 0.0), m.predict(GOLD))

    def test_signature_ignores_bench_and_skin(self):
        clean = ji.superpixel_signature(GOLD)
        hand = GOLD.copy()
        cv2.ellipse(hand, (40, 40), (30, 20), 0, 0, 360, ji.SKIN_BGR, -1)  # a finger beside the jewel
        covered = ji.superpixel_signature(hand)
        self.assertGreater(ji.signature_similarity(covered, clean), 0.85)
        self.assertLess(ji.signature_similarity(ji.superpixel_signature(RUBY), clean),
                        ji.signature_similarity(covered, clean))
        self.assertIsNone(ji.superpixel_signature(BENCH))

    def test_numpy_fallback_matches_the_index(self):
        rng = np.random.default_rng(5)
        vecs = rng.normal(size=(40, 16)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        query = vecs[:2] + 0.05 * rng.normal(size=(2, 16)).astype(np.float32)
        index = ji._Index(16)
        index.add(vecs)
        sims, idx = index.search(query, 3)
        expect = np.argsort(-(query @ vecs.T), axis=1)[:, :3]
        self.assertTrue((idx == expect).all())
        self.assertTrue(np.allclose(sims, np.take_along_axis(query @ vecs.T, expect, 1), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
