"""Checks for the 50 ms target grid, label overlap and frame accounting."""

import unittest

import numpy as np

from scripts.scoring import activity_targets, classification_counts, prf, target_grid


class ScoringTests(unittest.TestCase):
    def test_grid_matches_original_191_targets(self):
        starts, ends = target_grid(160000)
        self.assertEqual(len(starts), 191)
        self.assertEqual((starts[0], ends[0]), (7200, 8000))
        self.assertEqual((starts[-1], ends[-1]), (159200, 160000))
        indices = np.floor(((starts + ends) / 32000) / 0.064).astype(int)
        self.assertEqual(indices[-1], 155)

    def test_boundary_touch_is_not_positive_overlap(self):
        starts, ends = target_grid(8800)
        labels = activity_targets([("Dog", 0.5, 0.55)], starts, ends)
        self.assertEqual(labels[:, 4].tolist(), [False, True])
        self.assertEqual(int(labels.sum()), 1)

    def test_false_and_missed_predictions_are_in_f1(self):
        truth = np.array([[True, False], [False, True]])
        predicted = np.array([[True, True], [False, False]])
        counts = classification_counts(predicted, truth).sum(axis=0)
        self.assertEqual(counts.tolist(), [1, 1, 1])
        self.assertEqual(prf(counts)["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
