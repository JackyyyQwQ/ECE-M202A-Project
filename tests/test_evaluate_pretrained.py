"""Checks for label timing and one-to-one event evaluation."""

import unittest

import numpy as np

from scripts.evaluate_pretrained import (
    activity_targets, classification_counts, nominal_ten_seconds, overlap_matches, prf, target_grid,
)


class EvaluationTests(unittest.TestCase):
    def test_one_extra_source_sample_is_trimmed_without_shifting_audio(self):
        audio = np.arange(441001, dtype=np.float32)[:, None]
        trimmed, difference = nominal_ten_seconds(audio, 44100)
        self.assertEqual(difference, 1)
        np.testing.assert_array_equal(trimmed, audio[:441000])
        with self.assertRaises(ValueError):
            nominal_ten_seconds(np.zeros((441002, 1)), 44100)

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

    def test_matching_prioritizes_pair_count_over_iou(self):
        predicted = [("Dog", 0, 1.1), ("Dog", 0, 0.01)]
        reference = [("Dog", 0, 1), ("Dog", 1, 2)]
        self.assertEqual(set(overlap_matches(predicted, reference)), {(0, 1), (1, 0)})

    def test_fragments_cannot_claim_duplicate_credit_or_wrong_class(self):
        predicted = [("Dog", 0, 1), ("Dog", 1, 2), ("Cat", 0, 2)]
        reference = [("Dog", 0, 2)]
        matches = overlap_matches(predicted, reference)
        self.assertEqual(len(matches), 1)
        self.assertNotEqual(matches[0][0], 2)
        self.assertEqual(overlap_matches([], reference), [])


if __name__ == "__main__":
    unittest.main()
