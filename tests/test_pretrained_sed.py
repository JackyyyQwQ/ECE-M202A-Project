"""Check timestamp and event export without downloading model weights."""

import unittest

import numpy as np

from scripts.pretrained_sed import CLASSES, decode_events, frame_intervals, normalized_mel


class PretrainedExportTests(unittest.TestCase):
    def test_short_recording_excludes_padding_and_clips_last_interval(self):
        starts, ends = frame_intervals(156, 0.064, 7.0)
        self.assertEqual(len(starts), 110)
        self.assertAlmostEqual(starts[-1], 6.976)
        self.assertEqual(ends[-1], 7.0)
        self.assertTrue(np.all(ends > starts))

    def test_official_grid_does_not_stretch_frames_to_fill_ten_seconds(self):
        starts, ends = frame_intervals(156, 0.064, 10.0)
        self.assertAlmostEqual(ends[-1], 9.984)
        np.testing.assert_allclose(ends - starts, 0.064)

    def test_overlapping_classes_and_threshold_edges(self):
        scores = np.zeros((4, len(CLASSES)))
        scores[:, 0] = [0.9, 0.9, 0.5, 0.0]
        scores[:, 8] = [0.0, 0.8, 0.8, 0.8]
        starts, ends = frame_intervals(4, 0.064, 0.25)
        self.assertEqual(decode_events(scores, starts, ends, 0.5), [
            ("Alarm_bell_ringing", 0.0, 0.128),
            ("Speech", 0.064, 0.25),
        ])
        self.assertEqual(decode_events(np.zeros_like(scores), starts, ends, 0.5), [])

    def test_features_use_2023_training_range_zero_to_one(self):
        # Regression: the 2024 upstream scaler returns [-1, 1]; the 2023 checkpoint
        # was trained on [0, 1]. A [-1, 1] input silently suppresses detections.
        import torch
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=256, n_mels=16, power=1)
        generator = torch.Generator().manual_seed(0)
        audio = 0.1 * torch.randn(1, 16000, generator=generator)
        features = normalized_mel(audio, mel)
        self.assertAlmostEqual(features.min().item(), 0.0, places=6)
        self.assertAlmostEqual(features.max().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
