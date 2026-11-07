"""Synthetic correctness checks; these do not measure real detection accuracy."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from data.audio_dataset import AudioWindowDataset
from data.sequence_dataset import RecordingSequenceDataset, activity_targets
from models import CausalActivityModel


class BaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2026)

    def test_boundary_touch_is_not_overlap(self):
        labels, starts, ends = activity_targets(4, [(0.50, 0.55, 0)], 10)
        self.assertEqual(labels[:, 0].tolist(), [0, 1, 0, 0])
        self.assertEqual((starts[0], ends[0]), (0.45, 0.50))

    def test_partial_overlap_and_multiple_classes(self):
        labels, _, _ = activity_targets(
            4, [(0.499, 0.501, 0), (0.50, 0.55, 1), (0.0, 0.44, 2)], 10)
        self.assertEqual(labels[:, 0].tolist(), [1, 1, 0, 0])
        self.assertEqual(labels[:, 1].tolist(), [0, 1, 0, 0])
        self.assertEqual(labels[:, 2].tolist(), [0, 0, 0, 0])

    def test_loaders_agree_on_pcm_windows_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "processed_audio/wav").mkdir(parents=True)
            (root / "Synthetic_dataset").mkdir()
            (root / "processed_audio/classes.txt").write_text(
                "\n".join(f"class_{i}" for i in range(10)))
            (root / "processed_audio/recording_splits.csv").write_text(
                "filename,split\nexample.wav,train\n")
            (root / "Synthetic_dataset/synthetic_dataset.csv").write_text(
                "filename\tonset\toffset\tevent_label\n"
                "example.wav\t0.5\t0.55\tclass_0\n"
                "example.wav\t0.499\t0.651\tclass_1\n")
            # Known PCM16 values test conversion and sample alignment together.
            pcm = ((np.arange(160000) % 2000) - 1000).astype(np.int16)
            sf.write(root / "processed_audio/wav/example.wav", pcm, 16000,
                     subtype="PCM_16")
            sequence = RecordingSequenceDataset(root, "train", ["example.wav"])[0]
            previous = Path.cwd()
            try:
                os.chdir(root)  # The original window loader uses relative paths.
                windows = AudioWindowDataset("train")
                self.assertEqual(len(windows), 191)
                self.assertEqual(tuple(sequence["windows"].shape), (191, 8000))
                for i in range(len(windows)):
                    audio, target = windows[i]
                    torch.testing.assert_close(audio, sequence["windows"][i], rtol=0, atol=0)
                    torch.testing.assert_close(target, sequence["targets"][i], rtol=0, atol=0)
                expected = torch.from_numpy(pcm[:8000].astype(np.float32) / 32768)
                torch.testing.assert_close(windows[0][0], expected, rtol=0, atol=0)
            finally:
                os.chdir(previous)

    def test_model_outputs_and_gradients_are_finite(self):
        for feature_dim in (32, 128):
            with self.subTest(feature_dim=feature_dim):
                model = CausalActivityModel(feature_dim)
                # Include complete silence as well as nonzero audio.
                audio = torch.randn(2, 4, 8000) * 0.05
                audio[0] = 0
                logits = model(audio)
                self.assertEqual(tuple(logits.shape), (2, 4, 10))
                self.assertTrue(torch.isfinite(logits).all())
                targets = torch.zeros_like(logits)
                targets[1, :, 0] = 1
                F.binary_cross_entropy_with_logits(logits, targets).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_future_windows_cannot_change_past_predictions(self):
        model = CausalActivityModel().eval()
        audio = torch.randn(1, 6, 8000) * 0.05
        changed = audio.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:])
        with torch.no_grad():
            torch.testing.assert_close(model(audio)[:, :3], model(changed)[:, :3])

    def test_recording_state_does_not_leak_between_calls(self):
        model = CausalActivityModel().eval()
        audio = torch.randn(1, 4, 8000) * 0.05
        with torch.no_grad():
            before = model(audio)
            model(torch.randn_like(audio))
            torch.testing.assert_close(before, model(audio))

    def test_cached_frontend_matches_waveform_path(self):
        model = CausalActivityModel().eval()
        audio = torch.randn(2, 3, 8000) * 0.05
        with torch.no_grad():
            spectra = model.encoder.frontend(audio.flatten(0, 1)).reshape(2, 3, 1, 32, 16)
            torch.testing.assert_close(model(audio), model.forward_spectrograms(spectra))


if __name__ == "__main__":
    unittest.main()
