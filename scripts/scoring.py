"""Frame-level scoring helpers shared by the evaluation scripts.

Targets use a 50 ms grid at 16 kHz: windows end at 0.5 s, 0.55 s, ..., and each
covers the latest 50 ms ([end - 800, end) samples). A class is positive when an
annotated event overlaps that interval (touching boundaries do not count).
"""

import numpy as np

from scripts.pretrained_sed import CLASSES


def target_grid(sample_count, sample_rate=16000):
    """Half-open 50 ms target intervals in samples; the first 450 ms is unscored."""
    ends = np.arange(8000, sample_count + 1, 800)
    return ends - 800, ends


def activity_targets(events, starts, ends, sample_rate=16000):
    """Boolean [intervals, classes] labels from (label, onset_s, offset_s) events."""
    labels = np.zeros((len(starts), len(CLASSES)), dtype=bool)
    for label, onset, offset in events:
        active = (onset * sample_rate < ends) & (offset * sample_rate > starts)
        labels[active, CLASSES.index(label)] = True
    return labels


def classification_counts(predicted, reference):
    """Per-class [TP, FP, FN] counts from boolean [intervals, classes] arrays."""
    return np.stack((
        (predicted & reference).sum(axis=0),
        (predicted & ~reference).sum(axis=0),
        (~predicted & reference).sum(axis=0),
    ), axis=-1)


def prf(counts):
    """Precision, recall and F1 from [TP, FP, FN]; undefined ratios are 0."""
    tp, fp, fn = map(int, counts)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
