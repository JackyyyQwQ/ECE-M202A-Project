"""Render the completed full-corpus evaluation as a plain report and figures.

Requires requirements-report.txt. Does not run or modify the detector.
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)


ROOT = Path(__file__).resolve().parents[1]


def percent(value):
    return f"{value * 100:.2f}%"


def figures(report, source, data, output):
    raw, smooth = report["results"]["raw"], report["results"]["median7"]
    names = list(raw["per_class"])
    short = [n.replace("Electric_shaver_toothbrush", "Shaver/toothbrush")
             .replace("Alarm_bell_ringing", "Alarm").replace("_", " ") for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.0, 3.5), layout="constrained")
    ax.barh(y - .17, [raw["per_class"][n]["f1"] * 100 for n in names], height=.32, color="#286596", label="Raw (primary)")
    ax.barh(y + .17, [smooth["per_class"][n]["f1"] * 100 for n in names], height=.32, color="#999999", label="Median-7")
    ax.set(yticks=y, yticklabels=short, xlim=(0, 100), xlabel="Frame F1 (%)")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    chart = output / "pretrained_full_per_class.png"
    fig.savefig(chart, dpi=180)
    plt.close(fig)

    truth, predictions = defaultdict(list), defaultdict(list)
    with (data / "synthetic_dataset.csv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            truth[row["filename"]].append((row["event_label"], float(row["onset"]), float(row["offset"])))
    with (source / "events.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["mode"] == "raw":
                predictions[row["filename"]].append((row["event_label"], float(row["onset"]), float(row["offset"])))
    # Fixed annotation-only selection; do not select examples by favorable model scores.
    examples = []
    for target in ("Frying", "Speech"):
        examples.append(next(name for name in sorted(truth)
                             if name not in examples and any(e[0] == target for e in truth[name])))
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.0), layout="constrained")
    for ax, filename, target in zip(axes, examples, ("Frying", "Speech")):
        present = {e[0] for e in truth[filename] + predictions[filename]}
        labels = [n for n in names if n in present]
        for offset, events, color, legend in ((.16, truth[filename], "#286596", "Reference"),
                                             (-.16, predictions[filename], "#c47b25", "Prediction")):
            for index, (label, start, end) in enumerate(events):
                ax.barh(labels.index(label) + offset, end - start, left=start, height=.25,
                        color=color, label=legend if index == 0 else None)
        ax.set(yticks=range(len(labels)), yticklabels=[short[names.index(n)] for n in labels],
               xlim=(0, 10), xlabel="Recording time (s)",
               title=f"{filename}: first filename containing {target}")
        ax.axvline(.45, color="#777777", ls=":", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(9)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    timeline = output / "pretrained_full_examples.png"
    fig.savefig(timeline, dpi=180)
    plt.close(fig)
    return chart, timeline, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/pretrained_full")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/dcase2019")
    args = parser.parse_args()
    report = json.loads((args.input / "metrics.json").read_text())
    if report["recordings"] != 2045 or report["selection"]["count"] != 2045:
        raise ValueError("This report requires the complete 2045-recording evaluation")
    results, docs = ROOT / "results", ROOT / "docs"
    chart, timeline, examples = figures(report, args.input, args.data, results)
    raw, smooth = report["results"]["raw"], report["results"]["median7"]
    timing = report["timing"]
    summary = [
        ("Frame micro precision", percent(raw["frame_micro"]["precision"]), percent(smooth["frame_micro"]["precision"])),
        ("Frame micro recall", percent(raw["frame_micro"]["recall"]), percent(smooth["frame_micro"]["recall"])),
        ("Frame micro F1", percent(raw["frame_micro"]["f1"]), percent(smooth["frame_micro"]["f1"])),
        ("Frame macro F1", percent(raw["frame_macro_f1"]), percent(smooth["frame_macro_f1"])),
        ("Event overlap micro F1", percent(raw["event_overlap_micro"]["f1"]), percent(smooth["event_overlap_micro"]["f1"])),
        ("Matched onset MAE", f"{raw['onset_mae_ms']:.1f} ms (n={raw['onset_matched_count']})", f"{smooth['onset_mae_ms']:.1f} ms (n={smooth['onset_matched_count']})"),
        ("Matched offset MAE", f"{raw['offset_mae_ms']:.1f} ms (n={raw['offset_matched_count']})", f"{smooth['offset_mae_ms']:.1f} ms (n={smooth['offset_matched_count']})"),
    ]
    class_rows = [(name.replace("Electric_shaver_toothbrush", "Shaver/toothbrush").replace("_", " "),
                   percent(m["precision"]), percent(m["recall"]), percent(m["f1"]), str(m["tp"] + m["fn"]))
                  for name, m in raw["per_class"].items()]
    counts = raw["frame_micro"]
    events = raw["event_overlap_micro"]
    unscored_events = report["selected_annotation_events"] - events["tp"] - events["fn"]
    methods = [
        f"All 2,045 WAV files; {report['selected_annotation_events']:,} source event annotations; no quality-based exclusions. "
        f"Nominal audio: {report['audio_seconds']:,.0f} s ({report['audio_seconds']/3600:.2f} h). "
        f"Scored targets: {report['frames']:,}; first 450 ms of each file unscored. "
        "All source files are 44.1 kHz mono.",
        "Frozen official DCASE 2023 plain CNN+BiGRU student, 1,112,420 parameters. "
        "Original audio resampled to 16 kHz; official 128-bin mel, dB and instance normalization. "
        "No model training or threshold fitting. Primary threshold >=0.5; raw and median-7 both reported.",
        "Frame targets are 50 ms, positive for any annotation overlap. Assign the official 64 ms "
        "score bin containing each target midpoint. Event matching is same-class positive overlap, "
        "maximum pair count then IoU; it is permissive, not collar F1 or PSDS. "
        f"Only references intersecting [0.45, 10) s enter event scoring; {unscored_events} earlier-only events are excluded.",
        "Boundary MAE is conditional on matched events. Exclude reference onsets <=0.45 s and "
        "offsets >=10 s from timing errors; false/missed events remain in event F1. "
        f"Source-duration rounding of up to two samples was corrected in {len(report['duration_rounding'])} files.",
    ]
    limitations = [
        "These are public synthetic training recordings, not an independent official test set. "
        "Overlap with the pretrained model's training material has not been audited. "
        "The original custom-model validation list and weights remain unavailable.",
        "BiGRU, centered STFT, whole-window normalization and symmetric smoothing use future context. "
        "Compute time excludes audio collection, resampling, transfer, queueing and event decoding. "
        "This is not a deadline-aware or MCU measurement.",
        "The dataset is suitable for single-channel overlapping-event detection, but not direction "
        "localization. Nano 33 BLE Sense / Rev2 has one onboard microphone. Real board recordings "
        "are still needed to evaluate microphone, room and noise differences.",
        "Do not claim a gain over the old 21.39%/23.29% custom-model frame F1: those used a different "
        "32-recording subset, training setup and causal context. Future threshold/model choices "
        "need separate development data and a fresh evaluation set.",
    ]
    summary_text = (
        f"The full-corpus raw frame micro F1 is {percent(raw['frame_micro']['f1'])}; "
        f"precision {percent(raw['frame_micro']['precision'])}, recall {percent(raw['frame_micro']['recall'])}. "
        "This establishes a reproducible offline reference, not a completed online system."
    )
    ranked = sorted(raw["per_class"].items(), key=lambda item: item[1]["f1"])
    class_interpretation = (
        "Lowest raw class F1: " + ", ".join(f"{n} {percent(m['f1'])}" for n, m in ranked[:3])
        + ". Highest: " + ", ".join(f"{n} {percent(m['f1'])}" for n, m in reversed(ranked[-3:]))
        + f". Median filtering changes frame micro F1 by {(smooth['frame_micro']['f1'] - raw['frame_micro']['f1']) * 100:+.2f} percentage points. "
        "A change in matched-boundary MAE can also reflect a different matched population."
    )
    if all(m["f1"] > 0 for m in raw["per_class"].values()):
        class_interpretation += " All ten classes have nonzero F1 on the full corpus; zero F1 in the smaller pilot did not imply the model could never detect those classes."
    source_links = [
        ("Dataset", "https://zenodo.org/records/2583796"),
        ("Official model recipe", "https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline"),
        ("Checkpoint", "https://zenodo.org/records/7759146"),
        ("Nano Rev2 hardware", "https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2"),
    ]
    markdown = ["# DCASE2019 full-corpus pretrained BiGRU report", "",
                "P13 - Junchi Chen and Wei Zhang - November 7, 2026", "",
                "[Printable PDF](pretrained_full_report.pdf)", "", summary_text, "",
                "## Scope and methods", "", *["- " + s for s in methods], "",
                f"Source channels: `{report['source_audio_channels']}`; sample rates: `{report['source_audio_sample_rates']}`.", "",
                "## Results", "", "| Metric | Raw, primary | Median-7 |", "|---|---:|---:|",
                *["| " + " | ".join(row) + " |" for row in summary], "",
                f"Raw frame counts: {counts['tp']:,} TP, {counts['fp']:,} FP, {counts['fn']:,} FN. "
                f"Raw event counts: {events['tp']:,} matches, {events['fp']:,} false predictions, {events['fn']:,} misses.", "",
                "## Per-class frame results", "", "| Class | Precision | Recall | F1 | Positive frames |", "|---|---:|---:|---:|---:|",
                *["| " + " | ".join(row) + " |" for row in class_rows], "",
                "![Per-class raw and smoothed F1](../results/pretrained_full_per_class.png)", "",
                class_interpretation, "",
                "## Example timelines", "", "![Reference and prediction timelines](../results/pretrained_full_examples.png)", "",
                f"Examples `{examples[0]}` and `{examples[1]}` are the first filenames containing Frying and Speech respectively "
                "(second file distinct), chosen from annotations, not model performance. Blue is ground truth; orange is raw prediction. "
                "The dotted line marks the start of scored intervals at 0.45 s.", "",
                "## Computation time", "",
                f"CPU, four threads, one warm-up: median {timing['median_compute_ms']:.1f} ms, "
                f"p95 {timing['p95_compute_ms']:.1f} ms per ten-second input. This includes mel/dB/normalization/network, "
                "not end-to-end report latency. Hardware/OS and all measured calls are retained in the metrics artifact.", "",
                "## Interpretation and limits", "", *["- " + s for s in limitations], "",
                "## Reproduce", "", "```bash", "python scripts/fetch_dcase_pilot.py --all",
                "python -m scripts.evaluate_pretrained --all", "python scripts/report_pretrained.py", "```", "",
                "Inference dependencies: `requirements-pretrained.txt`; report dependencies: `requirements-report.txt`. "
                "Original pilot outputs are preserved. Full details remain in ignored `output/pretrained_full/`.", "",
                "## Evidence and sources", "",
                "- [Full metrics](../results/pretrained_full_metrics.json)",
                "- [All filenames and WAV hashes](../results/pretrained_full_manifest.csv)",
                "- [Per-class counts](../results/pretrained_full_per_class.tsv)",
                *[f"- [{label}]({url})" for label, url in source_links], "",
                f"Archive MD5: `06ebea7cc467ace800752bd21ec7178d`; checkpoint SHA256: `{report['checkpoint_sha256']}`. "
                f"Pinned model code: `{report['upstream_commit']}`. All archive members were CRC-checked. No model weights or raw audio are committed.", ""]
    (docs / "pretrained_full_report.md").write_text("\n".join(markdown))
    for name, origin in (("pretrained_full_metrics.json", args.input / "metrics.json"),
                         ("pretrained_full_manifest.csv", args.data / "full_manifest.csv"),
                         ("pretrained_full_per_class.tsv", args.input / "per_class.tsv")):
        shutil.copyfile(origin, results / name)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", fontName="Helvetica", fontSize=9, leading=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=12))
    styles["Heading2"].fontSize = 12
    styles["Heading2"].spaceBefore = 10
    story = []
    def text(value, style="BodySmall"):
        story.append(Paragraph(escape(value), styles[style]))
    def table(rows, widths):
        element = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        element.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("LINEBELOW", (0, 0), (-1, 0), .5, colors.grey),
            ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#dddddd")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        story.append(element)
        story.append(Spacer(1, 8))
    text("DCASE2019: full-corpus BiGRU evaluation", "ReportTitle")
    text("P13 | Junchi Chen and Wei Zhang | November 7, 2026")
    text(summary_text)
    text("Fixed model and threshold", "Heading2")
    table([("Metric", "Raw (primary)", "Median-7"), *summary], [195, 155, 155])
    text("Scope and measurement", "Heading2")
    for item in methods:
        text(item)
    text(f"Raw frame counts: {counts['tp']:,} TP / {counts['fp']:,} FP / {counts['fn']:,} FN. "
         f"CPU compute: median {timing['median_compute_ms']:.1f} ms, p95 {timing['p95_compute_ms']:.1f} ms; "
         "four threads, one warm-up. This is not end-to-end latency.")
    story.append(PageBreak())
    text("Class-level detection", "ReportTitle")
    text("Raw scores at threshold 0.5. Positive-frame counts show annotation support; "
         "a low score does not mean the class was absent. Scores were not calibrated on this corpus.")
    table([("Class", "Precision", "Recall", "F1", "Positive frames"), *class_rows], [195, 75, 75, 70, 90])
    story.append(Image(str(chart), width=7.0 * inch, height=3.0625 * inch))
    text(class_interpretation)
    text("Smoothing changes fragmentation and can suppress short events; "
         "it does not establish the benefit of requesting extra PCM.")
    story.append(PageBreak())
    text("Examples, interpretation and next steps", "ReportTitle")
    story.append(Image(str(timeline), width=7.0 * inch, height=3.5 * inch))
    text(f"Annotation-only example selection: {examples[0]} (first Frying file), {examples[1]} "
         "(first distinct Speech file). These are illustrative, not a representative performance estimate. "
         "Blue: reference; orange: raw prediction. Dotted line: 0.45 s scoring start.")
    for item in limitations:
        text(item)
    text("Reproduce", "Heading2")
    text("python scripts/fetch_dcase_pilot.py --all")
    text("python -m scripts.evaluate_pretrained --all")
    text("python scripts/report_pretrained.py")
    text("Sources and full configuration", "Heading2")
    for label, url in source_links[:3]:
        story.append(Paragraph(f'{escape(label)}: <link href="{url}" color="#1455a0">{escape(url)}</link>', styles["BodySmall"]))
    pdf_dir = ROOT / "output/pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_dir / "pretrained_full_report.pdf"
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(48, 27, "P13 | Offline reference - public synthetic training corpus")
        canvas.drawRightString(564, 27, str(document.page))
    SimpleDocTemplate(str(pdf), pagesize=(612, 792), leftMargin=48, rightMargin=48,
                      topMargin=42, bottomMargin=42,
                      title="DCASE2019 full-corpus pretrained BiGRU evaluation",
                      author="Junchi Chen and Wei Zhang").build(story, onFirstPage=footer, onLaterPages=footer)
    shutil.copyfile(pdf, docs / pdf.name)
    print(f"Saved report: {docs / 'pretrained_full_report.md'}")
    print(f"Saved PDF: {pdf}")


if __name__ == "__main__":
    main()
