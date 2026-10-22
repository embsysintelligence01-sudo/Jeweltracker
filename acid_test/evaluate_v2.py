#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

from analyzer_v2 import ANALYZER_VERSION, analyze_sequence_v2


LAB_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = LAB_ROOT / "dataset"
DEFAULT_OUTPUT = LAB_ROOT / "evaluation" / "latest_v2_report.json"


def _read_frames(sample_dir: Path, entries: list[str | dict]) -> list:
    frames = []
    for entry in entries:
        relative_path = entry if isinstance(entry, str) else entry["path"]
        frame = cv2.imread(str(sample_dir / relative_path))
        if frame is None or not frame.size:
            raise RuntimeError(f"Could not read {sample_dir / relative_path}")
        frames.append(frame)
    return frames


def evaluate(dataset_root: Path) -> dict:
    confusion: dict[str, Counter] = defaultdict(Counter)
    rows = []
    for manifest_path in sorted((dataset_root / "samples").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_version") != "acid_temporal_v2":
            continue
        sample_dir = manifest_path.parent
        metrics = analyze_sequence_v2(
            _read_frames(sample_dir, manifest["baseline_frames"]),
            _read_frames(sample_dir, manifest["observation_frames"]),
            _read_frames(sample_dir, manifest.get("transition_frames", [])),
        )
        actual = manifest["label"]
        predicted = metrics.suggestion
        confusion[actual][predicted] += 1
        rows.append({
            "sample_id": manifest["sample_id"],
            "item_or_control_id": manifest.get("metadata", {}).get(
                "item_or_control_id", ""
            ),
            "actual": actual,
            "v1_suggestion": manifest.get("analysis", {}).get("suggestion", ""),
            "v2_suggestion": predicted,
            **metrics.to_dict(),
        })

    correct = sum(row["actual"] == row["v2_suggestion"] for row in rows)
    false_gold = sum(
        row["actual"] != "gold_22k" and row["v2_suggestion"] == "gold_22k"
        for row in rows
    )
    return {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "analyzer_version": ANALYZER_VERSION,
        "dataset_version": "acid_temporal_v2",
        "sample_count": len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "false_gold_count": false_gold,
        "confusion": {
            actual: dict(predictions) for actual, predictions in confusion.items()
        },
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay acid temporal analyzer v2")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(report["samples"][0]) if report["samples"] else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(report["samples"])

    print(f"Analyzer: {report['analyzer_version']}")
    print(f"Samples: {report['sample_count']}")
    print(
        f"Accuracy: {report['correct_count']}/{report['sample_count']} "
        f"({report['accuracy']:.1%})"
    )
    print(f"False gold: {report['false_gold_count']}")
    print(f"JSON: {args.output}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
