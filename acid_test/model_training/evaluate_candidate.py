#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from session_dataset import (
    CLASSES,
    TemporalSessionDataset,
    build_index,
)


ROOT = Path(__file__).resolve().parent
LAB_ROOT = ROOT.parent
DEFAULT_DATASET = LAB_ROOT / "dataset"
DEFAULT_CURATION = ROOT / "curation.json"
GOLD_CONFIDENCE = 0.85
GENERAL_CONFIDENCE = 0.65


def _latest(pattern: str) -> Path:
    matches = sorted((ROOT / "artifacts").glob(pattern))
    if not matches:
        raise RuntimeError(f"No artifact matches {pattern}")
    return matches[-1]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def _prediction(probabilities: np.ndarray) -> tuple[str, float]:
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    label = CLASSES[index]
    if confidence < GENERAL_CONFIDENCE:
        return "invalid", confidence
    if label == "gold_22k" and confidence < GOLD_CONFIDENCE:
        return "invalid", confidence
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen temporal ONNX candidate")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--after", help="Evaluate sessions created after this ISO timestamp")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = args.model or _latest("temporal_mobilenet_v3_*.onnx")
    index = build_index(args.dataset, args.curation)
    sessions = index["sessions"]
    selection = "all curated sessions"
    if args.after:
        sessions = [session for session in sessions if session["created_at"] > args.after]
        selection = f"sessions after {args.after}"
    else:
        split_path = args.split or _latest("temporal_mobilenet_v3_*_split.json")
        validation_ids = set(
            json.loads(split_path.read_text(encoding="utf-8"))["validation_sample_ids"]
        )
        sessions = [session for session in sessions if session["sample_id"] in validation_ids]
        selection = f"development validation split {split_path.name}"
    if not sessions:
        raise RuntimeError(f"No sessions selected for {selection}")

    dataset = TemporalSessionDataset(sessions, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    runtime = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rows = []
    offset = 0
    for inputs, _targets in loader:
        logits = runtime.run(None, {"temporal_triplet": inputs.numpy()})[0]
        probabilities = _softmax(logits)
        for values in probabilities:
            session = sessions[offset]
            predicted, confidence = _prediction(values)
            rows.append({
                "sample_id": session["sample_id"],
                "item_or_control_id": session["item_or_control_id"],
                "actual": session["training_label"],
                "predicted": predicted,
                "confidence": confidence,
                "probabilities": {
                    label: float(values[index]) for index, label in enumerate(CLASSES)
                },
                "correct": predicted == session["training_label"],
            })
            offset += 1

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for row in rows:
        confusion[CLASSES.index(row["actual"]), CLASSES.index(row["predicted"])] += 1
    gold_index = CLASSES.index("gold_22k")
    false_gold = int(confusion[:, gold_index].sum() - confusion[gold_index, gold_index])
    correct = sum(row["correct"] for row in rows)
    report = {
        "status": "development_candidate_not_for_production",
        "model": str(model_path),
        "selection": selection,
        "sample_count": len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows),
        "false_gold_count": false_gold,
        "classes": list(CLASSES),
        "confusion": confusion.tolist(),
        "samples": rows,
    }
    output_path = args.output or model_path.with_name(
        f"{model_path.stem}_candidate_evaluation.json"
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selection: {selection}")
    print(f"Accuracy: {correct}/{len(rows)} ({correct / len(rows):.1%})")
    print(f"False gold: {false_gold}")
    for row in rows:
        print(
            f"{row['item_or_control_id']}: actual={row['actual']} "
            f"predicted={row['predicted']} confidence={row['confidence']:.3f}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
