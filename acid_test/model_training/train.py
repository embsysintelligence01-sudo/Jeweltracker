#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader

from session_dataset import (
    CLASSES,
    INPUT_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    TemporalSessionDataset,
    build_index,
    stratified_split,
)
from temporal_model import ARCHITECTURES, make_temporal_model


ROOT = Path(__file__).resolve().parent
LAB_ROOT = ROOT.parent
DEFAULT_DATASET = LAB_ROOT / "dataset"
DEFAULT_CURATION = ROOT / "curation.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
GOLD_CONFIDENCE = 0.85
GENERAL_CONFIDENCE = 0.65


def _counts(sessions: list[dict[str, Any]]) -> dict[str, int]:
    values = Counter(session["training_label"] for session in sessions)
    return {label: values[label] for label in CLASSES}


def _predictions(probabilities: torch.Tensor) -> torch.Tensor:
    confidence, prediction = probabilities.max(dim=1)
    invalid_index = CLASSES.index("invalid")
    gold_index = CLASSES.index("gold_22k")
    prediction = prediction.clone()
    prediction[confidence < GENERAL_CONFIDENCE] = invalid_index
    prediction[(prediction == gold_index) & (confidence < GOLD_CONFIDENCE)] = invalid_index
    return prediction


def sessions_from_split(
    sessions: list[dict[str, Any]],
    split: dict[str, Any],
    add_unlisted_to_training: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_ids = set(split["validation_sample_ids"])
    training_ids = (
        {session["sample_id"] for session in sessions} - validation_ids
        if add_unlisted_to_training
        else set(split["training_sample_ids"])
    )
    training_sessions = [
        session for session in sessions if session["sample_id"] in training_ids
    ]
    validation_sessions = [
        session for session in sessions if session["sample_id"] in validation_ids
    ]
    if len(training_sessions) != len(training_ids) or len(validation_sessions) != len(validation_ids):
        raise RuntimeError("The supplied index does not contain every split session")
    return training_sessions, validation_sessions


def evaluate(model, loader, criterion, device) -> dict[str, Any]:
    model.eval()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits, targets)) * targets.size(0)
            predictions = _predictions(torch.softmax(logits, dim=1))
            for target, prediction in zip(targets.tolist(), predictions.tolist()):
                confusion[target, prediction] += 1
    total = int(confusion.sum())
    gold_index = CLASSES.index("gold_22k")
    false_gold = int(confusion[:, gold_index].sum() - confusion[gold_index, gold_index])
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(np.trace(confusion)) / max(total, 1),
        "false_gold": false_gold,
        "confusion": confusion.tolist(),
        "sample_count": total,
    }


def _export_onnx(model: nn.Module, path: Path, architecture: str) -> None:
    exported = copy.deepcopy(model).cpu().eval()
    torch.onnx.export(
        exported,
        torch.zeros((1, 3, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32),
        path,
        input_names=["temporal_triplet"],
        output_names=["logits"],
        dynamic_axes={"temporal_triplet": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    graph = onnx.load(path)
    metadata = {
        "architecture": architecture,
        "classes": json.dumps(list(CLASSES)),
        "input_contract": "baseline median, last 3 transition median, last 5 observation median",
        "input_shape": f"N,3,3,{INPUT_SIZE},{INPUT_SIZE}",
        "resize": "aspect-preserving letterbox with RGB fill 24",
        "normalization_mean": ",".join(str(value) for value in NORMALIZATION_MEAN),
        "normalization_std": ",".join(str(value) for value in NORMALIZATION_STD),
        "gold_confidence": str(GOLD_CONFIDENCE),
        "general_confidence": str(GENERAL_CONFIDENCE),
        "status": "development_candidate_not_for_production",
    }
    del graph.metadata_props[:]
    for key, value in metadata.items():
        prop = graph.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, path)


def _verify_onnx(
    model: nn.Module, loader: DataLoader, path: Path
) -> dict[str, float]:
    inputs, _ = next(iter(loader))
    selected = inputs[: min(4, len(inputs))]
    model = model.cpu().eval()
    with torch.no_grad():
        expected = model(selected).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    actual = session.run(None, {"temporal_triplet": selected.numpy()})[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "maximum_logit_difference": float(np.max(np.abs(expected - actual))),
        "batch_size": int(len(selected)),
        "batch_latency_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal acid session classifier")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument(
        "--validation-split",
        type=Path,
        help="Keep this split's validation sessions fixed and train on every other session",
    )
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=ARCHITECTURES[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    index = (
        json.loads(args.index.read_text(encoding="utf-8"))
        if args.index
        else build_index(args.dataset, args.curation)
    )
    if args.split and args.validation_split:
        parser.error("--split and --validation-split cannot be used together")
    if args.split or args.validation_split:
        split_path = args.split or args.validation_split
        split = json.loads(split_path.read_text(encoding="utf-8"))
        training_sessions, validation_sessions = sessions_from_split(
            index["sessions"], split, add_unlisted_to_training=bool(args.validation_split)
        )
    else:
        training_sessions, validation_sessions = stratified_split(
            index["sessions"], args.seed
        )
    print(f"Curated counts: {index['counts']}", flush=True)
    print(f"Training counts: {_counts(training_sessions)}", flush=True)
    print(f"Validation counts: {_counts(validation_sessions)}", flush=True)

    training_dataset = TemporalSessionDataset(training_sessions, augment=True)
    validation_dataset = TemporalSessionDataset(validation_sessions, augment=False)
    training_loader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cpu")
    model = make_temporal_model(args.architecture, pretrained=True).to(device)
    class_counts = _counts(training_sessions)
    weights = torch.tensor(
        [len(training_sessions) / max(class_counts[label], 1) for label in CLASSES],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-3, weight_decay=1e-4)

    history = []
    best_state = None
    best_rank = None
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in training_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * targets.size(0)
            seen += targets.size(0)
        metrics = evaluate(model, validation_loader, criterion, device)
        metrics["epoch"] = epoch
        metrics["training_loss"] = running_loss / max(seen, 1)
        history.append(metrics)
        rank = (metrics["false_gold"], -metrics["accuracy"], metrics["loss"])
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch:02d} train_loss={metrics['training_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} val_accuracy={metrics['accuracy']:.1%} "
            f"false_gold={metrics['false_gold']}",
            flush=True,
        )

    model.load_state_dict(best_state)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    stem = f"temporal_{args.architecture}_{timestamp}"
    checkpoint_path = args.artifacts / f"{stem}.pt"
    onnx_path = args.artifacts / f"{stem}.onnx"
    index_path = args.artifacts / f"{stem}_dataset_index.json"
    split_path = args.artifacts / f"{stem}_split.json"
    report_path = args.artifacts / f"{stem}_report.json"
    torch.save({
        "architecture": args.architecture,
        "classes": list(CLASSES),
        "state_dict": model.state_dict(),
    }, checkpoint_path)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    split_path.write_text(json.dumps({
        "warning": "Development split only. Future sessions are the untouched holdout.",
        "seed": args.seed,
        "training_sample_ids": [s["sample_id"] for s in training_sessions],
        "validation_sample_ids": [s["sample_id"] for s in validation_sessions],
    }, indent=2), encoding="utf-8")
    _export_onnx(model, onnx_path, args.architecture)
    onnx_verification = _verify_onnx(model, validation_loader, onnx_path)
    report = {
        "status": "development_candidate_not_for_production",
        "architecture": args.architecture,
        "classes": list(CLASSES),
        "curated_counts": index["counts"],
        "training_counts": _counts(training_sessions),
        "validation_counts": _counts(validation_sessions),
        "best_validation": best_metrics,
        "history": history,
        "onnx_verification": onnx_verification,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"ONNX: {onnx_path}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
