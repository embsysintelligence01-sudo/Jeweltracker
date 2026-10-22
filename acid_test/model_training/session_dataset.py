from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CLASSES = ("gold_22k", "non_gold", "invalid")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
INPUT_SIZE = 224
NORMALIZATION_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
NORMALIZATION_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_paths(sample_dir: Path, entries: list[str | dict]) -> list[Path]:
    return [
        sample_dir / (entry if isinstance(entry, str) else entry["path"])
        for entry in entries
    ]


def build_index(dataset_root: Path, curation_path: Path) -> dict[str, Any]:
    curation = json.loads(curation_path.read_text(encoding="utf-8"))
    overrides = curation.get("invalid_overrides", {})
    sessions = []
    legacy_count = 0
    found_ids = set()
    for manifest_path in sorted((dataset_root / "samples").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_version") != "acid_temporal_v2":
            legacy_count += 1
            continue
        sample_id = manifest["sample_id"]
        original_label = manifest["label"]
        training_label = "invalid" if original_label == "inconclusive" else original_label
        reason = "saved as inconclusive" if original_label == "inconclusive" else "verified material"
        if sample_id in overrides:
            training_label = "invalid"
            reason = str(overrides[sample_id])
            found_ids.add(sample_id)
        if training_label not in CLASS_TO_INDEX:
            raise RuntimeError(f"Unsupported training label {training_label}: {manifest_path}")

        baseline = _frame_paths(manifest_path.parent, manifest["baseline_frames"])
        transition = _frame_paths(
            manifest_path.parent, manifest.get("transition_frames", [])
        )
        observation = _frame_paths(manifest_path.parent, manifest["observation_frames"])
        if not baseline or not transition or not observation:
            raise RuntimeError(f"Incomplete temporal session: {manifest_path.parent}")
        for frame_path in baseline + transition + observation:
            if not frame_path.is_file():
                raise RuntimeError(f"Missing session frame: {frame_path}")
        sessions.append({
            "sample_id": sample_id,
            "item_or_control_id": manifest.get("metadata", {}).get(
                "item_or_control_id", ""
            ),
            "created_at": manifest["created_at"],
            "original_label": original_label,
            "training_label": training_label,
            "class_index": CLASS_TO_INDEX[training_label],
            "curation_reason": reason,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "baseline_frames": [str(path) for path in baseline],
            "transition_frames": [str(path) for path in transition],
            "observation_frames": [str(path) for path in observation],
        })

    missing_overrides = sorted(set(overrides) - found_ids)
    if missing_overrides:
        raise RuntimeError(f"Curated session IDs were not found: {missing_overrides}")
    counts = Counter(session["training_label"] for session in sessions)
    return {
        "classes": list(CLASSES),
        "dataset_version": "acid_temporal_v2",
        "legacy_sessions_excluded": legacy_count,
        "counts": {label: counts[label] for label in CLASSES},
        "sessions": sessions,
    }


def stratified_split(
    sessions: list[dict[str, Any]], seed: int, validation_fraction: float = 0.25
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generator = random.Random(seed)
    training = []
    validation = []
    for label in CLASSES:
        members = [session for session in sessions if session["training_label"] == label]
        generator.shuffle(members)
        validation_count = max(2, round(len(members) * validation_fraction))
        validation.extend(members[:validation_count])
        training.extend(members[validation_count:])
    generator.shuffle(training)
    generator.shuffle(validation)
    return training, validation


def prepare_temporal_triplet(
    baseline_frames: list[np.ndarray],
    transition_frames: list[np.ndarray],
    observation_frames: list[np.ndarray],
) -> np.ndarray:
    if not baseline_frames or not transition_frames or not observation_frames:
        raise ValueError("Baseline, transition, and observation frames are required")
    if any(
        frame is None or not frame.size
        for frames in (baseline_frames, transition_frames, observation_frames)
        for frame in frames
    ):
        raise RuntimeError("Could not read a temporal session frame")

    def median(frames: list[np.ndarray], tail_count: int | None = None) -> np.ndarray:
        selected = frames[-tail_count:] if tail_count else frames
        return np.median(np.stack(selected), axis=0).astype(np.uint8)

    return np.stack((
        _letterbox(median(baseline_frames)),
        _letterbox(median(transition_frames, tail_count=3)),
        _letterbox(median(observation_frames, tail_count=5)),
    ))


def _letterbox(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 24, dtype=np.uint8)
    x = (INPUT_SIZE - resized_width) // 2
    y = (INPUT_SIZE - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def load_triplet(session: dict[str, Any]) -> np.ndarray:
    return prepare_temporal_triplet(
        [cv2.imread(path, cv2.IMREAD_COLOR) for path in session["baseline_frames"]],
        [cv2.imread(path, cv2.IMREAD_COLOR) for path in session["transition_frames"]],
        [cv2.imread(path, cv2.IMREAD_COLOR) for path in session["observation_frames"]],
    )


class TemporalSessionDataset(Dataset):
    def __init__(self, sessions: list[dict[str, Any]], augment: bool):
        self.sessions = sessions
        self.augment = augment
        self.triplets = [load_triplet(session) for session in sessions]

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int):
        images = self.triplets[index]
        if self.augment:
            if random.random() < 0.5:
                images = np.flip(images, axis=2)
            if random.random() < 0.5:
                images = np.flip(images, axis=1)
            gain = random.uniform(0.9, 1.1)
            images = np.clip(images.astype(np.float32) * gain, 0, 255)
        images = np.ascontiguousarray(images, dtype=np.float32) / 255.0
        images = (images - NORMALIZATION_MEAN) / NORMALIZATION_STD
        tensor = torch.from_numpy(images.transpose(0, 3, 1, 2))
        target = int(self.sessions[index]["class_index"])
        return tensor, target
