from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LABELS = ("gold_22k", "non_gold", "inconclusive")
LEGACY_LABELS = {"persisted": "gold_22k", "reacted": "non_gold"}
DATASET_VERSION = "acid_temporal_v2"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class TemporalDatasetStore:
    def __init__(self, root: Path, detector_path: Path):
        self.root = root
        self.detector_path = detector_path
        self.samples_dir = root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.detector_sha256 = sha256_file(detector_path)

    def counts(self) -> dict[str, int]:
        counts = {label: 0 for label in LABELS}
        for path in self.samples_dir.glob("*/manifest.json"):
            try:
                label = json.loads(path.read_text(encoding="utf-8"))["label"]
            except (OSError, KeyError, json.JSONDecodeError):
                continue
            label = LEGACY_LABELS.get(label, label)
            if label in counts:
                counts[label] += 1
        return counts

    def allocate_item_id(self, prefix: str, label: str) -> str:
        if label not in LABELS:
            raise ValueError(f"Label must be one of: {', '.join(LABELS)}")
        if not re.fullmatch(r"[a-z0-9-]+", prefix):
            raise ValueError("Item ID prefix contains unsupported characters")

        counter_path = self.root / "item_id_counters.json"
        try:
            counters = json.loads(counter_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            counters = {}
        except (OSError, json.JSONDecodeError):
            raise RuntimeError(f"Could not read automatic ID counters: {counter_path}")

        highest_saved = self.counts()[label]
        pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
        for manifest_path in self.samples_dir.glob("*/manifest.json"):
            try:
                metadata = json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"]
                item_id = metadata.get("item_or_control_id", "")
            except (OSError, KeyError, json.JSONDecodeError, AttributeError):
                continue
            match = pattern.fullmatch(str(item_id))
            if match:
                highest_saved = max(highest_saved, int(match.group(1)))

        next_number = max(int(counters.get(prefix, 0)), highest_saved) + 1
        counters[prefix] = next_number
        temporary_path = self.root / ".item_id_counters.json.tmp"
        temporary_path.write_text(json.dumps(counters, indent=2), encoding="utf-8")
        temporary_path.replace(counter_path)
        return f"{prefix}-{next_number:03d}"

    @staticmethod
    def _write_png(path: Path, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0 or not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not save image: {path}")

    def save(
        self,
        *,
        label: str,
        baseline_frames: list[np.ndarray],
        transition_frames: list[np.ndarray],
        transition_offsets: list[float],
        observation_frames: list[np.ndarray],
        observation_offsets: list[float],
        detector_bbox: tuple[int, int, int, int],
        reaction_bbox: tuple[int, int, int, int],
        detector_confidence: float,
        metadata: dict[str, str],
        metrics: dict[str, Any],
        camera_shape: tuple[int, int, int],
        baseline_seconds: float,
        observation_seconds: float,
        capture_fps: float,
    ) -> dict[str, Any]:
        if label not in LABELS:
            raise ValueError(f"Label must be one of: {', '.join(LABELS)}")
        if not baseline_frames or not observation_frames:
            raise ValueError("A temporal sample requires baseline and observation frames")
        if len(observation_frames) != len(observation_offsets):
            raise ValueError("Observation frames and timestamps do not match")
        if len(transition_frames) != len(transition_offsets):
            raise ValueError("Transition frames and timestamps do not match")

        sample_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
        temporary_dir = self.samples_dir / f".{sample_id}.tmp"
        sample_dir = self.samples_dir / sample_id
        temporary_dir.mkdir(parents=True)
        try:
            baseline_dir = temporary_dir / "baseline"
            transition_dir = temporary_dir / "transition"
            observation_dir = temporary_dir / "observation"
            baseline_dir.mkdir()
            transition_dir.mkdir()
            observation_dir.mkdir()
            for index, frame in enumerate(baseline_frames):
                self._write_png(baseline_dir / f"frame_{index:03d}.png", frame)
            transition_files = []
            for index, (frame, offset) in enumerate(zip(transition_frames, transition_offsets)):
                name = f"frame_{index:04d}_{offset:07.3f}s.png"
                self._write_png(transition_dir / name, frame)
                transition_files.append({
                    "path": f"transition/{name}",
                    "offset_seconds": offset,
                })
            observation_files = []
            for index, (frame, offset) in enumerate(zip(observation_frames, observation_offsets)):
                name = f"frame_{index:04d}_{offset:07.3f}s.png"
                self._write_png(observation_dir / name, frame)
                observation_files.append({"path": f"observation/{name}", "offset_seconds": offset})

            manifest = {
                "dataset_version": DATASET_VERSION,
                "sample_id": sample_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "label": label,
                "metadata": {str(key): str(value) for key, value in metadata.items()},
                "camera_shape": list(camera_shape),
                "baseline_seconds": float(baseline_seconds),
                "observation_seconds": float(observation_seconds),
                "capture_fps": float(capture_fps),
                "detector_model": self.detector_path.name,
                "detector_sha256": self.detector_sha256,
                "detector_bbox": list(detector_bbox),
                "reaction_bbox": list(reaction_bbox),
                "detector_confidence": float(detector_confidence),
                "baseline_frames": [
                    f"baseline/frame_{index:03d}.png"
                    for index in range(len(baseline_frames))
                ],
                "transition_frames": transition_files,
                "acid_prompt_to_detection_seconds": (
                    float(transition_offsets[-1]) if transition_offsets else 0.0
                ),
                "observation_frames": observation_files,
                "analysis": metrics,
            }
            (temporary_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            temporary_dir.replace(sample_dir)
            return manifest
        except Exception:
            for path in sorted(temporary_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if temporary_dir.exists():
                temporary_dir.rmdir()
            raise
