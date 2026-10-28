from __future__ import annotations
import hashlib
import time
from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
from .model_training.session_dataset import CLASSES as TEMPORAL_CLASSES, NORMALIZATION_MEAN, NORMALIZATION_STD, prepare_temporal_triplet

LAB_ROOT = Path(__file__).resolve().parent
DETECTOR_PATH = LAB_ROOT / "models" / "acid_detector.onnx"
MODEL_ARTIFACTS_ROOT = LAB_ROOT / "model_training" / "artifacts"
GOLD_CONFIDENCE = 0.85
GENERAL_CONFIDENCE = 0.65


def classify_probabilities(values):
    """Explain the existing thresholds without changing model decisions."""
    index = int(np.argmax(values))
    raw = TEMPORAL_CLASSES[index]
    confidence = float(values[index])
    if confidence < GENERAL_CONFIDENCE:
        return 'invalid', raw, confidence, f'Highest model confidence {confidence:.1%} is below the {GENERAL_CONFIDENCE:.0%} minimum.'
    if raw == 'gold_22k' and confidence < GOLD_CONFIDENCE:
        return 'invalid', raw, confidence, f'Gold confidence {confidence:.1%} is below the {GOLD_CONFIDENCE:.0%} gold threshold.'
    if raw == 'invalid':
        return raw, raw, confidence, 'The model selected its INVALID class; the model does not identify a specific physical cause.'
    return raw, raw, confidence, None

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def latest_temporal_model_path():
    candidates = list(MODEL_ARTIFACTS_ROOT.glob("temporal_mobilenet_v3_*.onnx"))
    candidates.extend(MODEL_ARTIFACTS_ROOT.glob("temporal_shared_mobilenet_v3_small_*.onnx"))
    if not candidates:
        raise RuntimeError("No local temporal MobileNet ONNX model was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)

class TemporalMobileNet:
    def __init__(self, path: Path):
        self.path = path
        self.sha256 = _sha256(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        if input_info.name != "temporal_triplet" or input_info.shape[-4:] != [3, 3, 224, 224]:
            raise RuntimeError("Temporal MobileNet input contract is incompatible")
        if output_info.name != "logits" or output_info.shape[-1] != len(TEMPORAL_CLASSES):
            raise RuntimeError("Temporal MobileNet output contract is incompatible")

    def predict(
        self,
        baseline_frames: list[np.ndarray],
        transition_frames: list[np.ndarray],
        observation_frames: list[np.ndarray],
    ) -> dict[str, Any]:
        triplet = prepare_temporal_triplet(
            baseline_frames, transition_frames, observation_frames
        ).astype(np.float32) / 255.0
        triplet = (triplet - NORMALIZATION_MEAN) / NORMALIZATION_STD
        tensor = np.ascontiguousarray(triplet.transpose(0, 3, 1, 2)[None])
        started = time.perf_counter()
        logits = self.session.run(["logits"], {"temporal_triplet": tensor})[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        values = probabilities[0]
        prediction, raw_prediction, confidence, invalid_reason = classify_probabilities(values)
        return {
            "model": self.path.name,
            "model_sha256": self.sha256,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "invalid_reason": invalid_reason,
            "thresholds": dict(general=GENERAL_CONFIDENCE, gold=GOLD_CONFIDENCE),
            "confidence": confidence,
            "probabilities": {
                name: float(values[position])
                for position, name in enumerate(TEMPORAL_CLASSES)
            },
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

