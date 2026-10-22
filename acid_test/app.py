#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PyQt6.QtCore import QProcess, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# OpenCV's Qt5 plugin paths must not be used by the standalone Qt6 UI.
if sys.platform.startswith("linux"):
    for _qt_env in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "/cv2/qt/" in os.environ.get(_qt_env, "").replace("\\", "/"):
            os.environ.pop(_qt_env, None)


LAB_ROOT = Path(__file__).resolve().parent
PURITY_ROOT = LAB_ROOT.parent
if str(PURITY_ROOT) not in sys.path:
    sys.path.insert(0, str(PURITY_ROOT))

from acid_test.flow import bbox_iou, pad_bbox  # noqa: E402
from acid_test.onnx_detector import OnnxAcidDetector  # noqa: E402

from acid_test.analyzer import median_frame  # noqa: E402
from acid_test.analyzer_v2 import analyze_sequence_v2  # noqa: E402
from acid_test.model_training.session_dataset import (  # noqa: E402
    CLASSES as TEMPORAL_CLASSES,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    prepare_temporal_triplet,
)
from acid_test.storage import LABELS, TemporalDatasetStore  # noqa: E402


DETECTOR_PATH = LAB_ROOT / "models" / "acid_detector.onnx"
DATASET_ROOT = LAB_ROOT / "dataset"
MODEL_TRAINING_ROOT = LAB_ROOT / "model_training"
MODEL_ARTIFACTS_ROOT = MODEL_TRAINING_ROOT / "artifacts"
TRAIN_SCRIPT = MODEL_TRAINING_ROOT / "train.py"
GOLD_CONFIDENCE = 0.85
GENERAL_CONFIDENCE = 0.65
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 1440
CAMERA_FPS = 30
BASELINE_SECONDS = 2.0
CAPTURE_FPS = 5.0
TRANSITION_CAPTURE_FPS = 15.0
TRANSITION_BUFFER_SECONDS = 15.0
TRANSITION_JPEG_QUALITY = 95
DETECTION_CONFIDENCE = 0.60
DETECTION_CONFIRMATIONS = 3
DETECTION_IOU = 0.45
REACTION_PADDING = 0.45
SAMPLE_TYPES = (
    ("22K", "22k", "gold_22k"),
    ("Non-gold", "non-gold", "non_gold"),
    ("Glare", "glare", "inconclusive"),
)


def latest_temporal_model_path() -> Path:
    candidates = list(MODEL_ARTIFACTS_ROOT.glob("temporal_mobilenet_v3_*.onnx"))
    candidates.extend(
        MODEL_ARTIFACTS_ROOT.glob("temporal_shared_mobilenet_v3_small_*.onnx")
    )
    if not candidates:
        raise RuntimeError("No trained temporal MobileNet ONNX model was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_companion_path(model_path: Path, suffix: str) -> Path:
    return model_path.with_name(f"{model_path.stem}_{suffix}.json")


def _session_ids(dataset_root: Path) -> set[str]:
    sample_ids = set()
    for manifest_path in (dataset_root / "samples").glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if manifest.get("dataset_version") == "acid_temporal_v2":
            sample_ids.add(str(manifest["sample_id"]))
    return sample_ids


from acid_test.runtime import TemporalMobileNet


class CameraWorker(QThread):
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._number = 0

    def snapshot(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._number, self._frame

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        capture = None
        for index in range(5):
            candidate = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not candidate.isOpened():
                candidate.release()
                continue
            candidate.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            candidate.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            candidate.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            candidate.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            candidate.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            for _ in range(6):
                candidate.grab()
            ok, frame = candidate.read()
            if ok and frame is not None and frame.size:
                capture = candidate
                self.status.emit(
                    f"Camera /dev/video{index}: {frame.shape[1]}x{frame.shape[0]}"
                )
                break
            candidate.release()
        if capture is None:
            self.status.emit("Camera unavailable")
            return
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok or frame is None or not frame.size:
                self.status.emit("Camera disconnected")
                break
            with self._lock:
                self._frame = frame
                self._number += 1
        capture.release()


class AcidWorker(QThread):
    ready = pyqtSignal(str)
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._stopping = False

    def submit(self, frame: np.ndarray) -> None:
        with self._condition:
            self._pending = frame.copy()
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def run(self) -> None:
        try:
            detector = OnnxAcidDetector(DETECTOR_PATH)
            self.ready.emit(f"{DETECTOR_PATH.name} ready (CPU)")
            while True:
                with self._condition:
                    while self._pending is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        return
                    frame = self._pending
                    self._pending = None
                detections, elapsed_ms = detector.predict(frame, DETECTION_CONFIDENCE)
                best = max(
                    detections,
                    key=lambda item: max(1, item.bbox[2] - item.bbox[0])
                    * max(1, item.bbox[3] - item.bbox[1])
                    * item.confidence,
                    default=None,
                )
                self.result.emit({
                    "bbox": None if best is None else best.bbox,
                    "confidence": 0.0 if best is None else best.confidence,
                    "elapsed_ms": elapsed_ms,
                })
        except Exception as exc:
            self.failed.emit(str(exc))


class TemporalLabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera = CameraWorker()
        self.detector = AcidWorker()
        self.store = TemporalDatasetStore(DATASET_ROOT, DETECTOR_PATH)
        self.state = "IDLE"
        self.state_started = time.monotonic()
        self.last_sample_at = 0.0
        self.last_transition_at = 0.0
        self.last_detector_submit_at = 0.0
        self.last_frame_number = -1
        self.baseline_full_frames: deque[np.ndarray] = deque(maxlen=20)
        self.baseline_crops: list[np.ndarray] = []
        self.transition_buffer: deque[tuple[float, np.ndarray]] = deque(
            maxlen=round(TRANSITION_CAPTURE_FPS * TRANSITION_BUFFER_SECONDS)
        )
        self.transition_crops: list[np.ndarray] = []
        self.transition_offsets: list[float] = []
        self.observation_crops: list[np.ndarray] = []
        self.observation_offsets: list[float] = []
        self.detected_at = 0.0
        self.acid_prompted_at = 0.0
        self.previous_detection: tuple[int, int, int, int] | None = None
        self.detection_hits = 0
        self.detector_bbox: tuple[int, int, int, int] | None = None
        self.reaction_bbox: tuple[int, int, int, int] | None = None
        self.detector_confidence = 0.0
        self.camera_shape: tuple[int, int, int] | None = None
        self.metrics: dict[str, Any] = {}
        self.candidate: TemporalMobileNet | None = None
        self.trained_session_ids: set[str] = set()
        self.current_item_id = ""
        self._build_ui()
        self.train_process = QProcess(self)
        self.train_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.train_process.readyReadStandardOutput.connect(self._training_output)
        self.train_process.finished.connect(self._training_finished)
        self._load_latest_candidate()
        self.camera.status.connect(self._status)
        self.detector.ready.connect(self._status)
        self.detector.failed.connect(self._detector_failed)
        self.detector.result.connect(self._detection_result)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self.camera.start()
        self.detector.start()
        self._refresh_counts()

    def _build_ui(self) -> None:
        self.setWindowTitle("Acid Temporal Lab")
        self.resize(1280, 850)
        central = QWidget()
        root = QVBoxLayout(central)
        title = QLabel("ACID TEMPORAL CAPTURE LAB")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)
        body = QHBoxLayout()
        self.preview = QLabel("Starting camera...")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(850, 600)
        body.addWidget(self.preview, 3)
        side = QVBoxLayout()
        form = QFormLayout()
        self.sample_type = QComboBox()
        for text, prefix, label in SAMPLE_TYPES:
            self.sample_type.addItem(text, (prefix, label))
        self.automatic_item_id = QLabel("Assigned when test starts")
        self.acid_id = QLineEdit()
        self.acid_id.setPlaceholderText("grade / bottle / batch")
        self.operator = QLineEdit()
        self.notes = QLineEdit()
        self.duration = QDoubleSpinBox()
        self.duration.setRange(5.0, 180.0)
        self.duration.setValue(5.0)
        self.duration.setSuffix(" seconds")
        form.addRow("Sample type", self.sample_type)
        form.addRow("Automatic test ID", self.automatic_item_id)
        form.addRow("Acid grade / batch", self.acid_id)
        form.addRow("Operator", self.operator)
        form.addRow("Notes", self.notes)
        form.addRow("Observation", self.duration)
        side.addLayout(form)
        self.start_button = QPushButton("START NEW TEST")
        self.start_button.clicked.connect(self._start_test)
        side.addWidget(self.start_button)
        self.instruction = QLabel("Select the sample type, enter the acid ID, then start before pouring acid.")
        self.instruction.setObjectName("instruction")
        self.instruction.setWordWrap(True)
        side.addWidget(self.instruction)
        self.roi_preview = QLabel("Reaction ROI sequence")
        self.roi_preview.setObjectName("roi")
        self.roi_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_preview.setFixedSize(330, 250)
        side.addWidget(self.roi_preview, 0, Qt.AlignmentFlag.AlignHCenter)
        self.analysis = QLabel("Analysis: not started")
        self.analysis.setWordWrap(True)
        side.addWidget(self.analysis)
        self.label_buttons = {}
        for label, text in (
            ("gold_22k", "SAVE: KNOWN 22K GOLD"),
            ("non_gold", "SAVE: KNOWN NON-GOLD"),
            ("inconclusive", "SAVE: INCONCLUSIVE"),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, value=label: self._save(value))
            button.setEnabled(False)
            self.label_buttons[label] = button
            side.addWidget(button)
        self.retrain_button = QPushButton("RETRAIN MODEL")
        self.retrain_button.clicked.connect(self._start_retraining)
        self.retrain_button.setEnabled(False)
        side.addWidget(self.retrain_button)
        self.model_status = QLabel("MobileNet: loading...")
        self.model_status.setWordWrap(True)
        side.addWidget(self.model_status)
        self.discard_button = QPushButton("DISCARD TEST")
        self.discard_button.clicked.connect(self._discard)
        self.discard_button.setEnabled(False)
        side.addWidget(self.discard_button)
        self.counts = QLabel()
        side.addWidget(self.counts)
        self.status = QLabel("Starting camera and detector...")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        side.addWidget(self.status, 1)
        body.addLayout(side, 1)
        root.addLayout(body, 1)
        self.setCentralWidget(central)
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#0f172a;color:#e2e8f0;font-family:Sans;}"
            "#title{font-size:24px;font-weight:700;padding:8px;}"
            "#preview,#roi{background:#020617;border:2px solid #475569;color:#94a3b8;}"
            "#instruction,#status{padding:10px;background:#111827;border:1px solid #334155;}"
            "QPushButton{font-size:14px;font-weight:700;padding:10px;background:#2563eb;border-radius:5px;}"
            "QPushButton:disabled{background:#334155;color:#64748b;}"
            "QLineEdit,QComboBox,QDoubleSpinBox{padding:6px;background:#1e293b;border:1px solid #475569;}"
        )

    def _status(self, message: str) -> None:
        self.status.setText(message)

    def _load_latest_candidate(self) -> bool:
        try:
            path = latest_temporal_model_path()
            candidate = TemporalMobileNet(path)
            index = json.loads(
                _model_companion_path(path, "dataset_index").read_text(encoding="utf-8")
            )
            trained_session_ids = {
                str(session["sample_id"]) for session in index["sessions"]
            }
        except Exception as exc:
            if self.candidate is None:
                self.model_status.setText(f"MobileNet unavailable: {exc}")
            else:
                self.model_status.setText(
                    f"Keeping {self.candidate.path.name}; new model failed to load: {exc}"
                )
            return False
        self.candidate = candidate
        self.trained_session_ids = trained_session_ids
        self.model_status.setText(f"MobileNet active: {path.name}")
        self._refresh_retrain_button()
        return True

    def _new_session_count(self) -> int:
        return len(_session_ids(DATASET_ROOT) - self.trained_session_ids)

    def _refresh_retrain_button(self) -> None:
        new_count = self._new_session_count() if self.candidate is not None else 0
        running = hasattr(self, "train_process") and (
            self.train_process.state() != QProcess.ProcessState.NotRunning
        )
        self.retrain_button.setText(
            f"RETRAIN WITH {new_count} NEW SAVED SESSION{'S' if new_count != 1 else ''}"
        )
        self.retrain_button.setEnabled(
            new_count > 0 and not running and self.state == "IDLE"
        )

    def _start_retraining(self) -> None:
        if (
            self.candidate is None
            or self._new_session_count() == 0
            or self.train_process.state() != QProcess.ProcessState.NotRunning
        ):
            return
        validation_split = _model_companion_path(self.candidate.path, "split")
        if not validation_split.is_file():
            self.status.setText(
                f"Cannot retrain: fixed validation split is missing: {validation_split.name}"
            )
            return
        self.start_button.setEnabled(False)
        self.sample_type.setEnabled(False)
        self.retrain_button.setEnabled(False)
        self.status.setText(
            f"Retraining MobileNet with {self._new_session_count()} new saved sessions..."
        )
        self.train_process.setWorkingDirectory(str(MODEL_TRAINING_ROOT))
        self.train_process.start(
            sys.executable,
            [
                str(TRAIN_SCRIPT),
                "--architecture",
                "shared_mobilenet_v3_small",
                "--validation-split",
                str(validation_split),
            ],
        )

    def _training_output(self) -> None:
        output = bytes(self.train_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            self.status.setText(f"Training: {lines[-1]}")

    def _training_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ) -> None:
        self.start_button.setEnabled(self.state == "IDLE")
        self.sample_type.setEnabled(self.state == "IDLE")
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            self.status.setText(
                f"Retraining failed (exit {exit_code}); the previous MobileNet remains active."
            )
            self._refresh_retrain_button()
            return
        previous_path = self.candidate.path if self.candidate is not None else None
        if not self._load_latest_candidate() or self.candidate.path == previous_path:
            self.status.setText(
                "Retraining finished but no new valid ONNX was activated; "
                "the previous MobileNet remains active."
            )
            self._refresh_retrain_button()
            return
        self.status.setText(
            f"Retraining complete. Active model: {self.candidate.path.name}"
        )
        self._refresh_retrain_button()

    def _start_test(self) -> None:
        if self.state != "IDLE":
            return
        if not self.acid_id.text().strip():
            self.status.setText("Acid grade/batch is required.")
            return
        prefix, expected_label = self.sample_type.currentData()
        try:
            self.current_item_id = self.store.allocate_item_id(prefix, expected_label)
        except Exception as exc:
            self.status.setText(f"Could not assign automatic test ID: {exc}")
            return
        self.automatic_item_id.setText(self.current_item_id)
        self._clear_capture()
        self.state = "BASELINE"
        self.state_started = time.monotonic()
        self.start_button.setEnabled(False)
        self.sample_type.setEnabled(False)
        self._refresh_retrain_button()
        self.discard_button.setEnabled(True)
        self.instruction.setText("BASELINE: do not pour acid yet. Keep the stone and camera still.")

    def _clear_capture(self) -> None:
        self.baseline_full_frames.clear()
        self.baseline_crops.clear()
        self.transition_buffer.clear()
        self.transition_crops.clear()
        self.transition_offsets.clear()
        self.observation_crops.clear()
        self.observation_offsets.clear()
        self.previous_detection = None
        self.detection_hits = 0
        self.detector_bbox = None
        self.reaction_bbox = None
        self.detector_confidence = 0.0
        self.metrics = {}
        self.roi_preview.clear()
        self.roi_preview.setText("Reaction ROI sequence")
        self.analysis.setText("Analysis: not started")
        for button in self.label_buttons.values():
            button.setEnabled(False)

    def _tick(self) -> None:
        frame_number, frame = self.camera.snapshot()
        if frame is None:
            return
        if frame_number != self.last_frame_number:
            self.last_frame_number = frame_number
            self.camera_shape = tuple(frame.shape)
            self._show(self.preview, self._annotated(frame), fast=True)
        now = time.monotonic()
        if self.state == "BASELINE":
            if now - self.last_sample_at < 1.0 / CAPTURE_FPS:
                return
            self.last_sample_at = now
            self.baseline_full_frames.append(frame.copy())
            remaining = max(0.0, BASELINE_SECONDS - (now - self.state_started))
            self.instruction.setText(
                f"BASELINE: do not pour acid yet. {remaining:.1f} seconds remaining."
            )
            if remaining <= 0.0:
                self.state = "WAIT_FOR_ACID"
                self.state_started = now
                self.acid_prompted_at = now
                self.last_transition_at = 0.0
                self.last_detector_submit_at = 0.0
                self.instruction.setText("NOW APPLY ACID. Keep the stone still and remove the dropper.")
        elif self.state == "WAIT_FOR_ACID":
            if now - self.last_transition_at >= 1.0 / TRANSITION_CAPTURE_FPS:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, TRANSITION_JPEG_QUALITY],
                )
                if ok:
                    self.transition_buffer.append(
                        (now - self.acid_prompted_at, encoded)
                    )
                self.last_transition_at = now
            if now - self.last_detector_submit_at >= 1.0 / CAPTURE_FPS:
                self.detector.submit(frame)
                self.last_detector_submit_at = now
        elif self.state == "OBSERVING" and self.reaction_bbox is not None:
            if now - self.last_sample_at < 1.0 / CAPTURE_FPS:
                return
            self.last_sample_at = now
            x1, y1, x2, y2 = self.reaction_bbox
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size:
                self.observation_crops.append(crop)
                self.observation_offsets.append(now - self.detected_at)
                self._show(self.roi_preview, crop)
            remaining = max(0.0, self.duration.value() - (now - self.detected_at))
            self.instruction.setText(
                f"OBSERVING REACTION: keep still; hands out. {remaining:.1f} seconds remaining."
            )
            if remaining <= 0.0:
                self._finish_observation()

    def _detection_result(self, result: dict[str, Any]) -> None:
        if self.state != "WAIT_FOR_ACID":
            return
        bbox = result["bbox"]
        if bbox is None:
            self.previous_detection = None
            self.detection_hits = 0
            return
        if self.previous_detection is not None and bbox_iou(self.previous_detection, bbox) >= DETECTION_IOU:
            self.detection_hits += 1
        else:
            self.detection_hits = 1
        self.previous_detection = bbox
        self.instruction.setText(
            f"ACID DETECTED {self.detection_hits}/{DETECTION_CONFIRMATIONS}. Remove hands and keep still."
        )
        if self.detection_hits < DETECTION_CONFIRMATIONS:
            return
        frame_number, frame = self.camera.snapshot()
        if frame is None:
            return
        self.detector_bbox = bbox
        self.detector_confidence = float(result["confidence"])
        self.reaction_bbox = pad_bbox(
            bbox, REACTION_PADDING, frame.shape[1], frame.shape[0]
        )
        x1, y1, x2, y2 = self.reaction_bbox
        self.baseline_crops = [
            baseline[y1:y2, x1:x2].copy()
            for baseline in self.baseline_full_frames
            if baseline[y1:y2, x1:x2].size
        ]
        self.transition_crops = []
        self.transition_offsets = []
        for offset, encoded in self.transition_buffer:
            transition = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if transition is None:
                continue
            crop = transition[y1:y2, x1:x2].copy()
            if crop.size:
                self.transition_crops.append(crop)
                self.transition_offsets.append(offset)
        if not self.baseline_crops:
            self._detector_failed("Could not recover a baseline crop; discard and repeat")
            return
        self.state = "OBSERVING"
        self.detected_at = time.monotonic()

    def _finish_observation(self) -> None:
        if not self.observation_crops:
            self._detector_failed("No observation crops were recorded")
            return
        metrics = analyze_sequence_v2(
            self.baseline_crops,
            self.observation_crops,
            self.transition_crops,
        )
        self.metrics = metrics.to_dict()
        prediction_text = "MobileNet prediction unavailable"
        if self.candidate is not None:
            try:
                prediction = self.candidate.predict(
                    self.baseline_crops,
                    self.transition_crops,
                    self.observation_crops,
                )
                self.metrics["mobilenet"] = prediction
                display_prediction = {
                    "gold_22k": "22K GOLD",
                    "non_gold": "NON-GOLD",
                    "invalid": "INCONCLUSIVE",
                }[prediction["prediction"]]
                probabilities = prediction["probabilities"]
                prediction_text = (
                    f"MobileNet prediction: {display_prediction} "
                    f"({prediction['confidence']:.1%})\n"
                    f"22K {probabilities['gold_22k']:.1%} | "
                    f"Non-gold {probabilities['non_gold']:.1%} | "
                    f"Invalid {probabilities['invalid']:.1%}\n"
                    f"Model: {prediction['model']}"
                )
            except Exception as exc:
                prediction_text = f"MobileNet prediction failed: {exc}"
        self.state = "REVIEW"
        self._refresh_retrain_button()
        self._show(self.roi_preview, median_frame(self.observation_crops[-5:]))
        self.analysis.setText(
            f"{prediction_text}\n"
            f"Rule diagnostic: {metrics.suggestion.upper()} ({metrics.quality})\n"
            f"Baseline feature points: {metrics.baseline_feature_points}\n"
            f"Early feature retention: {metrics.early_feature_retention:.2f}\n"
            f"Final feature retention: {metrics.final_feature_retention:.2f}\n"
            f"Final change: {metrics.final_change:.3f}\n"
            f"Baseline glare: {metrics.baseline_glare_fraction:.1%}"
        )
        self.instruction.setText(
            "REVIEW: compare the MobileNet prediction with the verified material, "
            "then save the actual result."
        )
        for button in self.label_buttons.values():
            button.setEnabled(True)

    def _save(self, label: str) -> None:
        if self.state != "REVIEW" or label not in LABELS:
            return
        try:
            manifest = self.store.save(
                label=label,
                baseline_frames=self.baseline_crops,
                transition_frames=self.transition_crops,
                transition_offsets=self.transition_offsets,
                observation_frames=self.observation_crops,
                observation_offsets=self.observation_offsets,
                detector_bbox=self.detector_bbox,
                reaction_bbox=self.reaction_bbox,
                detector_confidence=self.detector_confidence,
                metadata={
                    "item_or_control_id": self.current_item_id,
                    "selected_sample_type": self.sample_type.currentText(),
                    "acid_grade_or_batch": self.acid_id.text().strip(),
                    "operator": self.operator.text().strip(),
                    "notes": self.notes.text().strip(),
                },
                metrics=self.metrics,
                camera_shape=self.camera_shape,
                baseline_seconds=BASELINE_SECONDS,
                observation_seconds=self.duration.value(),
                capture_fps=CAPTURE_FPS,
            )
        except Exception as exc:
            self.status.setText(f"Could not save temporal sample: {exc}")
            return
        self._reset_idle()
        self.status.setText(f"Saved {manifest['sample_id']} as {label.upper()}")
        self._refresh_counts()
        self._refresh_retrain_button()

    def _discard(self) -> None:
        self._reset_idle()
        self.status.setText("Test discarded; no sample was saved")

    def _reset_idle(self) -> None:
        self.state = "IDLE"
        self._clear_capture()
        self.start_button.setEnabled(True)
        self.sample_type.setEnabled(True)
        self.discard_button.setEnabled(False)
        self.current_item_id = ""
        self.automatic_item_id.setText("Assigned when test starts")
        self.instruction.setText("Select the sample type, enter the acid ID, then start before pouring acid.")
        self._refresh_retrain_button()

    def _detector_failed(self, message: str) -> None:
        self.status.setText(f"Acid temporal capture error: {message}")

    def _annotated(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        if self.previous_detection is not None:
            x1, y1, x2, y2 = self.previous_detection
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)
        if self.reaction_bbox is not None:
            x1, y1, x2, y2 = self.reaction_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
        return display

    @staticmethod
    def _show(label: QLabel, frame: np.ndarray, fast: bool = False) -> None:
        image = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format.Format_BGR888,
        ).copy()
        label.setPixmap(QPixmap.fromImage(image).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation if fast else Qt.TransformationMode.SmoothTransformation,
        ))

    def _refresh_counts(self) -> None:
        counts = self.store.counts()
        self.counts.setText(
            "Saved sequences — "
            f"22K GOLD: {counts['gold_22k']} | NON-GOLD: {counts['non_gold']} | "
            f"INCONCLUSIVE: {counts['inconclusive']}"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        if self.train_process.state() != QProcess.ProcessState.NotRunning:
            self.train_process.terminate()
            if not self.train_process.waitForFinished(3000):
                self.train_process.kill()
                self.train_process.waitForFinished(3000)
        self.camera.stop()
        self.detector.stop()
        self.camera.wait(3000)
        self.detector.wait(15000)
        event.accept()


def main() -> int:
    application = QApplication(sys.argv)
    window = TemporalLabWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
