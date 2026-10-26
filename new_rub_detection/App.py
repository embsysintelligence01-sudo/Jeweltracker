"""Marker-free ONNX stone preview with two still-photo rubbed-mask checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import queue
import subprocess
import sys
from threading import Lock
from time import monotonic
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

if __package__:
    from .geometry import DEST, SIZE, align_to_reference, normalize, project_mask, stone_quad
    from .live_camera import LatestCamera
    from .rubbing_zones import (TARGET_RADIUS, analyze_zones, draw_recommendation,
                               rubbing_instruction, rubbing_result)
else:
    from geometry import DEST, SIZE, align_to_reference, normalize, project_mask, stone_quad
    from live_camera import LatestCamera
    from rubbing_zones import (TARGET_RADIUS, analyze_zones, draw_recommendation,
                               rubbing_instruction, rubbing_result)


ROOT = Path(__file__).resolve().parent
STONE_ONNX = ROOT / "models" / "newstone1.onnx"
PHOTO_WORKER = ROOT / "photo_worker.py"
RUB_ONNX = ROOT / "models" / "rubbedmask.onnx"
OVERLAY_MAX_AGE = 0.75


class OnnxStoneModel:
    """CPU ONNX Runtime wrapper for a YOLOv8-style instance-segmentation ONNX."""

    def __init__(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with: "
                "python -m pip install onnxruntime"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs")
        self.input = inputs[0]
        shape = self.input.shape
        if len(shape) != 4:
            raise RuntimeError(f"Expected a 4D image input, got {shape}")
        # Most YOLO ONNX exports are NCHW; also accept NHWC exports.
        if shape[1] == 3 or shape[1] == "3":
            self.nchw = True
            self.input_h = int(shape[2]) if isinstance(shape[2], int) else 640
            self.input_w = int(shape[3]) if isinstance(shape[3], int) else 640
        elif shape[3] == 3 or shape[3] == "3":
            self.nchw = False
            self.input_h = int(shape[1]) if isinstance(shape[1], int) else 640
            self.input_w = int(shape[2]) if isinstance(shape[2], int) else 640
        else:
            raise RuntimeError(f"Could not identify 3-channel image layout in {shape}")
        self.input_dtype = np.float32 if "float" in self.input.type else np.uint8
        self.output_names = [item.name for item in self.session.get_outputs()]

    def close(self):
        # ONNX Runtime releases its session resources when this object is collected.
        self.session = None


def _sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def infer_stone(model, frame, confidence_threshold=0.25, iou_threshold=0.45):
    """Return (bbox, confidence, 0/255 mask) detections from YOLOv8-seg ONNX.

    Expects the common Ultralytics segmentation export: a prediction tensor
    [1, 4 + classes + mask_coefficients, anchors] (or transposed), plus a
    prototype tensor [1, mask_coefficients, mask_h, mask_w].
    """
    original_h, original_w = frame.shape[:2]
    target_w, target_h = model.input_w, model.input_h
    scale = min(target_w / original_w, target_h / original_h)
    resized_w = max(1, int(round(original_w * scale)))
    resized_h = max(1, int(round(original_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if model.input_dtype == np.float32:
        tensor = rgb.astype(np.float32) / 255.0
    else:
        tensor = rgb
    if model.nchw:
        tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(np.ascontiguousarray(tensor), axis=0)

    outputs = model.session.run(model.output_names, {model.input.name: tensor})
    proto_index = next((i for i, out in enumerate(outputs) if np.asarray(out).ndim == 4), None)
    pred_index = next((i for i, out in enumerate(outputs) if np.asarray(out).ndim == 3), None)
    if proto_index is None or pred_index is None:
        shapes = [np.asarray(out).shape for out in outputs]
        raise RuntimeError(
            "Unsupported ONNX outputs; expected YOLO segmentation predictions and "
            f"prototypes, got {shapes}. A detection-only ONNX cannot provide stone masks."
        )
    proto = np.asarray(outputs[proto_index])[0]
    if proto.shape[0] > proto.shape[-1] and proto.shape[-1] <= 256:
        # Handle NHWC prototype tensors if an exporter produced them.
        proto = np.transpose(proto, (2, 0, 1))
    mask_channels, proto_h, proto_w = proto.shape
    pred = np.asarray(outputs[pred_index])[0]
    # Convert to [anchors, attributes]. Attributes = xywh + class scores + mask coefficients.
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    expected_min = 4 + mask_channels + 1
    if pred.shape[1] < expected_min:
        raise RuntimeError(
            f"Unexpected prediction shape {pred.shape} for {mask_channels} mask channels; "
            "confirm this is a YOLO segmentation ONNX model."
        )
    class_count = pred.shape[1] - 4 - mask_channels
    class_scores = pred[:, 4:4 + class_count]
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    keep = scores >= confidence_threshold
    if not np.any(keep):
        return []
    pred = pred[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    coeffs = pred[:, 4 + class_count:]

    # YOLO predicts center-x, center-y, width, height in model-input pixels.
    boxes_xywh = pred[:, :4].copy()
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_xyxy]
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.astype(float).tolist(),
                               confidence_threshold, iou_threshold)
    if indices is None or len(indices) == 0:
        return []
    indices = np.asarray(indices).reshape(-1)[:10]

    proto_flat = proto.reshape(mask_channels, -1)
    result = []
    for idx in indices:
        box = boxes_xyxy[idx]
        x1, y1, x2, y2 = box
        # Project mask coefficients onto the prototype tensor.
        logits = coeffs[idx] @ proto_flat
        mask_prob = _sigmoid(logits.reshape(proto_h, proto_w))
        mask_input = cv2.resize(mask_prob, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        ix1, iy1 = int(np.clip(np.floor(x1), 0, target_w)), int(np.clip(np.floor(y1), 0, target_h))
        ix2, iy2 = int(np.clip(np.ceil(x2), 0, target_w)), int(np.clip(np.ceil(y2), 0, target_h))
        cropped_mask = np.zeros_like(mask_input, dtype=np.float32)
        if ix2 > ix1 and iy2 > iy1:
            cropped_mask[iy1:iy2, ix1:ix2] = mask_input[iy1:iy2, ix1:ix2]
        unpadded = cropped_mask[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
        if unpadded.size == 0:
            continue
        frame_mask = cv2.resize(unpadded, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        frame_mask = (frame_mask > 0.5).astype(np.uint8) * 255

        fx1 = int(np.clip(round((x1 - pad_x) / scale), 0, original_w - 1))
        fy1 = int(np.clip(round((y1 - pad_y) / scale), 0, original_h - 1))
        fx2 = int(np.clip(round((x2 - pad_x) / scale), 0, original_w - 1))
        fy2 = int(np.clip(round((y2 - pad_y) / scale), 0, original_h - 1))
        if fx2 <= fx1 or fy2 <= fy1 or not np.any(frame_mask):
            continue
        result.append(((fx1, fy1, fx2, fy2), float(scores[idx]), frame_mask))
    return result



def merged_mark_regions(mask, stone_region):
    """Merge touching predictions and ignore tiny model specks on the stone."""
    joined = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((9, 9), np.uint8))
    joined[~stone_region] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    merged = np.zeros(mask.shape, np.uint8)
    regions = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 80:
            merged[labels == index] = 255
            regions += 1
    coverage = np.count_nonzero(merged) / max(1, np.count_nonzero(stone_region))
    return merged, regions, coverage


def new_mark_regions(mask, old_mask, stone_region):
    """Keep whole new regions; tolerate small shifts in old model masks."""
    current, _, _ = merged_mark_regions(mask, stone_region)
    near_old = cv2.dilate(old_mask, np.ones((25, 25), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(current, 8)
    new = np.zeros_like(current)
    for index in range(1, count):
        region = labels == index
        if np.count_nonzero(near_old & region) < 0.5 * stats[index, cv2.CC_STAT_AREA]:
            new[region] = 255
    return new


def new_mark_locations(mask, frame_to_stone):
    """Return each new region's center in saved-stone and current-frame coordinates."""
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    if count == 1:
        return []
    order = sorted(range(1, count), key=lambda index: stats[index, cv2.CC_STAT_AREA], reverse=True)
    stone_points = np.float32([centers[index] for index in order])
    frame_points = cv2.perspectiveTransform(stone_points[None],
                                            np.linalg.inv(frame_to_stone))[0]
    return [(tuple(np.rint(stone).astype(int)), tuple(np.rint(frame).astype(int)))
            for stone, frame in zip(stone_points, frame_points)]


def detect_tag(frame, detector):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or np.count_nonzero(ids == 1) != 1:
        return None
    return np.asarray(corners[int(np.where(ids.reshape(-1) == 1)[0][0])][0], np.float32)


def stone_region(stone_mask, transform):
    region = cv2.warpPerspective(stone_mask, transform, (SIZE, SIZE),
                                 flags=cv2.INTER_NEAREST)
    return cv2.erode(region, np.ones((11, 11), np.uint8)) > 0



def draw_stone_grid(image, quad, labels=True):
    """Draw a perspective-aware 2x2 Z1-Z4 grid inside the stone quadrilateral.

    `quad` is expected in perimeter order: top-left, top-right,
    bottom-right, bottom-left, as returned by stone_quad().
    """
    if quad is None:
        return image
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        return image

    matrix = cv2.getPerspectiveTransform(DEST, pts)
    half = SIZE // 2

    def project(points):
        return cv2.perspectiveTransform(np.float32([points]), matrix)[0]

    # Draw a subtle dark underlay, then bright grid lines for visibility.
    grid_lines = [project([(half, 0), (half, SIZE - 1)]),
                  project([(0, half), (SIZE - 1, half)])]
    for p1, p2 in grid_lines:
        a, b = tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int))
        cv2.line(image, a, b, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(image, a, b, (220, 220, 220), 1, cv2.LINE_AA)

    # Use the same perspective mapping as zone analysis and the rubbing target.
    centers = dict(zip(("Z1", "Z2", "Z3", "Z4"), project(
        [(half / 2, half / 2), (half * 1.5, half / 2),
         (half / 2, half * 1.5), (half * 1.5, half * 1.5)])))
    for name, point in centers.items():
        if labels:
            px, py = np.rint(point).astype(int)
            text_size, baseline = cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            tx = max(0, min(image.shape[1] - text_size[0] - 2, int(px - text_size[0] / 2)))
            ty = max(text_size[1] + 2, min(image.shape[0] - baseline - 2, int(py + text_size[1] / 2)))
            cv2.putText(image, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(image, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def marked_photo(frame, bbox, quad, tag, full_mask, stone_mask, *, show_grid=True,
                 grid_quad=None):
    shown = frame.copy()
    cv2.rectangle(shown, bbox[:2], bbox[2:], (0, 220, 0), 2)
    cv2.polylines(shown, [quad.astype(np.int32)], True, (255, 170, 0), 2)
    if show_grid:
        draw_stone_grid(shown, quad if grid_quad is None else grid_quad)
    if tag is not None:
        cv2.polylines(shown, [tag.astype(np.int32)], True, (0, 255, 255), 2)
    on_stone = (full_mask > 0) & (stone_mask > 0)
    green = np.full_like(shown, (0, 255, 0))
    blended = cv2.addWeighted(shown, 0.50, green, 0.50, 0)
    shown[on_stone] = blended[on_stone]
    return shown


def configure_camera(cap, width, height):
    """Configure a USB camera to reduce auto-exposure flicker where supported.

    Camera drivers differ: unsupported settings are safely ignored. For 50 Hz
    mains lighting, 25 FPS is a useful starting point; exposure is set to a
    common DirectShow manual value, but some cameras use different scales.
    """
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # On Windows DirectShow, 0.25 usually means manual exposure and 0.75 auto.
    # Try manual mode before setting exposure; backends may ignore these values.
    if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    if hasattr(cv2, "CAP_PROP_EXPOSURE"):
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    if hasattr(cv2, "CAP_PROP_GAIN"):
        cap.set(cv2.CAP_PROP_GAIN, 0)
    if hasattr(cv2, "CAP_PROP_AUTO_WB"):
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    # Return actual values for debugging; many webcams report -1 when unsupported.
    props = {}
    for name, prop in (("fps", cv2.CAP_PROP_FPS),
                       ("exposure", cv2.CAP_PROP_EXPOSURE),
                       ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
                       ("gain", cv2.CAP_PROP_GAIN),
                       ("auto_white_balance", cv2.CAP_PROP_AUTO_WB)):
        try:
            props[name] = cap.get(prop)
        except Exception:
            props[name] = None
    return props


class CameraWorker(QThread):
    photo_ready = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, camera, width, height, output_dir, stone_model=STONE_ONNX,
                 show_grid=False):
        super().__init__()
        self.camera, self.width, self.height = camera, width, height
        self.output_dir = output_dir
        self.stone_model_path = Path(stone_model).resolve()
        self.commands = queue.Queue()
        self.stage = "BEFORE"
        self.result = "Capture before-rubbing photo"
        self.photo_status = "No photo captured"
        self.reference = None
        self.reference_region = None
        self.clean_mask = None
        self.baseline_tag = None
        self.recommendation = None
        self.guidance = "Capture before rubbing to find a clear rubbing zone."
        self.session_dir = None
        self.photo_process = None
        self.pending = None
        self.preview_only_reason = None
        self.show_grid = show_grid
        self.reader = LatestCamera(camera, width, height, configure_camera)
        self.presentation_lock = Lock()
        self.inference_ms = None
        initial_info = self.info("marker-free mode", "loading model")
        initial_info["available"] = False
        self.presentation = (0.0, None, initial_info)
        # Temporal smoothing prevents the live stone outline/grid jumping between frames.
        self.smoothed_quad = None
        self.smoothed_bbox = None
        self.smoothing_alpha = 0.75

    def reset(self):
        if self.photo_process is not None:
            self.photo_process.terminate()
            self.photo_process.communicate()
        self.photo_process = self.pending = None
        self.stage = "BEFORE"
        self.result = "Capture before-rubbing photo"
        self.photo_status = "No photo captured"
        self.reference = self.reference_region = self.clean_mask = self.baseline_tag = None
        self.recommendation = None
        self.guidance = "Capture before rubbing to find a clear rubbing zone."
        self.session_dir = None
        self.smoothed_quad = None
        self.smoothed_bbox = None
        self.publish(0.0, None, self.info("marker-free mode", "waiting for stone"))
        self.photo_ready.emit(None, None)

    def run(self):
        runtime = None
        self.reader.start()
        try:
            stone = detector = None
            try:
                for path in (self.stone_model_path, PHOTO_WORKER, RUB_ONNX):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                runtime = OnnxStoneModel(self.stone_model_path)
                stone = runtime
                # Marker-free mode: stone geometry and image features provide alignment.
                detector = None
            except (ImportError, OSError, RuntimeError) as exc:
                self.preview_only_reason = f"ONNX stone inference unavailable: {exc}"
                if runtime is not None:
                    runtime.close()
                    runtime = None
                self.result = f"PREVIEW ONLY - {self.preview_only_reason}"
            sequence = 0
            while not self.isInterruptionRequested():
                packet = self.reader.wait_after(sequence)
                if packet is None:
                    continue
                sequence, captured_at, frame = packet
                self.finish_photo_if_ready()
                command = None
                while not self.commands.empty():
                    command = self.commands.get_nowait()
                    if command == "reset":
                        self.reset()
                        command = None
                if self.photo_process is not None:
                    info = self.info("Photo worker running", "checking captured photo")
                    self.publish(captured_at, None, info)
                    continue
                if stone is None:
                    info = self.info("not checked", "preview only", "model unavailable")
                    self.publish(captured_at, None, info)
                    continue
                # Photo alignment may have taken time; infer/capture the newest frame.
                sequence, captured_at, frame = self.reader.snapshot()
                start = monotonic()
                _, info, candidate = self.live_frame(frame, detector, stone)
                self.inference_ms = (monotonic() - start) * 1000
                if command in ("before", "after"):
                    self.capture(command, frame, candidate)
                    info = self.info(info["tag"], info["stone"], info["alignment"])
                info["prediction"] = f"{self.inference_ms:.0f} ms"
                self.publish(captured_at, None if self.photo_process else candidate, info)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if self.photo_process is not None:
                self.photo_process.terminate()
                self.photo_process.communicate()
            self.reader.stop()
            if runtime is not None:
                runtime.close()

    def publish(self, captured_at, candidate, info):
        # The GUI polls one snapshot; camera frames never accumulate in Qt's queue.
        display = None
        if candidate is not None:
            display = (self.smoothed_bbox.copy(), self.smoothed_quad.copy())
        with self.presentation_lock:
            self.presentation = (captured_at, display, info)

    def preview(self):
        packet = self.reader.snapshot()
        if packet is None:
            return None
        _, _, frame = packet
        with self.presentation_lock:
            captured_at, display, info = self.presentation
        shown = frame.copy()
        if display is not None and monotonic() - captured_at <= OVERLAY_MAX_AGE:
            bbox, quad = display
            bbox = np.rint(bbox).astype(int)
            cv2.rectangle(shown, tuple(bbox[:2]), tuple(bbox[2:]), (0, 220, 0), 2)
            cv2.polylines(shown, [np.rint(quad).astype(np.int32)], True, (255, 170, 0), 2)
            if self.show_grid:
                draw_stone_grid(shown, quad, labels=False)
        elif display is not None:
            info = {**info, "stone": "waiting for fresh stone prediction"}
        return shown, info

    def info(self, tag, stone, alignment="checked on captured photos"):
        return {"stage": self.stage, "tag": tag, "stone": stone,
                "alignment": alignment, "photo": self.photo_status,
                "result": self.result, "busy": self.photo_process is not None,
                "available": self.preview_only_reason is None,
                "guidance": self.guidance,
                "prediction": "starting" if self.inference_ms is None else f"{self.inference_ms:.0f} ms"}

    def live_frame(self, frame, detector, stone):
        # Reference-marker detection is intentionally disabled.
        # The four stone corners and feature-based alignment are used instead.
        tag = None
        tag_text = "marker-free mode"

        detections = infer_stone(stone, frame)
        detections = [d for d in detections if
                      (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) >= 0.02 * frame.shape[0] * frame.shape[1]]
        if not detections:
            self.smoothed_quad = self.smoothed_bbox = None
            return frame, self.info(tag_text, "stone not found", "RETAKE - waiting for stable detection"), None

        detections.sort(key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) * d[1], reverse=True)
        if len(detections) > 1 and (detections[1][0][2] - detections[1][0][0]) * (detections[1][0][3] - detections[1][0][1]) > 0.6 * (detections[0][0][2] - detections[0][0][0]) * (detections[0][0][3] - detections[0][0][1]):
            self.smoothed_quad = self.smoothed_bbox = None
            return frame, self.info(tag_text, "multiple possible stones", "RETAKE"), None

        bbox, score, stone_mask = detections[0]
        raw_quad = stone_quad(stone_mask, bbox)
        if raw_quad is None:
            self.smoothed_quad = self.smoothed_bbox = None
            return frame, self.info(tag_text, f"box {bbox}, {score:.0%}", "stone corners unclear"), None

        # Exponential moving average smooths small frame-to-frame coordinate changes.
        raw_quad = np.asarray(raw_quad, dtype=np.float32)
        raw_bbox = np.asarray(bbox, dtype=np.float32)
        if self.smoothed_quad is None or self.smoothed_bbox is None:
            self.smoothed_quad = raw_quad.copy()
            self.smoothed_bbox = raw_bbox.copy()
        else:
            a = self.smoothing_alpha
            self.smoothed_quad = (1.0 - a) * self.smoothed_quad + a * raw_quad
            self.smoothed_bbox = (1.0 - a) * self.smoothed_bbox + a * raw_bbox

        smooth_bbox = np.rint(self.smoothed_bbox).astype(int).tolist()

        # A valid detected quadrilateral is sufficient to capture in marker-free mode.
        # Keep geometry points as float32: OpenCV getPerspectiveTransform
        # requires CV_32F points. Use int32 only for drawing overlays.
        # Photo geometry must match this exact frame, without display smoothing.
        candidate_quad = np.asarray(raw_quad, dtype=np.float32).reshape(4, 2)
        candidate = {"bbox": tuple(bbox), "quad": candidate_quad,
                     "stone_mask": stone_mask, "tag": None}
        return frame, self.info(tag_text, f"box {tuple(smooth_bbox)}, {score:.0%}"), candidate

    def capture(self, command, frame, candidate):
        if self.stage != command.upper() or candidate is None:
            self.result = "RETAKE - capture needs a detected stone and four reliable corners"
            return
        if self.session_dir is None:
            self.session_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.session_dir.mkdir(parents=True, exist_ok=True)
        photo_path = self.session_dir / f"{command}.png"
        mask_path = self.session_dir / f"{command}_mask_full.png"
        if not cv2.imwrite(str(photo_path), frame):
            raise OSError(f"Cannot save captured photo: {photo_path}")
        self.pending = {"kind": command, "frame": frame.copy(), **candidate, "mask_path": mask_path}
        self.photo_process = subprocess.Popen(
            [sys.executable, str(PHOTO_WORKER), str(photo_path), str(mask_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self.photo_status = f"{command} photo saved; checking model once"
        self.result = "Processing captured photo"

    def finish_photo_if_ready(self):
        if self.photo_process is None or self.photo_process.poll() is None:
            return
        return_code = self.photo_process.returncode
        _, error = self.photo_process.communicate()
        self.photo_process = None
        pending, self.pending = self.pending, None
        if return_code:
            self.result = "RETAKE - photo model failed"
            self.photo_status = error.decode(errors="replace")[-400:].strip()
            return
        full_mask = cv2.imread(str(pending["mask_path"]), cv2.IMREAD_GRAYSCALE)
        frame = pending["frame"]
        if full_mask is None or full_mask.shape != frame.shape[:2]:
            self.result = "RETAKE - photo mask missing or wrong size"
            return
        kind, bbox, quad = pending["kind"], pending["bbox"], pending["quad"]
        stone_mask, tag = pending["stone_mask"], pending["tag"]
        photo = marked_photo(frame, bbox, quad, tag, full_mask, stone_mask,
                             show_grid=kind == "before")
        if kind == "before":
            quad = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
            if quad.shape != (4, 2) or not np.isfinite(quad).all():
                raise RuntimeError(f"Invalid stone quadrilateral for normalization: {quad.shape}")
            aligned, transform = normalize(frame, quad)
            inner = stone_region(stone_mask, transform)
            normalized_mask = cv2.warpPerspective(full_mask, transform, (SIZE, SIZE),
                                                   flags=cv2.INTER_NEAREST)
            normalized_mask, regions, coverage = merged_mark_regions(normalized_mask, inner)
            features = len(cv2.ORB_create(nfeatures=1500).detect(
                cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), inner.astype(np.uint8) * 255))
            self.photo_status = (f"Before: {regions} merged marks, {coverage:.2%} stone coverage; "
                                 f"{features} alignment features")
            zones, self.recommendation = analyze_zones(normalized_mask, inner)
            self.guidance = rubbing_instruction(zones, self.recommendation)
            self.photo_status += "; old marks per zone: " + ", ".join(
                f"{zone.name}={zone.old_marks}" for zone in zones)
            with (self.session_dir / "zone_recommendation.json").open("w") as saved:
                json.dump({"zones": [asdict(zone) for zone in zones],
                           "selected_zone": self.recommendation.name if self.recommendation else None,
                           "target_radius": TARGET_RADIUS, "instruction": self.guidance}, saved, indent=2)
            preview = aligned.copy()
            green = np.full_like(preview, (0, 255, 0))
            on_marks = normalized_mask > 0
            preview[on_marks] = cv2.addWeighted(preview, 0.5, green, 0.5, 0)[on_marks]
            draw_stone_grid(preview, DEST)
            if self.recommendation is None:
                self.reference = self.reference_region = self.clean_mask = None
                self.stage = "BEFORE"
                self.result = "CLEAN THE STONE - no zone has enough clear space"
            else:
                self.reference = aligned.copy()
                self.reference_region = inner.astype(np.uint8) * 255
                self.clean_mask = normalized_mask.copy()
                self.baseline_tag = None
                self.stage = "AFTER"
                self.result = ("Reference photo saved; capture after rubbing" if features >= 30
                               else "Reference photo saved; rotation may be RETAKE")
                cv2.imwrite(str(self.session_dir / "clean_reference.png"), aligned)
                draw_recommendation(photo, self.recommendation, transform)
                draw_recommendation(preview, self.recommendation)
            cv2.imwrite(str(self.session_dir / "before_zones.png"), preview)
        else:
            quad = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
            if quad.shape != (4, 2) or not np.isfinite(quad).all():
                raise RuntimeError(f"Invalid stone quadrilateral for normalization: {quad.shape}")
            _, local_transform = normalize(frame, quad)
            preview = cv2.warpPerspective(full_mask, local_transform, (SIZE, SIZE),
                                          flags=cv2.INTER_NEAREST)
            preview[~stone_region(stone_mask, local_transform)] = 0
            aligned, transform, detail = align_to_reference(
                frame, np.asarray(quad, dtype=np.float32).reshape(4, 2),
                self.reference, self.reference_region, stone_mask)
            if aligned is None:
                self.result = f"RETAKE - {detail}"
                self.photo_status = "After: model mask shown; feature-based alignment unresolved"
                self.guidance = "RETAKE - stone alignment unclear; no rubbing zone confirmed."
            else:
                # After rotation, draw zones in the saved reference orientation.
                reference_quad = cv2.perspectiveTransform(
                    DEST[None], np.linalg.inv(transform))[0]
                photo = marked_photo(frame, bbox, quad, tag, full_mask, stone_mask,
                                     grid_quad=reference_quad)
                inner = stone_region(stone_mask, transform)
                mark = cv2.warpPerspective(full_mask, transform, (SIZE, SIZE),
                                           flags=cv2.INTER_NEAREST)
                new = new_mark_regions(mark, self.clean_mask, inner)
                change = cv2.absdiff(aligned, self.reference).mean(axis=2)
                new[change < 12] = 0
                new, new_regions, fraction = merged_mark_regions(new, inner)
                preview = new
                self.photo_status = (f"After: {detail}; {new_regions} new merged marks, "
                                     f"{fraction:.2%} inside stone")
                if 0.001 <= fraction <= 0.15:
                    locations = new_mark_locations(new, transform)
                    projected = project_mask(new, frame.shape, transform) > 0
                    magenta = np.full_like(photo, (255, 0, 255))
                    photo[projected] = cv2.addWeighted(photo, 0.4, magenta, 0.6, 0)[projected]
                    contours, _ = cv2.findContours(projected.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(photo, contours, -1, (255, 0, 255), 2)
                    preview = aligned.copy()
                    stone_magenta = np.full_like(preview, (255, 0, 255))
                    preview[new > 0] = cv2.addWeighted(preview, 0.4, stone_magenta, 0.6, 0)[new > 0]
                    labels = []
                    for index, (stone_point, frame_point) in enumerate(locations, 1):
                        cv2.circle(photo, frame_point, 10, (255, 255, 255), 2)
                        cv2.circle(photo, frame_point, 4, (255, 0, 255), -1)
                        cv2.putText(photo, f"NEW {index}", (frame_point[0] + 12, frame_point[1] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                        cv2.putText(photo, f"NEW {index}", (frame_point[0] + 12, frame_point[1] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.circle(preview, stone_point, 8, (255, 255, 255), 2)
                        labels.append(f"{index}: x={stone_point[0]}, y={stone_point[1]}")
                    self.photo_status += "; new location (512 stone): " + ", ".join(labels)
                    self.result = "New rubbed mask detected"
                    self.guidance = rubbing_result(locations, self.recommendation)
                    cv2.imwrite(str(self.session_dir / "rubbed_mask.png"), new)
                else:
                    self.result = "RETAKE - new rubbed mask unclear"
                    self.guidance = "RETAKE - new rubbing unclear; no rubbing zone confirmed."
        cv2.imwrite(str(self.session_dir / f"{kind}_marked.png"), photo)
        self.photo_ready.emit(photo, preview)


class Window(QMainWindow):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.setWindowTitle("Stone SOP - Two Photo Rubbed Mask Check")
        body = QWidget()
        layout = QVBoxLayout(body)
        views = QHBoxLayout()
        self.video = QLabel("Starting camera...")
        self.video.setMinimumSize(640, 360)
        self.photo = QLabel("Captured photo and ONNX model marking")
        self.photo.setMinimumSize(480, 360)
        self.mask = QLabel("Stone crop / ONNX mask")
        self.mask.setFixedSize(200, 200)
        for label in (self.video, self.photo, self.mask):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background: #151515; color: white;")
        views.addWidget(self.video, 1)
        views.addWidget(self.photo, 1)
        views.addWidget(self.mask)
        layout.addLayout(views)
        legend = QLabel("Green = existing/model marks; blue circle = suggested rubbing target; "
                        "magenta NEW = confirmed new rubbing. Zones refer to the saved before photo.")
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.guidance = QLabel(worker.guidance)
        self.guidance.setWordWrap(True)
        self.guidance.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px; "
                                   "color: #1267ae; background: #e9f3ff;")
        layout.addWidget(self.guidance)
        self.status = QLabel("Starting camera and loading ONNX stone model...")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 17px; padding: 8px;")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.before = QPushButton("1. Capture before rubbing")
        self.before.clicked.connect(lambda: worker.commands.put("before"))
        self.after = QPushButton("2. Capture after rubbing")
        self.after.clicked.connect(lambda: worker.commands.put("after"))
        reset = QPushButton("New test / reset")
        reset.clicked.connect(lambda: worker.commands.put("reset"))
        for button in (self.before, self.after, reset):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.setCentralWidget(body)
        self.resize(1450, 720)
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(33)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.preview_timer.start()
        worker.photo_ready.connect(self.show_photo)
        worker.failed.connect(self.show_error)
        worker.start()

    def refresh_preview(self):
        preview = self.worker.preview()
        if preview is not None:
            self.show_frame(*preview)

    @staticmethod
    def pixmap(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                       QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def show_frame(self, frame, info):
        self.guidance.setText(info["guidance"])
        self.video.setPixmap(self.pixmap(frame).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.status.setText(" | ".join(f"{key}: {info[key]}" for key in
                                       ("stage", "tag", "stone", "alignment", "photo", "result", "prediction")))
        self.before.setEnabled(info["available"] and info["stage"] == "BEFORE" and not info["busy"])
        self.after.setEnabled(info["available"] and info["stage"] == "AFTER" and not info["busy"])

    def show_photo(self, photo, mask):
        if photo is None:
            self.photo.clear()
            self.photo.setText("Captured photo and ONNX model marking")
            self.mask.clear()
            self.mask.setText("Stone crop / ONNX mask")
            return
        self.photo.setPixmap(self.pixmap(photo).scaled(
            self.photo.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        if mask.ndim == 3:
            rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, SIZE, SIZE, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        else:
            image = QImage(mask.data, SIZE, SIZE, mask.strides[0], QImage.Format.Format_Grayscale8).copy()
        self.mask.setPixmap(QPixmap.fromImage(image).scaled(200, 200))

    def show_error(self, message):
        self.preview_timer.stop()
        self.before.setEnabled(False)
        self.after.setEnabled(False)
        self.status.setText(f"RETAKE | Camera/ONNX error: {message}")
        self.video.setText("Live preview stopped")

    def closeEvent(self, event):
        self.preview_timer.stop()
        self.worker.requestInterruption()
        self.worker.wait()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sessions")
    parser.add_argument("--stone-model", type=Path, default=STONE_ONNX,
                        help="Path to a YOLOv8-style segmentation ONNX model (default: newstone1.onnx beside app.py)")
    parser.add_argument("--show-grid", action="store_true",
                        help="Show the optional live grid (captured photos always show it)")
    args = parser.parse_args()
    qt = QApplication(sys.argv)
    cv2.setNumThreads(1)
    window = Window(CameraWorker(args.camera, args.width, args.height, args.output_dir,
                                 args.stone_model, args.show_grid))
    window.show()
    return qt.exec()

if __name__ == "__main__":
    raise SystemExit(main())
