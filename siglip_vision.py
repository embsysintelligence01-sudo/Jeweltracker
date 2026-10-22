"""SigLIP2 vision encoder (ONNX, CPU) for the jewel fingerprint and type classifier.
Copied from R:\\similarity_check\\desktop_backend\\app\\models\\siglip_onnx.py (+ letterbox_square from
app\\vision\\image_utils.py) so the app, and its .exe, don't depend on that folder."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox_square(image_rgb: np.ndarray, size: int, fill: int = 127) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


class SiglipVisionONNX:
    """Vision-only SigLIP2 ONNX wrapper."""

    def __init__(self, model_path: str | Path, input_size: int = 256) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"SigLIP2 model is missing: {model_path}")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        image_inputs = [value for value in inputs if len(value.shape) == 4]
        if not image_inputs:
            raise RuntimeError(f"No four-dimensional image input found; inputs={[(x.name, x.shape) for x in inputs]}")
        self.input = image_inputs[0]
        self.extra_inputs = [value for value in inputs if value.name != self.input.name]
        self.outputs = self.session.get_outputs()
        self.input_size = input_size
        self.patch_size = 32

    def _tensor(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = letterbox_square(rgb, self.input_size, fill=127)
        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        shape = self.input.shape
        nchw = len(shape) == 4 and (shape[1] == 3 or str(shape[1]) in {"3", "None"})
        if nchw:
            return x.transpose(2, 0, 1)[None].astype(np.float32)
        return x[None].astype(np.float32)

    @staticmethod
    def _select_embedding(outputs: list[np.ndarray]) -> np.ndarray:
        # Prefer an already pooled 2-D output. Otherwise pool token embeddings.
        two_d = [o for o in outputs if isinstance(o, np.ndarray) and o.ndim == 2 and o.shape[0] == 1]
        if two_d:
            candidate = two_d[-1]
        else:
            three_d = [o for o in outputs if isinstance(o, np.ndarray) and o.ndim == 3 and o.shape[0] == 1]
            if not three_d:
                raise RuntimeError(f"Cannot identify SigLIP embedding output; shapes={[getattr(o, 'shape', None) for o in outputs]}")
            candidate = three_d[-1].mean(axis=1)
        vector = candidate.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            raise RuntimeError("SigLIP produced a zero-length embedding.")
        return vector / norm

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        image_tensor = self._tensor(image_bgr)
        feed = {self.input.name: image_tensor}
        patch_grid = self.input_size // self.patch_size
        for value in self.extra_inputs:
            name = value.name.lower()
            if "spatial" in name and len(value.shape) == 2:
                feed[value.name] = np.asarray([[patch_grid, patch_grid]], dtype=np.int64)
            elif "attention" in name or "mask" in name:
                feed[value.name] = np.ones((1, patch_grid * patch_grid), dtype=np.int64)
            else:
                raise RuntimeError(f"Unsupported additional SigLIP input {value.name} shape={value.shape}.")
        outputs = self.session.run(None, feed)
        return self._select_embedding(outputs)

    def embed_rotations(self, image_bgr: np.ndarray) -> np.ndarray:
        images = (
            image_bgr,
            cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image_bgr, cv2.ROTATE_180),
            cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
        )
        return np.stack([self.embed(img) for img in images], axis=0)
