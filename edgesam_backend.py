"""Optional EdgeSAM ONNX adapter. Imported only by the recorded-evidence worker.

Uses the upstream two-file export (image -> embeddings; prompts -> scores/masks).
Segmentation refines a visual candidate; it never verifies an SOP operation.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np


class EdgeSAM:
    def __init__(self, encoder, decoder, threads=1, cache_size=4):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(2, int(threads)))
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.encoder = ort.InferenceSession(str(encoder), options, providers=['CPUExecutionProvider'])
        self.decoder = ort.InferenceSession(str(decoder), options, providers=['CPUExecutionProvider'])
        if self.encoder.get_inputs()[0].shape != [1, 3, 1024, 1024]:
            raise ValueError('Expected upstream EdgeSAM float32 1024 encoder')
        if {i.name for i in self.decoder.get_inputs()} != {'image_embeddings', 'point_coords', 'point_labels'}:
            raise ValueError('Unsupported EdgeSAM decoder inputs')
        self.cache = OrderedDict()
        self.cache_size = max(1, int(cache_size))
        self.stats = dict(encoder_seconds=0., decoder_seconds=0., encoder_calls=0,
                          decoder_calls=0, embedding_cache_hits=0)
        self.models = {kind: dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                       for kind, path in (('encoder', encoder), ('decoder', decoder))}

    def embedding(self, image):
        key = hashlib.sha256(image.tobytes()).digest(), image.shape
        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats['embedding_cache_hits'] += 1
            return self.cache[key]
        height, width = image.shape[:2]
        scale = 1024 / max(height, width)
        size = (int(width * scale + .5), int(height * scale + .5))
        rgb = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) - np.array([123.675, 116.28, 103.53], np.float32))
        tensor /= np.array([58.395, 57.12, 57.375], np.float32)
        tensor = np.pad(tensor.transpose(2, 0, 1)[None], ((0, 0), (0, 0), (0, 1024-size[1]), (0, 1024-size[0])))
        start = time.perf_counter()
        features = self.encoder.run(None, {self.encoder.get_inputs()[0].name: tensor})[0]
        self.stats['encoder_seconds'] += time.perf_counter()-start
        self.stats['encoder_calls'] += 1
        self.cache[key] = features, size
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return features, size

    def refine(self, image, seed, padding=.2, min_score=.85):
        """Box corners (labels 2/3), an interior positive and exterior negatives.

        All prompts use x,y coordinates. Masks must agree with the OpenCV seed;
        a confident mask of the entire stone is explicitly rejected.
        """
        seed = (seed > 0).astype(np.uint8)
        if not np.any(seed):
            return None
        height, width = seed.shape
        x, y, w, h = cv2.boundingRect(seed)
        px, py = max(2., w*padding), max(2., h*padding)
        box = [max(1., x-px), max(1., y-py), min(width-2., x+w+px), min(height-2., y+h+py)]
        distance = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
        cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
        coords = np.float32([[[cx, cy], box[:2], box[2:], [box[0], box[3]], [box[2], box[1]]]])
        labels = np.float32([[1, 2, 3, 0, 0]])
        features, size = self.embedding(image)
        coords *= np.float32([size[0]/width, size[1]/height])
        start = time.perf_counter()
        output = dict(zip((o.name for o in self.decoder.get_outputs()), self.decoder.run(None, dict(
            image_embeddings=features, point_coords=coords, point_labels=labels))))
        self.stats['decoder_seconds'] += time.perf_counter()-start
        self.stats['decoder_calls'] += 1
        scores = np.asarray(output['scores']).reshape(-1)
        logits = np.asarray(output['masks']).reshape((-1, *output['masks'].shape[-2:]))
        for index in np.argsort(scores)[::-1]:
            if not np.isfinite(scores[index]) or scores[index] < min_score:
                continue
            mask = cv2.resize(logits[index], (1024, 1024), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask[:size[1], :size[0]], (width, height), interpolation=cv2.INTER_LINEAR) > 0
            union = np.count_nonzero(mask | (seed > 0))
            hit = np.count_nonzero(mask & (seed > 0))
            if not .001 <= mask.mean() <= .15 or hit/max(1, seed.sum()) < .65 or hit/max(1, union) < .35:
                continue
            support = np.zeros_like(seed)
            x1, y1, x2, y2 = map(int, box)
            support[y1:y2+1, x1:x2+1] = 1
            if np.count_nonzero(mask & (support == 0))/max(1, mask.sum()) > .05:
                continue
            return dict(mask=mask.astype(np.uint8)*255, score=float(scores[index]),
                        prompt_box=box, point_coords=(coords / np.float32([size[0]/width, size[1]/height])).tolist(),
                        point_labels=labels.tolist())
        return None
