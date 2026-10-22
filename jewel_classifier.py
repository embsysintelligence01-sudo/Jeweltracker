"""
Jewel type classifier, following the Classification approach of
https://github.com/rishikumar-sn/Weight-Scale-SOP/tree/main/Classification (jewelry_classifier.py):

  - SigLIP2 (google/siglip2-base-patch32-256) vision ONNX image embedding, preprocessed like its
    SiglipImageProcessor (256x256 bilinear resize, (x/255 - 0.5) / 0.5)
  - the jewel isolated on a pure white background; the embedding is the mean of the whole crop and a
    padded square crop of its foreground
  - zero-shot scores against the cached prompt text embeddings (classification/*.npz), softmax per
    prompt group, then per class within the group
  - "Not Gold Jewelry" for an empty crop, a green-dominant object or specific foreign-object evidence
  - the correction gallery (labelled image vectors) can confirm or correct the zero-shot label

Only numpy + onnxruntime + Pillow are needed at run time (the text embeddings are precomputed).
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

# next to the .exe when built with PyInstaller, else next to this script
_BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_DIR = os.path.join(_BASE_DIR, "classification")
TEXT_CACHE = "jewelry_text_embeddings_cache.npz"
GALLERY_FILE = "jewelry_correction_gallery.npz"
# Operator corrections from this bench (double-click rename in Jewel Guard). They are matched before the
# repo's gallery (822 examples from other cameras), nearest example only: measured on this camera's
# captures, the same jewel re-captured in a later run is 0.85-0.98 similar, different jewel types <= 0.82;
# the repo gallery's rules (>= 0.90-0.94 with 2-3 agreeing neighbours) let re-runs miss the operator's fix.
OPERATOR_FILE = "operator_corrections.npz"
OPERATOR_MATCH_SIM = 0.84      # nearest operator example at least this similar -> its label
OPERATOR_MARGIN = 0.03         # ... and this much closer than the nearest example of any other label
REPO_GALLERY_SIZE = 822        # entries the copied repo gallery had; later ones are operator corrections
FOREIGN_CACHE = "foreign_object_text_embeddings_cache.npz"
FOREIGN_PROMPTS = "foreign_object_prompts.json"

# constants as in the repo's jewelry_classifier.py
GALLERY_THRESHOLD = 0.90
GALLERY_EXACT_MATCH_THRESHOLD = 0.975
GALLERY_DISAGREEMENT_MARGIN = 0.01
GALLERY_TOP_K = 5
WHITE_THRESHOLD = 245
EMPTY_BACKGROUND_RATIO = 0.98
STRONG_NON_JEWELRY_MAX_GOLD_SIMILARITY = 0.035
STRONG_NON_JEWELRY_MIN_MARGIN = 0.03
GREEN_DOMINANT_FOREGROUND_RATIO = 0.60
MIN_COLOR_GATE_FOREGROUND_PIXELS = 100
FOREGROUND_PADDING_RATIO = 0.12
SIMILARITY_SCALE = 100.0
COMBINED_NECK_LABEL = "Chain / Necklace"
OTHER_GOLD_LABEL = "Other Gold Jewellery"
NOT_GOLD_LABEL = "Not Gold Jewelry"
GALLERY_CLASS_THRESHOLDS = {
    "Bangle": 0.90, "Bracelet": 0.92, COMBINED_NECK_LABEL: 0.90, "Mattal": 0.92,
    "Earrings / Nosepin": 0.91, "Finger Ring": 0.91, OTHER_GOLD_LABEL: 0.94, NOT_GOLD_LABEL: 0.94,
}
PRODUCTION_LABELS = {
    "bangle": "Bangle", "bracelet": "Bracelet", "chain / necklace": COMBINED_NECK_LABEL,
    "chain": COMBINED_NECK_LABEL, "necklace": COMBINED_NECK_LABEL, "dollar": COMBINED_NECK_LABEL,
    "dollar chain": COMBINED_NECK_LABEL, "haram": COMBINED_NECK_LABEL, "kasu mala": COMBINED_NECK_LABEL,
    "kasu malai": COMBINED_NECK_LABEL, "mangalsutra": COMBINED_NECK_LABEL, "pendant": COMBINED_NECK_LABEL,
    "pendant chain": COMBINED_NECK_LABEL, "mattal": "Mattal", "earing / jumkha": "Earrings / Nosepin",
    "earring / jumkha": "Earrings / Nosepin", "earrings / nosepin": "Earrings / Nosepin",
    "earring": "Earrings / Nosepin", "earrings": "Earrings / Nosepin", "nose pin": "Earrings / Nosepin",
    "nosepin": "Earrings / Nosepin", "nose ring": "Earrings / Nosepin", "finger ring": "Finger Ring",
    "ring": "Finger Ring", "baby jewellery": OTHER_GOLD_LABEL, "other gold jewellery": OTHER_GOLD_LABEL,
    "not gold jewelry": NOT_GOLD_LABEL, "not gold jewellery": NOT_GOLD_LABEL,
}


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def canonical_label(label):
    cleaned = " ".join(str(label).split())
    return PRODUCTION_LABELS.get(cleaned.casefold(), cleaned)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class JewelClassifier:
    def __init__(self, onnx_path=None, data_dir=CLASSIFICATION_DIR, session=None):
        """session: an existing onnxruntime session of the same SigLIP2 vision model, to share memory."""
        with np.load(os.path.join(data_dir, TEXT_CACHE), allow_pickle=False) as z:
            self.labels = [str(x) for x in z["labels"].tolist()]
            self.text = z["embeddings"].astype(np.float32)
            self.class_groups = [str(x) for x in z["class_groups"].tolist()]
            self.group_labels = [str(x) for x in z["group_labels"].tolist()]
            self.group_emb = z["group_embeddings"].astype(np.float32)
        self.group_index = {g: i for i, g in enumerate(self.group_labels)}
        self.group_classes = {}
        for i, g in enumerate(self.class_groups):
            self.group_classes.setdefault(g, []).append(i)
        self.gallery_emb, self.gallery_labels = None, []
        self.gallery_path = gallery = os.path.join(data_dir, GALLERY_FILE)
        self.last_embedding = None  # embedding of the last classified crop (for learn())
        if os.path.exists(gallery):
            with np.load(gallery, allow_pickle=False) as z:
                self.gallery_emb = normalize(z["embeddings"].astype(np.float32))
                self.gallery_labels = [canonical_label(x) for x in z["labels"].tolist()]
        self.operator_path = os.path.join(data_dir, OPERATOR_FILE)
        self.operator_emb, self.operator_labels = None, []
        if os.path.exists(self.operator_path):
            with np.load(self.operator_path, allow_pickle=False) as z:
                self.operator_emb = normalize(z["embeddings"].astype(np.float32))
                self.operator_labels = [canonical_label(x) for x in z["labels"].tolist()]
        elif len(self.gallery_labels) > REPO_GALLERY_SIZE:  # renames made before the operator file existed
            self.operator_emb = self.gallery_emb[REPO_GALLERY_SIZE:].copy()
            self.operator_labels = self.gallery_labels[REPO_GALLERY_SIZE:]
            self.save_operator()
        with np.load(os.path.join(data_dir, FOREIGN_CACHE), allow_pickle=False) as z:
            self.foreign_emb = z["embeddings"].astype(np.float32)
        with open(os.path.join(data_dir, FOREIGN_PROMPTS), encoding="utf-8") as f:
            self.foreign = list(json.load(f)["objects"].items())
        self.session = session or ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        outs = self.session.get_outputs()
        self.out_index = next((i for i, o in enumerate(outs) if len(o.shape) == 2 and o.shape[-1] == 768),
                              len(outs) - 1)
        self.extra = {}
        for inp in self.session.get_inputs()[1:]:  # some exports also want a spatial shape / attention mask
            name = inp.name.lower()
            self.extra[inp.name] = (np.array([[8, 8]], np.int64) if "spatial" in name
                                    else np.ones((1, 64), np.int64))

    # ---- embedding (SiglipImageProcessor: 256x256 bilinear, rescale, normalize 0.5/0.5) ----
    def embed(self, rgb):
        img = Image.fromarray(rgb).convert("RGB").resize((256, 256), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5
        feed = {self.input.name: x.transpose(2, 0, 1)[None], **self.extra}
        out = self.session.run(None, feed)[self.out_index].astype(np.float32)
        return normalize(out.reshape(1, -1))[0]

    @staticmethod
    def crop_foreground(rgb):
        mask = np.any(rgb < WHITE_THRESHOLD, axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return rgb
        (y0, x0), (y1, x1) = coords.min(axis=0), coords.max(axis=0) + 1
        h, w = rgb.shape[:2]
        py, px = max(8, int((y1 - y0) * FOREGROUND_PADDING_RATIO)), max(8, int((x1 - x0) * FOREGROUND_PADDING_RATIO))
        crop = rgb[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
        side = max(crop.shape[:2])
        square = np.full((side, side, 3), 255, np.uint8)
        oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        square[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        return square

    # ---- checks ----
    @staticmethod
    def background_empty(rgb):
        white = np.all(rgb > WHITE_THRESHOLD - 10, axis=2).sum()
        return white / max(1, rgb.shape[0] * rgb.shape[1]) > EMPTY_BACKGROUND_RATIO

    @staticmethod
    def green_dominant(rgb):
        px = rgb.astype(np.int16)[np.any(rgb < WHITE_THRESHOLD, axis=2)]
        if len(px) < MIN_COLOR_GATE_FOREGROUND_PIXELS:
            return False
        r, g, b = px[:, 0], px[:, 1], px[:, 2]
        chroma = px.max(axis=1) - px.min(axis=1)
        return float(np.mean((chroma >= 25) & (g >= r + 12) & (g >= b + 8))) >= GREEN_DOMINANT_FOREGROUND_RATIO

    def group_sim(self, emb, group):
        i = self.group_index.get(group)
        return float(emb @ self.group_emb[i]) if i is not None else 0.0

    def non_jewelry(self, emb):
        jewel_sim = float(np.max(emb @ self.text.T))
        gold = self.group_sim(emb, "gold_verification")
        for (name, entry), sim in zip(self.foreign, emb @ self.foreign_emb.T):
            if (sim >= entry["min_similarity"] and sim - jewel_sim >= entry["min_margin"]
                    and gold < entry["max_gold_similarity"]):
                return True
        reject = max(self.group_sim(emb, "non_gold_metal"), self.group_sim(emb, "negative"))
        return gold < STRONG_NON_JEWELRY_MAX_GOLD_SIMILARITY and reject - gold >= STRONG_NON_JEWELRY_MIN_MARGIN

    def class_probabilities(self, sims, emb):
        if not self.group_labels:
            return softmax(sims * SIMILARITY_SCALE)
        group_p = softmax((emb @ self.group_emb.T) * SIMILARITY_SCALE)
        probs = np.zeros_like(sims)
        for group, idx in self.group_classes.items():
            gi = self.group_index.get(group)
            if gi is None:
                continue
            within = softmax(sims[idx] * SIMILARITY_SCALE)
            probs[idx] = group_p[gi] * len(idx) * within
        total = probs.sum()
        return probs / total if total > 0 else softmax(sims * SIMILARITY_SCALE)

    def gallery_search(self, emb, expected):
        if self.gallery_emb is None or not self.gallery_labels:
            return None, 0.0
        sims = self.gallery_emb @ emb
        order = np.argsort(sims)[::-1]
        best = float(sims[order[0]])
        votes = {}
        for i in order[:GALLERY_TOP_K]:
            votes.setdefault(self.gallery_labels[i], []).append(float(sims[i]))
        ranked = sorted(votes.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), max(kv[1]), len(kv[1])), reverse=True)
        winner, scores = ranked[0]
        win_sim = max(scores)
        margin = win_sim - (max(ranked[1][1]) if len(ranked) > 1 else -1.0)
        exact = best >= GALLERY_EXACT_MATCH_THRESHOLD
        thr = max(GALLERY_THRESHOLD, GALLERY_CLASS_THRESHOLDS.get(winner, GALLERY_THRESHOLD))
        consensus = len(scores) >= 2 and win_sim >= thr and margin >= 0.005
        if expected and winner != expected and not exact:
            consensus = len(scores) >= 3 and win_sim >= thr + GALLERY_DISAGREEMENT_MARGIN and margin >= GALLERY_DISAGREEMENT_MARGIN
        if exact:
            winner = self.gallery_labels[int(order[0])]
        return (winner, win_sim) if exact or consensus else (None, best)

    def operator_search(self, emb):
        """Nearest operator correction: (label, similarity) when close enough and clearly that label."""
        if self.operator_emb is None or not self.operator_labels:
            return None, 0.0
        sims = self.operator_emb @ emb
        best = {}
        for label, sim in zip(self.operator_labels, sims):
            best[label] = max(best.get(label, -1.0), float(sim))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        label, sim = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        if sim >= OPERATOR_MATCH_SIM and sim - runner_up >= OPERATOR_MARGIN:
            return label, sim
        return None, sim

    def merged_embedding(self, rgb_on_white):
        return normalize(self.embed(rgb_on_white) + self.embed(self.crop_foreground(rgb_on_white)))

    # ---- classification ----
    def classify(self, rgb_on_white):
        """rgb_on_white: RGB uint8 crop of one jewel with everything else pure white.
        Returns (label, confidence, source)."""
        original = rgb_on_white
        merged = self.merged_embedding(original)
        self.last_embedding = merged
        sims = merged @ self.text.T
        probs = self.class_probabilities(sims, merged)
        best = int(np.argmax(probs))
        label, conf, source = self.labels[best], float(probs[best]), "siglip"
        self.last_top = [self.labels[i] for i in np.argsort(probs)[::-1][:3]]  # for shape checks
        o_label, o_sim = self.operator_search(merged)
        if o_label is not None:  # the operator has named this jewel before: that wins
            return o_label, min(0.99, o_sim), "operator"
        if self.background_empty(original):
            return NOT_GOLD_LABEL, conf, "empty_background"
        if self.green_dominant(original):
            return NOT_GOLD_LABEL, conf, "green_non_gold"
        if self.non_jewelry(merged):
            return NOT_GOLD_LABEL, conf, "non_jewelry"
        g_label, g_sim = self.gallery_search(merged, label)
        if g_label is not None:
            source = "gallery_confirmation" if g_label == label else "gallery_correction"
            label, conf = g_label, max(conf, min(0.98, g_sim))
        return label, conf, source

    def label_choices(self):
        return list(self.labels) + [OTHER_GOLD_LABEL, NOT_GOLD_LABEL]

    def learn(self, embedding, label, crop_rgb=None):
        """Operator confirmation/correction, as the repo's CorrectionGallery.add + learn_correction: the
        jewel's embedding is stored with its true label, so similar crops are confirmed or corrected later.
        It also goes to the operator corrections (matched first), with the crop turned 90/180/270 degrees
        so the jewel is recognised however it lies next time."""
        label = canonical_label(label)
        vector = normalize(np.asarray(embedding, np.float32).reshape(1, -1))
        examples = [vector[0]]
        if crop_rgb is not None:
            for k in (1, 2, 3):
                examples.append(self.merged_embedding(np.ascontiguousarray(np.rot90(crop_rgb, k))))
        examples = np.stack(examples).astype(np.float32)
        self.operator_emb = examples if self.operator_emb is None else np.vstack([self.operator_emb, examples])
        self.operator_labels += [label] * len(examples)
        self.save_operator()
        if self.gallery_emb is not None and self.gallery_labels:
            same = np.array([x == label for x in self.gallery_labels])
            if same.any() and float(np.max(self.gallery_emb[same] @ vector[0])) >= 0.9995:
                return  # effectively identical example already stored
            self.gallery_emb = np.vstack([self.gallery_emb, vector])
        else:
            self.gallery_emb = vector
        self.gallery_labels.append(label)
        np.savez(self.gallery_path, embeddings=self.gallery_emb, labels=np.array(self.gallery_labels))

    def save_operator(self):
        np.savez(self.operator_path, embeddings=self.operator_emb, labels=np.array(self.operator_labels))
