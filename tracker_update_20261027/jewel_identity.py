"""Jewel identity matching that only commits when it is sure.

Two cues, both made to survive a hand over part of the jewel (rubbing):
  * FAISS gallery: many augmented views of each locked jewel (turned, scaled, re-lit, partly hidden by a "hand")
    as ResNet18 feature vectors, plus bench / stone patches as "not a jewel". A query is compared with every view;
    a jewel's score is the mean of its best few cosine similarities.
  * Superpixel colour signature: SLIC superpixels of the jewel's own pixels (bench, shadow and skin left out) are
    merged by agglomerative clustering into a handful of colour/texture regions. A query is scored by how much of
    it the jewel's regions explain (containment), which does not drop when part of the jewel is hidden.

decide() commits to an ID only when the best jewel is clearly better than the runner-up and than the
"not a jewel" patches; otherwise it answers "ambiguous" and the caller keeps what it had.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import faiss
except ImportError:  # numpy fallback, same results (exact inner product)
    faiss = None

# ---- pixel classes (same numbers as jewel_tracker_5.py) ----
BG_WHITE_S_MAX, BG_WHITE_V_MIN, BG_DARK_V_MAX = 40, 170, 60
SKIN_H_MAX, SKIN_S_MIN = 14, 110

# ---- superpixel signature ----
SIG_SIDE = 96             # crops are shrunk to this longest side first
SIG_SUPERPIXELS = 28      # target number of superpixels over the jewel
SIG_CLUSTERS = 4          # colour/texture regions per jewel
SIG_MIN_VALID_PX = 40     # fewer jewel pixels than this -> no signature
SIG_SIGMA = 14.0          # distance (Lab-ish units) at which a region stops being "explained"
SIG_TEX_W = 0.5           # weight of texture (std of L) in the distance
SIG_L_W = 0.4             # lighting shifts L most: count it less

# ---- FAISS gallery ----
VIEWS_PER_JEWEL = 40
TOP_K_MEAN = 3            # a jewel's embedding score = mean of its best K views

# ---- decision ----
W_EMB = 0.6               # fused = W_EMB * embedding + (1 - W_EMB) * signature
MIN_SCORE = 0.50          # best fused score at least this ...
MIN_MARGIN = 0.06         # ... and ahead of the runner-up by this much
NEG_MARGIN = 0.02         # ... and ahead of the best "not a jewel" patch by this much (embedding)
FULL_MARGIN = 0.20        # margin at which the answer is rated fully sure


def jewel_pixel_mask(img):
    """Pixels that belong to the jewel: not white bench, not dark stone/shadow, not skin."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bench = (s < BG_WHITE_S_MAX) & (v > BG_WHITE_V_MIN)
    dark = v < BG_DARK_V_MAX
    skin = (h <= SKIN_H_MAX) & (s >= SKIN_S_MIN)
    return ~(bench | dark | skin) & (img.max(axis=2) > 0)


def superpixel_signature(img, k=SIG_CLUSTERS):
    """(regions (k,4) = L, a, b, texture; weights (k,); jewel pixel count) or None when too few jewel pixels.
    SLIC superpixels over the jewel pixels, merged by agglomerative (Ward) clustering."""
    h, w = img.shape[:2]
    f = SIG_SIDE / max(h, w)
    if f < 1.0:
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    valid = jewel_pixel_mask(img)
    n_valid = int(valid.sum())
    if n_valid < SIG_MIN_VALID_PX:
        return None
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB)
    region = max(4, int(np.sqrt(n_valid / SIG_SUPERPIXELS)))
    slic = cv2.ximgproc.createSuperpixelSLIC(lab, cv2.ximgproc.SLICO, region, 10.0)
    slic.iterate(4)
    labels = slic.getLabels()
    feats, weights = [], []
    labf = lab.astype(np.float32)
    for sp in range(int(labels.max()) + 1):
        m = (labels == sp) & valid
        n = int(m.sum())
        if n < 6 or n < 0.5 * int((labels == sp).sum()):
            continue  # mostly bench / skin / stone: not the jewel
        px = labf[m]
        feats.append([px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean(), px[:, 0].std()])
        weights.append(n)
    if not feats:
        return None
    feats, weights = np.asarray(feats, np.float32), np.asarray(weights, np.float32)
    if len(feats) > k:
        from sklearn.cluster import AgglomerativeClustering
        scaled = feats * np.array([SIG_L_W, 1.0, 1.0, SIG_TEX_W], np.float32)
        groups = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
    else:
        groups = np.arange(len(feats))
    regions, rw = [], []
    for g in np.unique(groups):
        sel = groups == g
        wt = weights[sel]
        regions.append((feats[sel] * wt[:, None]).sum(0) / wt.sum())
        rw.append(wt.sum())
    rw = np.asarray(rw, np.float32)
    return np.asarray(regions, np.float32), rw / rw.sum(), n_valid


def _region_distance(a, b):
    """Pairwise distance between region sets a (n,4) and b (m,4): lighting (L) and texture count less."""
    scale = np.array([SIG_L_W, 1.0, 1.0, SIG_TEX_W], np.float32)
    d = (a[:, None, :] - b[None, :, :]) * scale
    return np.sqrt((d ** 2).sum(-1))


def signature_similarity(query, gallery):
    """0..1: share of the query's pixels whose colour/texture region the gallery jewel has (containment, so a
    hidden part of the jewel does not lower it), nudged by how much of the gallery jewel the query shows."""
    if query is None or gallery is None:
        return 0.0
    qr, qw, _ = query
    gr, gw, _ = gallery
    d = _region_distance(qr, gr)
    contain = float((qw * np.exp(-d.min(1) / SIG_SIGMA)).sum())
    cover = float((gw * np.exp(-d.min(0) / SIG_SIGMA)).sum())
    return 0.8 * contain + 0.2 * cover


# ---- augmentation: the jewel as it may look later, including under a hand ----
SKIN_BGR = (95, 125, 205)


def augment_view(crop, rng, occlude=True):
    h, w = crop.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(0, 360)), float(rng.uniform(0.85, 1.15)))
    m[:, 2] += rng.uniform(-0.06, 0.06, 2) * (w, h)
    out = cv2.warpAffine(crop, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-18, 18), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    if occlude and rng.random() < 0.6:  # fingers over part of it, or only part of it in view
        out = occlude_view(out, rng)
    return out


def occlude_view(img, rng):
    h, w = img.shape[:2]
    out = img.copy()
    kind = rng.integers(3)
    if kind == 0:  # skin-coloured blob (a finger) over up to ~35 % of the box
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        ax, ay = rng.uniform(0.12, 0.3) * w, rng.uniform(0.12, 0.3) * h
        skin = np.array(SKIN_BGR, np.float32) * rng.uniform(0.8, 1.15)
        cv2.ellipse(out, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(0, 180)), 0, 360,
                    tuple(int(c) for c in np.clip(skin, 0, 255)), -1)
    elif kind == 1:  # only a part of the jewel in the box
        x1, y1 = int(rng.uniform(0, 0.35) * w), int(rng.uniform(0, 0.35) * h)
        x2, y2 = int(w - rng.uniform(0, 0.35) * w), int(h - rng.uniform(0, 0.35) * h)
        out = cv2.resize(out[y1:max(y2, y1 + 8), x1:max(x2, x1 + 8)], (w, h))
    else:  # a blank strip (bench) where the hand lifted part of it
        x1, y1 = int(rng.uniform(0, 0.6) * w), int(rng.uniform(0, 0.6) * h)
        out[y1:y1 + int(0.3 * h), x1:x1 + int(0.3 * w)] = 255
    return out


class _Index:
    """Cosine nearest-neighbour index: FAISS when installed, numpy otherwise."""

    def __init__(self, dim):
        self.dim = dim
        self.vecs = np.zeros((0, dim), np.float32)
        self.index = faiss.IndexFlatIP(dim) if faiss is not None else None

    def add(self, v):
        v = np.ascontiguousarray(v, np.float32)
        self.vecs = np.concatenate([self.vecs, v])
        if self.index is not None:
            self.index.add(v)

    def search(self, q, k):
        q = np.ascontiguousarray(q, np.float32)
        k = min(k, len(self.vecs))
        if self.index is not None:
            return self.index.search(q, k)
        sims = q @ self.vecs.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx


class IdentityMatcher:
    """Drop-in for JewelIdModel (train / predict / ready) with the commit-only-when-sure rule."""

    def __init__(self, extractor):
        self.extractor = extractor      # JewelIdModel: .features(crops) -> unit-length (n, 512)
        self.ready = False
        self.labels = []
        self.index = None
        self.owner = np.zeros(0, np.int32)   # per gallery vector: jewel id, or -1 = "not a jewel"
        self.sigs = {}                  # jewel id -> [signature, ...]
        self.last = None                # the latest verdict, for logging

    # ---- build ----
    def train(self, crops_by_id, negatives, seed=0, views=VIEWS_PER_JEWEL):
        rng = np.random.default_rng(seed)
        self.labels = sorted(crops_by_id)
        crops, owner = [], []
        self.sigs = {}
        for jid in self.labels:
            c = crops_by_id[jid]
            # the same turns / scales / lights / hands for every jewel: two look-alikes then get exactly equal scores
            # (and are refused) instead of a margin made up by which random views each happened to get
            jrng = np.random.default_rng(seed + 1)
            views_ = [c] + [augment_view(c, jrng) for _ in range(views)]
            crops += views_
            owner += [jid] * len(views_)
            # signatures from the clean crop and a few re-lit / turned views (no occlusion: that is the query's job)
            self.sigs[jid] = [s for s in (superpixel_signature(v) for v in
                                          [c] + [augment_view(c, jrng, occlude=False) for _ in range(4)])
                              if s is not None]
        for c in negatives:
            crops.append(c)
            owner.append(-1)
        feats = self.extractor.features(crops)
        self.index = _Index(feats.shape[1])
        self.index.add(feats)
        self.owner = np.asarray(owner, np.int32)
        self.ready = True
        return self.self_check(crops_by_id, rng)

    def self_check(self, crops_by_id, rng, n=6):
        """Held-out augmented (and partly hidden) copies: share answered right among those answered at all, and
        the share answered at all."""
        right = answered = total = 0
        for jid, c in crops_by_id.items():
            for _ in range(n):
                total += 1
                got, _ = self.predict(augment_view(c, rng))
                if got is not None:
                    answered += 1
                    right += got == jid
        self.check_answered = answered / max(total, 1)
        return right / max(answered, 1)

    # ---- query ----
    def scores(self, crop):
        """{jewel id: fused score}, {id: embedding score}, {id: signature score}, best "not a jewel" similarity."""
        feat = self.extractor.features([crop])
        sims, idx = self.index.search(feat, max(len(self.owner), 1))
        sims, owners = sims[0], self.owner[idx[0]]
        emb = {}
        for jid in self.labels:
            s = sims[owners == jid][:TOP_K_MEAN]
            emb[jid] = float(s.mean()) if len(s) else 0.0
        neg = sims[owners == -1]
        neg_best = float(neg[:TOP_K_MEAN].mean()) if len(neg) else 0.0
        sig = self.signature_scores(crop)
        fused = {j: W_EMB * emb[j] + (1 - W_EMB) * sig[j] for j in self.labels}
        return fused, emb, sig, neg_best

    def signature_scores(self, crop):
        """{jewel id: signature score}: cheap (~ms), fit for every frame of a moving box."""
        q = superpixel_signature(crop)
        return {j: (max(signature_similarity(q, g) for g in self.sigs[j]) if q is not None and self.sigs[j] else 0.0)
                for j in self.labels}

    def decide(self, crop):
        """(verdict, jewel id or None, confidence, details). verdict: "jewel" (confident), "not_jewel" (the bench /
        stone patches fit better than any jewel), or "ambiguous" (no answer: keep what you had)."""
        fused, emb, sig, neg = self.scores(crop)
        order = sorted(fused, key=fused.get, reverse=True)
        best = order[0]
        second = fused[order[1]] if len(order) > 1 else 0.0
        margin = fused[best] - second
        info = dict(fused=fused, emb=emb, sig=sig, neg=neg, margin=margin)
        if neg > emb[best] + NEG_MARGIN:
            return "not_jewel", None, self._conf(neg - emb[best], NEG_MARGIN), info
        if fused[best] >= MIN_SCORE and margin >= MIN_MARGIN and emb[best] >= neg + NEG_MARGIN:
            return "jewel", best, self._conf(margin, MIN_MARGIN), info
        return "ambiguous", None, 0.0, info

    @staticmethod
    def _conf(margin, floor):
        return float(0.6 + 0.39 * np.clip((margin - floor) / FULL_MARGIN, 0.0, 1.0))

    def predict(self, crop):
        """Same answer shape as JewelIdModel: (jewel id, p), (None, p) for "not a jewel", (None, 0.0) when not sure."""
        if not self.ready or crop is None or crop.size == 0:
            return None, 0.0
        verdict, jid, p, self.last = self.decide(crop)
        return (jid, p) if verdict == "jewel" else (None, p if verdict == "not_jewel" else 0.0)
