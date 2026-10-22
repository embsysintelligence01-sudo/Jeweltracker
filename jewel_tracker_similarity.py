"""
Jewel Guard: RTSP jewel tracking with ROI-exit / disappearance alerts, Done-zone marking
and a multi-layer similarity check against jewel swaps.

Install:  pip install ultralytics opencv-python PyQt5 scipy numpy onnxruntime
          The similarity check uses the fraud-jewellery desktop backend in BACKEND_DIR
          (its app/ package, config.json and weights/siglip2-base-patch32-256_vision_encoder.sim.onnx).
Run:      python jewel_tracker_similarity.py   (nivagar_gold_v3.pt must sit next to this script)

Workflow:
  1. The camera connects on startup (Disconnect, edit the URL, Connect to change it; 0 = webcam)
  2. Draw ROI -> left-click corners -> right-click (or "Finish") to close.
     The ROI must cover the bench, the touchstone AND the Done zone.
  3. Draw Done zone the same way: the area inside the ROI where finished jewels are put.
     Both zones are saved to zones.json.
  4. Set "Jewels on bench" and place that many jewels in the ROI, outside the Done zone.
     Once exactly N are seen there for a short stable period they lock (green boxes, slots #1..#N).
  5. Move each jewel (touchstone, then Done zone). Alerts (message + 5 s beep) fire if a jewel
     leaves the ROI or stays unseen too long.
     Detection runs only inside the ROI, and detections that aren't gold-coloured (HSV check) are dropped.
  6. Jewels resting in the Done zone are marked DONE but stay tracked and guarded. Tracking only
     ends when you press "Reset (re-arm)", which then waits for the next batch.
  7. Similarity check: when a jewel first becomes DONE, a fingerprint of it is registered as its
     reference. Every time it vanishes (> VERIFY_AFTER_LOST_S) and lies still in the Done zone again,
     a new fingerprint is compared with that first one. The fingerprint uses only cues that survive
     a chain being crumpled and re-piled (no keypoint geometry):
       - type: chain (many link holes in the Otsu-on-saturation mask) or solid piece
       - surface area: jewel pixels at camera resolution (the camera is fixed)
       - colour: LAB histogram of the jewel pixels (bench removed by the backend's detail mask)
       - appearance: SigLIP2 embedding (0/90/180/270 deg) from the fraud-jewellery backend
     Type and area must agree; the combined score (geometric mean of area, colour, SigLIP2) is
     compared with the "Swap confidence" slider: MATCH -> OK, just below -> SIMILAR (orange warning,
     no beep), otherwise -> "POSSIBLE SWAP" alert. If the crop matches another slot's jewel instead,
     the tracker mixed up two labels and they are swapped back. Crops are saved in similarity_crops/.
"""
import os
import sys
import time
import json
import threading
import itertools

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# torch (via ultralytics) must load before PyQt5, otherwise on Windows its DLLs fail with WinError 1114
from ultralytics import YOLO
import queue
from pathlib import Path

# multi-layer similarity matcher of the fraud-jewellery desktop backend (loaded before PyQt5 too)
BACKEND_DIR = r"R:\similarity_check\desktop_backend"
try:
    sys.path.insert(0, BACKEND_DIR)
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)  # only errors, no graph-optimisation warnings
    from app.config import AppConfig
    from app.vision.features import prepare_detail_crop
    from app.models.siglip_onnx import SiglipVisionONNX
    BACKEND_CFG = AppConfig.load(Path(BACKEND_DIR) / "config.json")
    HAVE_BACKEND, BACKEND_ERROR = True, ""
except Exception as e:  # backend folder, models or onnxruntime missing: tracking still works
    BACKEND_CFG, HAVE_BACKEND, BACKEND_ERROR = None, False, str(e)
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from PyQt5 import QtCore, QtGui, QtWidgets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "nivagar_gold_v3.pt")
ZONES_FILE = os.path.join(SCRIPT_DIR, "zones.json")

# ---------------- tuning ----------------
# kept low on purpose: spread-out chains score only ~0.2-0.45; the gold colour check below
# removes the extra false positives a low threshold lets through
CONF = 0.15
IMGSZ = 640
LOCK_STABLE_FRAMES = 15    # exactly N jewels in the start area for this many frames -> lock
GATE_PX = 80               # max jump between consecutive frames
REACQ_SPEED_PX_S = 600     # search radius grows this fast while a jewel is unseen
SIZE_WEIGHT = 60           # penalty for size mismatch when matching
EDGE_MARGIN_PX = 40        # "lost close to the ROI boundary"
EDGE_GRACE_S = 1.0         # lost near boundary this long -> LEFT ROI
OUT_FRAMES = 3             # seen outside ROI this many frames -> LEFT ROI
DEFAULT_OCCLUSION_S = 5    # unseen inside ROI this long -> MISSING (editable in UI)
STREAM_TIMEOUT_S = 3.0     # no new camera frame this long while locked -> alert
DONE_HOLD_S = 5.0          # a jewel resting in the Done zone this long is shown as DONE
ALERT_BEEP_S = 5.0         # beep this long after an alert

# gold colour check (OpenCV HSV: H 0-179, S/V 0-255), applied to each detection's mask.
# White bench and very dark pixels (touchstone, shadows) are ignored; of the rest, at least
# GOLD_MIN_FRAC must be gold-coloured or the detection is dropped as a false positive.
GOLD_H = (8, 35)
GOLD_S_MIN = 40
GOLD_V_MIN = 80
BG_WHITE_S_MAX, BG_WHITE_V_MIN = 40, 170
BG_DARK_V_MAX = 60
GOLD_MIN_FRAC = 0.30
GOLD_MIN_PX = 30           # fewer non-background pixels than this -> not a jewel
MERGE_GAP_PX = 40          # detections whose masks come this close are pieces of one jewel
                           # (a spread-out chain's masks miss the thin strands, leaving ~30 px gaps)
                           # and are merged; ~1 cm at the current camera height, so keep
                           # separate jewels further apart than that
SHOW_REJECTED = False      # True: draw dropped (not gold) detections in dark red, for tuning

# similarity check for Done-zone jewels that vanish and come back (see workflow step 7)
SIM_DIR = os.path.join(SCRIPT_DIR, "similarity_crops")  # saved reference and check crops
VERIFY_AFTER_LOST_S = 0.3  # unseen longer than this (even a quick hand pass), then back -> compare
VERIFY_SETTLE_S = 1.0      # compare only after it has lain still in the Done zone this long (hand gone)
SETTLE_SPEED_PX_S = 40
CROP_PAD_PX = 20           # context around the jewel's box handed to the backend
# Measured on real pieces moved between sessions (bangle, small piece, re-piled chain):
#   same piece:  area ratio 0.59-0.85, colour 0.81-0.98, SigLIP2 0.79-0.94 -> combined 75-88 %
#   small vs bangle/chain: area ratio 0.06-0.15; bangle vs chain: similar area but different type
SIM_MIN_AREA_RATIO = 0.5   # smaller/larger than this factor of the reference -> different piece
DEFAULT_SWAP_CONF = 70     # "Swap confidence" slider start: minimum combined fingerprint score (%)
SIMILAR_BAND = 10          # this many points below the slider -> SIMILAR warning instead of SWAP
CHAIN_MIN_HOLES = 3        # link holes in the Otsu mask -> chain (measured: chains 7-8, solid pieces 0-1)
CHAIN_HOLE_MIN_PX = 12     # ignore holes smaller than this (noise)

COLORS = {"OK": (0, 200, 0), "OCCLUDED": (0, 200, 255),
          "LEFT_ROI": (0, 0, 255), "MISSING": (0, 0, 255)}
DONE_COLOR = (255, 255, 0)       # jewel resting in the Done zone
ROI_COLOR = (255, 200, 0)
DONE_ZONE_COLOR = (255, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def to_poly(pts):
    return np.array(pts, np.int32).reshape(-1, 1, 2) if pts and len(pts) >= 3 else None


# ---------------- camera ----------------
class RTSPGrabber(threading.Thread):
    """Always keeps only the newest frame, so processing never lags behind the stream."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.frame, self.ts = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.connected = False

    def run(self):
        while self.running:
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                time.sleep(2)
                continue
            self.connected = True
            while self.running:
                ok, f = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame, self.ts = f, time.time()
            cap.release()
            self.connected = False
            time.sleep(1)  # then reconnect

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.ts

    def stop(self):
        self.running = False


# ---------------- tracking ----------------
class Det:
    __slots__ = ("box", "c", "area", "conf", "gold", "mask", "origin")

    def __init__(self, box, c, area, conf, gold=1.0, mask=None, origin=(0, 0)):
        self.box, self.c, self.area, self.conf, self.gold = box, c, area, conf, gold
        self.mask, self.origin = mask, origin  # mask is local to its box; origin = its top-left in the frame


def touching(a, b, gap):
    """True if the masks of a and b come within gap pixels of each other."""
    # only pixels inside both boxes grown by gap can be that close, so work on that region alone
    x1 = max(a.origin[0], b.origin[0]) - gap
    y1 = max(a.origin[1], b.origin[1]) - gap
    x2 = min(a.origin[0] + a.mask.shape[1], b.origin[0] + b.mask.shape[1]) + gap
    y2 = min(a.origin[1] + a.mask.shape[0], b.origin[1] + b.mask.shape[0]) + gap
    if x2 <= x1 or y2 <= y1:
        return False

    def paste(d):
        canvas = np.zeros((y2 - y1, x2 - x1), np.uint8)
        ox, oy = d.origin
        sx1, sy1 = max(x1, ox), max(y1, oy)
        sx2, sy2 = min(x2, ox + d.mask.shape[1]), min(y2, oy + d.mask.shape[0])
        if sx2 > sx1 and sy2 > sy1:
            canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = d.mask[sy1 - oy:sy2 - oy, sx1 - ox:sx2 - ox]
        return canvas

    ca, cb = paste(a), paste(b)
    if not ca.any() or not cb.any():
        return False
    dist_to_a = cv2.distanceTransform(cv2.bitwise_not(ca), cv2.DIST_L2, 5)
    return bool(dist_to_a[cb > 0].min() <= gap)


def merge_dets(group):
    """One detection covering all of group (pieces of the same jewel)."""
    x1 = min(d.origin[0] for d in group)
    y1 = min(d.origin[1] for d in group)
    x2 = max(d.origin[0] + d.mask.shape[1] for d in group)
    y2 = max(d.origin[1] + d.mask.shape[0] for d in group)
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    for d in group:
        ox, oy = d.origin[0] - x1, d.origin[1] - y1
        mask[oy:oy + d.mask.shape[0], ox:ox + d.mask.shape[1]] |= d.mask
    m = cv2.moments(mask, binaryImage=True)
    area = max(m["m00"], 1.0)
    c = np.array([x1 + m["m10"] / area, y1 + m["m01"] / area])
    box = np.array([min(d.box[0] for d in group), min(d.box[1] for d in group),
                    max(d.box[2] for d in group), max(d.box[3] for d in group)])
    gold = sum(d.gold * d.area for d in group) / max(sum(d.area for d in group), 1.0)
    return Det(box, c, float(area), max(d.conf for d in group), gold, mask, (x1, y1))


def merge_pieces(dets):
    """Join detections whose masks touch (or overlap / sit inside each other) into one.
    A spread-out chain often comes back as a big detection plus a separate piece; the model's
    own IoU-based NMS keeps both because they barely overlap."""
    if len(dets) < 2 or any(d.mask is None for d in dets):
        return dets
    group_of = list(range(len(dets)))

    def root(i):
        while group_of[i] != i:
            i = group_of[i]
        return i

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if root(i) != root(j) and touching(dets[i], dets[j], MERGE_GAP_PX):
                group_of[root(j)] = root(i)
    groups = {}
    for i, d in enumerate(dets):
        groups.setdefault(root(i), []).append(d)
    return [g[0] if len(g) == 1 else merge_dets(g) for g in groups.values()]


def gold_fraction(hsv, mask):
    """Share of gold-coloured pixels among the non-background pixels under mask (None if too few)."""
    h, s, v = (c.astype(np.int16) for c in hsv[mask > 0].T)
    background = ((s < BG_WHITE_S_MAX) & (v > BG_WHITE_V_MIN)) | (v < BG_DARK_V_MAX)
    n = int((~background).sum())
    if n < GOLD_MIN_PX:
        return None
    gold = (h >= GOLD_H[0]) & (h <= GOLD_H[1]) & (s >= GOLD_S_MIN) & (v >= GOLD_V_MIN) & ~background
    return float(gold.sum()) / n


_track_uids = itertools.count()


class Track:
    def __init__(self, tid, det, t):
        self.id = tid                 # slot number shown on screen (can be corrected after a mix-up)
        self.uid = next(_track_uids)  # never changes
        self.pos = det.c.astype(float)
        self.vel = np.zeros(2)
        self.box = det.box
        self.area = det.area
        self.last_seen = t
        self.state = "OK"
        self.out_frames = 0
        self.done_since = None  # time it started resting in the Done zone
        self.has_ref = False         # similarity reference saved
        self.verify_pending = False  # came back after vanishing, waiting to lie still
        self.settle_since = None
        self.checking = False        # comparison running
        self.swap = False            # last comparison: not this jewel
        self.similar = False         # last comparison: same design, exact piece not confirmed
        self.mask, self.mask_origin = det.mask, det.origin  # detector mask (box-local) and its frame offset

    def predicted(self, t):
        # extrapolate only briefly; after that hold the last known position
        return self.pos + self.vel * min(t - self.last_seen, 0.3)

    def correct(self, det, t):
        dt = t - self.last_seen
        if dt < 0.5:
            v = (det.c - self.pos) / max(dt, 1e-3)
            self.vel = 0.7 * self.vel + 0.3 * v
            self.pos = 0.4 * self.pos + 0.6 * det.c
        else:  # re-acquired after a gap: jump to it, forget velocity
            self.vel[:] = 0
            self.pos = det.c.astype(float)
        if dt > VERIFY_AFTER_LOST_S and self.has_ref:
            self.verify_pending, self.settle_since = True, None
        self.box = det.box
        self.mask, self.mask_origin = det.mask, det.origin
        self.area = 0.9 * self.area + 0.1 * det.area
        self.last_seen = t


def jewel_region(frame, box, mask, origin, pad=CROP_PAD_PX):
    """Frame region around the jewel and the detector's segmentation mask in that region
    (the backend removes the bench inside it). Returns (image, mask) or (None, None)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    rx1, ry1, rx2, ry2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, w), min(y2 + pad, h)
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None, None
    img = frame[ry1:ry2, rx1:rx2].copy()
    m = np.zeros(img.shape[:2], np.uint8)
    if mask is not None:
        ox, oy = origin[0] - rx1, origin[1] - ry1
        sx1, sy1 = max(ox, 0), max(oy, 0)
        sx2, sy2 = min(ox + mask.shape[1], m.shape[1]), min(oy + mask.shape[0], m.shape[0])
        if sx2 > sx1 and sy2 > sy1:
            m[sy1:sy2, sx1:sx2] = mask[sy1 - oy:sy2 - oy, sx1 - ox:sx2 - ox]
    if not m.any():  # no detector mask: use the box
        m[max(y1 - ry1, 0):y2 - ry1, max(x1 - rx1, 0):x2 - rx1] = 255
    return img, m


def count_link_holes(img, mask):
    """Holes inside the jewel's Otsu-on-saturation mask: a chain's links leave many, however it is
    piled; bangles, rings and pendants almost none."""
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, otsu = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    otsu[cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0] = 0  # stay on this jewel
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for c, h in zip(contours, hierarchy[0]) if h[3] >= 0 and cv2.contourArea(c) >= CHAIN_HOLE_MIN_PX)


class SimilarityChecker(threading.Thread):
    """Fingerprint comparison (see workflow step 7), run off the video thread.
    Jobs (dicts):    kind "ref":   gen, slot, img, mask
                     kind "check": gen, slot, uid, img, mask   (slot = label of the track when queued)
    Results (dicts): kind "msg":   text, level
                     kind "ref":   gen, slot, chain
                     kind "check": gen, uid, scores = {slot: (verdict, score %, detail)} with verdict
                                   "match" | "similar" | "different"; only the queued slot when it
                                   matches, otherwise every reference of the batch (to spot mix-ups)
                     kind "error": gen, uid, text"""

    def __init__(self, min_conf=DEFAULT_SWAP_CONF):
        super().__init__(daemon=True)
        self.min_conf = min_conf  # %, set from the "Swap confidence" slider
        self.jobs, self.results = queue.Queue(), queue.Queue()
        self.refs = {}  # (gen, slot) -> fingerprint dict of the reference
        self.ready = False
        self.running = True

    def run(self):
        try:
            self.siglip = SiglipVisionONNX(str(Path(BACKEND_DIR) / BACKEND_CFG.models.siglip_encoder),
                                           BACKEND_CFG.preferred_provider)
            os.makedirs(SIM_DIR, exist_ok=True)
        except Exception as e:
            self.results.put(dict(kind="msg", text=f"Similarity check off: {e}", level="info"))
            return
        self.ready = True
        device = self.siglip.providers[0].replace("ExecutionProvider", "")
        self.results.put(dict(kind="msg", level="info", text=f"Similarity check ready (fingerprint, SigLIP2 on {device})"))
        while self.running:
            try:
                job = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if job["kind"] == "ref":
                    self.save_ref(job)
                else:
                    self.check(job)
            except Exception as e:
                self.results.put(dict(kind="error", gen=job["gen"], uid=job.get("uid"),
                                      text=f"Similarity check failed for #{job['slot']}: {e}"))

    def fingerprint(self, img, mask, rotations):
        """Crumple-proof description of one jewel; the reference gets 4 SigLIP2 rotations."""
        crop, crop_mask = prepare_detail_crop(img, mask)  # backend: bench removed, grey background
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        colour = cv2.calcHist([lab], [0, 1, 2], crop_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        colour /= max(float(colour.sum()), 1.0)
        embedding = self.siglip.embed_rotations(crop) if rotations else self.siglip.embed(crop)[None]
        return dict(area=max(int(np.count_nonzero(crop_mask)), 1), colour=colour, embedding=embedding,
                    chain=count_link_holes(img, mask) >= CHAIN_MIN_HOLES), crop

    def compare(self, q, ref):
        """(verdict, score %, detail) for query fingerprint q against a reference fingerprint."""
        area = min(q["area"], ref["area"]) / max(q["area"], ref["area"])
        colour = float(np.sum(np.sqrt(q["colour"] * ref["colour"])))  # Bhattacharyya coefficient
        siglip = float(np.clip(np.max(q["embedding"] @ ref["embedding"].T), 0.0, 1.0))
        score = round(100 * (area * colour * siglip) ** (1 / 3))
        kinds = lambda f: "chain" if f["chain"] else "solid"
        detail = (f"fingerprint {score}%: area x{q['area'] / ref['area']:.2f}, colour {colour:.2f}, "
                  f"SigLIP {siglip:.2f}, {kinds(q)}" + (f" vs {kinds(ref)}" if q["chain"] != ref["chain"] else ""))
        if q["chain"] != ref["chain"] or area < SIM_MIN_AREA_RATIO:
            return "different", score, detail
        if score >= self.min_conf:
            return "match", score, detail
        if score >= self.min_conf - SIMILAR_BAND:
            return "similar", score, detail
        return "different", score, detail

    def save_ref(self, job):
        gen, slot = job["gen"], job["slot"]
        self.refs = {k: v for k, v in self.refs.items() if k[0] == gen}  # forget earlier batches
        self.refs[(gen, slot)], crop = self.fingerprint(job["img"], job["mask"], rotations=True)
        cv2.imwrite(os.path.join(SIM_DIR, f"slot{slot}_ref.png"), crop)
        self.results.put(dict(kind="ref", gen=gen, slot=slot, chain=self.refs[(gen, slot)]["chain"]))

    def check(self, job):
        gen, slot = job["gen"], job["slot"]
        if (gen, slot) not in self.refs:
            return
        q, crop = self.fingerprint(job["img"], job["mask"], rotations=False)
        scores = {slot: self.compare(q, self.refs[(gen, slot)])}
        label = scores[slot][0]
        if label != "match":
            # not its own jewel: is it one of the others (tracker label mix-up)?
            for (g, other), ref in self.refs.items():
                if g == gen and other != slot:
                    scores[other] = self.compare(q, ref)
            mixed = [k for k, v in scores.items() if k != slot and v[0] == "match" and v[1] > scores[slot][1]]
            label = f"is{mixed[0]}" if mixed else ("similar" if label == "similar" else "SWAP")
        name = f"slot{slot}_check_{time.strftime('%H%M%S')}_{label}_{scores[slot][1]}.png"
        cv2.imwrite(os.path.join(SIM_DIR, name), crop)
        self.results.put(dict(kind="check", gen=gen, uid=job["uid"], scores=scores))


def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class Worker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    event = QtCore.pyqtSignal(str, str)  # message, level: "info" | "alert"
    status = QtCore.pyqtSignal(str)

    def __init__(self, source, roi_pts, done_pts, expected, occlusion_s):
        super().__init__()
        self.source = source
        self.running = True
        self.lock = threading.Lock()
        self.roi = to_poly(roi_pts)
        self.done = to_poly(done_pts)
        self.draft = []
        self.mode = "ARMING"  # ARMING -> LOCKED -> (Reset) -> ARMING
        self.expected = expected
        self.occlusion_s = occlusion_s
        self.tracks = []
        self.stable = 0
        self.arm_count = 0
        self.frame_wh = None
        self.rejected = []
        self.roi_mask, self.roi_mask_key = None, None
        self.swap_conf = DEFAULT_SWAP_CONF
        self.gen = 0      # bumped on every (re)arm so late similarity results of an old batch are ignored
        self.sim = None   # SimilarityChecker

    # ---- called from the GUI thread ----
    def set_zone(self, kind, pts):  # kind: "roi" or "done"
        with self.lock:
            setattr(self, kind, to_poly(pts))

    def set_draft(self, pts):
        with self.lock:
            self.draft = [tuple(p) for p in pts]

    def arm(self, n):
        with self.lock:
            self.expected = n
            self.gen += 1
            self.tracks, self.stable, self.mode = [], 0, "ARMING"

    def set_swap_conf(self, pct):
        self.swap_conf = pct
        if self.sim:
            self.sim.min_conf = pct

    def set_occlusion(self, s):
        with self.lock:
            self.occlusion_s = s

    def stop(self):
        self.running = False

    # ---- worker thread ----
    def run(self):
        self.status.emit("Loading model...")
        try:
            model = YOLO(MODEL_PATH)
        except Exception as e:
            self.event.emit(f"Model load failed: {e}", "alert")
            self.status.emit("Model load failed (see Events)")
            return
        if HAVE_BACKEND:
            self.sim = SimilarityChecker(self.swap_conf)
            self.sim.start()
        else:
            self.event.emit(f"Similarity check off: backend not available ({BACKEND_ERROR})", "info")
        grabber = RTSPGrabber(self.source)
        grabber.start()
        last_ts, fps, t_prev = 0.0, 0.0, time.time()
        stream_alerted, last_status = False, 0.0

        while self.running:
            frame, ts = grabber.latest()
            now = time.time()
            if frame is None or ts == last_ts:
                with self.lock:
                    locked = self.mode == "LOCKED"
                if locked and last_ts and now - last_ts > STREAM_TIMEOUT_S and not stream_alerted:
                    self.event.emit("CAMERA STREAM LOST while locked", "alert")
                    stream_alerted = True
                if now - last_status > 1.0:
                    self.status.emit("Connecting..." if not grabber.connected else "Waiting for frames...")
                    last_status = now
                self.msleep(5)
                continue
            if stream_alerted:
                self.event.emit("Camera stream restored", "info")
                stream_alerted = False
            last_ts = ts
            self.frame_wh = (frame.shape[1], frame.shape[0])

            with self.lock:
                self.ensure_roi()
                roi = self.roi.copy()
            dets, rejected = self.detect(model, frame, roi)
            with self.lock:
                self.rejected = rejected
                self.apply_sim_results()
                if self.mode == "ARMING":
                    self.try_lock(dets, ts)
                elif self.mode == "LOCKED":
                    self.update_tracks(dets, ts, frame)
                vis = self.draw(frame, dets, ts)
                mode = self.mode

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            self.status.emit(f"{mode}  |  {fps:.1f} FPS")
            self.frame_ready.emit(to_qimage(vis))
        grabber.stop()
        if self.sim:
            self.sim.running = False

    def detect(self, model, frame, roi):
        """Detect only inside the ROI, then drop detections that aren't gold-coloured.
        Returns (kept, rejected) in full-frame coordinates."""
        # run the model on the ROI's bounding box with everything outside the ROI blacked out
        ox, oy, w, h = cv2.boundingRect(roi)
        key = roi.tobytes()
        if self.roi_mask_key != key:  # rebuild the ROI mask only when the ROI changes
            self.roi_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(self.roi_mask, [(roi - (ox, oy)).astype(np.int32)], 255)
            self.roi_mask_key = key
        inside = self.roi_mask
        crop = cv2.bitwise_and(frame[oy:oy + h, ox:ox + w], frame[oy:oy + h, ox:ox + w], mask=inside)
        r = model.predict(crop, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        kept, rejected = [], []
        if r.boxes is None or len(r.boxes) == 0:
            return kept, rejected
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
        for b, cf, poly in zip(boxes, confs, polys):
            bx = b.astype(int)
            x1, y1 = max(bx[0], 0), max(bx[1], 0)
            x2, y2 = min(bx[2], w), min(bx[3], h)
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            area = max(float((b[2] - b[0]) * (b[3] - b[1])), 1.0)
            # colour check works on the detection's box only
            mask = inside[y1:y2, x1:x2].copy()
            if poly is not None and len(poly) >= 3:  # segmentation model: use mask centroid
                shape = np.zeros_like(mask)
                cv2.fillPoly(shape, [(poly - (x1, y1)).astype(np.int32)], 255)
                mask &= shape
                m = cv2.moments(poly.astype(np.float32))
                if m["m00"] > 1:
                    c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
                    area = m["m00"]
            gold = gold_fraction(cv2.cvtColor(crop[y1:y2, x1:x2], cv2.COLOR_BGR2HSV), mask) if mask.size else None
            det = Det(bx + (ox, oy, ox, oy), c + (ox, oy), float(area), float(cf), gold or 0.0,
                      mask, (x1 + ox, y1 + oy))
            if cv2.pointPolygonTest(roi, (float(det.c[0]), float(det.c[1])), False) <= 0:
                continue  # centre outside the ROI
            (kept if gold is not None and gold >= GOLD_MIN_FRAC else rejected).append(det)
        return merge_pieces(kept), rejected

    def ensure_roi(self):
        if self.roi is None and self.frame_wh:  # no ROI drawn -> whole frame
            w, h = self.frame_wh
            self.roi = to_poly([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            self.event.emit("No ROI drawn: using the whole frame", "info")

    def roi_dist(self, p):
        # >0 inside (distance to edge), <0 outside
        return cv2.pointPolygonTest(self.roi, (float(p[0]), float(p[1])), True)

    def in_done(self, p):
        return self.done is not None and cv2.pointPolygonTest(self.done, (float(p[0]), float(p[1])), False) > 0

    def try_lock(self, dets, t):
        # count only jewels in the start area, so jewels left in the Done zone by the last batch don't lock
        start = [d for d in dets if self.roi_dist(d.c) > 0 and not self.in_done(d.c)]
        self.arm_count = len(start)
        self.stable = self.stable + 1 if len(start) == self.expected else 0
        if self.stable >= LOCK_STABLE_FRAMES:
            start.sort(key=lambda d: d.c[0])  # slots #1..#N left to right
            self.tracks = [Track(i + 1, d, t) for i, d in enumerate(start)]
            self.mode = "LOCKED"
            self.event.emit(f"Locked {self.expected} jewel(s)", "info")
            if self.done is None:
                self.event.emit("No Done zone drawn: jewels can't be marked DONE", "info")

    def update_tracks(self, dets, t, frame=None):
        # Closed world: no new IDs after lock. Every detection can only be one of the N jewels.
        seen = set()
        if self.tracks and dets:
            BIG = 1e9
            cost = np.full((len(self.tracks), len(dets)), BIG)
            for i, tr in enumerate(self.tracks):
                lost = t - tr.last_seen
                gate = GATE_PX + REACQ_SPEED_PX_S * lost
                pred = tr.predicted(t)
                for j, d in enumerate(dets):
                    dist = float(np.hypot(*(d.c - pred)))
                    if dist > gate:
                        continue
                    cost[i, j] = dist + SIZE_WEIGHT * abs(np.log(d.area / tr.area))
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < BIG:
                    self.tracks[i].correct(dets[j], t)
                    seen.add(i)
        for i, tr in enumerate(self.tracks):
            self.evaluate(tr, i in seen, t)
            self.similarity(tr, i in seen, t, frame)

    def similarity(self, tr, seen, t, frame):
        if self.sim is None or not self.sim.ready or frame is None:
            return
        # reference: the first time the jewel has rested in the Done zone for DONE_HOLD_S
        if not tr.has_ref and seen and tr.done_since is not None and t - tr.done_since >= DONE_HOLD_S:
            img, mask = jewel_region(frame, tr.box, tr.mask, tr.mask_origin)
            if img is not None:
                tr.has_ref = True
                self.sim.jobs.put(dict(kind="ref", gen=self.gen, slot=tr.id, img=img, mask=mask))
            return
        # it vanished (even briefly) and came back: compare with the first reference once it lies
        # still in the Done zone, so the hand is out of the crop
        if tr.verify_pending:
            still = seen and float(np.hypot(*tr.vel)) <= SETTLE_SPEED_PX_S and self.in_done(tr.pos)
            if not still:
                tr.settle_since = None
            elif tr.settle_since is None:
                tr.settle_since = t
            elif t - tr.settle_since >= VERIFY_SETTLE_S:
                img, mask = jewel_region(frame, tr.box, tr.mask, tr.mask_origin)
                if img is not None:
                    tr.verify_pending, tr.checking = False, True
                    self.sim.jobs.put(dict(kind="check", gen=self.gen, slot=tr.id, uid=tr.uid, img=img,
                                           mask=mask))

    def apply_sim_results(self):
        while self.sim is not None:
            try:
                r = self.sim.results.get_nowait()
            except queue.Empty:
                return
            if r["kind"] == "msg":
                self.event.emit(r["text"], r["level"])
                continue
            if r["gen"] != self.gen:
                continue  # result for an earlier batch
            if r["kind"] == "ref":
                self.event.emit(f"Jewel #{r['slot']}: reference saved ({'chain' if r['chain'] else 'solid piece'})",
                                "info")
                continue
            tr = next((x for x in self.tracks if x.uid == r["uid"]), None)
            if tr is None:
                continue
            tr.checking = False
            if r["kind"] == "error":
                self.event.emit(r["text"], "info")
                continue
            self.judge(tr, r["scores"])

    def judge(self, tr, scores):
        own = scores.get(tr.id)
        if own is not None and own[0] == "match":
            tr.swap = tr.similar = False
            self.event.emit(f"Jewel #{tr.id} matches its reference ({own[2]})", "info")
            return
        own_score = own[1] if own is not None else -1
        others = {k: v for k, v in scores.items() if k != tr.id and v[0] == "match" and v[1] > own_score}
        if others:
            # it fits another jewel better: the tracker swapped two labels after a hand pass. Swap them
            # back, then re-check the jewel that now carries this track's old label.
            slot = max(others, key=lambda k: others[k][1])
            other = next((x for x in self.tracks if x.id == slot), None)
            old = tr.id
            tr.id, tr.swap, tr.similar = slot, False, False
            if other is not None:
                other.id, other.swap, other.similar = old, False, False
                if not other.checking:  # a check already running is judged against its new label
                    other.verify_pending, other.settle_since = True, None
            self.event.emit(f"Jewels #{old} and #{slot} were mixed up by the tracker - labels corrected "
                            f"({others[slot][2]})", "info")
            return
        if own is None:
            tr.verify_pending, tr.settle_since = True, None  # label changed while checking: check again
            return
        if own[0] == "similar":
            tr.swap, tr.similar = False, True
            self.event.emit(f"Jewel #{tr.id} SIMILAR DESIGN - exact piece not confirmed ({own[2]})", "warn")
            return
        tr.swap, tr.similar = True, False
        self.event.emit(f"Jewel #{tr.id} POSSIBLE SWAP ({own[2]}, need {self.swap_conf}%)", "alert")

    def evaluate(self, tr, seen, t):
        old = tr.state
        d = self.roi_dist(tr.pos)
        if seen:
            if d < 0:
                tr.out_frames += 1
                new = "LEFT_ROI" if tr.out_frames >= OUT_FRAMES else old
            else:
                tr.out_frames = 0
                new = "OK"
        else:
            lost = t - tr.last_seen
            if old in ("LEFT_ROI", "MISSING"):
                new = old  # latched until the jewel is seen inside the ROI again
            elif d < EDGE_MARGIN_PX and lost > EDGE_GRACE_S:
                new = "LEFT_ROI"
            elif lost > self.occlusion_s:
                new = "MISSING"
            elif lost > 0.3:
                new = "OCCLUDED"  # hand over it / in hand: no alert yet
            else:
                new = old
        if new != old:
            tr.state = new
            if new == "LEFT_ROI":
                self.event.emit(f"Jewel #{tr.id} LEFT THE ROI", "alert")
            elif new == "MISSING":
                self.event.emit(f"Jewel #{tr.id} MISSING ({self.occlusion_s}s unseen)", "alert")
            elif old in ("LEFT_ROI", "MISSING") and new == "OK":
                self.event.emit(f"Jewel #{tr.id} back inside ROI", "info")
        # the hold timer only runs while the jewel is actually visible in the Done zone,
        # so a jewel picked up again (hidden in the hand) can't count as done
        if tr.state == "OK" and self.in_done(tr.pos):
            if tr.done_since is None:
                tr.done_since = t
        else:
            tr.done_since = None

    def draw(self, vis, dets, t):
        if self.roi is not None:
            cv2.polylines(vis, [self.roi], True, ROI_COLOR, 2)
        if self.done is not None:
            cv2.polylines(vis, [self.done], True, DONE_ZONE_COLOR, 2)
            x, y = self.done[:, 0, :].min(axis=0)
            cv2.putText(vis, "DONE ZONE", (int(x) + 5, int(y) + 22), FONT, 0.7, DONE_ZONE_COLOR, 2)
        for p in self.draft:
            cv2.circle(vis, p, 5, (0, 255, 255), -1)
        if len(self.draft) > 1:
            cv2.polylines(vis, [np.array(self.draft, np.int32)], False, (0, 255, 255), 1)
        if SHOW_REJECTED:
            for d in self.rejected:
                x1, y1, x2, y2 = d.box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 140), 1)
                cv2.putText(vis, f"not gold {d.gold:.0%}", (x1, max(y1 - 4, 12)), FONT, 0.5, (0, 0, 140), 1)
        for d in dets:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (130, 130, 130), 1)
        alerts = []
        for tr in self.tracks:
            if tr.swap:
                col = (0, 0, 255)
                label = f"#{tr.id} SWAP?"
            elif tr.similar:
                col = (0, 140, 255)
                label = f"#{tr.id} SIMILAR?"
            elif tr.done_since is not None:
                col = DONE_COLOR
                held = t - tr.done_since
                label = f"#{tr.id} DONE" if held >= DONE_HOLD_S else f"#{tr.id} DONE {held:.0f}/{DONE_HOLD_S:.0f}s"
            else:
                col = COLORS[tr.state]
                label = f"#{tr.id}" if tr.state == "OK" else f"#{tr.id} {tr.state}"
            if tr.verify_pending or tr.checking:
                label += " checking"
            elif tr.has_ref and not (tr.swap or tr.similar):
                label += " (ref)"
            x1, y1, x2, y2 = tr.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.circle(vis, tuple(int(v) for v in tr.pos), 4, col, -1)
            cv2.putText(vis, label, (x1, max(y1 - 6, 14)), FONT, 0.7, col, 2)
            if tr.state in ("LEFT_ROI", "MISSING"):
                alerts.append(f"#{tr.id} {tr.state}")
            if tr.swap:
                alerts.append(f"#{tr.id} POSSIBLE SWAP")
        bottom = vis.shape[0] - 15
        if self.mode == "ARMING":
            cv2.putText(vis, f"Waiting for {self.expected} jewels in the start area: {self.arm_count} seen "
                             f"({self.stable}/{LOCK_STABLE_FRAMES})",
                        (10, bottom), FONT, 0.7, (0, 255, 255), 2)
        elif self.mode == "LOCKED":
            moved = sum(tr.done_since is not None for tr in self.tracks)
            cv2.putText(vis, f"In Done zone: {moved}/{len(self.tracks)}", (10, bottom), FONT, 0.8, DONE_COLOR, 2)
        if alerts:
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 42), (0, 0, 255), -1)
            cv2.putText(vis, "ALERT: " + ", ".join(alerts), (10, 30), FONT, 0.9, (255, 255, 255), 2)
        return vis


# ---------------- GUI ----------------
class VideoLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int, int, bool)  # x, y in frame pixels, is_left_button

    def __init__(self):
        super().__init__("No video")
        self.setMinimumSize(800, 450)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background:#111; color:#888;")
        self.frame_size = None

    def set_frame(self, img):
        self.frame_size = (img.width(), img.height())
        self.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def mousePressEvent(self, e):
        pm = self.pixmap()
        if not self.frame_size or pm is None or pm.isNull():
            return
        pw, ph = pm.width(), pm.height()
        ox, oy = (self.width() - pw) / 2, (self.height() - ph) / 2
        x = (e.x() - ox) * self.frame_size[0] / pw
        y = (e.y() - oy) * self.frame_size[1] / ph
        if 0 <= x < self.frame_size[0] and 0 <= y < self.frame_size[1]:
            self.clicked.emit(int(x), int(y), e.button() == QtCore.Qt.LeftButton)


def beep():
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(2500, 400), daemon=True).start()
    except ImportError:
        QtWidgets.QApplication.beep()


class MainWindow(QtWidgets.QWidget):
    ZONE_NAMES = {"roi": "ROI", "done": "Done zone"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jewel Guard")
        self.worker = None
        self.zones = self.load_zones()  # {"roi": [...], "done": [...]}
        self.draft = []
        self.drawing = None  # None, "roi" or "done"

        self.url = QtWidgets.QLineEdit("rtsp://admin:admin123@192.168.29.214:554/cam/realmonitor?channel=1&subtype=0")
        self.btn_conn = QtWidgets.QPushButton("Connect")
        self.video = VideoLabel()
        self.btn_roi = QtWidgets.QPushButton("Draw ROI")
        self.btn_done = QtWidgets.QPushButton("Draw Done zone")
        self.btn_clear = QtWidgets.QPushButton("Clear zones")
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(1, 30)
        self.spin_n.setValue(3)
        self.spin_occ = QtWidgets.QSpinBox()
        self.spin_occ.setRange(2, 300)
        self.spin_occ.setValue(DEFAULT_OCCLUSION_S)
        self.spin_occ.setSuffix(" s")
        self.slider_swap = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_swap.setRange(0, 100)
        self.slider_swap.setValue(DEFAULT_SWAP_CONF)
        self.slider_swap.setToolTip("Minimum fingerprint score (area, colour, SigLIP2) for a jewel that\n"
                                    "comes back to count as the same piece.\n"
                                    "Higher = stricter (more swap alerts), lower = more lenient.")
        self.lbl_swap = QtWidgets.QLabel(f"{DEFAULT_SWAP_CONF}%")
        self.lbl_swap.setFixedWidth(34)
        swap_row = QtWidgets.QHBoxLayout()
        swap_row.addWidget(self.slider_swap, 1)
        swap_row.addWidget(self.lbl_swap)
        self.btn_reset = QtWidgets.QPushButton("Reset (re-arm)")
        self.lbl_status = QtWidgets.QLabel("Not connected")
        self.log = QtWidgets.QListWidget()

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Camera:"))
        top.addWidget(self.url, 1)
        top.addWidget(self.btn_conn)

        form = QtWidgets.QFormLayout()
        form.addRow("Jewels on bench", self.spin_n)
        form.addRow("Missing after", self.spin_occ)
        form.addRow("Swap confidence", swap_row)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.btn_roi)
        side.addWidget(self.btn_done)
        side.addWidget(self.btn_clear)
        side.addLayout(form)
        side.addWidget(self.btn_reset)
        side.addWidget(self.lbl_status)
        side.addWidget(QtWidgets.QLabel("Events"))
        side.addWidget(self.log, 1)
        side_w = QtWidgets.QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(280)

        mid = QtWidgets.QHBoxLayout()
        mid.addWidget(self.video, 1)
        mid.addWidget(side_w)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid, 1)

        self.btn_conn.clicked.connect(self.toggle_connect)
        self.btn_roi.clicked.connect(lambda: self.toggle_draw("roi"))
        self.btn_done.clicked.connect(lambda: self.toggle_draw("done"))
        self.btn_clear.clicked.connect(self.clear_zones)
        self.btn_reset.clicked.connect(self.rearm)
        self.spin_n.valueChanged.connect(self.rearm)
        self.spin_occ.valueChanged.connect(lambda v: self.worker and self.worker.set_occlusion(v))
        self.slider_swap.valueChanged.connect(self.set_swap_conf)
        self.video.clicked.connect(self.on_click)

        self.beep_until = 0.0
        self.beep_timer = QtCore.QTimer(self)
        self.beep_timer.timeout.connect(self.beep_tick)
        self.set_running(False)
        loaded = [self.ZONE_NAMES[k] for k in ("roi", "done") if self.zones[k]]
        if loaded:
            self.add_log("Loaded saved " + " and ".join(loaded), "info")
        QtCore.QTimer.singleShot(0, self.toggle_connect)  # connect to the camera on startup

    # ---- zones ----
    def load_zones(self):
        try:
            with open(ZONES_FILE) as f:
                data = json.load(f)
            return {k: [tuple(p) for p in data.get(k, [])] for k in ("roi", "done")}
        except Exception:
            return {"roi": [], "done": []}

    def save_zones(self):
        # write to a temp file first so a crash mid-write can't corrupt the saved zones
        tmp = ZONES_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.zones, f)
            os.replace(tmp, ZONES_FILE)
        except OSError as e:
            self.add_log(f"Could not save zones: {e}", "alert")

    def zone_button(self, kind):
        return self.btn_roi if kind == "roi" else self.btn_done

    def toggle_draw(self, kind):
        if self.drawing == kind:
            self.finish_draw()
            return
        if self.drawing:
            self.finish_draw()
        self.drawing, self.draft = kind, []
        self.zone_button(kind).setText("Finish " + self.ZONE_NAMES[kind])

    def finish_draw(self):
        kind, self.drawing = self.drawing, None
        self.zone_button(kind).setText("Draw " + self.ZONE_NAMES[kind])
        if len(self.draft) >= 3:
            self.zones[kind] = self.draft
            self.save_zones()
            self.add_log(f"{self.ZONE_NAMES[kind]} saved (kept after restart)", "info")
            if self.worker:
                self.worker.set_zone(kind, self.zones[kind])
        else:
            self.add_log(f"{self.ZONE_NAMES[kind]} needs at least 3 points", "info")
        self.draft = []
        if self.worker:
            self.worker.set_draft([])

    def clear_zones(self):
        if QtWidgets.QMessageBox.question(self, "Clear zones", "Delete the saved ROI and Done zone?") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.zones = {"roi": [], "done": []}
        if os.path.exists(ZONES_FILE):
            os.remove(ZONES_FILE)
        if self.worker:
            self.worker.set_zone("roi", [])
            self.worker.set_zone("done", [])
        self.add_log("Zones cleared", "info")

    def on_click(self, x, y, left):
        if not self.drawing or not self.worker:
            return
        if left:
            self.draft.append((x, y))
            self.worker.set_draft(self.draft)
        else:
            self.finish_draw()

    # ---- control ----
    def set_running(self, running):
        for w in (self.btn_roi, self.btn_done, self.btn_clear, self.btn_reset):
            w.setEnabled(running)
        self.url.setEnabled(not running)
        self.btn_conn.setText("Disconnect" if running else "Connect")

    def toggle_connect(self):
        if self.worker:
            self.stop_worker()
            self.lbl_status.setText("Not connected")
            return
        text = self.url.text().strip()
        source = int(text) if text.isdigit() else text
        self.worker = Worker(source, self.zones["roi"], self.zones["done"],
                             self.spin_n.value(), self.spin_occ.value())
        self.worker.set_swap_conf(self.slider_swap.value())
        self.worker.frame_ready.connect(self.video.set_frame)
        self.worker.event.connect(self.add_log)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.start()
        self.set_running(True)

    def stop_worker(self):
        if self.drawing:
            self.finish_draw()
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.set_running(False)

    def set_swap_conf(self, v):
        self.lbl_swap.setText(f"{v}%")
        if self.worker:
            self.worker.set_swap_conf(v)

    def rearm(self):
        if not self.worker:
            return
        self.worker.arm(self.spin_n.value())
        self.add_log(f"Waiting for {self.spin_n.value()} jewel(s)...", "info")

    # ---- events / alarm ----
    def add_log(self, msg, level):
        item = QtWidgets.QListWidgetItem(f"{time.strftime('%H:%M:%S')}  {msg}")
        if level == "alert":
            item.setForeground(QtGui.QColor("#e53935"))
            self.beep_until = time.time() + ALERT_BEEP_S
            if not self.beep_timer.isActive():
                beep()
                self.beep_timer.start(600)
        elif level == "warn":
            item.setForeground(QtGui.QColor("#fb8c00"))
        self.log.insertItem(0, item)

    def beep_tick(self):
        if time.time() < self.beep_until:
            beep()
        else:
            self.beep_timer.stop()

    def closeEvent(self, e):
        self.stop_worker()
        e.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1280, 760)
    w.show()
    sys.exit(app.exec_())
