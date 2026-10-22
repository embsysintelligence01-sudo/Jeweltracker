"""
Jewel Guard: RTSP jewel tracking with ROI-exit / disappearance alerts and a per-jewel rubbing check.

Install:  pip install ultralytics opencv-python PyQt5 scipy numpy onnxruntime
Run:      python jewel_tracker.py   (nivagar_gold_v3.pt must sit next to this script; so must the
          rubbing models newstone1.onnx and rubbing_gold.onnx, copied from R:\Rubbing_Dept_Roha)

Workflow:
  1. The camera connects on startup. The top bar picks how it is reached:
       - Network (router/Wi-Fi): the Dahua camera's address on the LAN
       - Direct Ethernet: the camera cabled straight into this PC's Ethernet port; camera and PC need
         static IPs in the same subnet (this PC's Ethernet is 192.168.1.1/24, so e.g. camera 192.168.1.108)
       - Custom URL: any RTSP URL, or 0 for a webcam
     Switching it while connected reconnects to that feed. Settings are saved to camera.json.
  2. Draw ROI -> left-click corners -> right-click (or "Finish ROI") to close. It must cover the
     bench and the touchstone. "Draw stone" marks the touchstone the same way (recommended: the stone
     model of run-local4 barely finds the stone from this overhead camera). Both are saved to zones.json.
  3. Set "Jewels on bench" and place that many jewels in the ROI. Once exactly N are seen for a
     short stable period they lock (green boxes, slots #1..#N).
  4. Alerts (message + 5 s beep) fire if a jewel leaves the ROI or stays unseen too long.
     Detection runs only inside the ROI, and detections that aren't gold-coloured (HSV check) are dropped.
  5. Rubbing check (visual part of run-local4.py, no audio): the touchstone is found with its stone
     model, gold on the stone with its rubbing-gold model (mask clipped to the stone, >= 50 % inside),
     and rubbing is gold on the stone whose centroid keeps moving. Rubbing is credited to one jewel:
     the tracked jewel on that gold, else the jewel missing from the bench (in the hand) nearest the
     stone. After RUB_CONFIRM_S of rubbing that jewel is RUBBED; while it is being rubbed it shows
     RUBBING instead of going MISSING.
  6. IDs stay with their jewel: at lock each jewel gets an appearance fingerprint (chain or solid,
     surface area, colour, plus SigLIP2 when the fraud-jewellery backend is present). Jewels tracked
     continuously are matched by position (with colour/type so two jewels passing close can't swap);
     a jewel that was out of sight (in the hand, rubbing) is matched back by its fingerprint, since it
     can be put down anywhere.
     When the detector misses a jewel it saw a moment ago (flicker), OpenCV CSRT holds it in place
     for up to CSRT_MAX_GAP_S, as long as it doesn't move; then the normal OCCLUDED/MISSING logic applies.
  7. A jewel is DONE only when it was RUBBED and then comes back: seen off the stone, lying still for
     DONE_SETTLE_S. DONE jewels stay tracked and guarded. Tracking only ends when you press
     "Reset (re-arm)", which then waits for the next batch.
"""
import os
import sys
import time
import json
import threading
import socket
from urllib.parse import quote, urlparse

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# never let ultralytics pip-install packages on its own (it tries to when loading the ONNX models)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# torch (via ultralytics) must load before PyQt5, otherwise on Windows its DLLs fail with WinError 1114
from ultralytics import YOLO
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from PyQt5 import QtCore, QtGui, QtWidgets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# tracking model. rubbing_gold.onnx was tried here too, but on this camera it missed small pieces and
# ran ~4x slower (320 input, CPU), so it is only used for the gold-on-stone rubbing check.
MODEL_PATH = os.path.join(SCRIPT_DIR, "nivagar_gold_v3.pt")
ZONES_FILE = os.path.join(SCRIPT_DIR, "zones.json")
CAMERA_FILE = os.path.join(SCRIPT_DIR, "camera.json")
# camera connections (editable in the top bar, saved to camera.json)
CAMERA_MODES = {"network": "Network (router/Wi-Fi)", "direct": "Direct Ethernet", "custom": "Custom URL"}
DEFAULT_CAMERAS = {
    "mode": "network",
    "network": {"ip": "192.168.29.214", "user": "admin", "password": "admin123", "channel": 1, "subtype": 0},
    "direct": {"ip": "192.168.1.108", "user": "admin", "password": "admin123", "channel": 1, "subtype": 0},
    "custom": {"url": "0"},
}
RTSP_PORT = 554

# ---------------- tuning ----------------
# kept low on purpose: spread-out chains score only ~0.2-0.45; the gold colour check below
# removes the extra false positives a low threshold lets through
CONF = 0.15
# The end application runs on CPU only. Measured on a Ryzen 5 6600H CPU, detection on the ROI:
#   imgsz 640: ~275 ms/frame   imgsz 384: ~155 ms   imgsz 320: ~115 ms, still all 3 test jewels
#   (small piece confidence 0.62-0.73 vs 0.87-0.90 at 640). An ONNX export was no faster.
DEVICE = "cpu"             # "cpu" for the end application; "0" would use a CUDA GPU while developing
IMGSZ = 320
LOCK_WINDOW_S = 1.5        # lock once exactly N jewels were seen in LOCK_MIN_SHARE of the frames
LOCK_MIN_SHARE = 0.8       # over the last LOCK_WINDOW_S (a jewel near the confidence cut-off may flicker)
GATE_PX = 80               # max jump between consecutive frames
REACQ_SPEED_PX_S = 600     # search radius grows this fast while a jewel is unseen
SIZE_WEIGHT = 60           # penalty for size mismatch when matching
EDGE_MARGIN_PX = 40        # "lost close to the ROI boundary"
EDGE_GRACE_S = 1.0         # lost near boundary this long -> LEFT ROI
OUT_FRAMES = 3             # seen outside ROI this many frames -> LEFT ROI
DEFAULT_OCCLUSION_S = 5    # unseen inside ROI this long -> MISSING (editable in UI)
STREAM_TIMEOUT_S = 3.0     # no new camera frame this long while locked -> alert
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

# rubbing check: the visual part of R:\Rubbing_Dept_Roha\run-local4.py (no audio)
STONE_MODEL_PATH = os.path.join(SCRIPT_DIR, "newstone1.onnx")
RUB_GOLD_MODEL_PATH = os.path.join(SCRIPT_DIR, "rubbing_gold.onnx")
STONE_CONF = 0.03              # run-local4 uses 0.20; from this overhead camera the stone scores 0.04-0.30,
                               # so only large, roughly square boxes are accepted (a drawn stone is better)
STONE_ASPECT = (0.6, 1.6)      # width/height of an accepted stone box
STONE_MIN_AREA_RATIO = 0.02    # run-local4: stone box must cover this share of the frame
STONE_REFRESH_S = 3.0          # the stone doesn't move: look for it this often (~0.2 s on CPU)
RUB_GOLD_CONF = 0.70           # run-local4 GOLD_CONF_THRESH
RUB_GOLD_IMGSZ = 320           # rubbing_gold.onnx input size
RUB_CROP_PAD_RATIO = 0.90      # run-local4 GOLD_CROP_PAD_RATIO
RUB_MIN_MASK_PX = 80           # run-local4 GOLD_MIN_MASK_PIXELS
RUB_MIN_INSIDE_RATIO = 0.50    # run-local4 GOLD_MIN_INSIDE_STONE_RATIO
RUB_MOVE_PX = 1.5              # run-local4 RUBBING_MIN_CENTROID_MOVE is 0.5 px at 640x480; x3 for 1080p
RUB_MOVE_WINDOW = 3            # moves judged (run-local4: 5 at every 2nd frame; the rubbing thread checks
                               # ~20x/s, so 3 moves are ~0.15 s)
RUB_MAX_MISSES = 2             # checks without gold on the stone (fingers over it) before motion restarts
RUB_CONFIRM_S = 0.6            # seconds of rubbing credited to one jewel -> RUBBED (stands in for audio)
RUB_GAP_S = 0.5                # rubbing pauses shorter than this still count as continuous
RUB_MAX_CHECKS_PER_S = 10      # CPU budget for the rubbing thread (0 = as fast as possible); at ~45 ms
                               # per check, unlimited it starves the detector on a CPU-only machine
RUB_START_AFTER_S = 1.0        # rubbing is only looked for once a jewel is off the bench this long (or
                               # on the stone), not on a one-frame detection flicker
RUB_GRACE_S = 2.0              # rubbing seen this recently: an unseen jewel is RUBBING, not MISSING
STONE_MARGIN_PX = 40           # "on the stone" includes this margin around the stone box
DONE_SETTLE_S = 2.0            # rubbed jewel lying still off the stone this long -> DONE
SETTLE_SPEED_PX_S = 40

# keeping IDs on their jewel (the similarity cues): see Worker.update_tracks
APP_WEIGHT_PX = 150        # colour mismatch (1 - Bhattacharyya) in px of matching cost while tracked
TYPE_MISMATCH_PX = 400     # chain vs solid piece, in px of matching cost while tracked
REID_AFTER_S = 0.5         # unseen longer than this: matched back by fingerprint, not position
REID_MIN_SIM = 0.45        # a detection less similar than this is never taken for a lost jewel
# detector flicker: when nivagar misses a jewel it saw a moment ago, OpenCV CSRT follows it from its last
# detected box (~45-77 ms per jewel per frame on CPU, so only while it is missed)
CSRT_MAX_GAP_S = 1.0       # bridge at most this long after the last real detection
CSRT_MAX_SHIFT_PX = 40     # CSRT box moving further than this from the last detection (jewel carried
                           # off, or CSRT following the hand) ends the bridge -> normal OCCLUDED/MISSING
# Rub marks on the touchstone look like gold to the detector and can grow to a jewel's size, so on the
# stone area is ignored and colour AND SigLIP2 must each match the jewel. Measured: the red necklace
# moved 0.96 / 0.87, a large rub mark against it 0.45 / 0.68.
ON_STONE_MIN_COLOUR = 0.80
ON_STONE_MIN_SIGLIP = 0.80
BENCH_BEATS_STONE_SIM = 0.70  # a bench detection this similar to a jewel held on the stone takes its ID
CHAIN_MIN_HOLES = 3        # link holes in the Otsu mask -> chain (measured: chains 7-8, solid pieces 0-1)
CHAIN_HOLE_MIN_PX = 12
SIGLIP_BACKEND_DIR = r"R:\similarity_check\desktop_backend"  # optional: SigLIP2 for the fingerprint
SIGLIP_MODEL = os.path.join(SIGLIP_BACKEND_DIR, "weights", "siglip2-base-patch32-256_vision_encoder.sim.onnx")

COLORS = {"OK": (0, 200, 0), "OCCLUDED": (0, 200, 255), "RUBBING": (0, 140, 255),
          "LEFT_ROI": (0, 0, 255), "MISSING": (0, 0, 255)}
DONE_COLOR = (255, 255, 0)       # rubbed and placed back
RUBBED_COLOR = (255, 0, 255)     # rubbed, not placed back yet
ROI_COLOR = (255, 200, 0)
STONE_COLOR = (255, 120, 0)
RUB_GOLD_COLOR = (0, 215, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def to_poly(pts):
    return np.array(pts, np.int32).reshape(-1, 1, 2) if pts and len(pts) >= 3 else None


# ---------------- camera ----------------
def dahua_url(cam):
    """RTSP URL of a Dahua camera (subtype 0 = main stream, 1 = sub stream)."""
    auth = f"{quote(cam['user'], safe='')}:{quote(cam['password'], safe='')}@" if cam.get("user") else ""
    return (f"rtsp://{auth}{cam['ip']}:{RTSP_PORT}/cam/realmonitor"
            f"?channel={int(cam.get('channel', 1))}&subtype={int(cam.get('subtype', 0))}")


def local_ipv4s():
    try:
        return {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        return set()


def unreachable_hint(host, port):
    """None if host:port accepts a TCP connection, else a message on what to check."""
    try:
        socket.create_connection((host, port), timeout=2).close()
        return None
    except OSError as e:
        reason = str(e)
    subnet = ".".join(host.split(".")[:3])
    if host.count(".") == 3 and not any(ip.startswith(subnet + ".") for ip in local_ipv4s()):
        return (f"Camera {host} not reachable: this PC has no address in {subnet}.x. Plug the cable into the "
                f"Ethernet port, and give that port a static IP like {subnet}.10 / 255.255.255.0 "
                f"(or change the camera IP to this PC's subnet).")
    return f"Camera {host}:{port} not reachable ({reason}). Check the cable, the camera IP and that it is powered."


class RTSPGrabber(threading.Thread):
    """Always keeps only the newest frame, so processing never lags behind the stream."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.frame, self.ts = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.problem = ""  # why the camera can't be reached, shown in the status line

    def run(self):
        while self.running:
            if not isinstance(self.source, int):
                # quick TCP check first: opening an unreachable RTSP URL can hang for ~30 s
                url = urlparse(self.source)
                if url.hostname:
                    self.problem = unreachable_hint(url.hostname, url.port or RTSP_PORT) or ""
                    if self.problem:
                        time.sleep(2)
                        continue
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                self.problem = "Camera reachable but the stream did not open: check user, password and channel"
                time.sleep(2)
                continue
            self.connected, self.problem = True, ""
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
    __slots__ = ("box", "c", "area", "conf", "gold", "mask", "origin", "app")

    def __init__(self, box, c, area, conf, gold=1.0, mask=None, origin=(0, 0)):
        self.box, self.c, self.area, self.conf, self.gold = box, c, area, conf, gold
        self.mask, self.origin = mask, origin  # mask is local to its box; origin = its top-left in the frame
        self.app = None  # appearance fingerprint, filled in when needed


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


def count_link_holes(img, mask):
    """Holes inside the jewel's Otsu-on-saturation mask: a chain's links leave many however it is
    piled; bangles, rings and pendants almost none."""
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, otsu = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    otsu[mask == 0] = 0
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for c, h in zip(contours, hierarchy[0]) if h[3] >= 0 and cv2.contourArea(c) >= CHAIN_HOLE_MIN_PX)


def appearance(frame, det, siglip=None, rotations=False):
    """Fingerprint of one detection: chain or solid, jewel pixels, LAB colour histogram and optionally
    SigLIP2 embedding(s). None when no jewel pixels are visible."""
    ox, oy = (int(v) for v in det.origin)
    if det.mask is not None:
        h, w = det.mask.shape
        img, mask = frame[oy:oy + h, ox:ox + w], det.mask.copy()
    else:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        img = frame[max(y1, 0):y2, max(x1, 0):x2]
        mask = np.full(img.shape[:2], 255, np.uint8)
    if img.size == 0 or img.shape[:2] != mask.shape:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bench = (hsv[:, :, 1] < BG_WHITE_S_MAX) & (hsv[:, :, 2] > BG_WHITE_V_MIN)
    jewel = ((mask > 0) & ~bench & (img.max(axis=2) > 0)).astype(np.uint8) * 255
    pixels = int(np.count_nonzero(jewel))
    if pixels < GOLD_MIN_PX:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    colour = cv2.calcHist([lab], [0, 1, 2], jewel, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    colour /= max(float(colour.sum()), 1.0)
    app = dict(pixels=pixels, colour=colour, chain=count_link_holes(img, mask) >= CHAIN_MIN_HOLES, emb=None)
    if siglip is not None:
        app["emb"] = siglip_embed(siglip, img, jewel, rotations)
    return app


def siglip_embed(siglip, img, jewel, rotations):
    ys, xs = np.nonzero(jewel)
    m = int(0.1 * max(np.ptp(xs), np.ptp(ys)))
    y1, y2, x1, x2 = max(ys.min() - m, 0), ys.max() + m + 1, max(xs.min() - m, 0), xs.max() + m + 1
    crop = img[y1:y2, x1:x2].copy()
    crop[jewel[y1:y2, x1:x2] == 0] = 127  # grey background, as in the backend
    return siglip.embed_rotations(crop) if rotations else siglip.embed(crop)[None]


def similarity(a, b):
    """0..1 fingerprint similarity (geometric mean of area ratio, colour and SigLIP2 when both have it);
    0 for a chain vs a solid piece."""
    if a["chain"] != b["chain"]:
        return 0.0
    parts = [min(a["pixels"], b["pixels"]) / max(a["pixels"], b["pixels"]),
             float(np.sum(np.sqrt(a["colour"] * b["colour"])))]
    if a["emb"] is not None and b["emb"] is not None:
        parts.append(float(np.clip(np.max(a["emb"] @ b["emb"].T), 0.0, 1.0)))
    return float(np.prod(parts) ** (1 / len(parts)))


def colour_similarity(a, b):
    return float(np.sum(np.sqrt(a["colour"] * b["colour"])))


class Track:
    def __init__(self, tid, det, t):
        self.id = tid
        self.pos = det.c.astype(float)
        self.vel = np.zeros(2)
        self.box = det.box
        self.area = det.area
        self.last_seen = t
        self.state = "OK"
        self.out_frames = 0
        self.rub_time = 0.0       # seconds of rubbing credited to this jewel
        self.rub_last = 0.0       # last time rubbing was credited to it
        self.rubbed = False
        self.settle_since = None  # rubbed and lying still off the stone since
        self.done = False         # rubbed and placed back
        self.app = det.app        # appearance fingerprint taken at lock (never updated)
        self.det_seen = t         # last time the detector itself found it
        self.det_box = det.box    # its box at that moment
        self.csrt = None          # CSRT tracker while the detector misses it
        self.bridged = False      # held by CSRT in the current frame

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
        self.box = det.box
        self.area = 0.9 * self.area + 0.1 * det.area
        self.last_seen = t
        self.det_seen, self.det_box = t, det.box
        self.csrt, self.bridged = None, False


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


class StoneWatcher(threading.Thread):
    """Finds the touchstone with run-local4's stone model every STONE_REFRESH_S, off the video loop."""

    def __init__(self, grabber, model):
        super().__init__(daemon=True)
        self.grabber, self.model = grabber, model
        self.bbox = None
        self.misses = 0
        self.running = True
        self.paused = False  # a drawn stone makes the model unnecessary

    def run(self):
        while self.running:
            frame, _ = self.grabber.latest()
            if frame is not None and not self.paused:
                try:
                    self.update(frame)
                except Exception:
                    pass
            time.sleep(STONE_REFRESH_S)

    def update(self, frame):
        r = self.model.predict(frame, imgsz=640, conf=STONE_CONF, device=DEVICE, verbose=False)[0]
        h, w = frame.shape[:2]
        best = None
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            area = (b[2] - b[0]) * (b[3] - b[1])
            aspect = (b[2] - b[0]) / max(b[3] - b[1], 1)
            if area < STONE_MIN_AREA_RATIO * h * w or not STONE_ASPECT[0] <= aspect <= STONE_ASPECT[1]:
                continue
            if best is None or area * c > best[0]:
                best = (area * c, tuple(int(v) for v in b))
        if best:
            self.bbox, self.misses = best[1], 0
        else:  # a hand or rub marks hide it: the stone rarely moves, so keep the last box
            self.misses += 1


def rubbing_evidence(model, frame, stone):
    """run-local4 process_rubbing_frame: best gold mask from the rubbing model on a crop around the
    stone, clipped to the stone. Returns dict(c, box, mask, origin) or None."""
    sx1, sy1, sx2, sy2 = stone
    h, w = frame.shape[:2]
    pad = max(20, int(RUB_CROP_PAD_RATIO * max(sx2 - sx1, sy2 - sy1)))
    cx1, cy1, cx2, cy2 = max(0, sx1 - pad), max(0, sy1 - pad), min(w, sx2 + pad), min(h, sy2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    r = model.predict(crop, imgsz=RUB_GOLD_IMGSZ, conf=RUB_GOLD_CONF, device=DEVICE, verbose=False)[0]
    if r.masks is None:
        return None
    best = None
    for poly, conf in zip(r.masks.xy, r.boxes.conf.cpu().numpy()):
        if len(poly) < 3:
            continue
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        raw = int(mask.sum())
        if raw < RUB_MIN_MASK_PX:
            continue
        on_stone = np.zeros_like(mask)
        on_stone[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1] = mask[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1]
        on_stone = largest_component(on_stone)
        n = int(on_stone.sum())
        if n < RUB_MIN_MASK_PX or n / raw < RUB_MIN_INSIDE_RATIO:
            continue
        score = float(conf) * (n / raw) * n
        if best is None or score > best[0]:
            best = (score, on_stone)
    if best is None:
        return None
    mask = best[1]
    m = cv2.moments(mask, binaryImage=True)
    ys, xs = np.nonzero(mask)
    return dict(c=np.array([cx1 + m["m10"] / m["m00"], cy1 + m["m01"] / m["m00"]]),
                box=(cx1 + xs.min(), cy1 + ys.min(), cx1 + xs.max() + 1, cy1 + ys.max() + 1),
                mask=mask, origin=(cx1, cy1))


class RubbingWatcher(threading.Thread):
    """Gold-on-stone check (run-local4) on the newest camera frame in its own thread, so rubbing is
    judged as often as the CPU allows (~20x/s) without slowing the video loop. Runs only while the
    worker sets `active` (a jewel is in the hand or near the stone)."""

    def __init__(self, grabber, model, stone_box):
        super().__init__(daemon=True)
        self.grabber, self.model, self.stone_box = grabber, model, stone_box  # stone_box(): drawn or found
        self.active = False
        self.running = True
        self.lock = threading.Lock()
        self.ev, self.rubbing, self.ts = None, False, 0.0
        self.moves = deque(maxlen=10)
        self.prev, self.misses = None, 0

    def snapshot(self):
        with self.lock:
            return self.ev, self.rubbing, self.ts

    def reset(self):
        self.moves.clear()
        self.prev, self.misses = None, 0
        with self.lock:
            self.ev, self.rubbing = None, False

    def run(self):
        last_ts = 0.0
        while self.running:
            stone = self.stone_box()
            if not self.active or stone is None:
                self.reset()
                time.sleep(0.05)
                continue
            frame, ts = self.grabber.latest()
            if frame is None or ts == last_ts:
                time.sleep(0.005)
                continue
            last_ts = ts
            started = time.time()
            try:
                ev = rubbing_evidence(self.model, frame, stone)
            except Exception:
                ev = None
            self.judge(ev, ts)
            if RUB_MAX_CHECKS_PER_S:
                time.sleep(max(0.0, 1.0 / RUB_MAX_CHECKS_PER_S - (time.time() - started)))

    def judge(self, ev, ts):
        """run-local4 compute_rubbing: the gold centroid must keep moving; a couple of checks where the
        fingers hide the gold don't restart it."""
        if ev is None:
            self.misses += 1
            if self.misses > RUB_MAX_MISSES:
                self.moves.clear()
                self.prev = None
        else:
            self.misses = 0
            if self.prev is not None:
                self.moves.append(float(np.hypot(*(ev["c"] - self.prev))))
            self.prev = ev["c"]
        recent = list(self.moves)[-RUB_MOVE_WINDOW:]
        rubbing = (len(recent) == RUB_MOVE_WINDOW and max(recent) >= 1.2 * RUB_MOVE_PX
                   and sum(recent) / RUB_MOVE_WINDOW >= 0.6 * RUB_MOVE_PX)
        with self.lock:
            self.ev = ev if ev is not None else self.ev if self.misses <= RUB_MAX_MISSES else None
            self.rubbing, self.ts = rubbing, ts


def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class Worker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    event = QtCore.pyqtSignal(str, str)  # message, level: "info" | "alert"
    status = QtCore.pyqtSignal(str)

    def __init__(self, source, roi_pts, expected, occlusion_s, stone_pts=None):
        super().__init__()
        self.source = source
        self.running = True
        self.lock = threading.Lock()
        self.roi = to_poly(roi_pts)
        self.draft = []
        self.mode = "ARMING"  # ARMING -> LOCKED -> (Reset) -> ARMING
        self.expected = expected
        self.occlusion_s = occlusion_s
        self.tracks = []
        self.stable = 0.0         # share of recent frames with exactly N jewels
        self.arm_hist = deque()   # (time, exactly N seen) while arming
        self.arm_count = 0
        self.frame_wh = None
        self.rejected = []
        self.roi_mask, self.roi_mask_key = None, None
        self.stone = None         # StoneWatcher (stone model; only used when no stone is drawn)
        self.stone_drawn = None   # (x1, y1, x2, y2) of the drawn stone
        self.set_stone(stone_pts)
        self.siglip = None        # optional SigLIP2 for the appearance fingerprint
        self.rubber = None        # RubbingWatcher (run-local4 rubbing-gold model)
        self.rub_ev = None        # gold on the stone in the last checked frame
        self.rubbing_now = False
        self.rub_track = None     # jewel the current rubbing is credited to
        self.prev_dets = []       # last frame's detections with fingerprints (reused if unmoved)
        self.prev_frame, self.prev_t = None, 0.0  # for starting CSRT where the jewel was last detected

    # ---- called from the GUI thread ----
    def set_zone(self, kind, pts):  # kind: "roi" or "stone"
        with self.lock:
            if kind == "stone":
                self.set_stone(pts)
            else:
                setattr(self, kind, to_poly(pts))

    def set_stone(self, pts):
        poly = to_poly(pts)
        if poly is None:
            self.stone_drawn = None
        else:
            x, y, w, h = cv2.boundingRect(poly)
            self.stone_drawn = (x, y, x + w, y + h)
        if self.stone is not None:
            self.stone.paused = self.stone_drawn is not None

    def set_draft(self, pts):
        with self.lock:
            self.draft = [tuple(p) for p in pts]

    def arm(self, n):
        with self.lock:
            self.expected = n
            self.tracks, self.stable, self.mode = [], 0.0, "ARMING"
            self.arm_hist.clear()
            self.rub_track = None

    def set_occlusion(self, s):
        with self.lock:
            self.occlusion_s = s

    def stop(self):
        self.running = False

    # ---- worker thread ----
    def run(self):
        self.status.emit("Loading model...")
        try:
            model = YOLO(MODEL_PATH, task="segment")
        except Exception as e:
            self.event.emit(f"Model load failed: {e}", "alert")
            self.status.emit("Model load failed (see Events)")
            return
        try:  # optional: SigLIP2 from the fraud-jewellery backend makes the fingerprint stronger
            sys.path.insert(0, SIGLIP_BACKEND_DIR)
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)
            from app.models.siglip_onnx import SiglipVisionONNX
            self.siglip = SiglipVisionONNX(SIGLIP_MODEL)
        except Exception as e:
            self.event.emit(f"ID keeping without SigLIP2 (colour/area/type only): {e}", "info")
        stone_model = rub_model = None
        try:
            stone_model = YOLO(STONE_MODEL_PATH, task="segment")
            rub_model = YOLO(RUB_GOLD_MODEL_PATH, task="segment")
        except Exception as e:
            stone_model = rub_model = None
            self.event.emit(f"Rubbing check off: {e}", "info")
        grabber = RTSPGrabber(self.source)
        grabber.start()
        if stone_model is not None:
            self.stone = StoneWatcher(grabber, stone_model)
            self.stone.paused = self.stone_drawn is not None
            self.stone.start()
            self.rubber = RubbingWatcher(grabber, rub_model, self.stone_box)
            self.rubber.start()
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
                    self.status.emit(grabber.problem or ("Connecting..." if not grabber.connected
                                                         else "Waiting for frames..."))
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
                if self.mode == "ARMING":
                    self.try_lock(dets, ts, frame)
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
        if self.stone:
            self.stone.running = False
        if self.rubber:
            self.rubber.running = False

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
        r = model.predict(crop, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
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

    def stone_box(self):
        if self.stone_drawn is not None:
            return self.stone_drawn
        return self.stone.bbox if self.stone is not None else None

    def on_stone(self, p, margin=STONE_MARGIN_PX):
        b = self.stone_box()
        return b is not None and b[0] - margin <= p[0] <= b[2] + margin and b[1] - margin <= p[1] <= b[3] + margin

    def try_lock(self, dets, t, frame=None):
        # jewels are laid out on the bench; anything on the touchstone (rub marks) is not counted
        start = [d for d in dets if self.roi_dist(d.c) > 0 and not self.on_stone(d.c, margin=0)]
        self.arm_count = len(start)
        ok = len(start) == self.expected
        self.arm_hist.append((t, ok))
        while t - self.arm_hist[0][0] > LOCK_WINDOW_S:
            self.arm_hist.popleft()
        self.stable = sum(o for _, o in self.arm_hist) / len(self.arm_hist)
        window_full = t - self.arm_hist[0][0] >= 0.9 * LOCK_WINDOW_S
        if ok and window_full and self.stable >= LOCK_MIN_SHARE:
            start.sort(key=lambda d: d.c[0])  # slots #1..#N left to right
            if frame is not None:  # fingerprints now, while all jewels lie apart on the bench
                for d in start:
                    d.app = appearance(frame, d, self.siglip, rotations=True)
            self.tracks = [Track(i + 1, d, t) for i, d in enumerate(start)]
            self.mode = "LOCKED"
            self.event.emit(f"Locked {self.expected} jewel(s)", "info")
            if self.rubber is not None and self.stone_box() is None:
                self.event.emit("Touchstone not found yet: rubbing can't be checked", "info")

    def update_tracks(self, dets, t, frame=None):
        # Closed world: no new IDs after lock. Every detection can only be one of the N jewels.
        seen = set()
        free = list(range(len(dets)))
        self.cur_frame = frame
        self.reuse_fingerprints(dets)
        # stage 1: jewels tracked continuously -> position (+ colour and type, so two jewels passing
        # close to each other can't swap IDs). Fingerprints (~6 ms each) only where there is a choice:
        # a detection two jewels could claim, or a jewel with two candidate detections.
        # (by the last real detection: a jewel only held by CSRT counts as out of sight here)
        active = [i for i, tr in enumerate(self.tracks) if t - tr.det_seen < REID_AFTER_S]
        if frame is not None:
            near = np.array([[self.gate_dist(self.tracks[i], d, t)[1] for d in dets] for i in active], bool)
            if near.size:
                contested = near & ((near.sum(axis=0, keepdims=True) >= 2) | (near.sum(axis=1, keepdims=True) >= 2))
                for j in np.nonzero(contested.any(axis=0))[0]:
                    if dets[j].app is None:
                        dets[j].app = appearance(frame, dets[j])
        on_stone = {}  # track -> stone detection it was matched to this frame
        for i, j in self.assign(active, free, dets, t, self.motion_cost):
            self.tracks[i].correct(dets[j], t)
            seen.add(i)
            free.remove(j)
            if self.on_stone(dets[j].c, margin=0):
                on_stone[i] = j
        # stage 2: jewels that were out of sight (in the hand, rubbing) -> fingerprint, because they
        # can be put down anywhere; SigLIP2 only when several are out of sight at once (ambiguous)
        lost = [i for i in range(len(self.tracks)) if i not in seen]
        if lost and free and frame is not None:
            for j in free:
                if dets[j].app is None:
                    dets[j].app = appearance(frame, dets[j])
            if self.siglip is not None and sum(t - self.tracks[i].det_seen >= REID_AFTER_S for i in lost) >= 2:
                for j in free:
                    if dets[j].app is not None and dets[j].app["emb"] is None:
                        dets[j].app["emb"] = siglip_embed(*self.siglip_input(frame, dets[j]))
            for i, j in self.assign(lost, free, dets, t, self.reid_cost):
                self.tracks[i].correct(dets[j], t)
                seen.add(i)
                free.remove(j)
                if self.on_stone(dets[j].c, margin=0):
                    on_stone[i] = j
        # safety net: a jewel visible on the bench beats a doubtful match on the stone (rub mark)
        if on_stone and frame is not None:
            self.bench_beats_stone(on_stone, free, dets, frame, t)
        # detector flicker: follow recently detected jewels it missed this frame with CSRT
        if frame is not None:
            for i, tr in enumerate(self.tracks):
                if i not in seen and self.csrt_bridge(tr, frame, t):
                    seen.add(i)
            self.prev_frame, self.prev_t = frame, t
        self.prev_dets = [d for d in dets if d.app is not None]
        self.update_rubbing(t)
        for i, tr in enumerate(self.tracks):
            self.evaluate(tr, i in seen, t)
            self.check_done(tr, i in seen, t)

    def bench_beats_stone(self, on_stone, free, dets, frame, t):
        bench = [j for j in free if not self.on_stone(dets[j].c, margin=0)]
        for i in on_stone:
            tr = self.tracks[i]
            if tr.app is None or not bench:
                continue
            best, best_sim = None, BENCH_BEATS_STONE_SIM
            for j in bench:
                d = dets[j]
                if d.app is None:
                    d.app = appearance(frame, d)
                if d.app is None:
                    continue
                if self.siglip is not None and tr.app["emb"] is not None and d.app["emb"] is None:
                    d.app["emb"] = siglip_embed(*self.siglip_input(frame, d))
                sim = similarity(tr.app, d.app)
                if sim >= best_sim:
                    best, best_sim = j, sim
            if best is not None:
                tr.last_seen = t - 1.0  # a jump, not a motion: correct() then resets the velocity
                tr.correct(dets[best], t)
                free.remove(best)
                bench.remove(best)
                self.event.emit(f"Jewel #{tr.id} is on the bench (fingerprint {best_sim:.2f}); "
                                f"the detection on the stone was a rub mark", "info")

    def csrt_bridge(self, tr, frame, t):
        """nivagar sometimes misses a jewel for a few frames. Follow it with CSRT from its last detected box
        for up to CSRT_MAX_GAP_S, as long as it stays put; a jewel being carried off (or CSRT drifting onto
        the hand) moves and ends the bridge. Not on the stone: there the jewel is in the hand."""
        tr.bridged = False
        if (t - tr.det_seen > CSRT_MAX_GAP_S or tr.state in ("LEFT_ROI", "MISSING")
                or self.on_stone(tr.pos, margin=0)):
            tr.csrt = None
            return False
        x1, y1, x2, y2 = (int(v) for v in tr.det_box)
        if tr.csrt is None:
            # start only from the frame in which the detector last saw it
            if self.prev_frame is None or abs(self.prev_t - tr.det_seen) > 1e-6 or x2 - x1 < 4 or y2 - y1 < 4:
                return False
            tr.csrt = cv2.TrackerCSRT_create()
            tr.csrt.init(self.prev_frame, (x1, y1, x2 - x1, y2 - y1))
        ok, (bx, by, bw, bh) = tr.csrt.update(frame)
        center = np.array([bx + bw / 2, by + bh / 2])
        if not ok or np.hypot(*(center - np.array([(x1 + x2) / 2, (y1 + y2) / 2]))) > CSRT_MAX_SHIFT_PX:
            tr.csrt = None
            return False
        tr.box = np.array([bx, by, bx + bw, by + bh]).astype(int)
        tr.last_seen, tr.bridged = t, True
        return True

    def reuse_fingerprints(self, dets):
        """A detection that has not moved since the previous frame keeps that frame's fingerprint
        (SigLIP2 included), so fingerprints are only computed for new or moved detections."""
        for d in dets:
            for p in self.prev_dets:
                ix = min(d.box[2], p.box[2]) - max(d.box[0], p.box[0])
                iy = min(d.box[3], p.box[3]) - max(d.box[1], p.box[1])
                if ix <= 0 or iy <= 0:
                    continue
                inter = ix * iy
                union = ((d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
                         + (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]) - inter)
                if inter / union >= 0.85:
                    d.app = p.app
                    break

    def assign(self, track_idx, det_idx, dets, t, cost_fn):
        """Hungarian assignment of the given tracks to the given detections; returns (track, det) pairs."""
        if not track_idx or not det_idx:
            return []
        BIG = 1e9
        cost = np.array([[cost_fn(self.tracks[i], dets[j], t) for j in det_idx] for i in track_idx])
        rows, cols = linear_sum_assignment(cost)
        return [(track_idx[r], det_idx[c]) for r, c in zip(rows, cols) if cost[r, c] < BIG]

    @staticmethod
    def gate_dist(tr, d, t):
        dist = float(np.hypot(*(d.c - tr.predicted(t))))
        return dist, dist <= GATE_PX + REACQ_SPEED_PX_S * (t - tr.last_seen)

    def stone_det_ok(self, tr, d):
        """Rub marks on the touchstone look like gold to the detector, so a detection on the stone only
        counts as this jewel if its fingerprint (with SigLIP2 when available) matches the jewel's."""
        if not self.on_stone(d.c, margin=0):
            return True
        frame = getattr(self, "cur_frame", None)
        if tr.app is None or frame is None:
            return False
        if d.app is None:
            d.app = appearance(frame, d)
        if d.app is None:
            return False
        if tr.app["chain"] != d.app["chain"] or colour_similarity(tr.app, d.app) < ON_STONE_MIN_COLOUR:
            return False
        if self.siglip is not None and tr.app["emb"] is not None:
            if d.app["emb"] is None and d.mask is not None:
                d.app["emb"] = siglip_embed(*self.siglip_input(frame, d))
            if d.app["emb"] is None or float(np.max(tr.app["emb"] @ d.app["emb"].T)) < ON_STONE_MIN_SIGLIP:
                return False
        return True

    def motion_cost(self, tr, d, t):
        dist, inside = self.gate_dist(tr, d, t)
        if not inside or not self.stone_det_ok(tr, d):
            return 1e9
        cost = dist + SIZE_WEIGHT * abs(np.log(d.area / tr.area))
        if tr.app is not None and d.app is not None:
            cost += APP_WEIGHT_PX * (1 - colour_similarity(tr.app, d.app))
            cost += TYPE_MISMATCH_PX * (tr.app["chain"] != d.app["chain"])
        return cost

    def reid_cost(self, tr, d, t):
        dist, inside = self.gate_dist(tr, d, t)
        if not inside or not self.stone_det_ok(tr, d):
            return 1e9
        if tr.app is None or d.app is None:  # no fingerprint: fall back to position
            return 1 + dist / 1000
        sim = similarity(tr.app, d.app)
        if sim < REID_MIN_SIM:
            return 1e9
        return (1 - sim) + 0.1 * min(dist / 1000, 1)

    def siglip_input(self, frame, d):
        ox, oy = (int(v) for v in d.origin)
        h, w = d.mask.shape
        img = frame[oy:oy + h, ox:ox + w]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bench = (hsv[:, :, 1] < BG_WHITE_S_MAX) & (hsv[:, :, 2] > BG_WHITE_V_MIN)
        jewel = ((d.mask > 0) & ~bench).astype(np.uint8) * 255
        return self.siglip, img, jewel, False

    # ---- rubbing (visual part of run-local4.py) ----
    def update_rubbing(self, t):
        if self.rubber is None or self.stone_box() is None:
            self.rub_ev, self.rubbing_now = None, False
            return
        # a jewel can only be rubbed while it is off the bench (in the hand) or on/near the stone
        active = any(t - tr.det_seen > RUB_START_AFTER_S or self.on_stone(tr.pos) for tr in self.tracks)
        self.rubber.active = active
        ev, rubbing, ts = self.rubber.snapshot()
        fresh = active and abs(t - ts) < RUB_GAP_S
        self.rub_ev = ev if fresh else None
        self.rubbing_now = rubbing and fresh
        if not self.rubbing_now or ev is None:
            return
        tr = self.rub_track
        if tr is None or tr not in self.tracks or t - tr.rub_last > RUB_GRACE_S:
            tr = self.rub_track = self.pick_rubbed_jewel(ev, t)
        if tr is None:
            return
        if t - tr.rub_last < RUB_GAP_S:  # continuous rubbing
            tr.rub_time += t - tr.rub_last
        tr.rub_last = t
        if not tr.rubbed and tr.rub_time >= RUB_CONFIRM_S:
            tr.rubbed = True
            self.event.emit(f"Jewel #{tr.id} RUBBED on the stone", "info")

    def pick_rubbed_jewel(self, ev, t):
        """The jewel lying on the rubbed gold, else the one missing from the bench (in the hand)
        nearest the stone; jewels not rubbed yet are preferred."""
        x1, y1, x2, y2 = ev["box"]
        on_gold = [tr for tr in self.tracks if t - tr.det_seen <= 0.3
                   and tr.box[0] < x2 + 30 and tr.box[2] > x1 - 30 and tr.box[1] < y2 + 30 and tr.box[3] > y1 - 30]
        in_hand = [tr for tr in self.tracks if t - tr.det_seen > 0.3]
        pool = on_gold or in_hand or self.tracks
        return min(pool, key=lambda tr: (tr.rubbed, float(np.hypot(*(tr.pos - ev["c"])))), default=None)

    def check_done(self, tr, seen, t):
        """DONE = rubbed, then seen off the stone lying still for DONE_SETTLE_S."""
        if not tr.rubbed or tr.done:
            return
        still = (seen and tr.state == "OK" and not self.on_stone(tr.pos)
                 and float(np.hypot(*tr.vel)) <= SETTLE_SPEED_PX_S)
        if not still:
            tr.settle_since = None
        elif tr.settle_since is None:
            tr.settle_since = t
        elif t - tr.settle_since >= DONE_SETTLE_S:
            tr.done = True
            self.event.emit(f"Jewel #{tr.id} DONE (rubbed and placed back)", "info")

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
            elif t - tr.rub_last <= RUB_GRACE_S:
                new = "RUBBING"  # in the hand on the stone: no alert
            elif d < EDGE_MARGIN_PX and lost > EDGE_GRACE_S:
                new = "LEFT_ROI"
            elif t - max(tr.last_seen, tr.rub_last) > self.occlusion_s:  # counted from the end of rubbing
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

    def draw(self, vis, dets, t):
        if self.roi is not None:
            cv2.polylines(vis, [self.roi], True, ROI_COLOR, 2)
        stone = self.stone_box()
        if stone is not None:
            cv2.rectangle(vis, stone[:2], stone[2:], STONE_COLOR, 2)
            cv2.putText(vis, "STONE", (stone[0] + 5, stone[1] + 22), FONT, 0.7, STONE_COLOR, 2)
        if self.rub_ev is not None:
            ev = self.rub_ev
            ox, oy = ev["origin"]
            region = vis[oy:oy + ev["mask"].shape[0], ox:ox + ev["mask"].shape[1]]
            region[ev["mask"] > 0] = RUB_GOLD_COLOR
            if self.rubbing_now:
                who = f" #{self.rub_track.id}" if self.rub_track is not None else ""
                cv2.putText(vis, "RUBBING" + who, (int(ev["box"][0]), max(int(ev["box"][1]) - 8, 14)), FONT, 0.8,
                            RUB_GOLD_COLOR, 2)
        for p in self.draft:
            cv2.circle(vis, p, 5, (0, 255, 255), -1)
        if len(self.draft) > 1:
            cv2.polylines(vis, [np.array(self.draft, np.int32)], False, (0, 255, 255), 1)
        if SHOW_REJECTED:
            for d in self.rejected:
                x1, y1, x2, y2 = d.box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 140), 1)
                cv2.putText(vis, f"not gold {d.gold:.0%}", (x1, max(y1 - 4, 12)), FONT, 0.5, (0, 0, 140), 1)
        claimed = {tuple(tr.box) for tr in self.tracks if t - tr.last_seen < 0.05}
        for d in dets:
            x1, y1, x2, y2 = d.box
            if self.mode == "LOCKED" and self.on_stone(d.c, margin=0) and tuple(d.box) not in claimed:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 110, 140), 1)  # ignored: rub mark on the stone
                cv2.putText(vis, "stone mark?", (x1, max(y1 - 4, 12)), FONT, 0.5, (60, 110, 140), 1)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (130, 130, 130), 1)
        alerts = []
        for tr in self.tracks:
            col = COLORS[tr.state]
            label = f"#{tr.id}" if tr.state == "OK" else f"#{tr.id} {tr.state}"
            if tr.state in ("OK", "OCCLUDED", "RUBBING"):
                if tr.done:
                    col, label = DONE_COLOR, f"#{tr.id} DONE"
                elif tr.rubbed:
                    col, label = RUBBED_COLOR, f"#{tr.id} RUBBED"
                    if tr.settle_since is not None:
                        label += f" {t - tr.settle_since:.0f}/{DONE_SETTLE_S:.0f}s"
                elif tr.state == "RUBBING":
                    label += f" {tr.rub_time:.1f}/{RUB_CONFIRM_S:.1f}s"
            if tr.bridged:
                label += " (CSRT)"
            x1, y1, x2, y2 = tr.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.circle(vis, tuple(int(v) for v in tr.pos), 4, col, -1)
            cv2.putText(vis, label, (x1, max(y1 - 6, 14)), FONT, 0.7, col, 2)
            if tr.state in ("LEFT_ROI", "MISSING"):
                alerts.append(f"#{tr.id} {tr.state}")
        bottom = vis.shape[0] - 15
        if self.mode == "ARMING":
            cv2.putText(vis, f"Waiting for {self.expected} jewels in the ROI: {self.arm_count} seen "
                             f"({self.stable:.0%} of the last {LOCK_WINDOW_S:.1f}s)",
                        (10, bottom), FONT, 0.7, (0, 255, 255), 2)
        elif self.mode == "LOCKED":
            done = sum(tr.done for tr in self.tracks)
            cv2.putText(vis, f"Rubbed and back (DONE): {done}/{len(self.tracks)}", (10, bottom), FONT, 0.8,
                        DONE_COLOR, 2)
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
    ZONE_NAMES = {"roi": "ROI", "stone": "stone"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jewel Guard")
        self.worker = None
        self.zones = self.load_zones()  # {"roi": [...], "stone": [...]}
        self.draft = []
        self.drawing = None  # None, "roi" or "stone"

        self.cameras = self.load_cameras()
        self.cam_mode = QtWidgets.QComboBox()
        for key, name in CAMERA_MODES.items():
            self.cam_mode.addItem(name, key)
        self.cam_ip = QtWidgets.QLineEdit()
        self.cam_ip.setPlaceholderText("camera IP")
        self.cam_ip.setFixedWidth(120)
        self.cam_stream = QtWidgets.QComboBox()
        self.cam_stream.addItems(["Main stream", "Sub stream"])
        self.cam_user = QtWidgets.QLineEdit()
        self.cam_user.setPlaceholderText("user")
        self.cam_user.setFixedWidth(80)
        self.cam_pass = QtWidgets.QLineEdit()
        self.cam_pass.setPlaceholderText("password")
        self.cam_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.cam_pass.setFixedWidth(100)
        self.url = QtWidgets.QLineEdit()  # Custom URL mode only
        self.url.setPlaceholderText("rtsp://... or 0 for a webcam")
        self.cam_mode_key = None
        self.btn_conn = QtWidgets.QPushButton("Connect")
        self.video = VideoLabel()
        self.btn_roi = QtWidgets.QPushButton("Draw ROI")
        self.btn_stone = QtWidgets.QPushButton("Draw stone")
        self.btn_clear = QtWidgets.QPushButton("Clear zones")
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(1, 30)
        self.spin_n.setValue(3)
        self.spin_occ = QtWidgets.QSpinBox()
        self.spin_occ.setRange(2, 300)
        self.spin_occ.setValue(DEFAULT_OCCLUSION_S)
        self.spin_occ.setSuffix(" s")
        self.btn_reset = QtWidgets.QPushButton("Reset (re-arm)")
        self.lbl_status = QtWidgets.QLabel("Not connected")
        self.log = QtWidgets.QListWidget()

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Camera:"))
        top.addWidget(self.cam_mode)
        for w in (self.cam_ip, self.cam_stream, self.cam_user, self.cam_pass):
            top.addWidget(w)
        top.addWidget(self.url, 1)
        top.addStretch(1)
        top.addWidget(self.btn_conn)

        form = QtWidgets.QFormLayout()
        form.addRow("Jewels on bench", self.spin_n)
        form.addRow("Missing after", self.spin_occ)

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.btn_roi)
        side.addWidget(self.btn_stone)
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
        self.cam_mode.setCurrentIndex(max(0, list(CAMERA_MODES).index(self.cameras.get("mode", "network"))))
        self.show_camera(self.cam_mode.currentData())
        self.cam_mode.currentIndexChanged.connect(self.switch_camera)
        self.btn_roi.clicked.connect(lambda: self.toggle_draw("roi"))
        self.btn_stone.clicked.connect(lambda: self.toggle_draw("stone"))
        self.btn_clear.clicked.connect(self.clear_zones)
        self.btn_reset.clicked.connect(self.rearm)
        self.spin_n.valueChanged.connect(self.rearm)
        self.spin_occ.valueChanged.connect(lambda v: self.worker and self.worker.set_occlusion(v))
        self.video.clicked.connect(self.on_click)

        self.beep_until = 0.0
        self.beep_timer = QtCore.QTimer(self)
        self.beep_timer.timeout.connect(self.beep_tick)
        self.set_running(False)
        loaded = [self.ZONE_NAMES[k] for k in ("roi", "stone") if self.zones[k]]
        if loaded:
            self.add_log("Loaded saved " + " and ".join(loaded), "info")
        QtCore.QTimer.singleShot(0, self.toggle_connect)  # connect to the camera on startup

    # ---- zones ----
    def load_zones(self):
        try:
            with open(ZONES_FILE) as f:
                data = json.load(f)
            return {k: [tuple(p) for p in data.get(k, [])] for k in ("roi", "stone")}
        except Exception:
            return {"roi": [], "stone": []}

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
        return self.btn_stone if kind == "stone" else self.btn_roi

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
        if QtWidgets.QMessageBox.question(self, "Clear zones", "Delete the saved ROI and stone?") \
                != QtWidgets.QMessageBox.Yes:
            return
        self.zones = {"roi": [], "stone": []}
        if os.path.exists(ZONES_FILE):
            os.remove(ZONES_FILE)
        if self.worker:
            self.worker.set_zone("roi", [])
            self.worker.set_zone("stone", [])
        self.add_log("Zones cleared", "info")

    def on_click(self, x, y, left):
        if not self.drawing or not self.worker:
            return
        if left:
            self.draft.append((x, y))
            self.worker.set_draft(self.draft)
        else:
            self.finish_draw()

    # ---- camera connection ----
    def load_cameras(self):
        cams = json.loads(json.dumps(DEFAULT_CAMERAS))
        try:
            with open(CAMERA_FILE) as f:
                saved = json.load(f)
            cams["mode"] = saved.get("mode", cams["mode"])
            for key in CAMERA_MODES:
                cams[key].update(saved.get(key, {}))
        except Exception:
            pass
        return cams

    def save_cameras(self):
        tmp = CAMERA_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.cameras, f, indent=2)
            os.replace(tmp, CAMERA_FILE)
        except OSError as e:
            self.add_log(f"Could not save camera settings: {e}", "alert")

    def show_camera(self, key):
        """Fill the top bar with the settings of connection `key`."""
        self.cam_mode_key = key
        cam = self.cameras[key]
        custom = key == "custom"
        for w in (self.cam_ip, self.cam_stream, self.cam_user, self.cam_pass):
            w.setVisible(not custom)
        self.url.setVisible(custom)
        if custom:
            self.url.setText(str(cam.get("url", "")))
        else:
            self.cam_ip.setText(cam["ip"])
            self.cam_stream.setCurrentIndex(int(cam.get("subtype", 0)))
            self.cam_user.setText(cam.get("user", ""))
            self.cam_pass.setText(cam.get("password", ""))

    def read_camera(self):
        """Store the top-bar fields into the current connection's settings."""
        key = self.cam_mode_key
        if key == "custom":
            self.cameras[key]["url"] = self.url.text().strip()
        elif key:
            self.cameras[key].update(ip=self.cam_ip.text().strip(), subtype=self.cam_stream.currentIndex(),
                                     user=self.cam_user.text().strip(), password=self.cam_pass.text())
        self.cameras["mode"] = key

    def camera_source(self):
        cam = self.cameras[self.cam_mode_key]
        if self.cam_mode_key == "custom":
            text = str(cam.get("url", "")).strip()
            return int(text) if text.isdigit() else text
        return dahua_url(cam)

    def switch_camera(self):
        self.read_camera()  # keep what was typed for the previous connection
        self.show_camera(self.cam_mode.currentData())
        self.cameras["mode"] = self.cam_mode_key
        self.save_cameras()
        if self.worker:  # switching while connected: reconnect to the new feed
            self.add_log(f"Switching camera to {CAMERA_MODES[self.cam_mode_key]}", "info")
            self.stop_worker()
            self.toggle_connect()

    # ---- control ----
    def set_running(self, running):
        for w in (self.btn_roi, self.btn_stone, self.btn_clear, self.btn_reset):
            w.setEnabled(running)
        for w in (self.url, self.cam_ip, self.cam_stream, self.cam_user, self.cam_pass):
            w.setEnabled(not running)
        self.btn_conn.setText("Disconnect" if running else "Connect")

    def toggle_connect(self):
        if self.worker:
            self.stop_worker()
            self.lbl_status.setText("Not connected")
            return
        self.read_camera()
        self.save_cameras()
        source = self.camera_source()
        shown = source if isinstance(source, int) else urlparse(source).hostname or source
        self.add_log(f"Connecting to {CAMERA_MODES[self.cam_mode_key]}: {shown}", "info")
        self.worker = Worker(source, self.zones["roi"], self.spin_n.value(), self.spin_occ.value(),
                             self.zones["stone"])
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
