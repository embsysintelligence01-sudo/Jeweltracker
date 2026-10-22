"""
Jewel Guard: RTSP jewel tracking with ROI-exit / disappearance alerts and a per-jewel rubbing + acid check.

Install:  pip install ultralytics opencv-python PyQt5 scipy numpy onnxruntime
Run:      python jewel_tracker.py   (nivagar_gold_v3.pt must sit next to this script; so must the
          rubbing models newstone1.onnx and rubbing_gold.onnx and the acid model acid.onnx
          (acidbestmodel_hailo_ready.onnx), copied from R:\Rubbing_Dept_Roha)

Workflow:
  1. The camera connects on startup. The top bar picks how it is reached:
       - Network (router/Wi-Fi): the Dahua camera's address on the LAN
       - Direct Ethernet: the camera cabled straight into this PC's Ethernet port; camera and PC need
         static IPs in the same subnet (this PC's Ethernet is 192.168.1.1/24, so e.g. camera 192.168.1.108)
       - Custom URL: any RTSP URL, or 0 for a webcam
     Switching it while connected reconnects to that feed. Settings are saved to camera.json.
     The still capture is the median of the last frames (less sensor noise in the jewel crops).
  2. Draw ROI -> left-click corners -> right-click (or "Finish ROI") to close. It must cover the
     bench and the touchstone. The stone is found automatically every second (largest dark square in
     the ROI, run-local4's stone model as fallback), so it may be moved; "Draw stone" is only a fallback
     for a stone that can't be found. Both are saved to zones.json.
  3. Lay all jewels inside the ROI, then place the touchstone. Once the stone is present and the bench
     count has been steady for AUTO_COUNT_S, one still frame is captured and analysed as in
     SOP-monitoring-rpi / Weight-Scale-SOP: YOLO counts the jewels; each one is cut out on white from
     the repos' Otsu jewellery mask (jewel_capture.py) and classified once with SigLIP2 and its
     correction gallery (jewel_classifier.py), plus a closed-ring shape check for bangles. The type is
     locked to the jewel and listed in the side panel as "J1. Chain / Necklace"; double-click to rename.
     The live video only tracks J1..JN (no type prediction on the live video).
  4. The only jewel alert (message + 5 s beep) is a jewel leaving the ROI ("J1 MISSING - left the ROI"):
     it is seen outside the ROI, or vanishes at the ROI edge. A jewel that is merely unseen (in the
     hand, covered, rubbed) shows OCCLUDED without an alert.
     Detection runs only inside the ROI, and detections that aren't gold-coloured (HSV check) are dropped.
  5. Rubbing check (visual part of run-local4.py, no audio): the touchstone is found with its stone
     model, gold on the stone with its rubbing-gold model (mask clipped to the stone, >= 50 % inside),
     and rubbing is gold on the stone whose centroid keeps moving. Rubbing is credited to one jewel:
     the tracked jewel on that gold, else the jewel missing from the bench (in the hand) nearest the
     stone. After RUB_CONFIRM_S of rubbing that jewel is RUBBED; while it is being rubbed it shows
     RUBBING.
  6. IDs stay with their jewel: at lock each jewel gets an appearance fingerprint (chain or solid,
     surface area, colour, plus SigLIP2 when the fraud-jewellery backend is present). Jewels tracked
     continuously are matched by position (with colour/type so two jewels passing close can't swap);
     a jewel that was out of sight (in the hand, rubbing) is matched back by its fingerprint, since it
     can be put down anywhere.
     When the detector misses a jewel it saw a moment ago (flicker), OpenCV CSRT holds it in place
     for up to CSRT_MAX_GAP_S, as long as it doesn't move; then it shows OCCLUDED.
  7. Acid test (run-local4 process_acid_frame): once a jewel is RUBBED, the acid model looks for acid on
     the stone. A box counts when the model is sure enough, it is on the stone, part of it is a near-black
     wet patch and it is darker than the same spot before the acid (a clean, still view of the stone kept
     as reference). ACID_CONFIRM_CHECKS checks in a row -> the acid is credited to the rubbed jewel whose
     rub streak it covers: J1 ACID OK. run-local4's MobileNetV3 false-positive filter is used too when
     bestnewacid_mobilenet_v3.pt is placed next to this script.
     A jewel is DONE only when it was RUBBED, acid tested and comes back: seen off the stone, lying still for
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

# torch's OpenMP threads sleep between frames instead of spin-waiting (must be set before torch loads)
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# torch (via ultralytics) must load before PyQt5, otherwise on Windows its DLLs fail with WinError 1114
from ultralytics import YOLO
import onnxruntime
import torch

# CPU budget. Several models run side by side (jewel YOLO in torch; stone, rubbing, acid and SigLIP2 in
# onnxruntime). By default each grabs every core and keeps its threads spinning between runs: measured live
# (14-10 16:25) 8 threads at ~90 % each and the PC at 100 % while the jewels just lay there. So each
# onnxruntime model gets ORT_THREADS threads that sleep when idle, and torch TORCH_THREADS.
ORT_THREADS = 2
TORCH_THREADS = 4
torch.set_num_threads(TORCH_THREADS)
# ultralytics resets torch to its NUM_THREADS (8 here) in select_device() on every model's first predict
import ultralytics.utils.torch_utils as _ul_torch_utils
_ul_torch_utils.NUM_THREADS = TORCH_THREADS
_OrtSession = onnxruntime.InferenceSession


class _BudgetSession(_OrtSession):
    """onnxruntime.InferenceSession with the thread budget (ultralytics creates its ONNX sessions itself)."""

    def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
        so = sess_options or onnxruntime.SessionOptions()
        so.intra_op_num_threads = ORT_THREADS
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        super().__init__(path_or_bytes, so, *args, **kwargs)


onnxruntime.InferenceSession = _BudgetSession
from collections import deque
from jewel_capture import jewellery_mask, jewel_crop, ring_shape
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from PyQt5 import QtCore, QtGui, QtWidgets

# models, settings, captures and the log live next to the .exe when built with PyInstaller
# (build_jewel_tracker.ps1), else next to this script
SCRIPT_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "jewel_guard.log")
# tracking model. rubbing_gold.onnx was tried here too, but on this camera it missed small pieces and
# ran ~4x slower (320 input, CPU), so it is only used for the gold-on-stone rubbing check.
MODEL_PATH = os.path.join(SCRIPT_DIR, "nivagar_gold_v3.pt")
ZONES_FILE = os.path.join(SCRIPT_DIR, "zones.json")
CAPTURE_DIR = os.path.join(SCRIPT_DIR, "captures")  # stone-placed captures: annotated image + jewel crops
CAMERA_FILE = os.path.join(SCRIPT_DIR, "camera.json")
# camera connections (editable in the top bar, saved to camera.json)
CAMERA_MODES = {"network": "Network (router/Wi-Fi)", "direct": "Direct Ethernet", "custom": "Custom URL"}
DEFAULT_CAMERAS = {
    "mode": "direct",
    "network": {"ip": "192.168.29.214", "user": "admin", "password": "admin123", "channel": 1, "subtype": 0},
    # rtsp://admin:admin%4004@192.168.29.215:554/cam/realmonitor?channel=1&subtype=0
    "direct": {"ip": "192.168.29.215", "user": "admin", "password": "admin@04", "channel": 1, "subtype": 0},
    "custom": {"url": "0"},
}
RTSP_PORT = 554
CAPTURE_MEDIAN_FRAMES = 5     # the still capture is the median of this many recent frames

# ---------------- tuning ----------------
# kept low on purpose: spread-out chains score only ~0.2-0.45; the gold colour check below
# removes the extra false positives a low threshold lets through
CONF = 0.15
# The end application runs on CPU only. Measured on a Ryzen 5 6600H CPU, detection on the ROI:
#   imgsz 640: ~275 ms/frame   imgsz 384: ~155 ms   imgsz 320: ~115 ms, still all 3 test jewels
#   (small piece confidence 0.62-0.73 vs 0.87-0.90 at 640). An ONNX export was no faster.
DEVICE = "cpu"             # "cpu" for the end application; "0" would use a CUDA GPU while developing
IMGSZ = 320
RING_MAX_DIAMETER_PX = 150 # a closed loop smaller than this (outer diameter, px) is a finger ring
AUTO_COUNT_S = 3.0
CAPTURE_PRESENCE = 0.3     # a jewel counts when seen in this share of the frames of the last AUTO_COUNT_S
                           # (tiny pieces like earrings score ~0.2-0.3 and blink in and out of single frames)
CAPTURE_STRONG = 0.6       # jewels seen this often define "steady" (a blinking earring must not hold it up)
CAPTURE_STEADY_S = 1.0     # capture when the clearly present jewels have not changed for this long
CAPTURE_IMGSZ = 640        # one sharper YOLO pass on the still capture for small pieces (~0.4 s on CPU)
CLUSTER_PX = 40            # detections this close in different frames are the same jewel
CLUSTER_IOU = 0.5          # ... and so are detections whose boxes overlap this much (IoU)
CAPTURE_EXTRA_CONF = 0.5   # the sharper pass adds a piece never seen live only when this sure (and >= EDGE_MARGIN_PX
                           # inside the ROI: on 14-10 15:48 it added a gold-ish spot on the bench edge)         # auto count: the bench count must be this steady before locking
STONE_DARK_V = 100         # a drawn stone counts as placed when most of its area is this dark (HSV V)
STONE_PRESENT_RATIO = 0.3   # (only for a drawn stone, when the stone can't be found automatically)
GATE_PX = 80               # max jump between consecutive frames
REACQ_SPEED_PX_S = 600     # search radius grows this fast while a jewel is unseen
SIZE_WEIGHT = 60           # penalty for size mismatch when matching
EDGE_MARGIN_PX = 40        # "lost close to the ROI boundary"
EDGE_GRACE_S = 1.0         # lost near boundary this long -> LEFT ROI ...
STILL_THERE_NCC = 0.8      # ... unless its spot still looks as when last detected (correlation of the grey patch):
                           # a tiny earring near the ROI edge that the detector keeps missing is still lying there
                           # (14-10: repeated false "J3 left the ROI" / "back inside" for an earring 35-50 px inside)
OUT_FRAMES = 3             # seen outside ROI this many frames -> LEFT ROI
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
# The stone is found as the largest dark, solid, roughly square area in the ROI (~20 ms; it found the
# stone in every test frame, also after the stone was moved). run-local4's stone model is the fallback.
STONE_CHECK_S = 1.0            # look for the stone this often, so a moved stone is followed
STONE_CV_DARK_V = 90           # HSV brightness below this is "dark"
STONE_CV_MIN_AREA_RATIO = 0.015
STONE_CV_MIN_FILL = 0.6        # dark area / its box: a stone is solid, cables and hands are not
STONE_RING_PX = 30             # ring around a stone candidate that must be bright bench
STONE_BENCH_V = 150
STONE_RING_BRIGHT = 0.6
STONE_CONF = 0.20              # stone model fallback (run-local4 STONE_CONF_THRESH)
STONE_MODEL_EVERY = 3          # the model (~0.2 s on CPU) runs at most every 3rd failed check
STONE_ASPECT = (0.6, 1.6)      # width/height of an accepted stone box
STONE_MIN_AREA_RATIO = 0.02    # run-local4: stone box must cover this share of the frame
STONE_PRESENT_S = 2.5          # "the stone is placed" = found within this many seconds
STONE_FORGET_S = 30.0          # keep the last stone box this long (a hand over it while rubbing)
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
RUB_IDLE_DIFF = 3.0            # mean grey change of the stone between checks below this = nothing moving there
RUB_IDLE_S = 1.0               # stone still this long -> the rubbing model pauses (a jewel the detector keeps
                               # missing used to keep it running at ~4.6 cores with nobody rubbing)
STONE_MARGIN_PX = 40           # "on the stone" includes this margin around the stone box
# Rubbing focus: while a jewel is rubbed the others rest on the bench. The rubbing check (~100 ms, 10x/s) and the
# jewel YOLO (85-180 ms every frame) together saturated the CPU and the video hung, so while rubbing is seen the
# resting jewels are held in place and the jewel YOLO only runs once per RUB_FOCUS_CHECK_S to confirm they are
# still there. Any change on the bench (a resting jewel moved or gone, a jewel put back) -> full tracking.
RUB_FOCUS = True
RUB_FOCUS_CHECK_S = 1.0        # bench check while rubbing
RUB_FOCUS_EXIT_S = 2.0         # no rubbing for this long -> full tracking again
RUB_FOCUS_MATCH_PX = 40        # a resting jewel's detection must be within this of where it rests
RUB_FOCUS_COOLDOWN_S = 3.0     # after the bench changed, full tracking for at least this long
DONE_SETTLE_S = 2.0            # rubbed (and acid tested) jewel lying still off the stone this long -> DONE
SETTLE_SPEED_PX_S = 40

# acid test after rubbing: run-local4 process_acid_frame. bestnewacid.hef is a Hailo build; acid.onnx is the
# same YOLOv8s-seg acid model (class top_aci, 320 px) for the CPU.
ACID_MODEL_PATH = os.path.join(SCRIPT_DIR, "acid.onnx")
ACID_FP_MODEL_PATH = os.path.join(SCRIPT_DIR, "bestnewacid_mobilenet_v3.pt")  # run-local4's false-positive
ACID_FP_TRUE_THRESHOLD = 0.75  # filter (MobileNetV3); used only when the file is there
ACID_FP_CROP_PADDING_RATIO = 0.05
ACID_REQUIRED = True           # DONE needs the acid test (False: rubbed and placed back is enough)
ACID_IMGSZ = 320
ACID_CROP_PAD_RATIO = 0.90     # model input: the stone plus this share around it (as for the rubbing gold)
ACID_CONF = 0.50               # run-local4 uses 0.70 behind its MobileNet filter; here the two checks below
                               # replace that filter. Measured on the 128 recorded acid sessions (stone crops
                               # pasted on the .215 stone): acid on the streak found in 79% of single checks,
                               # 0% false on the same streaks before acid (model alone at 0.70: 26% false)
ACID_MIN_AREA_PX = 140         # run-local4 ACID_MIN_AREA_PX
ACID_DARK_RATIO = 0.55         # "near-black": darker than this share of the stone's own brightness
ACID_MIN_DARK = 0.05           # share of the acid box that must be near-black (wet acid on the stone)
ACID_DARKER_BY = 20            # grey levels darker than the stone before the acid ...
ACID_MIN_DARKER = 0.05         # ... over this share of the box
ACID_CONFIRM_CHECKS = 3        # run-local4 ACID_CONFIRM_FRAMES: acid in this many checks in a row
ACID_CHECKS_PER_S = 4          # ~70 ms per check on the CPU
ACID_REF_STILL = 4.0           # mean grey change between checks below which the stone is still
ACID_REF_MAX_SKIN = 0.02       # no hand over the stone in the reference
ACID_REF_MAX_NEW_BLACK = 0.003 # a new near-black patch (fresh acid) is kept out of the reference ...
ACID_REF_HOLD_S = 20.0         # ... unless it stays this long without being confirmed
ACID_USED_MARGIN_PX = 30       # acid already credited to a jewel is not counted again for the next one

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
SIGLIP_BACKEND_DIR = r"R:\similarity_check\desktop_backend"  # where the SigLIP2 model comes from
SIGLIP_FILE = "siglip2-base-patch32-256_vision_encoder.sim.onnx"   # SigLIP2 vision encoder: fingerprint + type
SIGLIP_MODEL = os.path.join(SCRIPT_DIR, "weights", SIGLIP_FILE)
if not os.path.exists(SIGLIP_MODEL):
    SIGLIP_MODEL = os.path.join(SIGLIP_BACKEND_DIR, "weights", SIGLIP_FILE)

COLORS = {"OK": (0, 200, 0), "OCCLUDED": (0, 200, 255), "RUBBING": (0, 140, 255),
          "LEFT_ROI": (0, 0, 255), "MISSING": (0, 0, 255)}
DONE_COLOR = (255, 255, 0)       # rubbed and placed back
RUBBED_COLOR = (255, 0, 255)     # rubbed, not placed back yet
ACID_COLOR = (0, 255, 255)       # rubbed and acid tested, not placed back yet
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
    others = sorted(ip for ip in local_ipv4s() if not ip.startswith(subnet + ".") and not ip.startswith("127."))
    direct = (f" If it is cabled straight into this PC: Windows sends {subnet}.x through the adapter that already "
              f"has a {subnet}.x address (Wi-Fi), not the cable; this PC's other address(es): {', '.join(others)}. "
              f"Give the camera an address in that subnet, or the Ethernet port a {subnet}.x address."
              if others else "")
    return (f"Camera {host}:{port} not reachable ({reason}). Check the cable, the camera IP and that it is "
            f"powered.{direct}")


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


def otsu_jewel_mask(img, mask):
    """Jewel pixels by Otsu on saturation (gold is saturated; the white bench and the dark stone are
    not), kept to the detection's mask."""
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, otsu = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    otsu[mask == 0] = 0
    return otsu


def count_link_holes(img, mask):
    """Holes inside the jewel's Otsu-on-saturation mask: a chain's links leave many however it is
    piled; bangles, rings and pendants almost none."""
    otsu = otsu_jewel_mask(img, mask)
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
        app["emb"] = siglip_embed(siglip, img, siglip_mask(img, mask, jewel), rotations)
    return app


def siglip_mask(img, mask, fallback):
    """What SigLIP2 sees: the Otsu jewel mask (on real pieces it separated jewels as well as the plain
    crop and slightly better than the seg mask), or the seg mask if Otsu finds nothing."""
    otsu = otsu_jewel_mask(img, mask)
    return otsu if np.count_nonzero(otsu) >= GOLD_MIN_PX else fallback


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
        self.done = False         # rubbed (and acid tested) and placed back
        self.rub_boxes = deque(maxlen=30)  # where its rub streak was drawn on the stone
        self.acid_done = False    # acid test seen on its rub streak
        self.acid_box = None
        self.app = det.app        # appearance fingerprint taken at lock (never updated)
        self.det_seen = t         # last time the detector itself found it
        self.det_box = det.box    # its box at that moment
        self.csrt = None          # CSRT tracker while the detector misses it
        self.bridged = False      # held by CSRT in the current frame
        self.kind, self.kind_conf = "", 0.0  # jewel type from the classifier, at lock
        self.kind_note = ""

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
        self.det_patch = None     # grey patch of det_box from that frame (set by the worker)


def box_iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = float(ix) * float(iy)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def spot_patch(frame, box, pad=6):
    """Small grey patch of a box (with a little bench around it), for "is it still lying there?"."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def find_stone_cv(frame, roi):
    """The touchstone on the white bench: the largest dark, solid, roughly square area inside the ROI.
    Returns (x1, y1, x2, y2) or None."""
    h, w = frame.shape[:2]
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = (cv2.GaussianBlur(v, (9, 9), 0) < STONE_CV_DARK_V).astype(np.uint8) * 255
    if roi is not None:
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [roi.astype(np.int32)], 255)
        dark &= inside
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))  # rub marks inside the stone
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))   # thin cables, beads
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if (area < STONE_CV_MIN_AREA_RATIO * h * w or not STONE_ASPECT[0] <= bw / max(bh, 1) <= STONE_ASPECT[1]
                or area / (bw * bh) < STONE_CV_MIN_FILL):
            continue
        if not bench_around(v, (x, y, x + bw, y + bh)):
            continue  # a stone lies on the bright bench; dark floor or cables past the bench edge do not
        if best is None or area > best[0]:
            best = (area, (int(x), int(y), int(x + bw), int(y + bh)))
    return best[1] if best else None


def bench_around(v, box, band=STONE_RING_PX):
    """Is the box surrounded by bright bench? (share of bright pixels in a ring around it, inside the frame)"""
    h, w = v.shape
    x1, y1, x2, y2 = box
    outer = v[max(0, y1 - band):min(h, y2 + band), max(0, x1 - band):min(w, x2 + band)]
    total = outer.size - (y2 - y1) * (x2 - x1)
    ring_bright = int(np.count_nonzero(outer > STONE_BENCH_V)) - int(np.count_nonzero(v[y1:y2, x1:x2] > STONE_BENCH_V))
    expected = (y2 - y1 + 2 * band) * (x2 - x1 + 2 * band) - (y2 - y1) * (x2 - x1)  # ring cut by the frame edge counts as dark
    return total > 0 and ring_bright / expected >= STONE_RING_BRIGHT


class StoneWatcher(threading.Thread):
    """Finds the touchstone every STONE_CHECK_S, off the video loop: dark-square search first, run-local4's
    stone model as fallback. Keeps the last box while a hand hides the stone."""

    def __init__(self, grabber, model, roi_fn):
        super().__init__(daemon=True)
        self.grabber, self.model, self.roi_fn = grabber, model, roi_fn
        self.bbox = None
        self.found_t = 0.0
        self.fails = 0
        self.running = True

    @property
    def present(self):
        return time.time() - self.found_t < STONE_PRESENT_S

    def run(self):
        while self.running:
            frame, _ = self.grabber.latest()
            if frame is not None:
                try:
                    self.update(frame)
                except Exception:
                    pass
            time.sleep(STONE_CHECK_S)

    def update(self, frame):
        box = find_stone_cv(frame, self.roi_fn())
        if box is None and self.model is not None and self.fails % STONE_MODEL_EVERY == 0:
            box = self.model_box(frame)
        if box is not None:
            self.bbox, self.found_t, self.fails = box, time.time(), 0
        else:
            self.fails += 1
            if time.time() - self.found_t > STONE_FORGET_S:
                self.bbox = None

    def model_box(self, frame):
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
        return best[1] if best else None


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
            if self.stone_still(frame, stone, ts):  # nothing moves on the stone: nobody is rubbing
                self.judge(None, ts)
                time.sleep(0.1)
                continue
            try:
                ev = rubbing_evidence(self.model, frame, stone)
            except Exception:
                ev = None
            self.judge(ev, ts)
            if RUB_MAX_CHECKS_PER_S:
                time.sleep(max(0.0, 1.0 / RUB_MAX_CHECKS_PER_S - (time.time() - started)))

    def stone_still(self, frame, stone, ts):
        """Cheap motion check on the stone (~1 ms): rubbing always moves a hand and a jewel there."""
        x1, y1, x2, y2 = stone
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return False
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        prev, self.stone_prev = getattr(self, "stone_prev", None), small
        if prev is None or float(np.mean(np.abs(small - prev))) >= RUB_IDLE_DIFF:
            self.moving_ts = ts
        return ts - getattr(self, "moving_ts", 0.0) > RUB_IDLE_S

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


class AcidFPFilter:
    """run-local4 AcidMobileNetV3Filter: accepts only true-acid crops (MobileNetV3-Small, 2 classes)."""

    def __init__(self, path):
        import torch
        from torch import nn
        from torchvision import models, transforms
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model, self.torch = model.eval(), torch
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, frame, box):
        # run-local4 make_acid_classifier_crop: the box padded by 5 %, letterboxed square in grey 114
        x1, y1, x2, y2 = box
        cw = int(np.ceil((x2 - x1) * (1 + 2 * ACID_FP_CROP_PADDING_RATIO)))
        ch = int(np.ceil((y2 - y1) * (1 + 2 * ACID_FP_CROP_PADDING_RATIO)))
        side = max(cw, ch, 1)
        cx1, cy1 = int((x1 + x2 - side) / 2), int((y1 + y2 - side) / 2)
        crop = np.full((side, side, 3), 114, np.uint8)
        fx1, fy1 = max(0, cx1), max(0, cy1)
        fx2, fy2 = min(frame.shape[1], cx1 + side), min(frame.shape[0], cy1 + side)
        if fx2 > fx1 and fy2 > fy1:
            crop[fy1 - cy1:fy2 - cy1, fx1 - cx1:fx2 - cx1] = frame[fy1:fy2, fx1:fx2]
        x = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with self.torch.inference_mode():
            prob = float(self.torch.softmax(self.model(x), dim=1)[0, 1])
        return prob >= ACID_FP_TRUE_THRESHOLD, prob


class AcidWatcher(threading.Thread):
    """Acid test on the stone (run-local4 process_acid_frame), in its own thread like RubbingWatcher.
    It always keeps a reference of the stone from clean, still frames (no hand, no fresh acid); while a
    rubbed jewel waits for its acid (`active`), the acid model runs and a box counts only when part of
    it is near-black and darker than that reference. ACID_CONFIRM_CHECKS in a row -> confirmed."""

    def __init__(self, grabber, model, stone_box, fp_filter=None):
        super().__init__(daemon=True)
        self.grabber, self.model, self.stone_box, self.fp = grabber, model, stone_box, fp_filter
        self.active = False
        self.running = True
        self.lock = threading.Lock()
        self.ref = self.ref_key = self.prev = None
        self.absorb = False           # take the next clean view as reference (after a confirmed acid)
        self.blocked_since = None     # a new dark patch has kept the reference from updating since
        self.streak = 0
        self.cand = None              # (box, conf, streak) of the current acid candidate, for drawing
        self.used = []                # acid boxes already credited to a jewel
        self.confirmed = []           # (ts, box) for the worker to collect

    def snapshot(self):
        with self.lock:
            out, self.confirmed = self.confirmed, []
            return out, self.cand, self.ref is not None

    def run(self):
        last_ts = 0.0
        while self.running:
            stone = self.stone_box()
            frame, ts = self.grabber.latest()
            if stone is None or frame is None or ts == last_ts:
                time.sleep(0.02)
                continue
            last_ts = ts
            started = time.time()
            try:
                self.step(frame, stone, ts)
            except Exception:
                pass
            time.sleep(max(0.0, 1.0 / ACID_CHECKS_PER_S - (time.time() - started)))

    def step(self, frame, stone, ts):
        sx1, sy1, sx2, sy2 = stone
        h, w = frame.shape[:2]
        pad = max(20, int(ACID_CROP_PAD_RATIO * max(sx2 - sx1, sy2 - sy1)))
        ox, oy = max(0, sx1 - pad), max(0, sy1 - pad)
        crop = frame[oy:min(h, sy2 + pad), ox:min(w, sx2 + pad)]
        if crop.size == 0:
            return
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        inner = (slice(sy1 - oy, sy2 - oy), slice(sx1 - ox, sx2 - ox))  # the stone inside the crop
        key = (ox, oy, gray.shape, stone)
        if key != self.ref_key:  # stone moved: start over
            self.ref_key, self.ref, self.prev, self.used = key, None, None, []
        level = float(np.median(gray[inner])) + 1.0
        dark = gray < ACID_DARK_RATIO * level
        self.update_reference(crop, gray, dark, inner, ts)
        if not self.active:
            self.streak = 0
            with self.lock:
                self.cand = None
            return
        best = None
        r = self.model.predict(crop, imgsz=ACID_IMGSZ, conf=ACID_CONF, device=DEVICE, verbose=False)[0]
        for b, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (int(v) for v in b)
            cx, cy = (x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy
            if (x2 - x1) * (y2 - y1) < ACID_MIN_AREA_PX or not (sx1 <= cx <= sx2 and sy1 <= cy <= sy2):
                continue
            if any(u[0] - ACID_USED_MARGIN_PX <= cx <= u[2] + ACID_USED_MARGIN_PX
                   and u[1] - ACID_USED_MARGIN_PX <= cy <= u[3] + ACID_USED_MARGIN_PX for u in self.used):
                continue  # acid already credited to an earlier jewel
            wet = float(np.mean(dark[y1:y2, x1:x2]))
            darker = float(np.mean(self.ref[y1:y2, x1:x2] - gray[y1:y2, x1:x2] > ACID_DARKER_BY)) \
                if self.ref is not None else 0.0
            if wet < ACID_MIN_DARK or darker < ACID_MIN_DARKER:
                continue
            box = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
            if self.fp is not None and not self.fp.predict(frame, box)[0]:
                continue
            if best is None or conf > best[1]:
                best = (box, float(conf))
        self.streak = self.streak + 1 if best else 0
        with self.lock:
            self.cand = (best[0], best[1], self.streak) if best else None
            if best and self.streak >= ACID_CONFIRM_CHECKS:
                self.confirmed.append((ts, best[0]))
                self.used.append(best[0])
                self.streak, self.absorb, self.cand = 0, True, None

    def update_reference(self, crop, gray, dark, inner, ts):
        small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4), interpolation=cv2.INTER_AREA)
        still = self.prev is not None and self.prev.shape == small.shape \
            and float(np.mean(np.abs(small - self.prev))) < ACID_REF_STILL
        self.prev = small
        if not still:
            return
        hsv = cv2.cvtColor(crop[inner], cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        skin = ((hue <= 25) | (hue >= 170)) & (sat > 40) & (val > 60)
        if float(np.mean(skin)) > ACID_REF_MAX_SKIN:
            return  # hand (or a jewel) over the stone
        if self.ref is not None and not self.absorb:
            new_black = float(np.mean(dark[inner] & (self.ref[inner] - gray[inner] > ACID_DARKER_BY)))
            if new_black > ACID_REF_MAX_NEW_BLACK:  # fresh acid: keep it out of the reference for a while
                self.blocked_since = self.blocked_since or ts
                if ts - self.blocked_since < ACID_REF_HOLD_S:
                    return
        self.ref, self.absorb, self.blocked_since = gray, False, None


def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class Worker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    jewel_list = QtCore.pyqtSignal(list)  # ["J1. Chain / Necklace", ...] from the capture
    jewel_table = QtCore.pyqtSignal(list)  # [(id, type, rubbed, acid tested, done), ...] status table
    event = QtCore.pyqtSignal(str, str)  # message, level: "info" | "alert"
    status = QtCore.pyqtSignal(str)

    def __init__(self, source, roi_pts, stone_pts=None, camera=None):
        super().__init__()
        self.source = source
        self.running = True
        self.lock = threading.Lock()
        self.roi = to_poly(roi_pts)
        self.draft = []
        self.mode = "ARMING"  # ARMING -> LOCKED -> (Reset) -> ARMING
        self.camera = camera          # Dahua camera settings (ip, user, password) or None
        self.recent = deque(maxlen=CAPTURE_MEDIAN_FRAMES)  # recent frames for the still capture
        self.arm_wait_stone = False
        self.classifier = None        # jewel type classifier (jewel_classifier.py)
        self.tracks = []
        self.stable = 0.0         # share of recent frames with exactly N jewels
        self.arm_frames = deque() # (time, bench detections) over the last AUTO_COUNT_S (auto count)
        self.arm_start = None     # when counting started (stone placed)
        self.persist_key = None   # clearly present jewel count, and since when it has not changed
        self.persist_since = 0.0
        self.model = None         # the YOLO jewel model (also used for the sharper capture pass)
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
        self.focus = False        # rubbing focus: resting jewels held, jewel YOLO only every RUB_FOCUS_CHECK_S
        self.focus_resting = set()  # ids of the jewels resting on the bench during the focus
        self.focus_last_rub = self.focus_check_t = self.focus_block_until = 0.0
        self.acider = None        # AcidWatcher (run-local4 acid model)
        self.acid_cand = None     # current acid candidate on the stone (box, conf, checks in a row)
        self.acid_ref = False     # the acid check has a clean view of the stone to compare with
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

    def set_draft(self, pts):
        with self.lock:
            self.draft = [tuple(p) for p in pts]

    def arm(self):
        self.jewel_list.emit([])
        self.jewel_table.emit([])
        with self.lock:
            self.tracks, self.stable, self.mode = [], 0.0, "ARMING"
            self.arm_frames.clear()
            self.arm_start = self.persist_key = None
            self.rub_track = None

    def stop(self):
        self.running = False

    # ---- worker thread ----
    def run(self):
        self.status.emit("Loading model...")
        try:
            model = self.model = YOLO(MODEL_PATH, task="segment")
        except Exception as e:
            self.event.emit(f"Model load failed: {e}", "alert")
            self.status.emit("Model load failed (see Events)")
            return
        try:  # optional: SigLIP2 (from the fraud-jewellery backend) makes the fingerprint stronger
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)
            from siglip_vision import SiglipVisionONNX
            self.siglip = SiglipVisionONNX(SIGLIP_MODEL)
        except Exception as e:
            self.event.emit(f"ID keeping without SigLIP2 (colour/area/type only): {e}", "info")
        try:  # jewel type, sharing the SigLIP2 session
            from jewel_classifier import JewelClassifier
            self.classifier = JewelClassifier(SIGLIP_MODEL, session=self.siglip.session if self.siglip else None)
        except Exception as e:
            self.event.emit(f"Jewel type classification off: {e}", "info")
        stone_model = rub_model = acid_model = acid_fp = None
        try:
            stone_model = YOLO(STONE_MODEL_PATH, task="segment")
            rub_model = YOLO(RUB_GOLD_MODEL_PATH, task="segment")
        except Exception as e:
            stone_model = rub_model = None
            self.event.emit(f"Rubbing check off: {e}", "info")
        try:
            acid_model = YOLO(ACID_MODEL_PATH, task="segment")
        except Exception as e:
            self.event.emit(f"Acid check off: {e}", "info")
        if acid_model is not None and os.path.exists(ACID_FP_MODEL_PATH):
            try:
                acid_fp = AcidFPFilter(ACID_FP_MODEL_PATH)
                self.event.emit("Acid check: MobileNetV3 false-positive filter loaded", "info")
            except Exception as e:
                self.event.emit(f"Acid false-positive filter not loaded: {e}", "info")
        grabber = RTSPGrabber(self.source)
        grabber.start()
        self.stone = StoneWatcher(grabber, stone_model, lambda: self.roi)
        self.stone.start()
        if rub_model is not None:
            self.rubber = RubbingWatcher(grabber, rub_model, self.stone_box)
            self.rubber.start()
        if acid_model is not None:
            self.acider = AcidWatcher(grabber, acid_model, self.stone_box, acid_fp)
            self.acider.start()
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
            self.recent.append(frame)

            with self.lock:
                self.ensure_roi()
                roi = self.roi.copy()
                focus = self.mode == "LOCKED" and self.rub_focus(ts)
                skip = focus and ts - self.focus_check_t < RUB_FOCUS_CHECK_S  # rubbing: no jewel YOLO now
            if skip:
                dets, rejected = [], self.rejected
            else:
                dets, rejected = self.detect(model, frame, roi)
                if focus:
                    self.focus_check_t = ts
            with self.lock:
                self.rejected = rejected
                if self.mode == "ARMING":
                    self.try_lock(dets, ts, frame)
                elif self.mode == "LOCKED" and focus:
                    self.update_focus(None if skip else dets, ts, frame)
                elif self.mode == "LOCKED":
                    self.update_tracks(dets, ts, frame)
                vis = self.draw(frame.copy(), dets, ts)  # never draw on the frame itself: it is also the
                                                         # capture / CSRT image and the watcher threads' frame
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
        if self.acider:
            self.acider.running = False

    def detect(self, model, frame, roi, imgsz=None):
        """Detect only inside the ROI, then drop detections that aren't gold-coloured.
        Returns (kept, rejected) in full-frame coordinates."""
        # run the model on the ROI's bounding box with everything outside the ROI blacked out
        ox, oy, w, h = cv2.boundingRect(roi)
        fh, fw = frame.shape[:2]
        x2, y2 = min(ox + w, fw), min(oy + h, fh)  # a ROI drawn on a larger image is clipped to this frame
        ox, oy = max(ox, 0), max(oy, 0)
        w, h = x2 - ox, y2 - oy
        key = roi.tobytes() + bytes(str((fw, fh)), "ascii")
        if self.roi_mask_key != key:  # rebuild the ROI mask only when the ROI changes
            self.roi_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(self.roi_mask, [(roi - (ox, oy)).astype(np.int32)], 255)
            self.roi_mask_key = key
            bx, by, bw, bh = cv2.boundingRect(roi)
            if bx < 0 or by < 0 or bx + bw > fw or by + bh > fh:
                self.event.emit(f"The ROI reaches outside this camera's {fw}x{fh} image (drawn for another "
                                f"camera?): it is clipped - redraw it", "info")
        inside = self.roi_mask
        crop = cv2.bitwise_and(frame[oy:oy + h, ox:ox + w], frame[oy:oy + h, ox:ox + w], mask=inside)
        r = model.predict(crop, conf=CONF, imgsz=imgsz or IMGSZ, device=DEVICE, verbose=False)[0]
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
        self.ensure_roi()  # zones cleared from the GUI while a frame was being processed -> whole frame
        if self.roi is None:
            return 1e9
        return cv2.pointPolygonTest(self.roi, (float(p[0]), float(p[1])), True)

    def stone_box(self):
        """Where the stone is: found automatically (it may be moved), else the drawn stone."""
        if self.stone is not None and self.stone.bbox is not None:
            return self.stone.bbox
        return self.stone_drawn

    def on_stone(self, p, margin=STONE_MARGIN_PX):
        b = self.stone_box()
        return b is not None and b[0] - margin <= p[0] <= b[2] + margin and b[1] - margin <= p[1] <= b[3] + margin

    def stone_present(self, frame):
        """Is the touchstone on the bench (found within STONE_PRESENT_S)? A drawn stone is only used when
        it can't be found automatically: then it counts when its area is partly dark."""
        if self.stone is not None and self.stone.present:
            return True
        if self.stone_drawn is None or frame is None:
            return False
        x1, y1, x2, y2 = self.stone_drawn
        v = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[:, :, 2]
        return v.size > 0 and float(np.mean(v < STONE_DARK_V)) >= STONE_PRESENT_RATIO

    def try_lock(self, dets, t, frame=None):
        """No count is entered: once the touchstone is placed, the jewels detected on the bench (steady for
        CAPTURE_STEADY_S) are the total, captured and locked as J1..JN."""
        # jewels are laid out on the bench; anything on the touchstone (rub marks) is not counted
        start = [d for d in dets if self.roi_dist(d.c) > 0 and not self.on_stone(d.c, margin=0)]
        self.arm_wait_stone = not self.stone_present(frame)
        if self.arm_wait_stone:  # count only once the stone is placed
            self.arm_count = len(start)
            self.arm_frames.clear()
            self.arm_start = self.persist_key = None
            self.stable = 0.0
            return
        # the jewels present over the last AUTO_COUNT_S, not one frame: a tiny piece may blink out of the
        # frame that happens to be captured
        if self.arm_start is None:
            self.arm_start = t
        self.arm_frames.append((t, start))
        while t - self.arm_frames[0][0] > AUTO_COUNT_S:
            self.arm_frames.popleft()
        seen, strong = self.persistent_jewels()
        self.arm_count = len(seen)
        if strong != self.persist_key:  # a jewel placed / taken / a hand in the way: wait again
            self.persist_key, self.persist_since = strong, t
        filled = min(1.0, (t - self.arm_start) / AUTO_COUNT_S)
        steady = min(1.0, (t - self.persist_since) / CAPTURE_STEADY_S)
        self.stable = min(filled, steady)
        if frame is None or not seen or self.stable < 1.0:
            return
        jewels = self.capture(self.still_frame(frame), seen)
        if not jewels:
            self.arm_frames.clear()
            self.arm_start = self.persist_key = None
            return
        self.lock_jewels(jewels, t, frame)

    def lock_jewels(self, jewels, t, frame=None):
        """jewels: [(det, item, label, confidence)] from the capture -> tracks J1..JN with their locked type."""
        if frame is not None:  # fingerprints now, while all jewels lie apart on the bench
            for d, *_ in jewels:
                d.app = appearance(frame, d, self.siglip, rotations=True)
        self.tracks = [Track(i + 1, d, t) for i, (d, *_) in enumerate(jewels)]
        for tr, (_, item, label, conf) in zip(self.tracks, jewels):
            # the type found at this first capture is locked to the jewel for the session (only the
            # operator can rename it); it is never re-predicted on the live video
            tr.kind, tr.kind_conf = label, conf
            tr.kind_embedding = item.get("embedding") if item else None
            tr.kind_crop = item.get("crop_bgr") if item else None
            src = (item or {}).get("source", "")
            tr.kind_note = (f"{conf:.0%}" if src == "siglip" else src.replace("_", " ")) if label else ""
        self.mode = "LOCKED"
        self.event.emit(f"Locked {len(jewels)} jewel(s)", "info")
        self.emit_jewel_list()
        if self.rubber is not None and self.stone_box() is None:
            self.event.emit("Touchstone not found yet: rubbing can't be checked", "info")

    def emit_jewel_list(self):
        tracks = sorted(self.tracks, key=lambda x: x.id)
        self.jewel_list.emit([f"J{tr.id}. {tr.kind or '?'}" + (f"  ({tr.kind_note})" if tr.kind_note else "")
                              for tr in tracks])
        self.emit_jewel_table()

    def emit_jewel_table(self):
        """Status table: updated when a jewel is locked, renamed, rubbed, acid tested or DONE."""
        self.jewel_table.emit([(tr.id, tr.kind or "?", tr.rubbed, tr.acid_done, tr.done)
                               for tr in sorted(self.tracks, key=lambda x: x.id)])

    def correct_type(self, jewel_id, label):
        """Operator confirms or corrects a jewel's type; the capture embedding goes into the correction
        gallery (classification/jewelry_correction_gallery.npz) like the repo's learn_correction."""
        with self.lock:
            tr = next((x for x in self.tracks if x.id == jewel_id), None)
            if tr is None:
                return
            confirmed = label == tr.kind
            emb, crop = getattr(tr, "kind_embedding", None), getattr(tr, "kind_crop", None)
            tr.kind, tr.kind_conf, tr.kind_note = label, 1.0, "set by operator"
            self.emit_jewel_list()
        if self.classifier is None or emb is None:
            return
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop is not None else None

        def learn():  # ~0.5 s of SigLIP for the turned copies: off the GUI thread
            try:
                self.classifier.learn(emb, label, rgb)
                self.event.emit(f"J{jewel_id} {'confirmed as' if confirmed else 'corrected to'} {label} "
                                f"(saved: recognised as {label} in the next runs)", "info")
            except Exception as e:
                self.event.emit(f"J{jewel_id}: correction not saved ({e})", "alert")
        threading.Thread(target=learn, daemon=True).start()

    def persistent_jewels(self):
        """Bench detections of the arming window grouped by position; a group is a jewel when it was seen
        in at least CAPTURE_PRESENCE of the frames. Returns each jewel's most confident detection, and how
        many jewels are clearly present (CAPTURE_STRONG) - only that count has to be steady before capture."""
        frames = list(self.arm_frames)
        groups = []
        for k, (_, ds) in enumerate(frames):
            for d in ds:
                # same jewel: centre close, or boxes mostly overlapping (a thin bangle's centre jumps around
                # between frames - 14-10 it was counted twice)
                for g in groups:
                    if np.hypot(*(d.c - g["c"])) <= CLUSTER_PX or box_iou(d.box, g["box"]) >= CLUSTER_IOU:
                        g["frames"].add(k)
                        if d.conf > g["best"].conf:
                            g["best"] = d
                        break
                else:
                    groups.append({"c": d.c.copy(), "box": d.box, "best": d, "frames": {k}})
        merged = []  # groups that still ended up on the same jewel
        for g in sorted(groups, key=lambda g: -len(g["frames"])):
            same = next((m for m in merged if box_iou(g["best"].box, m["best"].box) >= CLUSTER_IOU), None)
            if same is None:
                merged.append(g)
            else:
                same["frames"] |= g["frames"]
        groups = merged
        share = [len(g["frames"]) / max(1, len(frames)) for g in groups]
        return ([g["best"] for g, r in zip(groups, share) if r >= CAPTURE_PRESENCE],
                sum(r >= CAPTURE_STRONG for r in share))

    def still_frame(self, frame):
        """The capture image: median of the last frames (the scene is steady by now), which removes
        sensor noise before the Otsu cut-out and SigLIP."""
        if len(self.recent) < 3:
            return frame
        return np.median(np.stack(list(self.recent)), axis=0).astype(np.uint8)

    def capture(self, frame, yolo_dets):
        """Still capture once the stone is placed: YOLO counts the jewels; each one is cut out on white from
        the repos' Otsu jewellery mask (inside its YOLO mask) and classified once - SigLIP2 with the
        correction gallery, plus a closed-ring shape check for bangles / finger rings.
        Returns [(det, item, label, confidence)] in reading order (top to bottom, then left to right)."""
        self.status.emit("Capturing: counting and classifying the jewels...")
        if self.model is not None:  # one sharper pass on the still image for small pieces the live pass misses
            sharp, _ = self.detect(self.model, frame, self.roi, imgsz=CAPTURE_IMGSZ)

            def known(d):  # already counted, or a piece of a counted jewel
                return any(np.hypot(*(d.c - y.c)) <= CLUSTER_PX
                           or (y.box[0] - 20 <= d.c[0] <= y.box[2] + 20 and y.box[1] - 20 <= d.c[1] <= y.box[3] + 20)
                           for y in yolo_dets)
            def seen_live(d):  # detected at least once while counting
                return any(np.hypot(*(d.c - y.c)) <= CLUSTER_PX for _, ds in self.arm_frames for y in ds)
            extra = [d for d in sharp if self.roi_dist(d.c) >= EDGE_MARGIN_PX and not self.on_stone(d.c, margin=0)
                     and not known(d) and (seen_live(d) or d.conf >= CAPTURE_EXTRA_CONF)]
            if extra:
                self.event.emit(f"Capture: the sharper pass found {len(extra)} more small jewel(s)", "info")
                yolo_dets = list(yolo_dets) + extra
        otsu = jewellery_mask(frame, self.roi, self.stone_box())
        dets = sorted(yolo_dets, key=lambda d: (int(d.box[1]), int(d.box[0])))
        jewels = []
        for d in dets:
            if d.mask is not None:
                crop, cmask, bbox = jewel_crop(frame, otsu, d.mask, d.origin)
            else:
                x1, y1, x2, y2 = (int(v) for v in d.box)
                crop, cmask, bbox = frame[y1:y2, x1:x2].copy(), np.ones((y2 - y1, x2 - x1), np.uint8), (x1, y1, x2, y2)
            item = {"bbox": bbox, "crop_bgr": crop}
            label, conf, source = self.classify_type(crop, cmask, item)
            item["source"] = source
            jewels.append((d, item, label, conf))
        self.event.emit(f"Capture: YOLO counted {len(jewels)} jewel(s)", "info")
        self.save_capture(frame, jewels)
        return jewels

    def classify_type(self, crop_bgr, crop_mask, item):
        """SigLIP2 (+ gallery) type; a closed elliptical loop is a Bangle / Finger Ring when SigLIP itself has
        that in its top 3 (on real pieces SigLIP called 2 of 3 bangles a chain, with Bangle 2nd/3rd)."""
        if self.classifier is None:
            return "", 0.0, ""
        try:
            label, conf, source = self.classifier.classify(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            self.event.emit(f"Jewel type unknown: {e}", "info")
            return "", 0.0, ""
        item["embedding"] = self.classifier.last_embedding
        if source == "siglip":  # an operator-taught gallery match always wins
            is_ring, diameter = ring_shape(crop_mask)
            loop_labels = {"Bangle", "Finger Ring"}
            if is_ring and loop_labels & set(self.classifier.last_top) and label not in loop_labels:
                label = "Finger Ring" if diameter < RING_MAX_DIAMETER_PX else "Bangle"
                source = "ring shape"
        return label, conf, source

    def save_capture(self, frame, jewels):
        """Evidence like the repos: the annotated capture and each jewel's Otsu crop, in captures/."""
        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            annotated = frame.copy()
            for i, (_, item, label, conf) in enumerate(jewels, 1):
                if item.get("source") not in (None, "", "siglip"):
                    label = f"{label} ({item['source']})"
                x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 184, 166), 3)
                cv2.putText(annotated, f"J{i} {label} {conf:.0%}", (x1, max(24, y1 - 8)), FONT, 0.8, (20, 184, 166), 2)
                cv2.imwrite(os.path.join(CAPTURE_DIR, f"{stamp}_J{i}.png"), item["crop_bgr"])
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"{stamp}_capture.jpg"), annotated)
        except Exception as e:
            self.event.emit(f"Could not save the capture: {e}", "info")

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
        if frame is not None:  # how each jewel's spot looks when detected, to tell "missed" from "gone"
            for tr in self.tracks:
                if tr.det_seen == t:
                    tr.det_patch = spot_patch(frame, tr.det_box)
        self.update_rubbing(t)
        self.update_acid(t)
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
                self.event.emit(f"J{tr.id} is on the bench (fingerprint {best_sim:.2f}); "
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
        return self.siglip, img, siglip_mask(img, d.mask, jewel), False

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
        tr.rub_boxes.append(ev["box"])
        if not tr.rubbed and tr.rub_time >= RUB_CONFIRM_S:
            tr.rubbed = True
            self.event.emit(f"J{tr.id} RUBBED on the stone", "info")
            self.emit_jewel_table()

    def rub_focus(self, t):
        """True while a jewel is being rubbed: the others rest on the bench (see RUB_FOCUS)."""
        if not RUB_FOCUS or self.rubber is None:
            return False
        self.update_rubbing(t)
        if self.rubbing_now:
            self.focus_last_rub = t
        if not self.focus:
            if self.rubbing_now and t >= self.focus_block_until:
                self.focus, self.focus_check_t = True, t
                self.focus_resting = {tr.id for tr in self.tracks
                                      if t - tr.det_seen < REID_AFTER_S and tr.state == "OK"
                                      and not self.on_stone(tr.pos, margin=0)}
                for tr in self.tracks:
                    tr.focus_miss = 0
        elif t - self.focus_last_rub > RUB_FOCUS_EXIT_S:
            self.focus = False
        return self.focus

    def update_focus(self, dets, t, frame):
        """Rubbing focus step. dets is None between the bench checks: resting jewels are taken as lying where
        they are. At a check each must still be detected at its place; anything else on the bench (a resting
        jewel moved or gone, a jewel put back, a hand reaching in) ends the focus and full tracking takes over."""
        resting = [tr for tr in self.tracks if tr.id in self.focus_resting]
        if dets is not None:
            bench = [d for d in dets if not self.on_stone(d.c, margin=0)]
            used = set()
            for tr in resting:
                x1, y1, x2, y2 = tr.box
                near = [j for j, d in enumerate(bench) if j not in used
                        and (np.hypot(*(d.c - tr.pos)) <= RUB_FOCUS_MATCH_PX
                             or (x1 <= d.c[0] <= x2 and y1 <= d.c[1] <= y2))]
                if near:
                    used.update(near)  # a chain may come as several pieces
                    tr.focus_miss = 0
                else:
                    tr.focus_miss += 1
            if any(tr.focus_miss >= 2 for tr in resting) or len(used) < len(bench):
                self.focus = False
                self.focus_block_until = t + RUB_FOCUS_COOLDOWN_S
                self.update_tracks(dets, t, frame)
                return
        seen = set()
        for i, tr in enumerate(self.tracks):
            if tr.id in self.focus_resting and tr.focus_miss == 0:
                tr.last_seen = tr.det_seen = t
                tr.csrt, tr.bridged = None, False
                seen.add(i)
        self.update_acid(t)
        for i, tr in enumerate(self.tracks):
            self.evaluate(tr, i in seen, t)
            self.check_done(tr, i in seen, t)

    def pick_rubbed_jewel(self, ev, t):
        """The jewel lying on the rubbed gold, else the one missing from the bench (in the hand)
        nearest the stone; jewels not rubbed yet are preferred."""
        x1, y1, x2, y2 = ev["box"]
        on_gold = [tr for tr in self.tracks if t - tr.det_seen <= 0.3
                   and tr.box[0] < x2 + 30 and tr.box[2] > x1 - 30 and tr.box[1] < y2 + 30 and tr.box[3] > y1 - 30]
        in_hand = [tr for tr in self.tracks if t - tr.det_seen > 0.3]
        pool = on_gold or in_hand or self.tracks
        return min(pool, key=lambda tr: (tr.rubbed, float(np.hypot(*(tr.pos - ev["c"])))), default=None)

    # ---- acid test (run-local4 process_acid_frame) ----
    def update_acid(self, t):
        if self.acider is None:
            self.acid_cand = None
            return
        waiting = [tr for tr in self.tracks if tr.rubbed and not tr.acid_done]
        self.acider.active = bool(waiting) and self.stone_box() is not None
        confirmed, self.acid_cand, self.acid_ref = self.acider.snapshot()
        for _, box in confirmed:
            waiting = [tr for tr in self.tracks if tr.rubbed and not tr.acid_done]
            tr = self.pick_acid_jewel(box, waiting)
            if tr is None:
                continue
            tr.acid_done, tr.acid_box = True, box
            self.event.emit(f"J{tr.id} ACID TEST seen on its rub streak", "info")
            self.emit_jewel_table()

    @staticmethod
    def pick_acid_jewel(box, waiting):
        """The rubbed jewel whose rub streak the acid covers; else the one rubbed last."""
        if not waiting:
            return None
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        def streak_dist(tr):
            if not tr.rub_boxes:
                return 1e9
            return min(float(np.hypot(*(c - [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]))) for b in tr.rub_boxes)
        near = min(waiting, key=streak_dist)
        if streak_dist(near) <= 0.6 * max(box[2] - box[0], box[3] - box[1]) + 60:
            return near
        return max(waiting, key=lambda tr: tr.rub_last)

    def check_done(self, tr, seen, t):
        """DONE = rubbed and acid tested, then seen off the stone lying still for DONE_SETTLE_S."""
        if not tr.rubbed or tr.done or (ACID_REQUIRED and self.acider is not None and not tr.acid_done):
            return
        still = (seen and tr.state == "OK" and not self.on_stone(tr.pos)
                 and float(np.hypot(*tr.vel)) <= SETTLE_SPEED_PX_S)
        if not still:
            tr.settle_since = None
        elif tr.settle_since is None:
            tr.settle_since = t
        elif t - tr.settle_since >= DONE_SETTLE_S:
            tr.done = True
            acid = ", acid tested" if tr.acid_done else ""
            self.event.emit(f"J{tr.id} DONE (rubbed{acid} and placed back)", "info")
            self.emit_jewel_table()

    def still_there(self, tr):
        """The jewel's spot looks as it did when it was last detected: it is lying there, only missed."""
        frame = getattr(self, "cur_frame", None)
        ref = getattr(tr, "det_patch", None)
        if frame is None or ref is None:
            return False
        cur = spot_patch(frame, tr.det_box)
        if cur is None:
            return False
        a, b = cur - cur.mean(), ref - ref.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return den > 1e-6 and float((a * b).sum()) / den >= STILL_THERE_NCC

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
            elif d < EDGE_MARGIN_PX and lost > EDGE_GRACE_S and not self.still_there(tr):
                new = "LEFT_ROI"
            elif lost > 0.3:
                new = "OCCLUDED"  # in the hand / covered: no alert however long - only leaving the ROI alerts
            else:
                new = old
        if new != old:
            tr.state = new
            if new == "LEFT_ROI":
                self.event.emit(f"J{tr.id} MISSING - left the ROI", "alert")
            elif old in ("LEFT_ROI", "MISSING") and new == "OK":
                self.event.emit(f"J{tr.id} back inside ROI", "info")

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
                who = f" J{self.rub_track.id}" if self.rub_track is not None else ""
                cv2.putText(vis, "RUBBING" + who, (int(ev["box"][0]), max(int(ev["box"][1]) - 8, 14)), FONT, 0.8,
                            RUB_GOLD_COLOR, 2)
        for tr in self.tracks:  # acid credited to each jewel
            if tr.acid_box is not None:
                x1, y1, x2, y2 = (int(v) for v in tr.acid_box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), ACID_COLOR, 2)
                cv2.putText(vis, f"ACID J{tr.id}", (x1, max(y1 - 6, 14)), FONT, 0.6, ACID_COLOR, 2)
        if self.acid_cand is not None:
            (x1, y1, x2, y2), conf, n = self.acid_cand
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), ACID_COLOR, 1)
            cv2.putText(vis, f"acid? {n}/{ACID_CONFIRM_CHECKS}", (int(x1), int(y2) + 18), FONT, 0.6, ACID_COLOR, 2)
        for p in self.draft:
            cv2.circle(vis, p, 5, (0, 255, 255), -1)
        if len(self.draft) > 1:
            cv2.polylines(vis, [np.array(self.draft, np.int32)], False, (0, 255, 255), 1)
        if SHOW_REJECTED:
            for d in self.rejected:
                x1, y1, x2, y2 = d.box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 140), 1)
                cv2.putText(vis, f"not gold {d.gold:.0%}", (x1, max(y1 - 4, 12)), FONT, 0.5, (0, 0, 140), 1)
        for d in dets:  # raw detections, thin grey (on the stone they may be the jewel or a rub mark: no label)
            x1, y1, x2, y2 = d.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (130, 130, 130), 1)
        alerts = []
        for tr in self.tracks:
            col = COLORS[tr.state]
            name = f"J{tr.id}"  # the type is only listed in the side panel, not predicted on the video
            label = name if tr.state == "OK" else f"{name} {tr.state}"
            if tr.state in ("OK", "OCCLUDED", "RUBBING"):
                if tr.done:
                    col, label = DONE_COLOR, f"{name} DONE"
                elif tr.rubbed and tr.acid_done:
                    col, label = ACID_COLOR, f"{name} ACID OK"
                    if tr.settle_since is not None:
                        label += f" {t - tr.settle_since:.0f}/{DONE_SETTLE_S:.0f}s"
                elif tr.rubbed and ACID_REQUIRED and self.acider is not None:
                    col, label = RUBBED_COLOR, f"{name} RUBBED - acid?"
                elif tr.rubbed:
                    col, label = RUBBED_COLOR, f"{name} RUBBED"
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
                alerts.append(f"J{tr.id} {tr.state}")
        bottom = vis.shape[0] - 15
        if self.mode == "ARMING":
            if self.arm_wait_stone:
                text = f"Lay the jewels in the ROI, then place the touchstone ({self.arm_count} on the bench)"
            else:
                text = (f"Stone placed: {self.arm_count} jewels detected - capturing when steady "
                        f"({self.stable:.0%})")
            cv2.putText(vis, text, (10, bottom), FONT, 0.7, (0, 255, 255), 2)
        elif self.mode == "LOCKED":
            done = sum(tr.done for tr in self.tracks)
            steps = "Rubbed + acid + back" if ACID_REQUIRED and self.acider is not None else "Rubbed and back"
            text = f"{steps} (DONE): {done}/{len(self.tracks)}"
            if self.focus:
                text += "   |   rubbing: other jewels held in place"
            if self.acider is not None and self.acider.active:
                text += "   |   apply acid on the streak" if self.acid_ref else \
                    "   |   acid check: waiting for a clear view of the stone"
            cv2.putText(vis, text, (10, bottom), FONT, 0.8, DONE_COLOR, 2)
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
        self.jewels = QtWidgets.QListWidget()  # "J1. Chain / Necklace" from the stone-placed capture
        self.jewels.setFixedHeight(130)
        self.jewels.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.jewels.setToolTip("Double-click a jewel to confirm or correct its type (saved to the correction gallery)")
        self.table = QtWidgets.QTableWidget(0, 4)  # S.No | Type | Rubbed | Acid test, ticked as they happen
        self.table.setHorizontalHeaderLabels(["S.No", "Type", "Rubbed", "Acid test"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setStyleSheet("font-size: 14px;")
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setMinimumHeight(160)
        self.table.setToolTip("Green row = DONE (rubbed, acid tested and placed back)")
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

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.btn_roi)
        side.addWidget(self.btn_stone)
        side.addWidget(self.btn_clear)
        side.addLayout(form)
        side.addWidget(self.btn_reset)
        side.addWidget(self.lbl_status)
        self.lbl_jewels = QtWidgets.QLabel("Jewels: counted when the stone is placed")
        side.addWidget(self.lbl_jewels)
        side.addWidget(self.jewels)
        side.addWidget(QtWidgets.QLabel("Testing status"))
        side.addWidget(self.table)
        side.addWidget(QtWidgets.QLabel("Events"))
        side.addWidget(self.log, 1)
        side_w = QtWidgets.QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(330)

        mid = QtWidgets.QHBoxLayout()
        mid.addWidget(self.video, 1)
        mid.addWidget(side_w)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid, 1)

        self.btn_conn.clicked.connect(self.toggle_connect)
        mode = self.cameras.get("mode", "network")  # "usb" is only known to jewel_tracker_2.py
        self.cam_mode.setCurrentIndex(list(CAMERA_MODES).index(mode) if mode in CAMERA_MODES else 0)
        self.show_camera(self.cam_mode.currentData())
        self.cam_mode.currentIndexChanged.connect(self.switch_camera)
        self.btn_roi.clicked.connect(lambda: self.toggle_draw("roi"))
        self.btn_stone.clicked.connect(lambda: self.toggle_draw("stone"))
        self.btn_clear.clicked.connect(self.clear_zones)
        self.jewels.itemDoubleClicked.connect(self.edit_jewel_type)
        self.btn_reset.clicked.connect(self.rearm)
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
            self.cameras[key].pop("antiflicker", None)
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
        camera = None if self.cam_mode_key == "custom" else self.cameras[self.cam_mode_key]
        self.worker = Worker(source, self.zones["roi"], self.zones["stone"], camera=camera)
        self.worker.frame_ready.connect(self.video.set_frame)
        self.worker.jewel_list.connect(self.show_jewels)
        self.worker.jewel_table.connect(self.show_table)
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

    def show_jewels(self, lines):
        self.jewels.clear()
        self.jewels.addItems(lines)
        self.lbl_jewels.setText(f"Jewels: {len(lines)} detected when the stone was placed" if lines
                                else "Jewels: counted when the stone is placed")

    def show_table(self, rows):
        self.table.setRowCount(len(rows))
        for r, (jid, kind, rubbed, acid, done) in enumerate(rows):
            cells = [f"J{jid}", kind, "\u2714" if rubbed else "", "\u2714" if acid else ""]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter if c != 1 else QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                if c >= 2 and text:
                    item.setForeground(QtGui.QColor(0, 150, 0))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                if done:
                    item.setBackground(QtGui.QColor(200, 240, 200))
                self.table.setItem(r, c, item)

    def edit_jewel_type(self, item):
        if not self.worker or self.worker.classifier is None:
            return
        head, _, rest = item.text().partition(". ")
        current = rest.split("  (")[0]
        choices = self.worker.classifier.label_choices()
        label, ok = QtWidgets.QInputDialog.getItem(
            self, "Jewel type", f"Type of {head} (saved to the correction gallery):", choices,
            choices.index(current) if current in choices else 0, False)
        if ok:
            self.worker.correct_type(int(head[1:]), label)

    def rearm(self):
        if not self.worker:
            return
        self.worker.arm()
        self.add_log("Re-armed: lay the jewels in the ROI, then place the touchstone", "info")

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
        write_log(f"{level.upper():5}  {msg}")

    def beep_tick(self):
        if time.time() < self.beep_until:
            beep()
        else:
            self.beep_timer.stop()

    def closeEvent(self, e):
        self.stop_worker()
        e.accept()


def write_log(line):
    """Events also go to jewel_guard.log next to the app (the .exe has no console)."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")
    except OSError:
        pass


def log_crash(kind, value, tb):
    import traceback
    write_log("CRASH  " + "".join(traceback.format_exception(kind, value, tb)))
    sys.__excepthook__(kind, value, tb)


if __name__ == "__main__":
    sys.excepthook = log_crash
    write_log("START  Jewel Guard")
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1280, 760)
    w.show()
    sys.exit(app.exec_())
