"""
Jewel Guard: RTSP jewel tracking with ROI-exit / disappearance alerts and a per-jewel rubbing + acid check.

Install:  pip install ultralytics opencv-python PyQt5 scipy numpy onnxruntime openvino
Run:      ./run_tracker.sh (or .venv/bin/python jewel_tracker_5.py)
          Jewel/stone/rubbing models sit next to this script. Temporal acid
          code and models are bundled under acid_test/. See README-linux.md.

Workflow:
  1. The camera connects on startup. The top bar picks how it is reached:
       - Network (router/Wi-Fi): the Dahua camera's address on the LAN
       - Direct Ethernet: the camera cabled straight into this PC's Ethernet port; camera and PC need
         static IPs in the same subnet (this PC's Ethernet is 192.168.1.1/24, so e.g. camera 192.168.1.108)
       - Custom URL: any RTSP URL, or 0 for a webcam
     Switching it while connected reconnects to that feed. Settings are saved to camera.json.
     The still capture is the median of the last frames (less sensor noise in the jewel crops).
  2. Draw ROI -> left-click corners -> right-click (or "Finish ROI") to close. It must cover the
     bench and the touchstone. Automatic stone boxes come from INT8_openvino_model/best.xml,
     checked continuously on fresh frames; "Draw stone" is a manual fallback
     for a stone that can't be found. Both are saved to zones.json.
  3. Lay all jewels inside the ROI, then place the touchstone. Once the stone is present and the bench
     count has been steady for AUTO_COUNT_S, one still frame is captured and analysed as in
     SOP-monitoring-rpi / Weight-Scale-SOP: YOLO counts the jewels; each one is cut out on white from
     the repos' Otsu jewellery mask (jewel_capture.py) and classified once with SigLIP2 and its
     correction gallery (jewel_classifier.py), plus a closed-ring shape check for bangles. The type is
     locked to the jewel and listed in the side panel as "J1. Chain / Necklace"; double-click to rename.
     The live video only tracks J1..JN (no type prediction on the live video).
  4. The only jewel alert (message + 5 s beep) is a jewel leaving the ROI ("J1 MISSING - left the ROI"):
     its box centre is seen outside the ROI (the detector also looks OUTSIDE_MARGIN_PX past it). A jewel
     that is merely unseen (in the hand, covered, rubbed) shows a dashed box, never an alert.
     Detections that aren't gold-coloured (HSV check) are dropped.
  5. Rubbing check (visual part of run-local4.py, no audio): the touchstone is found with its stone
     model, gold on the stone with its rubbing-gold model (mask clipped to the stone, >= 50 % inside),
     and rubbing is gold on the stone whose centroid keeps moving. Rubbing is credited to one jewel:
     the jewel seen at the stone that moves most (movement score: rubbing jiggles it), else the tracked
     jewel on that gold, else the jewel missing from the bench (in the hand) nearest the stone. After RUB_CONFIRM_S of rubbing that jewel is RUBBED; while it is being rubbed it shows
     RUBBING.
  6. IDs stay with their jewel (v4): the gold model's boxes are followed by position from frame to frame. At
     lock a per-session ResNet18 ID model is trained in the background (~8 s) on each jewel's capture
     crop (turned / scaled / re-lit copies; one class per jewel, plus "not a jewel" from bench and stone
     crops). A box that stands still for STABLE_S is identified by it: a jewel put back anywhere gets its
     ID back, and IDs swapped when jewels slid over each other are corrected. No CSRT.
  7. Rub any untested jewel first. Once rubbing is confirmed, that jewel owns the next acid test.
     Click Start acid test for Jn, show fresh yellow pH paper, and add acid to that
     same paper until its pink reaction is verified using OpenVINO. Repeat for EVERY jewel
     and retry. Remove paper/hands, click Ready for stone test, wait for the 2-second clean stone baseline,
     and only apply acid to the streak at NOW APPLY ACID. The local acid_test package records pouring and
     post-detection frames and runs the temporal MobileNet classifier. The same jewel's row
     receives 22K GOLD, NON-GOLD or INCONCLUSIVE and confidence. Then rub any next jewel.
     Capture sequences and predictions are saved locally; predictions are never ground-truth labels.
     A jewel is DONE after its acid result is ready and it rests off the stone for DONE_SETTLE_S.

"""
import math
import os
import sys
import time
import json
import threading
import uuid
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

# CPU budget. Several models run side by side (jewel YOLO, stone, rubbing, acid and SigLIP2, all in
# onnxruntime). By default each grabs every core and keeps its threads spinning between runs: measured live
# (14-10 16:25) 8 threads at ~90 % each and the PC at 100 % while the jewels just lay there. So each
# onnxruntime model gets ORT_THREADS threads that sleep when idle, and torch TORCH_THREADS.
ORT_THREADS = 2
JEWEL_MODEL_THREADS = 4   # the jewel model runs on every frame: it gets more
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
        jewel = isinstance(path_or_bytes, str) and os.path.basename(path_or_bytes) == JEWEL_MODEL_FILE
        so.intra_op_num_threads = JEWEL_MODEL_THREADS if jewel else ORT_THREADS
        so.inter_op_num_threads = 1
        for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
            try:
                spinning = so.get_session_config_entry(key)
            except RuntimeError:
                spinning = None
            if spinning != "0":
                so.add_session_config_entry(key, "0")
        super().__init__(path_or_bytes, so, *args, **kwargs)


onnxruntime.InferenceSession = _BudgetSession
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from jewel_capture import jewellery_mask, jewel_crop, ring_shape
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from PyQt5 import QtCore, QtGui, QtWidgets
from acid_test.integration import AcidTestThread, AcidModelCache
from live_pipeline import InferenceDispatcher, StaleInference, LatestPreview
from stone_onnx import StoneOnnxModel
from rub_preparation import RubPhotoModels, RubCycle, StoneCleanup, draw_acid_roi, render_rub_preview
from auto_testing import StoneSettling
from linux_camera_controls import LINUX_AF_DEFAULT, apply_linux_antiflicker

# OpenCV's Linux wheel sets paths to its own Qt plugins during import. The
# GUI uses PyQt5's Qt build, so those plugins cannot be used by QApplication.
if sys.platform.startswith("linux"):
    for _qt_env in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "/cv2/qt/" in os.environ.get(_qt_env, "").replace("\\", "/"):
            os.environ.pop(_qt_env, None)

# models, settings, captures and the log live next to the .exe when built with PyInstaller
# (build_jewel_tracker.ps1), else next to this script
SCRIPT_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "jewel_guard.log")
# One fixed-320 gold segmentation model serves both bench tracking and
# gold-on-stone rubbing. No fallback to the old jewel/rubbing models.
JEWEL_MODEL_FILE = "rojeweltrack.onnx"
MODEL_PATH = os.path.join(SCRIPT_DIR, JEWEL_MODEL_FILE)
ZONES_FILE = os.path.join(SCRIPT_DIR, "zones.json")
CAPTURE_DIR = os.path.join(SCRIPT_DIR, "captures")  # stone-placed captures: annotated image + jewel crops
CAMERA_FILE = os.path.join(SCRIPT_DIR, "camera.json")
# camera connections (editable in the top bar, saved to camera.json)
CAMERA_MODES = {"usb": "USB camera", "network": "Network (router/Wi-Fi)", "direct": "Direct Ethernet",
                "custom": "Custom URL"}
USB_RESOLUTIONS = ["2560x1440", "1920x1080", "1280x720"]
DEFAULT_CAMERAS = {
    "mode": "usb",
    # USB camera over the bench: device number and the resolution asked for (MJPG through DirectShow; without
    # that Windows opens it at a low default resolution). Measured 15-10: camera 0 gives 2560x1440 at 20 fps.
    "usb": {"index": 0, "resolution": "2560x1440"},
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
CAPTURE_IMGSZ = 320        # fixed input size of rojeweltrack.onnx for the still capture
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
ALERT_BEEP_S = 2.0         # beep this long after an alert

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
# Skin passes the gold check above (hue 8-35 covers it). Measured on the 14-10 11:53 recording: the operator's
# fingers H 10-13, S ~190 (60-99 % of their pixels match below); the jewels H 16-24, S 60-90 (0-6 % match).
# A low-confidence detection on the fingertip beside the stone was taken for the jewel being rubbed and then
# raised a false "J2/J3 MISSING - left the ROI".
SKIN_H_MAX = 14            # skin: hue <= this ...
SKIN_S_MIN = 110           # ... and strongly saturated (gold here is S 60-90)
SKIN_MAX_FRAC = 0.35       # more skin than this among the non-background pixels -> a hand, not a jewel
MERGE_GAP_PX = 40          # detections whose masks come this close are pieces of one jewel
                           # (a spread-out chain's masks miss the thin strands, leaving ~30 px gaps)
                           # and are merged; ~1 cm at the current camera height, so keep
                           # separate jewels further apart than that
SHOW_REJECTED = False      # True: draw dropped (not gold) detections in dark red, for tuning

# rubbing check: the visual part of R:\Rubbing_Dept_Roha\run-local4.py (no audio)
STONE_MODEL_PATH = os.path.join(SCRIPT_DIR, "INT8_openvino_model", "best.xml")
RUB_GOLD_MODEL_PATH = MODEL_PATH  # same loaded model as jewellery tracking
# Automatic stone detection runs the INT8 model on each newest available frame.
STONE_RESULT_MAX_AGE_S = 1.0   # discard delayed results during CPU contention
STONE_CONF = 0.20              # minimum stone model confidence
STONE_PRESENT_S = 2.5          # maximum age for a successful model observation
RUB_GOLD_CONF = 0.15           # gold-on-stone confidence, matching bench detection
RUB_MIN_MASK_PX = 80           # run-local4 GOLD_MIN_MASK_PIXELS
RUB_MIN_INSIDE_RATIO = 0.50    # run-local4 GOLD_MIN_INSIDE_STONE_RATIO
RUB_MOVE_PX = 1.5              # run-local4 RUBBING_MIN_CENTROID_MOVE is 0.5 px at 640x480; x3 for 1080p
RUB_MOVE_WINDOW = 3            # three fresh jewellery movements relative to the stone
RUB_CONFIRM_S = 0.6            # seconds of rubbing credited to one jewel -> RUBBED (stands in for audio)
RUB_GAP_S = 0.5                # rubbing pauses shorter than this still count as continuous
# Stone masks reuse the full-frame bench prediction; no second model call.
RUB_STONE_MIN_IOU = 0.70
RUB_STONE_MAX_CENTER_RATIO = 0.15  # displacement / average stone diagonal during inference
RUB_STONE_JITTER_PX = 3.0          # stabilize small box changes for relative motion
RUB_GRACE_S = 2.0              # rubbing seen this recently: an unseen jewel is RUBBING, not MISSING
# One scheduled jewellery inference serves tracking and stone masking.
LIVE_RESULT_MAX_AGE_S = 0.5
PREVIEW_INTERVAL_MS = 40
STONE_MARGIN_PX = 40           # "on the stone" includes this margin around the stone box
# Rubbing focus: while a jewel is rubbed the others rest on the bench. The rubbing check (~100 ms, 10x/s) and the
# jewel YOLO (85-180 ms every frame) together saturated the CPU and the video hung, so while rubbing is seen the
# resting jewels are held in place and the jewel YOLO only runs once per RUB_FOCUS_CHECK_S to confirm they are
# still there. Any change on the bench (a resting jewel moved or gone, a jewel put back) -> full tracking.
RUB_FOCUS = False              # off: since the CPU fixes it is not needed, and it held boxes of jewels taken away
RUB_FOCUS_CHECK_S = 1.0        # bench check while rubbing
RUB_FOCUS_EXIT_S = 2.0         # no rubbing for this long -> full tracking again
RUB_FOCUS_MATCH_PX = 40        # a resting jewel's detection must be within this of where it rests
RUB_FOCUS_COOLDOWN_S = 3.0     # after the bench changed, full tracking for at least this long
DONE_SETTLE_S = 2.0            # rubbed (and acid tested) jewel lying still off the stone this long -> DONE
SETTLE_SPEED_PX_S = 40

# Temporal acid capture and models are local to acid_test/.
ACID_REQUIRED = True  # DONE requires a completed temporal classification.

# keeping IDs on their jewel (the similarity cues): see Worker.update_tracks
APP_WEIGHT_PX = 150        # colour mismatch (1 - Bhattacharyya) in px of matching cost while tracked
TYPE_MISMATCH_PX = 400     # chain vs solid piece, in px of matching cost while tracked
# ---- v4 tracking: gold-model boxes followed by position; identity from a per-session ResNet18 ID model ----
OUTSIDE_MARGIN_PX = 150    # the jewel detector also looks this far outside the ROI, so a jewel's centre can be
                           # seen leaving it: centre outside the ROI for OUT_FRAMES frames = "MISSING - left the
                           # ROI". That is the only missing alert (16-10). The detector input grows with the
                           # strip so jewels keep their size in it (150 px at 320 alone lost an earring in 20/20
                           # frames; at 384 both earrings 20/20, 166 ms vs 125 ms per frame).
TRACK_KEEP_S = 1.0         # a jewel unseen longer than this is no longer followed by position
# A jewel out of sight (16-10): its own pixels from the last detection are looked at. Same as before -> still lying
# there (only not detected). Mostly white bench -> removed from there. Anything else (dark, skin, sandal) -> a hand
# over it: no alert. Removed and neither rubbed nor seen again within REMOVED_ALERT_S -> "MISSING - removed".
SPOT_WHITE = 0.7           # share of the jewel's pixels showing white bench -> removed
# "Missing" only when no hand is there (17-10): hands over the bench move, so the bench is "still" when nothing
# moves in the ROI between frames (1/8 scale grey, exposure-matched). Cables and other still things never count.
MOTION_DIFF = 25           # grey-level change between frames that counts as movement
MOTION_MIN_FRAC = 0.002    # moving pixels (share of the ROI) above this = a hand / arm moving
BENCH_STILL_S = 1.0        # "MISSING - removed" only after the bench has been still this long
REMOVED_ALERT_S = 2.0      # time to carry a jewel to the stone (rubbing seen -> no alert) or to put it back
OUT_FOLLOW_GAP_S = 0.3     # a box outside the ROI is only taken by a jewel seen this recently ...
OUT_FOLLOW_PX = 120        # ... and this close to where it was going (followed out, not jumped to)
STABLE_S = 0.8             # a detection that has not moved for this long is identified by the ID model
STABLE_MOVE_PX = 15        # ... "not moved": centre within this of the previous frame's (and boxes overlapping)
ID_MIN_PROB = 0.6          # ID model: a stable detection is taken for a jewel at this probability or more
ID_SWAP_PROB = 0.8         # a followed jewel the model says is another one this surely -> IDs corrected
ID_SAMPLES = 48            # turned / scaled / re-lit copies of each capture crop to learn from
ID_NEG_SAMPLES = 60        # bench and stone crops (rub marks): the "not a jewel" class
ID_CROP_PAD = 0.12         # crop = detection box plus this share around it
RESNET18_FILE = "resnet18-f37072fd.pth"  # torchvision's ImageNet ResNet18, kept in weights\ ...
RESNET18_ONNX = "resnet18_features.onnx"  # ... and exported without its last layer (512 features) for
                                          # onnxruntime: 20 ms per crop vs 34 ms in torch, same output
ID_CROP_MAX = 256          # crops are shrunk to this first (ResNet18 sees 224 px): turning a 1000 px necklace crop
                           # made training take 25 s
MOVE_EMA = 0.4             # movement score: smoothed centre jitter (px per frame); the rubbed jewel moves most
REID_AFTER_S = 0.5         # unseen longer than this: matched back by fingerprint, not position
REID_MIN_SIM = 0.45        # a detection less similar than this is never taken for a lost jewel ...
REACQ_COVER = 0.6          # ... unless it covers this share of the lost jewel's last box: then it is that jewel
                           # (16-10: J1's track had followed one piece of a necklace while it was lifted; put back,
                           # the whole necklace was detected around that box but its fingerprint did not match)
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


class SharedGoldModel:
    """One predictor for full-frame live observations and still captures.

    A live result supplies both tracking and stone-region masking; stone
    masking never submits a second prediction.
    """
    def __init__(self, path=MODEL_PATH):
        self.backend = YOLO(path, task="segment")
        self.lock = threading.Lock()
        self.dispatcher = InferenceDispatcher(self._predict)

    def _predict(self, source, **kwargs):
        with self.lock:
            return self.backend.predict(source, **kwargs)

    def predict(self, source, **kwargs):
        kind = kwargs.pop('_live_job', 'bench')
        captured_at = kwargs.pop('_captured_at', None)
        expires_at = (time.monotonic() + LIVE_RESULT_MAX_AGE_S - (time.time() - captured_at)
                      if captured_at is not None else None)
        kwargs['imgsz'] = 320
        kwargs['stream'] = False
        return self.dispatcher.submit(kind, source, kwargs, expires_at).result()

    def close(self):
        self.dispatcher.close()


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


# Anti-flicker for the USB camera, ported from R:\prototype_app\vision_app.py (verified there): a manual shutter
# that is a whole multiple of the mains light's half-period (10 ms at 50 Hz, 8.33 ms at 60 Hz) catches the same
# amount of light in every row and frame. Through DirectShow CAP_PROP_EXPOSURE is a log2 stop (1/100 s -> -6.64);
# some drivers want absolute 0.1 ms units instead, tried when the log2 write does not stick.
AF_DENOM_RANGE = (8, 200)      # shutter 1/N s: 1/8 s (125 ms, brightest) .. 1/200 s (5 ms, darkest)
AF_DEFAULT = (dict(LINUX_AF_DEFAULT) if sys.platform.startswith("linux") else
              {"enabled": False, "mains_hz": 50, "shutter_denom": 100})
AF_CHECK_AFTER_S = 4.0         # after applying, compare the measured flicker with before
# A fixed shutter is longer than what auto exposure picked under bright light: measured 15-10 on the Dell WB3023
# 13 % (15.6 ms) to 46 % (31 ms) of the frame burnt pure white. Gain is already at its minimum there, so the
# camera's brightness is stepped down until at most AF_MAX_CLIPPED of the frame is clipped.
AF_MAX_CLIPPED = 0.02
AF_BRIGHTNESS_STEPS = (96, 64, 32, 0)
AF_SETTLE_S = 1.2              # the webcam needs ~1 s to settle after a change


def af_flicker_note(denom, mains_hz):
    """(exposure ms, flicker-safe?): safe when the shutter is ~a whole multiple of the light's half-period."""
    ms = 1000.0 / denom
    half = 1000.0 / (2 * mains_hz)
    frac = (ms % half) / half
    return ms, ms >= 0.8 * half and (frac < 0.2 or frac > 0.8)  # shorter than a half-period catches part of it


def apply_usb_antiflicker(cap, af, state, index=0):
    """Push the anti-flicker setting onto an open USB camera (DirectShow). Returns a read-back message.
    `state` keeps the camera's own brightness so switching off restores it."""
    if sys.platform.startswith("linux"):
        return apply_linux_antiflicker(cap, af, state, index)
    if not af.get("enabled"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto exposure (DSHOW/UVC)
        if state.get("brightness") is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, state["brightness"])
        return "Anti-flicker off: camera back on auto exposure"
    if state.get("brightness") is None:
        state["brightness"] = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    else:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, state["brightness"])  # start from the camera's own brightness
    denom = max(AF_DENOM_RANGE[0], min(AF_DENOM_RANGE[1], int(af.get("shutter_denom", 100))))
    hz = int(af.get("mains_hz", 50))

    def settle():  # several UVC drivers ignore controls until samples flow; a read-back right after set() is stale
        for _ in range(5):
            cap.grab()
        time.sleep(0.12)

    def manual_mode():
        for val in (0.25, 1.0, 0.0, 3.0):  # the manual-exposure code differs between drivers
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            time.sleep(0.12)
            rb = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            if rb != -1 and abs(rb - val) < 0.5:
                return True
        return False

    settle()
    manual = manual_mode()
    want = math.log2(1.0 / denom)
    ok = cap.set(cv2.CAP_PROP_EXPOSURE, want)
    time.sleep(0.12)
    rb = cap.get(cv2.CAP_PROP_EXPOSURE)
    scale = "log2"
    if ok and rb != -1 and abs(rb - want) >= 0.3 and abs(rb - round(rb)) < 0.05:
        # the camera only takes whole stops (asked 1/100 s = -6.64, it used -6 = 15.6 ms): use the whole stop
        # closest to the asked shutter that is flicker-safe at this mains frequency instead
        safe = [st for st in range(-13, 0) if af_flicker_note(2.0 ** -st, hz)[1]]
        if safe:
            st = min(safe, key=lambda v: abs(v - want))
            ok2 = cap.set(cv2.CAP_PROP_EXPOSURE, float(st))
            time.sleep(0.12)
            rb2 = cap.get(cv2.CAP_PROP_EXPOSURE)
            if ok2 and rb2 != -1:
                want, ok, rb, scale = float(st), ok2, rb2, "nearest flicker-safe whole stop"
    if not ok or rb == -1 or abs(rb - want) > 1.5:  # absolute 0.1 ms units (MSMF-style drivers)
        manual = manual_mode() or manual
        absolute = max(1, int(round(10000 / denom)))
        ok3 = cap.set(cv2.CAP_PROP_EXPOSURE, float(absolute))
        time.sleep(0.12)
        rb3 = cap.get(cv2.CAP_PROP_EXPOSURE)
        if ok3 and rb3 != -1 and abs(rb3 - absolute) <= max(2.0, absolute * 0.5):
            want, ok, rb, scale = float(absolute), ok3, rb3, "absolute"
    settle()
    took = bool(ok) and rb != -1
    # the exposure the camera really uses (read back), not the one asked for
    real_ms = 1000.0 * 2.0 ** rb if took and scale.startswith(("log2", "nearest")) else (
        rb / 10.0 if took else 1000.0 / denom)
    safe = af_flicker_note(1000.0 / real_ms, hz)[1] if real_ms > 0 else False
    # keep the white bench from burning out: lower the brightness until little of the frame is clipped
    clipped, bright = clipped_share(cap), cap.get(cv2.CAP_PROP_BRIGHTNESS)
    for b in AF_BRIGHTNESS_STEPS:
        if clipped <= AF_MAX_CLIPPED or b >= bright:
            if clipped <= AF_MAX_CLIPPED:
                break
            continue
        cap.set(cv2.CAP_PROP_BRIGHTNESS, b)
        clipped, bright = clipped_share(cap), cap.get(cv2.CAP_PROP_BRIGHTNESS)
    return (f"Anti-flicker {'applied' if took else 'NOT accepted by the camera'}: asked 1/{denom} s, camera uses "
            f"{real_ms:.1f} ms ({'flicker-safe' if safe else 'may band'} at {hz} Hz; {scale} {want:.2f}, read-back "
            f"{rb:.2f}) | brightness {bright:.0f}, {clipped:.0%} of the frame burnt white"
            f"{'' if manual else ' | manual mode unconfirmed'}")


def clipped_share(cap):
    """Share of pure-white pixels once the camera has settled after a change."""
    t = time.time()
    while time.time() - t < AF_SETTLE_S:
        cap.grab()
    shares = []
    for _ in range(3):
        ok, f = cap.read()
        if ok:
            g = cv2.cvtColor(f[::4, ::4], cv2.COLOR_BGR2GRAY)
            shares.append(float(np.mean(g >= 250)))
    return float(np.median(shares)) if shares else 0.0


class FlickerMeter:
    """Measures flicker on the live feed (~1 ms per frame): banding = rows brighter/darker than their running
    average (rolling shutter under AC light), pulsing = whole-frame brightness changing between frames."""

    def __init__(self, n=30):
        self.n = n
        self.reset()

    def reset(self):
        self.ref = None
        self.band = deque(maxlen=self.n)
        self.level = deque(maxlen=self.n)

    def add(self, frame):
        g = cv2.cvtColor(frame[::4, ::16], cv2.COLOR_BGR2GRAY).astype(np.float32)
        prof = np.median(g, axis=1) + 1.0
        self.level.append(float(g.mean()))
        if self.ref is not None and self.ref.shape == prof.shape:
            self.band.append(float(np.std(prof / self.ref)))
            self.ref = 0.9 * self.ref + 0.1 * prof
        else:
            self.ref = prof

    def value(self):
        """(banding %, pulsing %, brightness) over the last frames, or None while too few."""
        if len(self.band) < 10:
            return None
        lv = np.array(self.level)
        return float(np.median(self.band)) * 100, float(lv.std() / max(lv.mean(), 1e-6)) * 100, float(lv.mean())


def open_usb_camera(index, resolution=None):
    """A USB camera at the asked resolution: DirectShow with MJPG (Windows otherwise opens it at a low default
    resolution, or uncompressed at a few fps)."""
    linux = sys.platform.startswith("linux")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2 if linux else cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if resolution:
        w, h = (int(v) for v in str(resolution).lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if linux:
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


class RTSPGrabber(threading.Thread):
    """Always keeps only the newest frame, so processing never lags behind the stream."""

    def __init__(self, source, camera=None):
        super().__init__(daemon=True)
        self.source = source      # RTSP URL, or the device number of a USB camera
        self.camera = camera or {}
        self.af = self.camera.get("antiflicker", dict(AF_DEFAULT) if sys.platform.startswith("linux") else None)
        # USB anti-flicker is applied on every (re)connect, before publishing frames.
        self.af_dirty = True
        self.af_state = {}        # the camera's own brightness, restored when anti-flicker is switched off
        self.af_busy_until = 0.0  # applying anti-flicker pauses the frames for a few seconds: not a lost stream
        self.on_af = None         # called with the read-back message after applying
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
                cap = open_usb_camera(self.source, self.camera.get("resolution"))
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                self.problem = (f"USB camera {self.source} not found: check the cable or try another camera number"
                                if isinstance(self.source, int) else
                                "Camera reachable but the stream did not open: check user, password and channel")
                time.sleep(2)
                continue
            self.connected, self.problem = True, ""
            self.af_dirty = True
            while self.running:
                if self.af_dirty and isinstance(self.source, int) and self.af is not None:
                    self.af_dirty = False  # on this thread: it owns the camera handle
                    self.af_busy_until = time.time() + 30
                    try:
                        msg = apply_usb_antiflicker(cap, self.af, self.af_state, self.source)
                    except Exception as e:
                        msg = f"Anti-flicker could not be applied: {e}"
                    self.af_busy_until = time.time() + 2.0
                    if self.on_af:
                        self.on_af(msg)
                ok, f = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame, self.ts = f, time.time()
            cap.release()
            self.connected = False
            time.sleep(1)  # then reconnect

    def latest(self, after_ts=None, copy_frame=True):
        with self.lock:
            if self.frame is None:
                return None, 0.0
            if self.ts == after_ts:
                return None, self.ts
            return (self.frame.copy() if copy_frame else self.frame), self.ts

    def request_antiflicker(self, af):
        self.af, self.af_dirty = af, True

    def stop(self):
        self.running = False


# ---------------- tracking ----------------
class Det:
    __slots__ = ("box", "c", "area", "conf", "gold", "mask", "origin", "app", "stable_since", "idres")

    def __init__(self, box, c, area, conf, gold=1.0, mask=None, origin=(0, 0)):
        self.box, self.c, self.area, self.conf, self.gold = box, c, area, conf, gold
        self.mask, self.origin = mask, origin  # mask is local to its box; origin = its top-left in the frame
        self.app = None  # appearance fingerprint, filled in when needed
        self.stable_since = None  # standing still since (set by the worker)
        self.idres = None         # (jewel id or None, probability) from the ID model, once it is stable


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
    skin = (h <= SKIN_H_MAX) & (s >= SKIN_S_MIN) & ~background
    if float(skin.sum()) / n > SKIN_MAX_FRAC:
        return 0.0  # the operator's fingers, not gold
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
        self.rub_masks = deque(maxlen=30)  # observed gold (mask, origin) during rubbing
        self.rub_marks = deque(maxlen=30)  # original masks and stone poses, for moving-stone projection
        self.rub_img = None       # fixed, unannotated photo at rubbing confirmation
        self.new_rub_target = None
        self.acid_roi_mask = self.acid_roi_stone = None
        self.acid_done = False    # acid test seen on its rub streak
        self.acid_box = None
        self.acid_result = None
        self.acid_crop = None
        self.app = det.app        # appearance fingerprint taken at lock (never updated)
        self.det_seen = t         # last time the detector itself found it
        self.det_box = det.box    # its box at that moment
        self.bridged = False      # (no CSRT any more in v4: always False)
        self.kind, self.kind_conf = "", 0.0  # jewel type from the classifier, at lock
        self.kind_note = ""
        self.move = 0.0           # movement score (smoothed centre jitter); the rubbed jewel moves most
        self.det_mask, self.det_origin = det.mask, det.origin
        self.spot, self.removed_since = "jewel", None  # out of sight: "jewel" / "removed" / "hand"

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
        self.det_mask, self.det_origin = det.mask, det.origin  # its own pixels at that moment


def box_iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = float(ix) * float(iy)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def dashed_rect(img, p1, p2, col, dash=14):
    """Dashed rectangle: a jewel that is not where its box is drawn (out of sight)."""
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1])))
        for s0 in range(0, length, 2 * dash):
            s1 = min(length, s0 + dash)
            pa = (int(a[0] + (b[0] - a[0]) * s0 / length), int(a[1] + (b[1] - a[1]) * s0 / length))
            pb = (int(a[0] + (b[0] - a[0]) * s1 / length), int(a[1] + (b[1] - a[1]) * s1 / length))
            cv2.line(img, pa, pb, col, 2)


def crop_box(frame, box, pad=ID_CROP_PAD):
    """The detection box plus a little bench around it, shrunk to at most ID_CROP_MAX px."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * pad) + 4, int((y2 - y1) * pad) + 4
    c = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
    if c.size and max(c.shape[:2]) > ID_CROP_MAX:
        f = ID_CROP_MAX / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * f)), max(1, int(c.shape[0] * f))), interpolation=cv2.INTER_AREA)
    return c


def augment(crop, rng):
    """A capture crop as it may look later: turned any way, mirrored, a bit bigger/smaller, lit differently."""
    h, w = crop.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(0, 360)), float(rng.uniform(0.85, 1.15)))
    m[:, 2] += rng.uniform(-0.06, 0.06, 2) * (w, h)
    out = cv2.warpAffine(crop, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-18, 18), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    return out


class JewelIdModel:
    """Per-session jewel identities (16-10): each jewel captured when the stone was placed is a class, plus a
    "not a jewel" class from bench and stone crops (rub marks). ImageNet ResNet18 (frozen) turns a crop into 512
    features; a linear softmax classifier on them is trained from ID_SAMPLES augmented copies per jewel."""

    def __init__(self):
        self.session = self.net = None
        onnx_path = os.path.join(SCRIPT_DIR, "weights", RESNET18_ONNX)
        if os.path.exists(onnx_path):
            self.session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        else:  # torch fallback
            import torchvision
            net = torchvision.models.resnet18(weights=None)
            path = os.path.join(SCRIPT_DIR, "weights", RESNET18_FILE)
            if not os.path.exists(path):
                path = os.path.join(torch.hub.get_dir(), "checkpoints", RESNET18_FILE)
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            net.fc = torch.nn.Identity()
            self.net = net.eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.labels, self.W, self.b = [], None, None
        self.ready = False

    def features(self, crops, batch=32):
        out = []
        for k in range(0, len(crops), batch):
            xs = []
            for c in crops[k:k + batch]:
                h, w = c.shape[:2]
                side = max(h, w)
                sq = cv2.copyMakeBorder(c, (side - h) // 2, side - h - (side - h) // 2, (side - w) // 2,
                                        side - w - (side - w) // 2, cv2.BORDER_REPLICATE)
                x = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
                xs.append(((x - self.mean) / self.std).transpose(2, 0, 1))
            if self.session is not None:
                f = self.session.run(None, {self.session.get_inputs()[0].name: np.stack(xs).astype(np.float32)})[0]
            else:
                with torch.inference_mode():
                    f = self.net(torch.from_numpy(np.stack(xs))).numpy()
            out.append(f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-6))
        return np.concatenate(out)

    def train(self, crops_by_id, negatives, seed=0):
        """crops_by_id: {jewel id: capture crop}; negatives: bench / stone crops. Returns the training accuracy
        on fresh augmented copies (a self-check)."""
        rng = np.random.default_rng(seed)
        labels = sorted(crops_by_id) + [None]
        xs, ys = [], []
        for k, jid in enumerate(labels[:-1]):
            c = crops_by_id[jid]
            xs += [c] + [augment(c, rng) for _ in range(ID_SAMPLES)]
            ys += [k] * (ID_SAMPLES + 1)
        for c in negatives:
            xs.append(c)
            ys.append(len(labels) - 1)
        X = torch.from_numpy(self.features(xs))
        y = torch.tensor(ys)
        lin = torch.nn.Linear(X.shape[1], len(labels))
        opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lin(X) * 10.0, y)  # x10: features are unit length
            loss.backward()
            opt.step()
        self.labels = labels
        self.W, self.b = lin.weight.detach().numpy() * 10.0, lin.bias.detach().numpy() * 10.0
        check = [augment(crops_by_id[j], rng) for j in labels[:-1] for _ in range(6)]
        truth = [j for j in labels[:-1] for _ in range(6)]
        pred = [self.predict_features(f)[0] for f in self.features(check)]
        self.ready = True
        return float(np.mean([a == b for a, b in zip(pred, truth)]))

    def predict_features(self, f):
        z = self.W @ f + self.b
        p = np.exp(z - z.max())
        p /= p.sum()
        k = int(np.argmax(p))
        return self.labels[k], float(p[k])

    def predict(self, crop):
        """(jewel id or None for "not a jewel", probability)"""
        if not self.ready or crop is None or crop.size == 0:
            return None, 0.0
        return self.predict_features(self.features([crop])[0])


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


class StoneWatcher(threading.Thread):
    """Predict stones with the INT8 OpenVINO model on each latest camera frame.

    A missed, expired or failed prediction clears the automatic box. No dark
    shape search, brightness/shape validation or retained occlusion box is used.
    """

    def __init__(self, grabber, model, roi_fn):
        super().__init__(daemon=True)
        self.grabber, self.model, self.roi_fn = grabber, model, roi_fn
        self.bbox = None
        self.found_t = 0.0
        self.running = True

    @property
    def present(self):
        return self.bbox is not None and time.time() - self.found_t < STONE_PRESENT_S

    def run(self):
        last_ts = None
        while self.running:
            frame, ts = self.grabber.latest(copy_frame=False)
            if frame is None or time.time() - ts > STONE_RESULT_MAX_AGE_S:
                self.bbox, self.found_t = None, 0.0
                time.sleep(0.005)
                continue
            if ts == last_ts:
                time.sleep(0.005)  # wait only when the camera has no new frame
                continue
            last_ts = ts
            try:
                self.update(frame, captured_at=ts)
            except Exception as exc:
                self.bbox, self.found_t = None, 0.0
                write_log(f"STONE ERROR  OpenVINO prediction failed: {exc}")
            # Immediately take the newest frame; never queue frames or repeat one.

    def update(self, frame, captured_at=None):
        observed_at = time.time() if captured_at is None else captured_at
        roi = self.roi_fn()
        roi = roi.copy() if roi is not None else None
        box = self.model_box(frame, roi)
        current_roi = self.roi_fn()
        same_roi = ((roi is None and current_roi is None)
                    or (roi is not None and current_roi is not None
                        and np.array_equal(roi, current_roi)))
        if time.time() - observed_at > STONE_RESULT_MAX_AGE_S or not same_roi:
            box = None
        self.bbox = box
        self.found_t = observed_at if box is not None else 0.0

    def model_box(self, frame, roi=None):
        """Highest-confidence stone detection whose centre lies in the ROI."""
        if self.model is None:
            return None
        r = self.model.predict(frame, imgsz=640, conf=STONE_CONF, device=DEVICE, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        h, w = frame.shape[:2]
        best = None
        for b, confidence in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if confidence < STONE_CONF or not np.isfinite(b).all():
                continue
            x1, y1, x2, y2 = np.clip(b, (0, 0, 0, 0), (w, h, w, h)).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            centre = (float(x1 + x2) / 2, float(y1 + y2) / 2)
            if roi is not None and cv2.pointPolygonTest(roi, centre, False) < 0:
                continue
            if best is None or confidence > best[0]:
                best = (float(confidence), (int(x1), int(y1), int(x2), int(y2)))
        return best[1] if best else None


def stone_observation_matches(before, after):
    """Accept normal detector jitter; a moved stone only invalidates this sample."""
    if before is None or after is None:
        return False
    a, b = np.asarray(before, float), np.asarray(after, float)
    diagonal = 0.5 * (np.linalg.norm(a[2:] - a[:2]) + np.linalg.norm(b[2:] - b[:2]))
    displacement = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
    return (box_iou(before, after) >= RUB_STONE_MIN_IOU
            and displacement <= RUB_STONE_MAX_CENTER_RATIO * max(diagonal, 1.0))


def rubbing_evidence(polys, confs, frame_shape, stone):
    """Clip full-frame model masks to the current stone; never invoke a model."""
    h, w = frame_shape[:2]
    sx1, sy1, sx2, sy2 = np.clip(stone, (0, 0, 0, 0), (w, h, w, h)).astype(int)
    best = None
    for poly, confidence in zip(polys, confs):
        if poly is None or len(poly) < 3 or confidence < RUB_GOLD_CONF:
            continue
        px, py, pw, ph = cv2.boundingRect(poly.astype(np.int32))
        x1, y1, x2, y2 = max(0, px), max(0, py), min(w, px + pw), min(h, py + ph)
        ix1, iy1, ix2, iy2 = max(x1, sx1), max(y1, sy1), min(x2, sx2), min(y2, sy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        # Local mask allocation avoids a full-camera-sized buffer per detection.
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(mask, [(poly - (x1, y1)).astype(np.int32)], 1)
        raw = int(mask.sum())
        if raw < RUB_MIN_MASK_PX:
            continue
        on_stone = np.zeros_like(mask)
        on_stone[iy1-y1:iy2-y1, ix1-x1:ix2-x1] = mask[iy1-y1:iy2-y1, ix1-x1:ix2-x1]
        on_stone = largest_component(on_stone)
        n = int(on_stone.sum())
        if n < RUB_MIN_MASK_PX or n / raw < RUB_MIN_INSIDE_RATIO:
            continue
        score = float(confidence) * (n / raw) * n
        if best is None or score > best[0]:
            best = (score, on_stone, (x1, y1))
    if best is None:
        return None
    _, mask, (ox, oy) = best
    m = cv2.moments(mask, binaryImage=True)
    ys, xs = np.nonzero(mask)
    return dict(c=np.array([ox + m['m10'] / m['m00'], oy + m['m01'] / m['m00']]),
                box=(ox + xs.min(), oy + ys.min(), ox + xs.max() + 1, oy + ys.max() + 1),
                mask=mask, origin=(ox, oy), stone=tuple(stone))


def projected_rub_geometry(tr, stone):
    """Place saved rubbing masks on the stone's current translation and scale."""
    marks = getattr(tr, 'rub_marks', None)
    if not marks or stone is None:
        return list(tr.rub_boxes), list(tr.rub_masks)
    current = np.asarray(stone, float)
    boxes, masks = [], []
    for mask, origin, recorded_stone in marks:
        previous = np.asarray(recorded_stone, float)
        scale = np.maximum(current[2:] - current[:2], 1.) / np.maximum(previous[2:] - previous[:2], 1.)
        origin = np.rint(current[:2] + (np.asarray(origin) - previous[:2]) * scale).astype(int)
        size = np.maximum(1, np.rint(np.asarray(mask.shape[::-1]) * scale).astype(int))
        if tuple(size) != mask.shape[::-1]:
            mask = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(mask)
        if w and h:
            ox, oy = map(int, origin)
            boxes.append((ox+x, oy+y, ox+x+w, oy+y+h))
            masks.append((mask, (ox, oy)))
    return boxes, masks


class RubbingObserver:
    """Consume the same full-frame observation as tracking, on the worker thread.

    Motion and box jitter do not reset history. Invalid/missed observations
    cannot contribute credit. Stone disappearance and explicit Reset clear it.
    """
    def __init__(self, stone_box, generation_fn=None):
        self.stone_box = stone_box
        self.generation_fn = generation_fn or (lambda: 0)
        self.active = False
        self.reset()

    def snapshot(self):
        return self.ev, self.rubbing, self.ts

    def reset(self):
        self.moves = deque(maxlen=10)
        self.prev = None
        self.motion_stone = self.reference_size = None
        self.ev, self.rubbing, self.ts = None, False, 0.0

    def suspend(self, ts):
        # Preserve progress, but do not bridge an unknown movement with a jump.
        self.prev = None
        self.ev, self.rubbing, self.ts = None, False, ts

    def observe(self, polys, confs, frame_shape, ts, captured_stone, revision):
        if revision != self.generation_fn():
            return  # explicit Reset already owns the new generation
        stone = self.stone_box()
        if stone is None:
            self.reset()
            return
        if not 0 <= time.time() - ts <= LIVE_RESULT_MAX_AGE_S:
            self.suspend(ts)
            return
        if not stone_observation_matches(captured_stone, stone):
            self.suspend(ts)
            return
        ev = rubbing_evidence(polys, confs, frame_shape, stone)
        self.judge(ev, ts)

    def judge(self, ev, ts):
        self.ev, self.rubbing, self.ts = ev, False, ts
        if ev is None:
            self.prev = None
            return  # no gold mask: preserve history, never display an old mask
        stone = np.asarray(ev['stone'], float)
        if self.motion_stone is not None:
            previous = self.motion_stone
            displacement = np.linalg.norm((stone[:2] + stone[2:] - previous[:2] - previous[2:]) / 2)
            if box_iou(stone, previous) >= .95 and displacement <= RUB_STONE_JITTER_PX:
                stone = previous
        self.motion_stone = stone
        size = np.maximum(stone[2:] - stone[:2], 1.)
        if self.reference_size is None:
            self.reference_size = size.copy()
        # Translation and scale of the stone do not count as jewel movement.
        point = (ev['c'] - stone[:2]) / size * self.reference_size
        previous, self.prev = self.prev, (point, ts)
        if not self.active or previous is None or not 0 < ts - previous[1] < RUB_GAP_S:
            return
        move = float(np.linalg.norm(point - previous[0]))
        self.moves.append((ts, move))
        recent = [distance for when, distance in self.moves if 0 <= ts - when < RUB_GAP_S][-RUB_MOVE_WINDOW:]
        self.rubbing = (len(recent) == RUB_MOVE_WINDOW and max(recent) >= 1.2 * RUB_MOVE_PX
                        and sum(recent) / RUB_MOVE_WINDOW >= .6 * RUB_MOVE_PX)


def to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class Worker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    jewel_list = QtCore.pyqtSignal(list)  # ["J1. Chain / Necklace", ...] from the capture
    jewel_table = QtCore.pyqtSignal(list)  # [(id, jewel crop, rubbed, rub spot image, acid tested, done), ...]
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
        self.camera = camera          # camera settings (Dahua ip/user/password, or USB index/resolution) or None
        self.grabber = None
        # Match overlay expiry to the accepted observation age, including inference.
        self.preview = LatestPreview(max_age=LIVE_RESULT_MAX_AGE_S)
        self.flow_revision = 0
        self.flicker_meter = FlickerMeter()
        self.af_before, self.af_check_at = None, None
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
        self.gold_observation = ([], np.empty(0))
        self.stone = None         # ONNX-only automatic stone watcher
        self.stone_drawn = None   # (x1, y1, x2, y2) of the drawn stone
        self.set_stone(stone_pts)
        self.siglip = None        # optional SigLIP2 for the appearance fingerprint
        self.rubber = None        # consumes full-frame masks without a second inference
        self.rub_ev = None        # gold on the stone in the last checked frame
        self.rubbing_now = False
        self.rub_track = None     # jewel the current rubbing is credited to
        self.rub_reference_ready = False
        self.rub_cycle_id = 0
        self.stone_cleanup_pending = False
        try:
            self.id_model = JewelIdModel()  # per-session jewel IDs (ResNet18), trained at lock
        except Exception as e:
            self.id_model = None
            print(f"[jewel ID model] not available: {e}")
        self.focus = False        # rubbing focus: resting jewels held, jewel YOLO only every RUB_FOCUS_CHECK_S
        self.focus_resting = set()  # ids of the jewels resting on the bench during the focus
        self.focus_last_rub = self.focus_check_t = self.focus_block_until = 0.0
        self.acid_pending_id = None
        self.acid_ph_required = True  # snapshotted for each acid-test attempt
        self.acid_token = None
        self.acid_batch_id = uuid.uuid4().hex
        self.id_training_thread = None
        self.prev_dets = []       # last frame's detections with fingerprints (reused if unmoved)
        self.prev_frame, self.prev_t = None, 0.0  # for starting CSRT where the jewel was last detected

    # ---- called from the GUI thread ----
    def set_zone(self, kind, pts):  # kind: "roi" or "stone"
        with self.lock:
            self.flow_revision += 1
            self.preview.clear()
            self.invalidate_rub_reference()
            if kind == "stone":
                self.set_stone(pts)
            else:
                setattr(self, kind, to_poly(pts))

    def invalidate_rub_reference(self, keep_cycle=False):
        self.rub_reference_ready = False
        if not keep_cycle:
            self.rub_cycle_id += 1
        self.rub_track = None
        for tr in self.tracks:
            if not tr.rubbed:
                tr.rub_time, tr.rub_last = 0., 0.
                for field in ('rub_boxes', 'rub_masks', 'rub_marks'):
                    marks = getattr(tr, field, None)
                    if marks is not None:
                        marks.clear()

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
            self.stone_cleanup_pending = False
            self.invalidate_rub_reference()
            self.flow_revision += 1
            self.preview.clear()
            if self.rubber is not None:
                self.rubber.active = False
                self.rubber.reset()
            self.arm_frames.clear()
            self.arm_start = self.persist_key = None
            self.rub_track = None
            self.acid_pending_id = self.acid_token = None
            self.acid_batch_id = uuid.uuid4().hex

    def stop(self):
        self.running = False
        if self.grabber is not None:
            self.grabber.stop()
        for watcher in (self.stone,):
            if watcher is not None:
                watcher.running = False

    def request_antiflicker(self, af):
        if self.grabber is not None:
            self.grabber.request_antiflicker(af)

    def antiflicker_applied(self, msg):
        """From the camera thread: log the read-back, then compare the measured flicker AF_CHECK_AFTER_S later."""
        self.event.emit(msg, "info")
        self.af_before = self.flicker_meter.value()
        self.flicker_meter.reset()
        self.af_check_at = time.time() + AF_CHECK_AFTER_S

    # ---- worker thread ----
    def run(self):
        self.status.emit("Loading model...")
        try:
            model = self.model = SharedGoldModel(MODEL_PATH)
        except Exception as e:
            self.event.emit(f"Model load failed: {e}", "alert")
            self.status.emit("Model load failed (see Events)")
            return
        # 17-10: no jewel type (bangle, chain...) any more - the ID (J1, J2...) is enough; SigLIP2 is not loaded
        stone_model = None
        try:
            stone_model = StoneOnnxModel(STONE_MODEL_PATH)
            self.event.emit("Stone: INT8_openvino_model/best.xml | OpenVINO CPU, INT8, 640, 2 threads, continuous", "info")
        except Exception as e:
            self.event.emit(f"Stone INT8 OpenVINO model unavailable: {e}", "alert")
        if not self.running:
            model.close()
            return
        grabber = self.grabber = RTSPGrabber(self.source, self.camera)
        grabber.on_af = self.antiflicker_applied
        grabber.start()
        try:
            self.stone = StoneWatcher(grabber, stone_model, lambda: self.roi)
            self.stone.start()
            self.rubber = RubbingObserver(self.stone_box, generation_fn=lambda: self.flow_revision)
            last_ts, fps, t_prev = 0.0, 0.0, time.time()
            stream_alerted, last_status = False, 0.0

            while self.running:
                # Camera frames are immutable: only preview consumers need a copy.
                frame, ts = grabber.latest(copy_frame=False)
                now = time.time()
                if frame is None or now - ts > LIVE_RESULT_MAX_AGE_S or ts == last_ts:
                    with self.lock:
                        locked = self.mode == "LOCKED"
                    if (locked and last_ts and now - last_ts > STREAM_TIMEOUT_S and not stream_alerted
                            and now > grabber.af_busy_until):
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
                self.flicker_meter.add(frame)

                with self.lock:
                    revision = self.flow_revision
                    self.ensure_roi()
                    roi = self.roi.copy()
                    captured_stone = self.stone_box()
                    focus = self.mode == "LOCKED" and self.rub_focus(ts)
                    skip = focus and ts - self.focus_check_t < RUB_FOCUS_CHECK_S  # rubbing: no jewel YOLO now
                if skip:
                    dets, rejected = [], self.rejected
                else:
                    try:
                        dets, rejected = self.detect(model, frame, roi, captured_at=ts)
                    except StaleInference:
                        continue  # no observation, never a false missing detection
                    if focus:
                        self.focus_check_t = ts
                with self.lock:
                    if (not self.running or revision != self.flow_revision
                            or time.time() - ts > LIVE_RESULT_MAX_AGE_S):
                        continue
                    self.rejected = rejected
                    if not skip:
                        self.rubber.active = (self.mode == "LOCKED" and self.acid_pending_id is None
                                              and self.rub_reference_ready)
                        polys, confs = self.gold_observation
                        self.rubber.observe(polys, confs, frame.shape, ts, captured_stone, revision)
                    if self.mode == "ARMING":
                        self.try_lock(dets, ts, frame)
                    elif self.mode == "LOCKED" and focus:
                        self.update_focus(None if skip else dets, ts, frame)
                    elif self.mode == "LOCKED":
                        self.update_tracks(dets, ts, frame)
                    # Publish drawing only; the GUI supplies fresh camera pixels.
                    overlay = self.draw(np.zeros_like(frame), dets, ts)
                    self.preview.publish(overlay, ts)
                    mode = self.mode

                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
                t_prev = now
                flick = self.flicker_meter.value()
                self.status.emit(f"{mode}  |  {fps:.1f} detection FPS" + (f"  |  flicker: banding {flick[0]:.1f}%, "
                                                                f"pulsing {flick[1]:.1f}%" if flick else ""))
                if self.af_check_at and now >= self.af_check_at and flick:
                    self.af_check_at = None
                    before = self.af_before
                    self.event.emit("Anti-flicker check on the live feed: " + (
                        f"banding {before[0]:.2f}% -> {flick[0]:.2f}%, pulsing {before[1]:.2f}% -> {flick[1]:.2f}%, "
                        f"brightness {before[2]:.0f} -> {flick[2]:.0f}" if before else
                        f"banding {flick[0]:.2f}%, pulsing {flick[1]:.2f}%, brightness {flick[2]:.0f}"), "info")
        finally:
            grabber.stop()
            for watcher in (self.stone,):
                if watcher is not None:
                    watcher.running = False
            # Native OpenCV/ORT/PyTorch work must end before Python/Qt tears
            # down their objects. The GUI keeps this Worker alive until done.
            for watcher in (self.stone, grabber, self.id_training_thread):
                if watcher is not None and watcher.is_alive():
                    watcher.join()
            model.close()

    def detect(self, model, frame, roi, imgsz=None, captured_at=None):
        """One unmodified full-frame prediction; ROI filtering happens afterward.

        Raw model polygons are shared with stone masking before bench-only
        colour/skin filtering. Returns the usual (kept, rejected) bench lists.
        """
        h, w = frame.shape[:2]
        key = roi.tobytes() + bytes(str((w, h)), 'ascii')
        if self.roi_mask_key != key:
            self.roi_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(self.roi_mask, [roi.astype(np.int32)], 255)
            margin = OUTSIDE_MARGIN_PX
            self.roi_mask = cv2.dilate(self.roi_mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
            self.roi_mask_key = key
            bx, by, bw, bh = cv2.boundingRect(roi)
            if bx < 0 or by < 0 or bx + bw > w or by + bh > h:
                self.event.emit(f"The ROI reaches outside this camera's {w}x{h} image: redraw it", "info")
        r = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False,
                          _live_job='bench', _captured_at=captured_at)[0]
        kept, rejected = [], []
        self.gold_observation = ([], np.empty(0))
        if r.boxes is None or len(r.boxes) == 0:
            return kept, rejected
        confs = r.boxes.conf.cpu().numpy()
        polys = r.masks.xy if r.masks is not None else [None] * len(confs)
        self.gold_observation = (polys, confs)
        for b, cf, poly in zip(r.boxes.xyxy.cpu().numpy(), confs, polys):
            x1, y1, x2, y2 = np.clip(b, (0, 0, 0, 0), (w, h, w, h)).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            mask = self.roi_mask[y1:y2, x1:x2].copy()
            c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            area = max(float((b[2] - b[0]) * (b[3] - b[1])), 1.)
            if poly is not None and len(poly) >= 3:
                shape = np.zeros_like(mask)
                cv2.fillPoly(shape, [(poly - (x1, y1)).astype(np.int32)], 255)
                mask &= shape
                m = cv2.moments(poly.astype(np.float32))
                if m['m00'] > 1:
                    c = np.array([m['m10'] / m['m00'], m['m01'] / m['m00']])
                    area = m['m00']
            gold = gold_fraction(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV), mask)
            det = Det(np.array([x1, y1, x2, y2]), c, float(area), float(cf), gold or 0., mask, (x1, y1))
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
        self.tracks = [Track(i + 1, d, t) for i, (d, *_) in enumerate(jewels)]
        if frame is not None and self.id_model is not None:
            self.learn_ids(frame)
        for tr, (d, item, _, _) in zip(self.tracks, jewels):
            # the jewel as captured, shown in the status table
            tr.thumb = (item or {}).get("thumb")
            if tr.thumb is None and frame is not None:
                tr.thumb = crop_box(frame, d.box).copy()
        self.mode = "LOCKED"
        self.event.emit(f"Locked {len(jewels)} jewel(s)", "info")
        self.emit_jewel_list()
        if self.rubber is not None and self.stone_box() is None:
            self.event.emit("Touchstone not found yet: rubbing can't be checked", "info")

    def learn_ids(self, frame):
        """Train the per-session ID model from the jewels as captured (in the background, a few seconds)."""
        crops = {tr.id: crop_box(frame, tr.box).copy() for tr in self.tracks}
        negatives = self.negative_crops(frame)
        self.id_model.ready = False

        def train():
            t0 = time.time()
            try:
                acc = self.id_model.train(crops, negatives)
                self.event.emit(f"Jewel IDs learned (ResNet18, {len(crops)} jewels + bench/stone, "
                                f"{time.time() - t0:.0f} s, self-check {acc:.0%})", "info")
            except Exception as e:
                self.event.emit(f"Jewel ID model could not be trained: {e}", "alert")
        self.id_training_thread = threading.Thread(target=train, daemon=True)
        self.id_training_thread.start()

    def negative_crops(self, frame, n=ID_NEG_SAMPLES):
        """Bench and stone (rub marks) crops of jewel-like sizes: the "not a jewel" class."""
        rng = np.random.default_rng(1)
        x0, y0, w, h = cv2.boundingRect(self.roi)
        fh, fw = frame.shape[:2]
        sizes = [(b[2] - b[0], b[3] - b[1]) for b in (tr.box for tr in self.tracks)] or [(150, 150)]
        stone = self.stone_box()
        out, tries = [], 0
        while len(out) < n and tries < 40 * n:
            tries += 1
            bw, bh = sizes[rng.integers(len(sizes))]
            s = rng.uniform(0.5, 1.2)
            bw, bh = max(24, int(bw * s)), max(24, int(bh * s))
            if stone is not None and len(out) < n // 3:  # a third on the stone, where rub marks are
                cx, cy = rng.uniform(stone[0], stone[2]), rng.uniform(stone[1], stone[3])
            else:
                cx, cy = rng.uniform(x0, x0 + w), rng.uniform(y0, y0 + h)
            box = (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2))
            if box[0] < 0 or box[1] < 0 or box[2] > fw or box[3] > fh:
                continue
            if any(box_iou(box, tr.box) > 0.0 for tr in self.tracks):
                continue
            out.append(crop_box(frame, box, pad=0).copy())
        return out

    def identify(self, frame, d):
        """ID model on a detection, once per standing-still episode."""
        if d.idres is None:
            d.idres = self.id_model.predict(crop_box(frame, d.box))
        return d.idres

    def emit_jewel_list(self):
        self.jewel_list.emit([f"J{tr.id}" for tr in sorted(self.tracks, key=lambda x: x.id)])
        self.emit_jewel_table()

    def emit_jewel_table(self):
        """Status table: updated when a jewel is locked, renamed, rubbed, acid tested or DONE."""
        self.jewel_table.emit([(tr.id, getattr(tr, "thumb", None), tr.rubbed, getattr(tr, "rub_img", None),
                                tr.acid_done, tr.done, tr.acid_result, tr.id == self.acid_pending_id, getattr(tr, "acid_crop", None))
                               for tr in sorted(self.tracks, key=lambda x: x.id)])

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
            item = {"bbox": bbox, "crop_bgr": crop, "source": "",
                    "thumb": crop_box(frame, d.box).copy()}  # the jewel as it lies, for the status table
            jewels.append((d, item, "", 0.0))
        self.event.emit(f"Capture: YOLO counted {len(jewels)} jewel(s)", "info")
        self.save_capture(frame, jewels)
        return jewels

    def save_capture(self, frame, jewels):
        """Evidence like the repos: the annotated capture and each jewel's Otsu crop, in captures/."""
        try:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            annotated = frame.copy()
            for i, (_, item, label, conf) in enumerate(jewels, 1):
                x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 184, 166), 3)
                cv2.putText(annotated, f"J{i}", (x1, max(24, y1 - 8)), FONT, 0.9, (20, 184, 166), 2)
                cv2.imwrite(os.path.join(CAPTURE_DIR, f"{stamp}_J{i}.png"), item["crop_bgr"])
            cv2.imwrite(os.path.join(CAPTURE_DIR, f"{stamp}_capture.jpg"), annotated)
        except Exception as e:
            self.event.emit(f"Could not save the capture: {e}", "info")

    def update_tracks(self, dets, t, frame=None):
        """v4 (16-10): rojeweltrack finds the jewels; boxes are followed by position from frame to frame; a box that
        stands still is identified by the per-session ResNet18 ID model (puts a jewel back under its ID, and
        corrects IDs swapped when jewels slid over each other). Closed world: no new IDs after lock."""
        self.cur_frame = frame
        if frame is not None:
            self.update_motion(frame, t)
        self.mark_stable(dets, t)
        free = list(range(len(dets)))
        match = {}  # track index -> detection index
        ready = self.id_model is not None and self.id_model.ready and frame is not None
        # 1: jewels seen a moment ago -> by position (Hungarian on distance + size)
        active = [i for i, tr in enumerate(self.tracks) if t - tr.det_seen < TRACK_KEEP_S]
        for i, j in self.assign(active, free, dets, t, self.motion_cost):
            match[i] = j
            free.remove(j)
        # the ID model overrules position: a jewel followed onto a box the model is sure is "not a jewel" (clutter,
        # a rub mark) lets go of it; boxes outside the ROI are checked before they can raise "left the ROI"
        if ready:
            for i, j in list(match.items()):
                d = dets[j]
                if not (self.is_stable(d, t) or self.roi_dist(d.c) < 0):
                    continue
                jid, p = self.identify(frame, d)
                if jid is None and p >= ID_MIN_PROB:
                    del match[i]
        # 2: a standing-still detection nobody took -> the jewel the ID model says it is (put back after being
        # carried / rubbed). If that jewel is on another box that the model does not see as it (still, or outside
        # the ROI), it moves to the recognised one.
        if ready:
            for j in list(free):
                d = dets[j]
                if not self.is_stable(d, t) or self.roi_dist(d.c) <= 0:
                    continue  # outside the ROI a box is never given a jewel's ID (only followed out, step 1)
                jid, p = self.identify(frame, d)
                i = next((k for k, tr in enumerate(self.tracks) if tr.id == jid), None)
                if i is None or p < ID_MIN_PROB:
                    continue
                if i in match:
                    other = dets[match[i]]
                    if not (self.is_stable(other, t) or self.roi_dist(other.c) < 0)                             or self.identify(frame, other)[0] == jid:
                        continue
                    free.append(match[i])  # hand the wrong box back
                match[i] = j
                free.remove(j)
        # 3: a lost jewel whose last box lies inside a detection nobody took is that jewel (its track followed a
        # piece of it, or it was put back on the same spot); not on the stone (rub marks)
        for i in sorted((i for i in range(len(self.tracks)) if i not in match), key=lambda i: -self.tracks[i].area):
            tr, best = self.tracks[i], None
            bx = tr.box
            area = max(1.0, float((bx[2] - bx[0]) * (bx[3] - bx[1])))
            for j in free:
                d = dets[j]
                if self.on_stone(d.c, margin=0) or self.roi_dist(d.c) <= 0:
                    continue  # not on the stone (rub marks), not outside the ROI
                ix = min(bx[2], d.box[2]) - max(bx[0], d.box[0])
                iy = min(bx[3], d.box[3]) - max(bx[1], d.box[1])
                cover = max(0, ix) * max(0, iy) / area
                if cover >= REACQ_COVER and (best is None or cover > best[0]):
                    best = (cover, j)
            if best is not None:
                match[i] = best[1]
                free.remove(best[1])
        # closed world: if only one jewel is out of sight, a still, unclaimed box on the bench that the ID model does
        # not call "not a jewel" is that jewel (16-10: an earring put down a little away was left unclaimed)
        lost = [i for i in range(len(self.tracks)) if i not in match]
        if ready and len(lost) == 1:
            tr = self.tracks[lost[0]]
            cands = []
            for j in free:
                d = dets[j]
                if (not self.is_stable(d, t) or self.roi_dist(d.c) <= 0 or self.on_stone(d.c, margin=0)
                        or not 0.25 <= d.area / max(tr.area, 1.0) <= 4.0):
                    continue
                jid, p = self.identify(frame, d)
                if jid is None and p >= ID_MIN_PROB:
                    continue  # the model is sure it is not a jewel
                if jid is not None and jid != tr.id and p >= ID_MIN_PROB:
                    continue  # the model is sure it is another jewel
                cands.append(j)
            if cands:
                j = min(cands, key=lambda j: float(np.hypot(*(dets[j].c - tr.pos))))
                match[lost[0]] = j
                free.remove(j)
        # 4: IDs swapped? a followed jewel that stands still is checked once by the ID model
        if ready:
            self.check_swaps(match, dets, frame, t)
        # apply, with the movement score (how much the centre jitters: the rubbed jewel moves most)
        seen = set(match)
        for i, tr in enumerate(self.tracks):
            if i in match:
                d = dets[match[i]]
                step = float(np.hypot(*(d.c - tr.pos))) if t - tr.last_seen < 0.5 else 0.0
                tr.move = (1 - MOVE_EMA) * tr.move + MOVE_EMA * step
                tr.correct(d, t)
            else:
                tr.move *= 1 - MOVE_EMA
        for i, tr in enumerate(self.tracks):
            if i in match:
                tr.spot, tr.removed_since = "jewel", None
                if frame is not None:  # how its spot looks with the jewel there
                    tr.det_patch = spot_patch(frame, tr.det_box)
            elif frame is not None and t - tr.last_seen > 0.3:
                tr.spot = self.spot_state(tr, frame)
                tr.removed_since = (tr.removed_since or t) if tr.spot == "removed" else None
        self.update_rubbing(t)
        self.update_acid(t)
        for i, tr in enumerate(self.tracks):
            self.evaluate(tr, i in seen, t)
            self.check_done(tr, i in seen, t)

    def update_motion(self, frame, t):
        """Is anything (a hand) moving over the bench? Sets self.still_since (None while moving)."""
        h, w = frame.shape[:2]
        g = cv2.cvtColor(cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
        key = (self.roi.tobytes() if self.roi is not None else b"", g.shape)
        if getattr(self, "motion_key", None) != key:
            self.motion_key, self.motion_prev = key, None
            self.motion_roi = np.zeros(g.shape, bool)
            if self.roi is not None:
                m = np.zeros(g.shape, np.uint8)
                cv2.fillPoly(m, [(self.roi.reshape(-1, 2) / 8).astype(np.int32)], 1)
                self.motion_roi = m > 0
            else:
                self.motion_roi[:] = True
            self.still_since = t
        prev, self.motion_prev = self.motion_prev, g
        if prev is None or not self.motion_roi.any():
            return
        inside = self.motion_roi
        g = g * (float(np.median(prev[inside])) / max(float(np.median(g[inside])), 1.0))  # exposure changes
        moving = float(np.mean(np.abs(g - prev)[inside] > MOTION_DIFF)) >= MOTION_MIN_FRAC
        self.moving_now = moving
        if moving:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = t

    def bench_still_for(self, t):
        since = getattr(self, "still_since", None)
        return 0.0 if since is None else t - since

    def spot_state(self, tr, frame):
        """Where a jewel out of sight was last seen: "jewel" (looks as before: still there), "removed" (its pixels
        now show white bench) or "hand" (covered: dark, skin, sandal...)."""
        if self.still_there(tr):
            return "jewel"
        m = tr.det_mask
        if m is None:
            return "hand"
        ox, oy = (int(v) for v in tr.det_origin)
        h, w = m.shape
        fh, fw = frame.shape[:2]
        region = frame[max(0, oy):min(fh, oy + h), max(0, ox):min(fw, ox + w)]
        m = m[max(0, -oy):max(0, -oy) + region.shape[0], max(0, -ox):max(0, -ox) + region.shape[1]]
        if region.size == 0 or not (m > 0).any():
            return "hand"
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[m > 0]
        # "white bench" relative to the bench right around the box (shade and light differ across the bench)
        x1, y1, x2, y2 = (int(v) for v in tr.det_box)
        pad = 40
        ring = cv2.cvtColor(frame[max(0, y1 - pad):min(fh, y2 + pad), max(0, x1 - pad):min(fw, x2 + pad)],
                            cv2.COLOR_BGR2HSV).reshape(-1, 3)
        bench = ring[(ring[:, 1] < BG_WHITE_S_MAX) & (ring[:, 2] > 90)]
        bench_v = float(np.median(bench[:, 2])) if len(bench) > 50 else float(BG_WHITE_V_MIN)
        white = (hsv[:, 1] < BG_WHITE_S_MAX) & (hsv[:, 2] > 0.85 * bench_v)
        return "removed" if float(white.mean()) >= SPOT_WHITE else "hand"

    def mark_stable(self, dets, t):
        """Each detection: since when it stands still (same place as in the previous frames)."""
        prev = getattr(self, "prev_stable", [])
        for d in dets:
            d.stable_since, d.idres = t, None
            for p in prev:
                if np.hypot(*(d.c - p.c)) <= STABLE_MOVE_PX and box_iou(d.box, p.box) >= 0.6:
                    d.stable_since = p.stable_since
                    d.idres = p.idres  # same still detection: keep its ID-model answer
                    break
        self.prev_stable = dets

    @staticmethod
    def is_stable(d, t):
        return d.stable_since is not None and t - d.stable_since >= STABLE_S

    def check_swaps(self, match, dets, frame, t):
        """A jewel that is followed and now stands still: if the ID model is sure it is another jewel, correct the
        IDs (jewels slid over each other / were swapped while in the hand)."""
        for i, j in list(match.items()):
            d = dets[j]
            if self.roi_dist(d.c) <= 0:
                continue  # a jewel followed out of the ROI keeps its ID (its "left the ROI" alert is about it)
            if not self.is_stable(d, t) or getattr(d, "idres", None) is not None and d.idres[0] == self.tracks[i].id:
                continue
            jid, p = self.identify(frame, d)
            if jid is None or jid == self.tracks[i].id or p < ID_SWAP_PROB:
                continue
            k = next((n for n, tr in enumerate(self.tracks) if tr.id == jid), None)
            if k is None:
                continue
            if k in match:  # the other jewel is visible too: swap only if its detection does not look like it
                other = dets[match[k]]
                if self.is_stable(other, t) and self.identify(frame, other)[0] == jid:
                    continue
                match[i], match[k] = match[k], j
            else:
                match[k] = j
                del match[i]
            self.tracks[k].last_seen = t - 1.0  # a jump: correct() resets the velocity
            self.tracks[i].last_seen = min(self.tracks[i].last_seen, t - 1.0)
            self.event.emit(f"IDs corrected: this jewel is J{jid} (ResNet18 {p:.2f}), "
                            f"not J{self.tracks[i].id}", "info")

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

    def motion_cost(self, tr, d, t):
        dist, inside = self.gate_dist(tr, d, t)
        if not inside:
            return 1e9
        if self.roi_dist(d.c) < 0 and (t - tr.last_seen > OUT_FOLLOW_GAP_S or dist > OUT_FOLLOW_PX):
            # outside the ROI only by being followed there step by step; never by a search jump (16-10: J3's
            # track jumped onto gold-looking clutter beside the bench -> false "left the ROI")
            return 1e9
        if self.on_stone(d.c, margin=0) and not self.on_stone(tr.pos):
            # onto the stone only by moving there (carried / rubbed), never by jumping onto a still rub mark
            if self.is_stable(d, t) and self.id_model is not None and self.id_model.ready \
                    and getattr(self, "cur_frame", None) is not None:
                jid, p = self.identify(self.cur_frame, d)
                if jid != tr.id:
                    return 1e9
        return dist + SIZE_WEIGHT * abs(np.log(d.area / tr.area))

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
            if self.rubber is not None:
                self.rubber.reset()
            if self.stone_box() is None:
                for tr in self.tracks:
                    if not tr.rubbed:
                        tr.rub_time, tr.rub_last = 0.0, 0.0
                        for field in ('rub_boxes', 'rub_masks', 'rub_marks'):
                            marks = getattr(tr, field, None)
                            if marks is not None:
                                marks.clear()
                self.rub_track = None
            self.rub_ev, self.rubbing_now = None, False
            return
        ev, rubbing, ts = self.rubber.snapshot()
        fresh = 0 <= t - ts < RUB_GAP_S
        # Display gold immediately on entry, independently of rubbing credit.
        self.rub_ev = ev if fresh and self.acid_pending_id is None else None
        self.rubbing_now = (rubbing and fresh and self.acid_pending_id is None
                            and self.rub_reference_ready)
        if self.acid_pending_id is not None:
            self.rubber.active = False
            return
        if not self.rubbing_now or ev is None:
            return
        tr = self.rub_track
        if tr is None or tr.acid_done or tr not in self.tracks or t - tr.rub_last > RUB_GRACE_S:
            tr = self.rub_track = self.pick_rubbed_jewel(ev, t)
        if tr is None:
            return
        if t - tr.rub_last < RUB_GAP_S:  # continuous rubbing
            tr.rub_time += t - tr.rub_last
        tr.rub_last = t
        tr.rub_boxes.append(ev["box"])
        tr.rub_masks.append((ev["mask"], ev["origin"]))
        if not hasattr(tr, 'rub_marks'):
            tr.rub_marks = deque(maxlen=30)
        tr.rub_marks.append((ev["mask"], ev["origin"], ev.get('stone', self.stone_box())))
        if not tr.rubbed and tr.rub_time >= RUB_CONFIRM_S:
            tr.rubbed = True
            self.acid_pending_id = tr.id
            self.acid_token = None
            self.rubber.active = False
            tr.rub_img = self.rub_spot_image(ev)
            self.event.emit(f"J{tr.id} RUBBED: return it to the bench, outside the stone, and let it settle for testing", "info")
            self.emit_jewel_table()

    def rub_spot_image(self, ev):
        """Freeze an unannotated crop from the current rubbing observation's frame."""
        frame = getattr(self, "cur_frame", None)
        if frame is None:
            return None
        # Use this observation's box, not historical masks projected onto a new frame.
        x1, y1, x2, y2 = map(int, ev["box"])
        h, w = frame.shape[:2]
        pad = 40  # include the hand and stone around the jewellery
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

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
                seen.add(i)
        self.update_acid(t)
        for i, tr in enumerate(self.tracks):
            self.evaluate(tr, i in seen, t)
            self.check_done(tr, i in seen, t)

    def pick_rubbed_jewel(self, ev, t):
        """Only a jewel that left its place can be rubbed (17-10: the RUBBED label landed on a jewel lying beside the
        stone - its box jumped about under the hand, so it looked like it was moving). In order:
        a jewel seen ON the stone (centre inside it) that moves; else the jewel out of sight (in the hand) that went
        out of sight last; else a jewel lying on the stone. A jewel lying beside the stone is never credited."""
        on_stone = [tr for tr in self.tracks if not tr.acid_done and t - tr.det_seen <= 0.5 and self.on_stone(tr.pos, margin=0)]
        movers = [tr for tr in on_stone if tr.move > 2.0]
        if movers:
            return max(movers, key=lambda tr: (not tr.rubbed, tr.move))
        in_hand = [tr for tr in self.tracks if not tr.acid_done and t - tr.det_seen > 0.3 and tr.state not in ("LEFT_ROI", "MISSING")]
        if in_hand:
            return max(in_hand, key=lambda tr: (not tr.rubbed, tr.det_seen))  # the one picked up last
        if on_stone:
            return max(on_stone, key=lambda tr: (not tr.rubbed, tr.move))
        return None

    # ---- local temporal acid test, bound to the confirmed rubbed jewel ----
    def update_acid(self, t):
        # Capture/detection/classification runs on AcidTestThread after the click.
        # Legacy acid-application detection must never complete a jewel here.
        return

    def set_new_rub_target(self, jewel_id, target):
        with self.lock:
            tr = next((tr for tr in self.tracks if tr.id == jewel_id), None)
            if (tr is None or not tr.rubbed or tr.acid_done or self.acid_pending_id != jewel_id
                    or self.acid_token is not None):
                raise RuntimeError("This rubbing check no longer belongs to the pending jewel")
            tr.new_rub_target = target
            tr.acid_roi_mask, tr.acid_roi_stone = target.roi_mask.copy(), target.stone
            # Keep the rubbing photo; the confirmed mark remains in the acid target.
            self.emit_jewel_table()

    def update_acid_target(self, payload):
        with self.lock:
            if payload['token'] != self.acid_token or payload['jewel_id'] != self.acid_pending_id:
                return False
            tr = next((tr for tr in self.tracks if tr.id == self.acid_pending_id), None)
            if tr is None or getattr(tr, 'new_rub_target', None) is None:
                return False
            tr.acid_roi_mask = payload['roi_mask'].copy()
            tr.acid_roi_stone = tuple(payload['stone'])
            self.preview.clear()
            return True

    def begin_acid_test(self, ph_verification_enabled=True):
        with self.lock:
            if self.acid_token is not None:
                raise RuntimeError("An acid test is already running")
            tr = next((tr for tr in self.tracks if tr.id == self.acid_pending_id), None)
            if tr is None or not tr.rubbed or tr.acid_done:
                raise RuntimeError("Rub a jewel first")
            target = getattr(tr, 'new_rub_target', None)
            if target is None:
                raise RuntimeError("Confirm the new rubbed mask before starting acid verification")
            stone = target.stone  # verified by the Start capture, even if the live detector misses
            if stone is None:
                raise RuntimeError("A confirmed stone snapshot is required")
            if self.grabber is None:
                raise RuntimeError("Camera is unavailable")
            latest_frame, latest_ts = self.grabber.latest()
            if latest_frame is None or time.time() - latest_ts > STREAM_TIMEOUT_S:
                raise RuntimeError("Camera is unavailable or its last frame is stale")
            if target.roi_mask.shape != latest_frame.shape[:2] or not np.any(target.roi_mask):
                raise RuntimeError("The confirmed new-rub ROI does not match this camera frame")
            streak = target.streak
            self.acid_token = uuid.uuid4().hex
            self.acid_ph_required = bool(ph_verification_enabled)
            tr.acid_crop = None
            context = dict(jewel_id=tr.id, token=self.acid_token, batch_id=self.acid_batch_id,
                           stone=tuple(stone), streak=streak, grabber=self.grabber,
                           stone_getter=self.stone_box, ph_verification_enabled=self.acid_ph_required,
                           rub_target=target, roi_mask=target.roi_mask.copy())
            self.emit_jewel_table()
            return context

    def update_acid_crop(self, payload):
        with self.lock:
            if payload['token'] != self.acid_token or payload['jewel_id'] != self.acid_pending_id:
                return False
            tr = next((tr for tr in self.tracks if tr.id == self.acid_pending_id), None)
            if tr is None:
                return False
            tr.acid_crop = payload['crop']
            return True

    def finish_acid_test(self, payload):
        with self.lock:
            if payload['token'] != self.acid_token or payload['jewel_id'] != self.acid_pending_id:
                return False  # cancelled/reset/reconnected results cannot change a new batch
            tr = next((tr for tr in self.tracks if tr.id == self.acid_pending_id), None)
            if tr is None:
                return False
            record = payload['result']
            ph_result = record.get('ph_verification', {}).get('result')
            if (ph_result != 'ACIDIC_REACTION_VERIFIED'
                    and not (not self.acid_ph_required and ph_result == 'SKIPPED_BY_OPERATOR')):
                return False
            tr.acid_result = record
            tr.acid_done, tr.acid_box = True, tuple(record['detector_bbox'])
            self.acid_pending_id = self.acid_token = None
            self.stone_cleanup_pending = True
            self.invalidate_rub_reference()
            prediction = record['prediction']
            label = {'gold_22k': '22K GOLD', 'non_gold': 'NON-GOLD', 'invalid': 'INCONCLUSIVE'}[prediction['prediction']]
            self.event.emit(f"J{tr.id}: {label} ({prediction['confidence']:.1%}). Clean and dry the stone; its next reference will be captured automatically", "info")
            self.emit_jewel_table()
            return True

    def fail_acid_test(self, token, message):
        with self.lock:
            if token != self.acid_token:
                return False
            self.acid_token = None
            self.event.emit(f"J{self.acid_pending_id}: {message}. Prepare the streak and retry", "alert")
            self.emit_jewel_table()
            return True

    def check_done(self, tr, seen, t):
        """DONE = rubbed and acid tested, then seen off the stone lying still for DONE_SETTLE_S."""
        if not tr.rubbed or tr.done or (ACID_REQUIRED and not tr.acid_done):
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
            elif old == "LEFT_ROI" and d < EDGE_MARGIN_PX:
                new = old  # back only once well inside: a box centre hopping on the edge alarmed twice
            else:
                tr.out_frames = 0
                new = "OK"
        else:
            lost = t - tr.last_seen
            if old in ("LEFT_ROI", "MISSING"):
                new = old  # latched until the jewel is seen inside the ROI again
            elif t - tr.rub_last <= RUB_GRACE_S:
                new = "RUBBING"  # in the hand on the stone: no alert
            elif (tr.spot == "removed" and tr.removed_since is not None and t - tr.rub_last > RUB_GRACE_S
                  and t - max(tr.removed_since, tr.rub_last) >= REMOVED_ALERT_S
                  and self.bench_still_for(t) >= BENCH_STILL_S):
                new = "MISSING"  # its place shows bare bench, it was not rubbed nor put back, and no hand is moving
                                 # over the bench any more (the hands have gone)
            elif lost > 0.3:
                new = "OCCLUDED"  # out of sight: in the hand / under a hand
            else:
                new = old
        if new != old:
            tr.state = new
            if new == "LEFT_ROI":
                self.event.emit(f"J{tr.id} MISSING - left the ROI", "alert")
            elif new == "MISSING":
                self.event.emit(f"J{tr.id} MISSING - removed from the bench (not rubbed, not put back)", "alert")
            elif old in ("LEFT_ROI", "MISSING") and new == "OK":
                self.event.emit(f"J{tr.id} back on the bench", "info")

    def draw(self, vis, dets, t):
        if self.roi is not None:
            cv2.polylines(vis, [self.roi], True, ROI_COLOR, 2)
        stone = self.stone_box()
        pending = next((tr for tr in self.tracks if tr.id == self.acid_pending_id), None)
        roi_mask = getattr(pending, 'acid_roi_mask', None)
        roi_stone = getattr(pending, 'acid_roi_stone', None)
        if roi_stone is not None and getattr(pending, "new_rub_target", None) is not None:
            stone = roi_stone  # keep the confirmed region through hand/model interference
        if stone is not None:
            cv2.rectangle(vis, stone[:2], stone[2:], STONE_COLOR, 2)
            cv2.putText(vis, "STONE", (stone[0] + 5, stone[1] + 22), FONT, 0.7, STONE_COLOR, 2)
        if (roi_mask is not None and roi_mask.shape == vis.shape[:2] and stone is not None
                and roi_stone is not None and box_iou(stone, roi_stone) >= .95):
            draw_acid_roi(vis, roi_mask)
        if self.rub_ev is not None:
            ev = self.rub_ev
            ox, oy = ev["origin"]
            region = vis[oy:oy + ev["mask"].shape[0], ox:ox + ev["mask"].shape[1]]
            region[ev["mask"] > 0] = RUB_GOLD_COLOR
            if self.rubbing_now:
                who = f" J{self.rub_track.id}" if self.rub_track is not None else ""
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
        for d in dets:  # raw detections, thin grey (on the stone they may be the jewel or a rub mark: no label)
            if self.roi is not None and self.roi_dist(d.c) <= 0:
                continue  # outside the ROI: not a jewel of ours (only used to see a tracked jewel leave)
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
                    col, label = ACID_COLOR, f"{name} ACID RESULT READY"
                    if tr.settle_since is not None:
                        label += f" {t - tr.settle_since:.0f}/{DONE_SETTLE_S:.0f}s"
                elif tr.rubbed and ACID_REQUIRED:
                    col, label = RUBBED_COLOR, f"{name} RUBBED - acid?"
                elif tr.rubbed:
                    col, label = RUBBED_COLOR, f"{name} RUBBED"
                    if tr.settle_since is not None:
                        label += f" {t - tr.settle_since:.0f}/{DONE_SETTLE_S:.0f}s"
                elif tr.state == "RUBBING":
                    label += f" {tr.rub_time:.1f}/{RUB_CONFIRM_S:.1f}s"
            x1, y1, x2, y2 = (int(v) for v in tr.box)
            if t - tr.last_seen < 0.3:  # seen now: solid box
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv2.circle(vis, tuple(int(v) for v in tr.pos), 4, col, -1)
            else:  # out of sight: dashed box where it was last seen
                away = t - max(tr.last_seen, tr.rub_last)
                if tr.state == "OCCLUDED":
                    if tr.spot == "hand":
                        label = f"{name} under a hand"
                    elif tr.spot == "removed" and tr.removed_since is not None:
                        label = (f"{name} removed (hands at work)" if self.bench_still_for(t) < BENCH_STILL_S
                                 else f"{name} removed {t - max(tr.removed_since, tr.rub_last):.0f}/{REMOVED_ALERT_S:.0f}s")
                    else:
                        label = f"{name} not seen (still there?)"
                dashed_rect(vis, (x1, y1), (x2, y2), col)
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
            steps = "Rubbed + acid result + back" if ACID_REQUIRED else "Rubbed and back"
            text = f"{steps} (DONE): {done}/{len(self.tracks)}"
            if self.focus:
                text += "   |   rubbing: other jewels held in place"
            if self.id_model is not None and not self.id_model.ready:
                text += "   |   learning the jewels (ResNet18)..."
            if self.acid_pending_id is not None:
                text += f"   |   J{self.acid_pending_id}: " + ("acid test running" if self.acid_token else "click Start acid test")
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

    def set_frame(self, img, source_size=None):
        self.frame_size = source_size or (img.width(), img.height())
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
        self.cam_usb = QtWidgets.QComboBox()  # USB mode: which camera ...
        self.cam_usb.addItems([f"Camera {i}" for i in range(4)])
        self.cam_usb.setToolTip("USB camera number (0 = the first one Windows found)")
        self.cam_res = QtWidgets.QComboBox()  # ... and its resolution
        self.cam_res.addItems(USB_RESOLUTIONS)
        self.btn_af = QtWidgets.QPushButton("Anti-flicker")
        self.btn_af.setToolTip("Camera shutter matched to the mains light (USB camera)")
        self.af_dialog, self.af_last = None, ""
        self.url = QtWidgets.QLineEdit()  # Custom URL mode only
        self.url.setPlaceholderText("rtsp://... or 0 for a webcam")
        self.cam_mode_key = None
        self.btn_conn = QtWidgets.QPushButton("Connect")
        self.video = VideoLabel()
        self.btn_roi = QtWidgets.QPushButton("Draw ROI")
        self.btn_stone = QtWidgets.QPushButton("Draw stone")
        self.btn_clear = QtWidgets.QPushButton("Clear zones")
        # J | jewel (crop at capture) | Rubbed | rub spot (image, once rubbed) | Acid test - updated as they happen
        self.acid_models = AcidModelCache()
        self.rub_models = RubPhotoModels()
        self.rub_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rub-photos')
        self.rub_job = self.rub_cycle = self.rub_cycle_key = None
        self.rub_cleanup = None
        self.rub_display_result = None
        self.auto_gate = self.auto_gate_key = self.auto_hold = None
        self.chk_auto_test = QtWidgets.QCheckBox("Automatic tests (ON)")
        self.chk_auto_test.setChecked(True)
        self.chk_auto_test.setToolTip("Return the rubbed jewel to the bench and keep it and the stone settled for 2 seconds. pH runs first when ON; Ready follows automatically.")
        self.btn_rub_reference = QtWidgets.QPushButton("Capture / retake rubbing reference")
        self.rub_instruction = QtWidgets.QLabel("Waiting for a visible stone to capture its starting marks.")
        self.rub_instruction.setWordWrap(True)
        self.chk_existing_rub_marks = QtWidgets.QCheckBox("Show existing rub marks (OFF)")
        self.chk_existing_rub_marks.setToolTip("Show green starting-mark overlays for development. Detection stays active when hidden.")
        self.rub_preview = QtWidgets.QLabel("Before / after rubbing")
        self.rub_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.rub_preview.setMinimumHeight(160)
        self.rub_preview.setMaximumHeight(210)
        self.acid_thread = self.acid_owner = None
        self.acid_ui_token = None
        self.acid_error_message = None
        self.acid_logged_phase = None
        self.btn_start_acid = QtWidgets.QPushButton("Rub a jewel to enable acid test")
        self.btn_start_acid.setEnabled(False)
        self.btn_stone_ready = QtWidgets.QPushButton("Ready for stone test")
        self.btn_stone_ready.setEnabled(False)
        self.btn_cancel_acid = QtWidgets.QPushButton("Cancel acid test")
        self.btn_cancel_acid.setEnabled(False)
        self.acid_instruction = QtWidgets.QLabel("Rub any jewel first. Verify fresh pH paper for every jewel.")
        self.acid_instruction.setWordWrap(True)
        self.chk_ph_verification = QtWidgets.QCheckBox("pH verification (ON)")
        self.chk_ph_verification.setChecked(True)
        self.chk_ph_verification.setToolTip("On: verify pH paper before the stone test. Off: skip pH paper and continue to stone readiness.")
        self.acid_duration = QtWidgets.QDoubleSpinBox()
        self.acid_duration.setRange(1, 120)
        self.acid_duration.setValue(5)
        self.acid_duration.setSuffix(" s observation")
        self.acid_grade = QtWidgets.QLineEdit()
        self.acid_grade.setPlaceholderText("Acid grade / batch (optional)")
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["J", "Jewel", "Rubbed", "Rub spot", "Acid", "Result", "pH / Acid crop"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setStyleSheet("font-size: 14px;")
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        for c in (1, 3, 6):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)
        self.table.setMinimumHeight(260)
        self.table.setToolTip("Green row = DONE (rubbed, acid tested and placed back)")
        self.btn_reset = QtWidgets.QPushButton("Reset (re-arm)")
        self.lbl_status = QtWidgets.QLabel("Not connected")
        self.log = QtWidgets.QListWidget()

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Camera:"))
        top.addWidget(self.cam_mode)
        for w in (self.cam_usb, self.cam_res, self.btn_af, self.cam_ip, self.cam_stream, self.cam_user, self.cam_pass):
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
        side.addWidget(self.btn_rub_reference)
        side.addWidget(self.rub_instruction)
        side.addWidget(self.chk_existing_rub_marks)
        side.addWidget(self.rub_preview)
        side.addWidget(self.chk_auto_test)
        side.addWidget(self.chk_ph_verification)
        side.addWidget(self.acid_duration)
        side.addWidget(self.acid_grade)
        side.addWidget(self.btn_start_acid)
        side.addWidget(self.btn_stone_ready)
        side.addWidget(self.btn_cancel_acid)
        side.addWidget(self.acid_instruction)
        side.addWidget(self.lbl_status)
        self.lbl_jewels = QtWidgets.QLabel("Jewels: counted when the stone is placed")
        side.addWidget(self.lbl_jewels)
        side.addWidget(self.table, 2)
        side.addWidget(QtWidgets.QLabel("Events"))
        side.addWidget(self.log, 1)
        side_w = QtWidgets.QWidget()
        side_w.setLayout(side)
        side_w.setMinimumWidth(540)

        mid = QtWidgets.QHBoxLayout()
        mid.addWidget(self.video, 1)
        side_scroll = QtWidgets.QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setWidget(side_w)
        side_scroll.setMinimumWidth(560)
        mid.addWidget(side_scroll)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid, 1)

        self.btn_conn.clicked.connect(self.toggle_connect)
        self.btn_af.clicked.connect(self.open_antiflicker)
        self.cam_mode.setCurrentIndex(max(0, list(CAMERA_MODES).index(self.cameras.get("mode", "network"))))
        self.show_camera(self.cam_mode.currentData())
        self.cam_mode.currentIndexChanged.connect(self.switch_camera)
        self.btn_roi.clicked.connect(lambda: self.toggle_draw("roi"))
        self.btn_stone.clicked.connect(lambda: self.toggle_draw("stone"))
        self.btn_clear.clicked.connect(self.clear_zones)
        self.btn_reset.clicked.connect(self.rearm)
        self.btn_rub_reference.clicked.connect(lambda: self.capture_rub_reference(manual=True))
        self.chk_auto_test.toggled.connect(self.auto_test_changed)
        self.chk_ph_verification.toggled.connect(self.ph_verification_changed)
        self.chk_existing_rub_marks.toggled.connect(self.existing_rub_marks_changed)
        self.btn_start_acid.clicked.connect(self.start_acid_test)
        self.btn_stone_ready.clicked.connect(self.ready_for_stone_test)
        self.btn_cancel_acid.clicked.connect(self.cancel_acid_test)
        self.video.clicked.connect(self.on_click)

        self.beep_until = 0.0
        self.beep_timer = QtCore.QTimer(self)
        self.beep_timer.timeout.connect(self.beep_tick)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.preview_timer.start(PREVIEW_INTERVAL_MS)
        self.rub_timer = QtCore.QTimer(self)
        self.rub_timer.timeout.connect(self.rub_guidance_tick)
        self.rub_timer.start(200)
        self.preview_key = None
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
            cams["mode"] = saved.get("mode", cams["mode"]) if "usb" in saved else "usb"  # USB is now the default
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
        custom, usb = key == "custom", key == "usb"
        for w in (self.cam_ip, self.cam_stream, self.cam_user, self.cam_pass):
            w.setVisible(not custom and not usb)
        for w in (self.cam_usb, self.cam_res, self.btn_af):
            w.setVisible(usb)
        self.url.setVisible(custom)
        if custom:
            self.url.setText(str(cam.get("url", "")))
        elif usb:
            self.cam_usb.setCurrentIndex(int(cam.get("index", 0)))
            self.cam_res.setCurrentText(cam.get("resolution", USB_RESOLUTIONS[0]))
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
        elif key == "usb":
            self.cameras[key].update(index=self.cam_usb.currentIndex(), resolution=self.cam_res.currentText())
        elif key:
            self.cameras[key].update(ip=self.cam_ip.text().strip(), subtype=self.cam_stream.currentIndex(),
                                     user=self.cam_user.text().strip(), password=self.cam_pass.text())
            self.cameras[key].pop("antiflicker", None)
        self.cameras["mode"] = key

    def camera_source(self):
        cam = self.cameras[self.cam_mode_key]
        if self.cam_mode_key == "usb":
            return int(cam.get("index", 0))
        if self.cam_mode_key == "custom":
            text = str(cam.get("url", "")).strip()
            return int(text) if text.isdigit() else text
        return dahua_url(cam)

    def open_antiflicker(self):
        """Anti-flicker settings (as vision_app.py): on/off, mains frequency, shutter 1/N s; applied live."""
        if self.af_dialog is not None and self.af_dialog.isVisible():
            self.af_dialog.raise_()
            return
        af = {**AF_DEFAULT, **self.cameras["usb"].get("antiflicker", {})}
        dlg = self.af_dialog = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Anti-flicker (USB camera shutter)")
        dlg.setMinimumWidth(380)
        lay = QtWidgets.QVBoxLayout(dlg)
        chk = QtWidgets.QCheckBox("Enable anti-flicker (fixed shutter for the whole feed)")
        chk.setChecked(bool(af["enabled"]))
        lay.addWidget(chk)
        hz_row = QtWidgets.QHBoxLayout()
        hz_row.addWidget(QtWidgets.QLabel("Mains frequency:"))
        cmb = QtWidgets.QComboBox()
        cmb.addItem("50 Hz", 50)
        cmb.addItem("60 Hz", 60)
        cmb.setCurrentIndex(0 if int(af["mains_hz"]) == 50 else 1)
        hz_row.addWidget(cmb, 1)
        lay.addLayout(hz_row)
        lbl_shutter, lbl_note = QtWidgets.QLabel(), QtWidgets.QLabel()
        sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sld.setRange(*AF_DENOM_RANGE)
        sld.setValue(max(AF_DENOM_RANGE[0], min(AF_DENOM_RANGE[1], int(af["shutter_denom"]))))
        sld.setPageStep(10)
        lay.addWidget(lbl_shutter)
        lay.addWidget(sld)
        lay.addWidget(lbl_note)

        def refresh():
            ms, safe = af_flicker_note(sld.value(), cmb.currentData())
            lbl_shutter.setText(f"Shutter: 1/{sld.value()} s")
            lbl_note.setText(f"Exposure ~{ms:.1f} ms ({'flicker-safe' if safe else 'may band under mains light'})")
            lbl_note.setStyleSheet("color: #2e7d32;" if safe else "color: #b26a00;")
        sld.valueChanged.connect(refresh)
        cmb.currentIndexChanged.connect(refresh)
        refresh()
        presets = QtWidgets.QHBoxLayout()
        for denom in (100, 50, 25):
            b = QtWidgets.QPushButton(f"1/{denom} s")
            b.clicked.connect(lambda _, d=denom: sld.setValue(d))
            presets.addWidget(b)
        lay.addLayout(presets)
        dlg.readback = QtWidgets.QLabel(self.af_last or "Camera: not applied yet")
        dlg.readback.setWordWrap(True)
        dlg.readback.setStyleSheet("color: #555; font-size: 11px;")
        lay.addWidget(dlg.readback)
        btn = QtWidgets.QPushButton("Apply")

        def apply():
            new = {**af, "enabled": chk.isChecked(), "mains_hz": int(cmb.currentData()),
                   "shutter_denom": int(sld.value())}
            self.cameras["usb"]["antiflicker"] = new
            self.save_cameras()
            if self.worker and self.cam_mode_key == "usb":
                self.worker.request_antiflicker(new)
                dlg.readback.setText("Applying on the live feed...")
            else:
                dlg.readback.setText("Saved: applied when the USB camera connects")
        btn.clicked.connect(apply)
        lay.addWidget(btn)
        dlg.show()

    def switch_camera(self):
        self.read_camera()  # keep what was typed for the previous connection
        self.show_camera(self.cam_mode.currentData())
        self.cameras["mode"] = self.cam_mode_key
        self.save_cameras()
        if self.worker:  # switching while connected: reconnect to the new feed
            self.add_log(f"Switching camera to {CAMERA_MODES[self.cam_mode_key]}", "info")
            if self.stop_worker():
                self.toggle_connect()

    # ---- control ----
    def set_running(self, running):
        for w in (self.btn_roi, self.btn_stone, self.btn_clear, self.btn_reset):
            w.setEnabled(running)
        for w in (self.url, self.cam_usb, self.cam_res, self.cam_ip, self.cam_stream, self.cam_user, self.cam_pass):
            w.setEnabled(not running)
        self.btn_conn.setText("Disconnect" if running else "Connect")

    def toggle_connect(self):
        if self.worker:
            if not self.stop_worker():
                return
            self.lbl_status.setText("Not connected")
            return
        self.read_camera()
        self.save_cameras()
        source = self.camera_source()
        shown = (f"camera {source}" if isinstance(source, int) else urlparse(source).hostname or source)
        self.add_log(f"Connecting to {CAMERA_MODES[self.cam_mode_key]}: {shown}", "info")
        camera = None if self.cam_mode_key == "custom" else self.cameras[self.cam_mode_key]
        self.worker = Worker(source, self.zones["roi"], self.zones["stone"], camera=camera)
        self.worker.jewel_list.connect(self.show_jewels)
        self.worker.jewel_table.connect(self.show_table)
        self.worker.event.connect(self.add_log)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.start()
        self.set_running(True)

    def refresh_preview(self):
        worker = self.worker
        if worker is None or worker.grabber is None:
            return
        after_ts = self.preview_key[1] if self.preview_key and self.preview_key[0] == id(worker) else None
        frame, ts = worker.grabber.latest(after_ts=after_ts)
        key = (id(worker), ts)
        if frame is None or key == self.preview_key:
            return
        self.preview_key = key
        frame = worker.preview.render(frame, ts)
        h, w = frame.shape[:2]
        scale = min(1.0, self.video.width() / w, self.video.height() / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_LINEAR)
        self.video.set_frame(to_qimage(frame), source_size=(w, h))

    def stop_worker(self):
        if not self.cancel_acid_test():
            return False
        if self.drawing:
            self.finish_draw()
        if self.worker:
            self.worker.stop()
            if not self.worker.wait(15000):
                self.lbl_status.setText("Stopping camera and model workers; please wait, then retry")
                return False
            self.worker = None
        self.set_running(False)
        self.refresh_acid_controls()
        return True

    def show_jewels(self, lines):
        self.lbl_jewels.setText(f"Jewels: {len(lines)} detected when the stone was placed" if lines
                                else "Jewels: counted when the stone is placed")

    def show_table(self, rows):
        """J | jewel crop | Rubbed | rub spot image | Acid; a DONE row (rubbed, acid tested, put back) is green."""
        th = 70  # thumbnail height
        self.table.setRowCount(len(rows))
        for r, (jid, thumb, rubbed, rub_img, acid, done, record, pending, acid_image) in enumerate(rows):
            bg = "#c8f0c8" if done else "transparent"
            prediction = record['prediction'] if record else None
            result_text = (f"{ {'gold_22k': '22K GOLD', 'non_gold': 'NON-GOLD', 'invalid': 'INCONCLUSIVE'}[prediction['prediction']]}\n{prediction['confidence']:.1%}"
                           if prediction else "")
            acid_text = "\u2714" if acid else ("Testing" if pending and self.worker and self.worker.acid_token else "Ready" if pending else "")
            for c, text in ((0, f"J{jid}"), (2, "\u2714" if rubbed else ""), (4, acid_text), (5, result_text)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                if c:
                    item.setForeground(QtGui.QColor(0, 150, 0))
                if done:
                    item.setBackground(QtGui.QColor(200, 240, 200))
                if c == 5 and record:
                    item.setToolTip(f"Prediction only; verified label not set.\nCapture: {record['record_path']}")
                    if prediction['prediction'] == 'invalid':
                        item.setForeground(QtGui.QColor(180, 110, 0))
                    elif prediction['prediction'] == 'non_gold':
                        item.setForeground(QtGui.QColor(190, 30, 30))
                self.table.setItem(r, c, item)
            for c, img in ((1, thumb), (3, rub_img), (6, acid_image)):
                lbl = QtWidgets.QLabel()
                lbl.setAlignment(QtCore.Qt.AlignCenter)
                lbl.setStyleSheet(f"background: {bg};")
                if img is not None and img.size:
                    pix = QtGui.QPixmap.fromImage(to_qimage(img))
                    lbl.setPixmap(pix.scaled(110, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                    lbl.setToolTip(f"J{jid} " + ("as captured" if c == 1 else "pH paper / acid region" if c == 6 else "photo captured when rubbing was confirmed"))
                else:
                    lbl.setText("" if c == 1 else "-")
                self.table.setCellWidget(r, c, lbl)
            self.table.setRowHeight(r, th + 6)
        self.refresh_acid_controls()

    def current_rub_key(self):
        worker = self.worker
        if worker is None:
            return None
        return (id(worker), worker.acid_batch_id, worker.rub_cycle_id, worker.flow_revision)

    def sync_rub_cycle(self):
        key = self.current_rub_key()
        if key != self.rub_cycle_key:
            self.rub_cycle_key = key
            self.rub_cycle = RubCycle(self.rub_models) if key is not None else None
            self.rub_cleanup = (StoneCleanup(self.rub_models, self.acid_models)
                                if self.worker is not None and self.worker.stone_cleanup_pending else None)
            self.auto_gate = self.auto_gate_key = self.auto_hold = None
            self.rub_display_result = None
            self.rub_preview.clear()
            self.rub_instruction.setText(
                "Clean and dry the stone, then uncover it. Waiting to save the next reference."
                if self.rub_cleanup is not None else
                "Keep hands and jewellery off the stone while its reference is captured.")
        return self.rub_cycle

    def show_rub_result(self, result):
        self.rub_instruction.setText(result['message'])
        self.rub_display_result = result
        self.refresh_rub_preview()

    def existing_rub_marks_changed(self, enabled):
        self.chk_existing_rub_marks.setText(f"Show existing rub marks ({'ON' if enabled else 'OFF'})")
        self.refresh_rub_preview()

    def refresh_rub_preview(self):
        result = self.rub_display_result
        if result is None:
            self.rub_preview.clear()
            return
        image = render_rub_preview(result, self.chk_existing_rub_marks.isChecked())
        if image is None:
            self.rub_preview.clear()
            return
        pix = QtGui.QPixmap.fromImage(to_qimage(image))
        self.rub_preview.setPixmap(pix.scaled(480, 200, QtCore.Qt.KeepAspectRatio,
                                             QtCore.Qt.SmoothTransformation))

    def rub_guidance_tick(self):
        job = self.rub_job
        if job is not None and job['future'].done():
            self.rub_job = None
            valid = (not job.get('cancelled') and job['owner'] is self.worker
                     and job['key'] == self.current_rub_key())
            if valid:
                try:
                    value = job['future'].result()
                    if job['kind'] in ('before', 'cleanup'):
                        self.show_rub_result(value)
                        with self.worker.lock:
                            self.worker.rub_reference_ready = value['status'] == 'READY'
                            if job['kind'] == 'cleanup' and value['status'] == 'READY':
                                self.worker.stone_cleanup_pending = False
                        if value['status'] != 'WAIT_FOR_CLEANUP':
                            self.add_log(value['message'], 'info')
                    else:
                        result, target = value
                        self.worker.set_new_rub_target(job['jewel_id'], target)
                        self.show_rub_result(result)
                        self.add_log(f"J{job['jewel_id']}: new-rub mask confirmed; acid must be inside its ROI", 'info')
                        self.launch_acid_test(job['ph_enabled'], job['duration'], job['grade'], auto_test=job.get('auto_test', False))
                except Exception as exc:
                    if job['kind'] == 'cleanup' and self.rub_cleanup is not None:
                        self.rub_cleanup.waiting("Cleanup check could not complete; waiting to retry.")
                    self.rub_instruction.setText(str(exc))
                    self.add_log(f"Rubbing photo check: {exc}", 'info')
            self.refresh_acid_controls()
        cycle = self.sync_rub_cycle()
        worker = self.worker
        if (worker is not None and worker.stone_cleanup_pending
                and self.rub_job is None and self.acid_thread is None):
            self.capture_cleanup_reference()
        elif (cycle is not None and self.rub_job is None and self.acid_thread is None
                and not cycle.attempted and worker.acid_pending_id is None
                and worker.grabber is not None and worker.stone_box() is not None):
            self.capture_rub_reference()
        self.auto_test_tick()

    def capture_rub_reference(self, manual=False):
        worker = self.worker
        if (worker is None or self.rub_job is not None or self.acid_thread is not None
                or worker.acid_pending_id is not None or worker.grabber is None):
            return
        if worker.stone_cleanup_pending:
            self.rub_instruction.setText("Finish cleaning and uncover the stone. Its reference will be captured automatically.")
            return
        if manual:
            with worker.lock:
                worker.invalidate_rub_reference(keep_cycle=True)
        cycle = self.sync_rub_cycle()
        frame, ts = worker.grabber.latest()
        if frame is None or time.time()-ts > STREAM_TIMEOUT_S or worker.stone_box() is None:
            self.rub_instruction.setText("Expose the stone and restore a fresh camera frame before capturing.")
            return
        if worker.stone is not None and worker.stone.model is not None:
            self.rub_models.use_stone_model(worker.stone.model, worker.roi)
        cycle.attempted = True
        self.rub_instruction.setText("Saving the starting marks. Keep hands and jewellery off the stone.")
        self.rub_job = dict(kind='before', owner=worker, key=self.current_rub_key(),
                            future=self.rub_pool.submit(cycle.before, frame))
        self.refresh_acid_controls()

    def capture_cleanup_reference(self):
        worker, cleanup, cycle = self.worker, self.rub_cleanup, self.rub_cycle
        if (worker is None or cleanup is None or cycle is None or worker.grabber is None
                or self.rub_job is not None or self.acid_thread is not None):
            return
        frame, ts = worker.grabber.latest()
        if frame is None or not 0 <= time.time()-ts <= .75:
            self.rub_instruction.setText("Waiting for fresh camera frames after cleaning.")
            cleanup.waiting(self.rub_instruction.text())
            self.refresh_acid_controls()
            return
        if cleanup.last_ts is not None and ts <= cleanup.last_ts:
            return
        if worker.stone is not None and worker.stone.model is not None:
            self.rub_models.use_stone_model(worker.stone.model, worker.roi)
        self.rub_job = dict(kind='cleanup', owner=worker, key=self.current_rub_key(),
                            future=self.rub_pool.submit(lambda image: cleanup.before(cycle, image, ts), frame))
        self.refresh_acid_controls()

    def new_rub_target_ready(self, payload):
        if self.current_acid_payload(payload):
            self.worker.update_acid_target(payload)

    def auto_test_changed(self, enabled):
        self.chk_auto_test.setText("Automatic tests (ON)" if enabled else "Automatic tests (OFF)")
        self.auto_gate = self.auto_gate_key = self.auto_hold = None
        if enabled:
            self.acid_error_message = None
        self.refresh_acid_controls()

    def auto_test_tick(self):
        worker, cycle = self.worker, self.rub_cycle
        if (not self.chk_auto_test.isChecked() or worker is None or cycle is None
                or self.rub_job is not None or self.acid_thread is not None
                or worker.mode != "LOCKED" or not worker.rub_reference_ready
                or worker.acid_pending_id is None or worker.acid_token is not None):
            return
        key = (self.current_rub_key(), worker.acid_pending_id)
        if self.auto_hold == key:
            return
        with worker.lock:
            tr = next((tr for tr in worker.tracks if tr.id == worker.acid_pending_id), None)
            if tr is None or not tr.rubbed or tr.acid_done:
                return
            center = getattr(tr, 'pos', None)
            center = None if center is None else np.asarray(center, float).copy()
            detected = getattr(tr, 'det_seen', 0.)
            box = getattr(tr, 'det_box', None)
            state = getattr(tr, 'state', None)
            target = getattr(tr, 'new_rub_target', None)
            stone = target.stone if target is not None else cycle.stone
            stone_visible = target is not None or worker.stone_box() is not None
            roi = None if worker.roi is None else worker.roi.copy()
        if stone is None or worker.grabber is None:
            return
        frame, ts = worker.grabber.latest()
        if frame is None or not 0 <= time.time()-ts <= .75:
            return
        visible = (center is not None and state == "OK" and 0 <= time.time()-detected <= .75)
        if visible:
            x, y = map(float, center)
            visible = not (stone[0] <= x <= stone[2] and stone[1] <= y <= stone[3])
            if roi is not None and cv2.pointPolygonTest(roi, (x, y), False) < 0:
                visible = False
        if self.auto_gate_key != key:
            if not visible or not stone_visible:
                self.acid_instruction.setText(f"J{worker.acid_pending_id}: return this jewel to the bench and keep the stone visible. Acid OFF.")
                return
            tolerance = 8. if box is None else max(6., .03*np.linalg.norm(np.asarray(box[2:])-box[:2]))
            self.auto_gate = StoneSettling(stone, center_tolerance=tolerance)
            self.auto_gate_key = key
        if self.auto_gate.update(frame, ts, visible=visible and stone_visible, center=center):
            self.add_log(f"J{worker.acid_pending_id} returned; jewel and stone settled for 2 seconds. Starting automatic test.", "info")
            self.start_acid_test(automatic=True)
        elif not self.auto_gate.latched:
            message = (f"J{worker.acid_pending_id}: return this jewel to the bench, outside the stone."
                       if not visible else
                       f"J{worker.acid_pending_id}: settling {self.auto_gate.elapsed:.1f}/2.0 s; keep jewel and stone still.")
            self.acid_instruction.setText(message + " Acid OFF.")
        else:
            self.acid_instruction.setText("The automatic photo check needs a retry. Clear the stone and move/return the jewel to retry automatically, or use Start.")

    def ph_verification_changed(self, enabled):
        self.chk_ph_verification.setText("pH verification (ON)" if enabled else "pH verification (OFF)")
        self.refresh_acid_controls()

    def refresh_acid_controls(self):
        pending = self.worker.acid_pending_id if self.worker else None
        busy = self.acid_thread is not None or (
            self.rub_job is not None and self.rub_job['kind'] != 'cleanup')
        self.btn_start_acid.setText(f"Start acid test for J{pending}" if pending is not None else "Rub a jewel to enable acid test")
        self.btn_start_acid.setEnabled(pending is not None and not busy)
        cleaning = self.worker is not None and self.worker.stone_cleanup_pending
        self.btn_rub_reference.setEnabled(self.worker is not None and pending is None and not busy and not cleaning)
        waiting = (busy and self.worker is not None and self.acid_owner is self.worker
                   and self.acid_ui_token == self.worker.acid_token
                   and self.acid_logged_phase == 'WAIT_FOR_STONE_READY')
        self.btn_stone_ready.setEnabled(waiting)
        self.btn_cancel_acid.setEnabled(self.acid_thread is not None or (
            self.rub_job is not None and self.rub_job['kind'] == 'after' and not self.rub_job.get('cancelled')))
        self.acid_duration.setEnabled(not busy)
        self.acid_grade.setEnabled(not busy)
        self.chk_ph_verification.setEnabled(not busy)
        self.chk_auto_test.setEnabled(not busy)
        if cleaning:
            self.acid_instruction.setText("Next jewel paused: " + self.rub_instruction.text())
        elif pending is not None and not busy and self.acid_error_message:
            self.acid_instruction.setText(self.acid_error_message)
        elif pending is not None and not busy:
            ph_step = "verify pH paper, then " if self.chk_ph_verification.isChecked() else ""
            if self.chk_auto_test.isChecked():
                if self.auto_hold == (self.current_rub_key(), pending):
                    self.acid_instruction.setText(f"Automatic testing paused for J{pending}. Use Start to retry, or switch Automatic tests OFF and ON to resume.")
                else:
                    self.acid_instruction.setText(f"J{pending} rubbed. Return it to the bench and keep it and the stone still for 2 seconds; testing starts automatically. Acid OFF.")
            else:
                self.acid_instruction.setText(f"J{pending} rubbed. Click Start, {ph_step}click Ready for stone test. Apply acid only at NOW APPLY ACID.")
        elif not busy and self.worker is not None and not self.worker.rub_reference_ready:
            if self.rub_display_result is not None and self.rub_display_result['status'] == 'CLEAN_THE_STONE':
                self.acid_instruction.setText(self.rub_display_result['message'])
            else:
                self.acid_instruction.setText("Wait for the before-rubbing reference. Keep acid OFF the stone.")
        elif not busy:
            self.acid_instruction.setText("Rub a jewel on the stone. " + (
                "Verify fresh pH paper for every jewel." if self.chk_ph_verification.isChecked() else "pH verification is OFF."))

    def start_acid_test(self, automatic=False):
        if self.worker is None or self.acid_thread is not None or self.rub_job is not None:
            return
        worker = self.worker
        if not automatic:
            self.auto_hold = None
        tr = next((tr for tr in worker.tracks if tr.id == worker.acid_pending_id), None)
        if tr is None:
            return
        if getattr(tr, 'new_rub_target', None) is not None:
            self.launch_acid_test()
            return
        cycle = self.sync_rub_cycle()
        if not worker.rub_reference_ready or cycle is None or cycle.before_result is None:
            self.rub_instruction.setText("A before-rubbing reference is required. Reset and capture it before rubbing.")
            return
        frame, ts = worker.grabber.latest()
        if frame is None or time.time()-ts > STREAM_TIMEOUT_S:
            self.rub_instruction.setText("Camera frame is unavailable; restore the feed and retry.")
            return
        self.rub_instruction.setText("Checking the new rub mark. Keep hands away and acid OFF the stone.")
        self.rub_job = dict(kind='after', owner=worker, key=self.current_rub_key(), jewel_id=tr.id,
                            auto_test=self.chk_auto_test.isChecked(), ph_enabled=self.chk_ph_verification.isChecked(),
                            duration=self.acid_duration.value(), grade=self.acid_grade.text().strip(),
                            future=self.rub_pool.submit(cycle.after, frame))
        self.refresh_acid_controls()

    def launch_acid_test(self, ph_enabled=None, duration=None, grade=None, auto_test=None):
        if self.worker is None or self.acid_thread is not None:
            return
        self.acid_error_message = None
        self.acid_logged_phase = None
        try:
            context = self.worker.begin_acid_test(ph_verification_enabled=(
                self.chk_ph_verification.isChecked() if ph_enabled is None else ph_enabled))
            context["auto_test"] = self.chk_auto_test.isChecked() if auto_test is None else auto_test
            thread = AcidTestThread(context, self.acid_duration.value() if duration is None else duration,
                                    self.acid_grade.text().strip() if grade is None else grade, models=self.acid_models)
        except Exception as exc:
            self.auto_hold = (self.current_rub_key(), self.worker.acid_pending_id)
            if self.worker.acid_token:
                self.worker.fail_acid_test(self.worker.acid_token, str(exc))
            self.acid_instruction.setText(str(exc))
            return
        self.acid_thread, self.acid_owner = thread, self.worker
        self.acid_ui_token = context['token']
        thread.progress.connect(self.acid_progress)
        thread.crop_ready.connect(self.show_acid_crop)
        thread.target_ready.connect(self.new_rub_target_ready)
        thread.completed.connect(self.acid_completed)
        thread.failed.connect(self.acid_failed)
        thread.finished.connect(self.acid_thread_finished)
        thread.finished.connect(thread.deleteLater)
        stage = "loading pH-paper verification..." if context['ph_verification_enabled'] else "pH verification OFF; preparing stone readiness..."
        self.acid_instruction.setText(f"J{context['jewel_id']}: {stage}")
        self.refresh_acid_controls()
        thread.start()

    def ready_for_stone_test(self):
        if (self.acid_thread is None or not self.btn_stone_ready.isEnabled()
                or self.acid_owner is not self.worker or self.worker is None
                or self.acid_ui_token != self.worker.acid_token):
            return
        if self.acid_thread.confirm_stone_ready():
            self.acid_logged_phase = 'STONE_READY'
            self.btn_stone_ready.setEnabled(False)
            self.acid_instruction.setText("Stone test confirmed. Keep paper/hands away; wait for the baseline and NOW APPLY ACID.")

    def current_acid_payload(self, payload):
        return (self.worker is not None and self.worker is self.acid_owner
                and payload['token'] == self.acid_ui_token
                and payload['token'] == self.worker.acid_token)

    def acid_progress(self, payload):
        if self.current_acid_payload(payload):
            self.acid_instruction.setText(f"J{payload['jewel_id']}: {payload['message']}")
            phase = payload.get('phase')
            if phase != self.acid_logged_phase:
                self.acid_logged_phase = phase
                detail = f" | {payload['model_path']}" if payload.get('model_path') else ""
                self.add_log(f"J{payload['jewel_id']} {phase}: {payload['message']}{detail}", "info")
            self.refresh_acid_controls()

    def show_acid_crop(self, payload):
        if not self.current_acid_payload(payload) or not self.worker.update_acid_crop(payload):
            return
        # Update this thumbnail only; don't rebuild all table rows every frame.
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.text() == f"J{payload['jewel_id']}":
                label = self.table.cellWidget(row, 6)
                if label is not None:
                    pix = QtGui.QPixmap.fromImage(to_qimage(payload['crop']))
                    label.setPixmap(pix.scaled(110, 70, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                    label.setToolTip(f"J{payload['jewel_id']} " + ("pH paper" if payload.get("kind") == "ph_paper" else "acid region"))
                break

    def acid_completed(self, payload):
        if self.current_acid_payload(payload) and self.worker.finish_acid_test(payload):
            prediction = payload['result']['prediction']
            label = {'gold_22k': '22K GOLD', 'non_gold': 'NON-GOLD', 'invalid': 'INCONCLUSIVE'}[prediction['prediction']]
            self.acid_instruction.setText(f"J{payload['jewel_id']}: {label} ({prediction['confidence']:.1%}). Clean and dry the stone. The next reference will be captured after it is uncovered and still.")

    def acid_failed(self, payload):
        if self.current_acid_payload(payload):
            self.acid_logged_phase = None
            self.auto_hold = (self.current_rub_key(), self.worker.acid_pending_id)
            self.btn_stone_ready.setEnabled(False)
            self.acid_error_message = f"J{payload['jewel_id']}: {payload['message']}. Retry when ready."
            if payload.get('traceback'):
                write_log(f"ACID ERROR J{payload['jewel_id']}  {payload['traceback']}")
            self.worker.fail_acid_test(payload['token'], payload['message'])
            self.acid_instruction.setText(self.acid_error_message)

    def acid_thread_finished(self):
        if self.sender() is self.acid_thread:
            self.acid_thread = self.acid_owner = None
            self.acid_ui_token = None
            self.refresh_acid_controls()

    def cancel_acid_test(self):
        if self.worker is not None and self.worker.acid_pending_id is not None:
            self.auto_hold = (self.current_rub_key(), self.worker.acid_pending_id)
        if self.rub_job is not None:
            self.rub_job['cancelled'] = True
            self.rub_job['future'].cancel()
        thread = self.acid_thread
        if thread is not None:
            thread.cancel()
            if not thread.wait(15000):
                self.acid_instruction.setText("Stopping acid test; please wait...")
                return False
            if self.acid_owner is self.worker and self.worker:
                self.worker.fail_acid_test(self.acid_ui_token, "Acid test cancelled")
            self.acid_thread = self.acid_owner = None
            self.acid_ui_token = None
            self.refresh_acid_controls()
        return True

    def rearm(self):
        if not self.worker:
            return
        if not self.cancel_acid_test():
            return
        self.acid_error_message = None
        self.worker.arm()
        self.refresh_acid_controls()
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
        if msg.startswith("Anti-flicker"):  # read-back / live check into the open settings window
            self.af_last = msg
            if self.af_dialog is not None and self.af_dialog.isVisible():
                self.af_dialog.readback.setText(msg)
        write_log(f"{level.upper():5}  {msg}")

    def beep_tick(self):
        if time.time() < self.beep_until:
            beep()
        else:
            self.beep_timer.stop()

    def closeEvent(self, e):
        if self.stop_worker():
            self.rub_timer.stop()
            self.rub_pool.shutdown(wait=True, cancel_futures=True)
            self.rub_models.close()
            e.accept()
        else:
            e.ignore()


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
    write_log(f"START  Jewel Guard | app={os.path.abspath(__file__)} | python={sys.executable} | pH gate enabled")
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1280, 760)
    w.show()
    sys.exit(app.exec_())
