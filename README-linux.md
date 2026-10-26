# Jewel Tracker 5 on this PC

Application folder: `/home/embsys/Desktop/jewel_tracker/jewel_tracker`.
Python environment: `.venv` (Python 3.10, CPU PyTorch).

## Run

```bash
cd /home/embsys/Desktop/jewel_tracker/jewel_tracker
./run_tracker.sh
```

Or activate the environment and run the script directly:

```bash
source .venv/bin/activate
python jewel_tracker_5.py
```

Select `.venv/bin/python` as the Python interpreter in your IDE.
Open `jewel_tracker.code-workspace` for a dedicated VS Code workspace with
that interpreter selected by default. The launcher also activates this
project's environment before starting Python, so `VIRTUAL_ENV` and command
lookup use Jewel Tracker even if a camera-app environment was active in the
parent terminal. Restart an already running tracker to pick up this change.

To reinstall Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The script clears Qt environment paths set by the OpenCV Linux wheel so that
the PyQt5 GUI loads its own Qt platform plugins.

### USB anti-flicker on Ubuntu

Install the native camera utility if setting up another PC:

```bash
sudo apt install v4l-utils
```

On Linux, USB capture uses V4L2 with MJPG and requests 30 FPS. At startup and
every reconnect, the tracker applies the saved `camera.json` anti-flicker
profile before publishing frames. The verified Dell WB3023 bench profile is
50 Hz, manual 20 ms exposure, brightness 96, gain 0, and dynamic frame rate
off. Focus and white balance controls are untouched. The Anti-flicker dialog
still changes the shutter live; Off restores the controls saved before the
first application during that connection's capture-thread lifetime.

`linux_camera_controls.py` reads native controls back and reports acceptance
in Events. Missing utilities, unsupported controls, or rejected values produce
an explicit failure message; capture continues. The Windows DirectShow path
retains its existing exposure logic. These values were measured for this
camera and bench lighting; another camera may require a different profile.

Real capture verification (two opens, 90 frames each) is saved in
[`diagnostics/tracker_antiflicker_20261021/verification.json`](diagnostics/tracker_antiflicker_20261021/verification.json).
Both sessions confirmed 50 Hz / 20 ms / brightness 96 / gain 0 and delivered
2560x1440 at 29.3–29.7 FPS. Moving-row variation measured 0.16–0.23%; stone
brightness variation measured 0.14–0.25%. These checks verify capture and
camera settings; an operator still needs to validate a physical acid test.

## File check and verification

`rojeweltrack.onnx` serves both jewellery tracking and gold-on-stone
segmentation from **one unmodified full-camera-frame prediction** per live
observation. The ROI is applied afterward for bench guarding. Stone masking
reuses the same raw model polygons, before bench-only colour/skin filtering.
There is no separate stone-crop jewellery inference or rubbing inference thread.
A bounded inference scheduler owns the predictor on one thread.
A separate GUI timer previews fresh camera frames without waiting for inference;
old overlays expire. The export has fixed float32 input [1,3,320,320]
and one class, gold. Full-frame live and still-capture passes use 320.
The older `nivagar_gold_v3.onnx` and `rubbing_gold.onnx` files are retained
but are not loaded. Automatic stone detection uses only `newstone1.onnx`; the temporal
acid detector/classifier remain under `acid_test/`.

Automatic stone checks use fresh frames with a target start-to-start interval
of 0.5 seconds, including inference time. The highest-confidence model box
whose centre lies inside the ROI is used (minimum confidence 0.20). Dark-shape
search, darkness/shape/white-bench filters and retention of an old automatic
box during occlusion have been removed. A missed, failed or expired prediction
clears the automatic box. Covering the stone can therefore interrupt an acid
test that requires a stable stone box. The explicit **Draw stone** manual
fallback is still available. Stone inference runs on its own thread. OpenVINO
reads the original `newstone1.onnx` directly, retaining float32 precision and
the fixed 640x640 input, with one stream and two CPU threads. YOLO letterboxing,
confidence, NMS and mask-validity decoding match the original predictor.
Weights and class labels are unchanged. This reduces contention with the
jewellery/rubbing model; actual update speed still depends on CPU load.

Preview overlays now share the 0.5-second observation age limit. Previously,
their 0.3-second expiry included model processing time, causing boxes to blink
off between slow predictions. Camera pixels still come from the latest frame;
reset clears the overlay immediately. The status labels the worker rate as
**detection FPS** to distinguish it from camera preview speed. Model misses
still clear the automatic stone box; expired observations are never extended.

Stone-region gold is displayed as soon as a model mask satisfies confidence
0.15, minimum 80 pixels and at least 50% inside the stone rectangle. Display
does not wait for a one-second missing-jewel timer or a grayscale motion gate.
Rubbing confirmation still requires three recent qualifying movements and
0.6 seconds of credit, associated with a tracked jewel.

Exact stone-coordinate equality is removed. During an inference, stone boxes
must have IoU >= 0.70 and centre displacement <= 15% of their average diagonal;
the jewellery result must be no older than 0.5 seconds. An inconsistent or
expired observation is skipped without erasing rubbing history. Missing gold
masks stop credit and hide the mask while preserving history. Stone disappearance
clears the observer and unfinished rubbing credit; explicit batch Reset also
clears history. Already-confirmed rubbing remains recorded.

Movement is measured in stone-relative coordinates with scale compensation;
box jitter up to 3 pixels with IoU >= 0.95 uses the prior motion reference.
Carrying stone and jewel together should not itself count as rubbing. Large
in-flight stone movements skip one observation; the next fresh observation
can continue at the new position. The **Rub spot** table cell freezes a plain
photo crop around the gold detection when rubbing is confirmed, including
40 pixels of context for the hand and stone. It uses that observation's box
and camera frame without painting masks or outlines. Later frames and the
before/after new-mark check do not replace this photo. The separate confirmed
new-mark mask remains available for acid targeting and the rubbing guidance
preview. Hand occlusion, rotation, perspective and small-jewel accuracy still
require physical testing. The separate acid capture continues to require a
stable stone.

CPU inference and full-frame mask reuse checks passed, along with model
parity and reset/expiry checks. Full physical rubbing accuracy must be checked
with the model on the live camera.
`python -m pip check` passed during environment setup.

The missing identity weights were restored as
`weights/resnet18-f37072fd.pth` from the
[official TorchVision source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
Their SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
The corresponding `weights/resnet18_features.onnx` was generated with its
final classifier replaced by an identity layer. Its 512-dimensional features
were compared against PyTorch on a batch of two inputs and agreed within
1e-4 tolerance (maximum absolute difference 4.06e-6).

The integrated temporal acid test is bundled under `acid_test/`, including
its detector, MobileNet ONNX classifier and checkpoint, preprocessing, and
training source. There are no missing model files required by the active
flow. The legacy `bestnewacid_mobilenet_v3.pt` filter is no longer used.
See [acid_test/README.md](acid_test/README.md) for the copied files and flow.

The acid classifier was updated on 21 October 2026 to
`temporal_shared_mobilenet_v3_small_20261021_144911.onnx` from the original
lab's training artifacts. The tracker loads this local model through its
existing acid runtime; the previous classifier is retained. Restart the
tracker after the update because acid models are cached for the window's
lifetime. Recorded-sequence parity checks and deployment hashes are under
`diagnostics/acid_model_update_20261021/`.

SigLIP2 weights, `jewel_classifier.py`, and the classification caches are
present, but jewel-type classification is disabled in this version's worker.
The tracker requires the supplied `rojeweltrack.onnx`; it does not fall back to a different jewellery model.

## Before/after rubbing guidance and acid ROI

The main application uses `new_rub_detection.flow.RubCheck` with the existing
camera. Its separate Qt6 demo is not launched. When the stone is first visible,
the tracker captures a before-rubbing reference in a background job, detects old
marks with `new_rub_detection/models/rubbedmask.onnx`, and displays the Z1–Z4
grid with a blue recommended clear target. Wait for this guidance before rubbing.
Use **Capture / retake rubbing reference** after clearing hands, or after cleaning
when the result says **CLEAN THE STONE**. An invalid reference blocks rubbing
credit; it does not disable bench guarding or the fresh video preview.

After RUBBED, **Start acid test** first captures an after-rubbing image. Remove
hands/jewellery and keep acid off the stone. The reference and current image are
aligned, including supported stone rotation, and only newly added marks are
accepted. **RETAKE** holds the acid flow and allows another after capture using
the same before reference. The suggested grid zone is guidance; a valid new mark
in another zone is accepted. Once confirmed, the preview and live outline show
only the new-rub acid ROI, rather than the whole grid or old marks.

The pH ON/OFF selection still applies. At **Ready for stone test**, the saved
new-mark mask is re-aligned onto a fresh, clear stone image. If alignment is
unclear or the stone is covered, the flow waits and retains pH progress. Apply
acid only at **NOW APPLY ACID**, over the magenta mark; spreading beyond it is
allowed. The detected acid box must overlap at least 50% of the smaller of the
box and saved mask areas. Its centre must stay inside the stone, but may be
outside the mask and padded streak area. A larger acid box containing the
mark is accepted. Three consistent detections at confidence >= 0.60 are still
required. Detections elsewhere without sufficient overlap, including gaps
between new marks inside their shared bounding box, do not confirm acid
application. Matching uses the detected box rather than a segmented acid outline.

Each completed test saves before/after photos, old/new normalized masks,
`new_rub_roi.png`, grid recommendation and mask acceptance settings in its
capture directory. The next jewel gets a fresh before reference including
previous marks. Cancel, Reset, changed zones and camera reconnection invalidate
in-flight snapshot results. Native snapshot work finishes safely on shutdown.

Snapshot models load lazily and are reused. They run on before/after captures;
only the stone/geometry check repeats at readiness retries. Continuous jewellery
tracking still uses its single full-frame pass at confidence 0.15. The supplied
snapshot rubbed-mask model retains its separate 0.50 threshold. No second camera
reader is opened and no model is retrained.

## Camera and flow

The main worker, ResNet identity model and rubbing watcher load on this PC.
The integrated temporal acid worker starts only after a confirmed rubbed
jewel's **Start acid test** button is clicked. A USB camera became available
during the final check: the live GUI received 2560x1440 frames and reached
LOCKED tracking at about 7.5 FPS. Startup and shutdown both completed cleanly.
The earlier setup had no camera connected. Full physical rubbing and acid
reaction verification still needs an operator test.

Use the top bar to select USB/network/custom camera sources. Redraw ROI/stone
zones if the view differs from the saved `zones.json` view.

Rub any untested jewel first. Once its rubbing is confirmed, remove the
jewel/hands and press **Start acid test for Jn**. Show fresh yellow pH paper
in the camera, then add acid to that same paper until its pink reaction is
verified. This OpenVINO CPU check is required for every jewel and every retry.
After pH verification the test pauses. Remove paper/hands and click
**Ready for stone test**, wait through the two-second stone baseline, apply acid to the streak
only at **NOW APPLY ACID**, and keep still for observation. The same jewel's
row receives **22K GOLD**, **NON-GOLD**, or **INCONCLUSIVE**, with confidence.
Then rub any next untested jewel. The original detection/identity guarding
continues during the acid capture. Only one jewel's cycle can be active.
The next rubbing is not credited until the current classification finishes.
Cancel/error allows retry for the same jewel; Reset starts a fresh batch.

Captures and prediction manifests are saved under `acid_test/captures/`.
Verified labels are left empty; these records are not automatically used as
training ground truth. See the local module README for review/conversion
requirements before training. The optional acid grade/batch and observation
duration are recorded with each test.

Run integration checks without a camera:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

These checks cover J2-before-J1 ordering, rubbing gating, per-jewel results,
reset/cancel token rejection, safe worker shutdown, capture stages, off-streak acid rejection,
model inference, archival, and the background worker. Full live operation
still requires a physical camera and operator test.

The OpenVINO pH model loads on the first acid-test click; the temporal acid
models load after pH verification. Model sessions remain cached for later jewels,
while pH verification restarts for every attempt. Idle acid sessions do no live inference. The table's
Acid crop column displays the selected jewel's detected acid region during
observation and retains its final image. See the module README for details.

## Live scheduling update (21 October 2026)

See [LIVE_PIPELINE_CHANGE.md](LIVE_PIPELINE_CHANGE.md) for the change scope,
replay measurements, regression coverage, and backup. Launch with the same
`./run_tracker.sh` command; no model replacement or new dependencies are needed.


## pH-paper gate (22 October 2026)

The side panel now has a **pH verification (ON/OFF)** checkbox, defaulting to
**ON** each time the application opens. ON keeps the existing pH-paper flow.
OFF skips pH-model loading and verification and proceeds to **Ready for stone
test**, followed by the normal baseline, acid observation and classification.
Choose before starting an attempt; the checkbox is locked while it runs.
Results record skipped verification as `SKIPPED_BY_OPERATOR`, with
`ph_verification_enabled: false`. The choice is retained for later attempts
in the same application session.

The integrated flow bundles the supplied `acidphintegration/acidph_openvino/`
XML/BIN model and the original headless color SOP. Install the added
`openvino==2026.4.1` dependency through `requirements.txt`. The tracker reuses
its camera feed; the standalone pH GUI is not opened. The pH model uses one
CPU inference stream with two threads. It only runs during an active pH check.
Evidence is saved in the same capture directory as the jewel's acid result.
See [acid_test/README.md](acid_test/README.md) for the full operator sequence.


The pH gate is installed in both the active `jewel_tracker` application and
`jewel_tracker_full_backup`. Each launcher uses its own `.venv`; both require
OpenVINO. Startup logs now identify the script and Python interpreter. After
Start acid test, Events logs `PH_LOADING`, `PH_READY`, `PH_FRESH`,
`PH_REACTION`, `PH_VERIFIED`, and `WAIT_FOR_STONE_READY` before the stone
baseline. The **Ready for stone test** button is enabled only in the waiting
phase, and the baseline starts only after that button is clicked. Model-loading
errors remain visible for retry, and their tracebacks go to `jewel_guard.log`.
The active application's CPU loading check is recorded in
`diagnostics/ph_gate_loading_20261022/model_check.json` (no physical acid test).
