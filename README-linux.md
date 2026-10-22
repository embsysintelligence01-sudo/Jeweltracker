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

`rojeweltrack.onnx` now serves both jewellery tracking and gold-on-stone
rubbing segmentation. One `SharedGoldModel` instance is passed to both
paths. A bounded inference scheduler owns the predictor on one thread and
prioritizes pending bench checks. Bench and rubbing mask processing still run
independently, retaining the original rubbing check limit and decision rules.
A separate GUI timer previews fresh camera frames without waiting for inference;
old overlays expire. The export has fixed float32 input [1,3,320,320]
and one class, gold. Bench, still-capture and stone-crop passes all use 320.
The older `nivagar_gold_v3.onnx` and `rubbing_gold.onnx` files are retained
but are not loaded. The stone detector remains `newstone1.onnx`; the temporal
acid detector/classifier remain under `acid_test/`.

CPU inference on a saved capture and the rubbing crop path passed, along
with concurrent-call and shared-instance checks. Full physical rubbing
accuracy must be checked with the new model on the live camera.
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
jewel/hands and press **Start acid test for Jn**. Wait through the baseline,
apply acid when prompted, and keep still for observation. The same jewel's
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

Acid models load and warm up only on the first acid-test click, then remain
cached for later jewels. Idle acid sessions do no live inference. The table's
Acid crop column displays the selected jewel's detected acid region during
observation and retains its final image. See the module README for details.

## Live scheduling update (21 October 2026)

See [LIVE_PIPELINE_CHANGE.md](LIVE_PIPELINE_CHANGE.md) for the change scope,
replay measurements, regression coverage, and backup. Launch with the same
`./run_tracker.sh` command; no model replacement or new dependencies are needed.
