# Jewel Tracker 5 on this PC

Application folder: `/home/embsys/Desktop/jewel_tracker/jewel_tracker`.
Python environment: `.venv` (Python 3.10, CPU PyTorch).

Version 5 now uses `int_8_GOLD_JEWEL/gold_int8.xml` and its companion `.bin`
for jewellery segmentation (320x320), and
`INT_8_RUBBEDMASK/rubbedmask_det_v3_int8.xml` plus `.bin` for mark detection
(640x640). Both run through OpenVINO on CPU with two inference threads and
one stream, reusing loaded sessions. The jewellery pass still supplies live
tracking and gold-on-stone contact together. Acid and stone models are unchanged.

The supplied rubbed-mark export is a **box detector**, not a segmentation model.
Its candidate rectangles go through the existing stone-relative before/after
change, old-mark exclusion, clear-frame persistence and contact-location checks.
The resulting changed region is not a model-generated pixel segmentation;
the audit records this as `DETECTION_BOX_WITH_BEFORE_AFTER_CHANGE`.
The PDF's new-mark panel shows the rubbed region without the jewellery-contact
outline; the separate contact panel keeps its contact evidence.

Acid detections now start a provisional recording/verification stage. One
freshness-qualified model observation seeds a candidate; subsequent fresh
camera frames perform lightweight checks in stone-relative coordinates without
another model pass. Two distinct clear observations, at least 30 ms apart,
must show a persistent changed patch with low local movement. Skin/glare,
unknown pose, broad surface change and straight elongated tool-like changes
prevent confirmation. A stationary needle-like line is not accepted merely
because it stopped moving. This is a visual heuristic, not a needle model or
proof of chemistry.

The timestamped journal keeps recording throughout. Candidates have a five-second
verification window; rejected candidates leave acid search active, with no manual
handoff. A confirmed application is latched once and its reaction ends five seconds
after the first supported clear-patch frame. If the onset was obscured, the report
records that timing limitation rather than using an unverified needle timestamp.
Live verification and bounded recorded recovery apply the same checks. Candidate
events, verification/onset times and provisional video links are included in the
audit. At most two candidate states and sixty audit events are retained per cycle.

## Run

The tracker now opens in **continuous capture** mode. Configure the bench ROI
and optionally **Draw test position ROI** once; otherwise the first visible
stone establishes a displayed test-position rectangle.
The automatic rectangle expands during initial setup as the complete stone
becomes visible, then freezes when tracking starts. A manually drawn ROI stays
fixed throughout. IDs/crops target two seconds after jewels first appear with
the stone present. There is no per-jewel
Ready, Retake or Retry action. Old manual adapters remain available to their
standalone tools, but do not control the main window's continuous flow.

The camera records timestamped JPEG evidence at a target 15 FPS on an independent,
bounded writer. Processing uses the newest frame; queue drops and camera gaps
are saved. A visible held jewel uses the existing skin/contact heuristic, and a
covered jewel uses the existing under-hand/bench-motion evidence. Absence alone
does not establish an in-hand identity. Ambiguous or completely missed IDs stay
explicit in the evidence record.

Trial pickup policy: **UNDER_HAND alone does not start a jewel's recording/test
cycle**, and does not seed its recent pickup history. IN_HAND evidence can start
the cycle; a cycle already in progress continues through later hand coverage.
The full session recording and under-hand visibility notes remain available.
`PICKUP_STATES` in `continuous_testing.py` currently contains only `IN_HAND`.

Gold-on-stone contact acceptance is scoped to that recording cycle. With jewels
resting on the bed, the loaded jewellery model continues ID tracking, but its
stone polygons are not interpreted as rubbing contact. A pickup begins a fresh
contact/movement history. Contacts from another cycle or an inference started
before the gate opened are rejected. A jewel appearing back on the bed immediately
suppresses new contact updates; two distinct fresh return observations spanning
at least 120 ms confirm the return and close the gate. Confirmed new masks,
baseline start (including a failed-mask decision), acid capture, cycle completion
and reset also close it. The best saved contact photo/mask stays frozen. Only
the next pickup cycle can reopen acceptance; recorded contact recovery observes
the same closing timestamp. No new model is loaded at pickup.

The before-stone reference now starts as soon as a clear stone is available,
including while initial IDs are still being assigned. Its image is saved under
the session's `references/` folder, shown in the side panel and included in the
PDF for each jewel. Development text on the video shows the current process,
new-mask outcome, acid detection state, reaction countdown and completed IDs.

After jewel return and stone placement inside the displayed ROI, the new-rub
check completes with a detected/not-detected decision (maximum one-second
decision window). The baseline stage starts after that decision; camera evidence
continues recording throughout. Missing masks can still be recovered later.
Model preparation happens before the test. Corner continuity handles a plain
stone whose texture cannot be aligned; its orientation through complete
occlusion remains an assumption recorded in the manifest. Missing evidence never
asks the operator to repeat the process.

One qualifying fresh-acid detection at confidence 0.60 seeds a provisional candidate.
After clear-patch verification, further acid detection stops for that jewel
and reaction timing uses the first supported **camera frame timestamp**. The reaction is
then classified and the jewel's DONE indicator stays visible during the next
test. When acid is not confirmed, its search closes at a reliably observed next
jewel pickup (0.3-second debounce and distinct fresh detections), or after 12
seconds of genuine SOP inactivity. The last jewel also finalizes automatically.
Fresh camera frames drive the inactivity timer; stream gaps do not count.
Meaningful movement on the stone and supported rubbing reset that timer;
stationary residue and repeated model misses do not. These values are configurable
in `sop_config.json`. J1's final recovery is detached from J2's live processing;
J1's recovery interval ends at J2's first reliable pickup, excluding J2's footage. Recorded-frame recovery also runs
while waiting and can confirm an application missed live. Acid inference uses a
padded stone crop. Failed live rubbing,
mask and acid checks are recovered from the recorded sequence in a background
worker. Pre-acid frames alone can establish a new rubbing mask. Two seconds
before detected onset and five seconds after it are exported with timestamp
sidecars; incomplete footage, coverage and wiping are recorded separately.
A source-frame gap or insufficient usable reaction coverage prevents material
classification; the report preserves the acid event and explains the missing result. A
two-second video buffer does not guarantee two clear baseline seconds. Models,
thresholds and classifier preprocessing are unchanged; the faster temporal capture
protocol still requires field validation on labelled sequences.

Previous acid footprints persist in stone coordinates between jewels and through
hand coverage. The fresh-application gate runs for live detections, background
recovery and final export. Unchanged baseline patches and detections overlapping
previous acid do not start another timer. A new detector footprint must have a
localized appearance change outside remembered acid, with recent known stone
geometry. The present detector cannot establish a new dose in an already wet
patch; that case stays unconfirmed until handoff/inactivity closes the search. It adds no fixed wait and
does not request a repeat. Footprints use boxes rather than chemical masks;
camera/stone orientation through complete occlusion remains an assumption.

Residue history is cleared only after at least three stable clear negative
observations over 0.6 seconds AND visual similarity to an earlier observed dry
surface. Missing detections or hand coverage alone never clear it. This is visual
evidence, not a measurement of chemical cleanliness. A new rubbing overlapping
previous acid receives a contamination warning independently of the model result.

Recorded rubbing recovery checks only the current jewel's pickup-to-return,
pre-acid interval. It reuses the jewel segmentation model at low priority,
clips gold evidence to the stone, rejects skin-dominated masks, and saves a
stone close-up with the visible gold marked. Back-and-forth motion is required
to call rubbing recovered. Contact-only and not-visible cases are distinguished;
a new mark alone is not proof of visible rubbing contact. Photos include source
frame timestamps. Background work samples at most 24 frames per jewel.

The live UI has separate timestamped **Rubbing contact** and **New rubbing mask**
panels. Contact boxes/masks originate from the same immutable frame used by live
jewellery inference; later preview frames cannot substitute for that source.
The contact mask, origin, box, source frame and time are saved per cycle.

Recorded recovery tries jewellery segmentation, then an in-stone jewellery box
from the same inference, then localized image movement. Box-only and motion-only
fallbacks are unverified. New marks need three distinct clear observations spanning
at least 0.12 seconds, remain within the stone and fixed test ROI, exclude old
marks, and overlap a segmentation-supported contact region. Unrelated candidates
are rejected; missing contact leaves the candidate unverified. Baseline/acid
capture continues after the one-second mask decision target without a manual retry.
The same checks apply during final video recovery. A new mark alone never verifies
rubbing motion.

The visual PDF follows the proposed audit layout: session overview, one visual
page per test, then full-detail appendices. It includes the original jewellery,
contact, verified/rejected candidate mark, acid placement and first/final usable
reaction frames. Audit outcomes are EVIDENCE VERIFIED, REVIEW REQUIRED, or
INCOMPLETE; material predictions remain separate. Missing frames use explicit
placeholders and every warning/timestamp remains available in the appendix and
JSON. No model or retraining is added.

Final contact recovery samples at most 24 frames and acid recovery at most 60.
The live model dispatcher retains bench priority. Two finalization snapshots can
wait in RAM; overload is serialized to a private session-local recovery spool,
which the single recovery worker drains. PDF writes run in one separate worker
with coalesced requests. Cycle tokens plus ROI/reset generations prevent late
jobs from overwriting a different cycle, even when jewel IDs are reused.

The PDF includes a same-frame comparison: magenta new-rub contour and cyan acid
detector footprint, with the overlap status and a warning when locations differ.
The 50% overlap-of-the-smaller-region check does not block the classifier or
replace its result. A missing mask makes the comparison unavailable. Runtime
INVALID explanations distinguish a model INVALID class from the existing 65%
general and 85% gold confidence thresholds; they do not invent a physical cause.

Outputs are in `captures/sessions/<session>/`: `report.pdf`, `session.json`, a
frame index and recorded frames, plus per-test masks, pictures, reaction crops,
pickup/rubbing/return video and acid video. Reports update automatically after
each result and at disconnect/reset. They contain only IDs, actual pictures,
tracking visibility, rubbing/mask evidence and acid-model output. No unsupported
stone-risk, thread, weight or sealing sections are added. Process warnings never
replace `gold_22k`, `non_gold` or `invalid` from the acid runtime; a missing runtime
result is displayed as **No acid result generated**. The runtime's `invalid`
class is displayed as **INVALID (acid model)**. Continuous capture does not wait
for the old pH-paper stage; the manifest/PDF records that pH was not observed.

Initial analysis and mask timing are targets, dependent on actual camera frames
and CPU load. Recording continues when a deadline cannot produce confirmed
evidence. `reportlab` is required and included in `requirements.txt`.

Continuous mode uses position tracking plus recent hand history for a unique
returned-jewel association. It does not start the older per-session ResNet
training job. Corner tracking supplies the photo geometry without repeating
stone-model inference for each mask. The stone model refreshes at up to 5 FPS
while corner tracking continues on fresh frames; idle acid checks run at up to
2 FPS and armed acid checks at up to 5 FPS to keep CPU available for mask capture.

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

## Earlier implementation and validation notes

The sections below describe earlier releases and the retained legacy adapters.
For the current main-window workflow, use the continuous-capture instructions
above; manual rubbing gates, Retake/Ready actions and per-session ResNet training
described below are superseded in continuous mode.

`int_8_GOLD_JEWEL/gold_int8.xml` serves both jewellery tracking and gold-on-stone
segmentation from **one unmodified full-camera-frame prediction** per live
observation. The ROI is applied afterward for bench guarding. Stone masking
reuses the same raw model polygons, before bench-only colour/skin filtering.
There is no separate stone-crop jewellery inference or rubbing inference thread.
A bounded inference scheduler owns the predictor on one thread.
A separate GUI timer previews fresh camera frames without waiting for inference;
old overlays expire. The export has fixed float32 input [1,3,320,320]
and one class, gold. Full-frame live and still-capture passes use 320.
The older `nivagar_gold_v3.onnx` and `rubbing_gold.onnx` files are retained
but are not loaded. Live stone detection uses `INT8_openvino_model/best.xml`
and its companion `best.bin`; the temporal acid detector/classifier remain
under `acid_test/`.

The integrated acid-on-stone detector currently uses
`acid_test/models/acid_detector.onnx` (ONNX Runtime CPU). The experimental
`int_8_acid_model/` export is retained but is not loaded by the application.

Automatic stone checks run continuously on the newest available camera frame.
There is no fixed 0.5-second interval and no queue of old frames. A 5 ms idle
wait applies only when the camera has no new frame. The highest-confidence model box
whose centre lies inside the ROI is used (minimum confidence 0.20). Dark-shape
search, darkness/shape/white-bench filters and retention of an old automatic
box during occlusion have been removed. A missed, failed or expired prediction
clears the automatic box. A confirmed acid snapshot keeps its saved box/mask
through these live model misses using the fixed region described below. The explicit **Draw stone** manual
fallback is still available. Stone inference runs on its own thread. OpenVINO
loads the supplied INT8 IR at fixed 640x640 input, with one stream and two CPU
threads. Inputs remain normalized float32 BGR-to-RGB tensors; the quantized
graph runs integer operations internally. YOLO letterboxing, confidence, NMS
and mask-validity decoding match the INT8 model's Ultralytics predictor.
The original ONNX files remain available for reference comparisons. Before-rubbing
snapshots share the live worker's INT8 model, including its segmentation output.
Actual update speed and detection accuracy depend on CPU load and the new weights.

The 26 October saved-frame replay measured stone updates increasing from 2.0
to 4.8 per second, with concurrent mean stone processing falling from 376 to
210 ms. Jewellery updates changed from 11.0 to 9.3 per second because the stone
worker is now continuous. See the [INT8 measurements](diagnostics/stone_int8_20261026/REPORT.md)
for the method, detection differences and live-validation limits.

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
The tracker requires the supplied `int_8_GOLD_JEWEL/gold_int8.xml` and `.bin`;
it does not silently fall back to a different jewellery model.

## Earlier before/after rubbing guidance and acid ROI

The main application uses `new_rub_detection.flow.RubCheck` with the existing
camera. Its separate Qt6 demo is not launched. When the stone is first visible,
the tracker captures a before-rubbing reference in a background job. It uses the
highest-confidence INT8 stone prediction inside the configured ROI, derives the
oriented surface quadrilateral using the photo app's contour geometry, and keeps
the actual segmentation mask inside that outline. The enclosing rectangular box
is never filled as the usable surface: test-bed pixels outside the mask are
excluded even when inside that box. This cannot repair a model mask that itself
incorrectly includes the bed; unclear surface geometry requires a retake.

The flow uses the stone model and image alignment without AprilTag detection.
A clear before-rubbing image stores the stone mask, oriented quad and starting
marks. The flow detects old
marks with `new_rub_detection/models/rubbedmask.onnx`. It saves these marks as the
baseline without recommending a zone or drawing a target. Wait for the reference
before rubbing. The original clear-space check is retained: when existing marks
leave no usable clear patch, **CLEAN THE STONE** asks the operator to clean the
stone and capture a new reference. Rubbing credit and acid testing stay blocked
until a valid reference is captured. This check runs even when the old-mark
overlay is OFF, and does not display a zone recommendation or blue target.
Use **Capture / retake rubbing reference** after clearing hands or cleaning the
stone. An invalid reference blocks rubbing
credit; it does not disable bench guarding or the fresh video preview.
An unsuccessful manual retake keeps the previous reference/masks in memory while
blocking new rubbing credit until the retake succeeds. A validated dirty surface
still requires cleaning; Reset, camera/ROI changes and a completed jewel start a
new reference cycle. Captures process one fresh image in the background, with no
marker search, capture burst or marker-related retake requirement.

**Automatic tests (ON)** is enabled by default. After Jn is RUBBED, return that
same tracked jewel to the bench, outside the stone. Two seconds of observed
settling automatically starts the after-rubbing check. The jewel check uses its
smoothed centre with a fixed tolerance (minimum 6 pixels, or 3% of the initial
box diagonal); changing box width/height does not restart the timer. The stone
check uses a fixed reference crop and image changes, ignoring later predicted
box dimensions. It tolerates small uniform brightness changes. Brief detection
misses pause settling, and gaps over 0.75 seconds restart unfinished settling;
only fresh camera timestamps count. The saved reference/masks are retained.

The automatic trigger captures an after-rubbing image. With Automatic tests OFF,
use **Start acid test** to capture it manually. Remove
hands/jewellery and keep acid off the stone. The reference and current image are
aligned using the detected stone corners and existing image matching. Both photos
need a clear stone surface for the new-mark comparison. Only newly added marks are
accepted. **RETAKE** holds the acid flow and allows another after capture using
the same before reference. A new mark can appear anywhere on the stone. Regions
touching old marks, including a 12-pixel buffer in the normalized reference, are
excluded completely. Rubbing over an old mark or leaving the stone unchanged
shows **No new rub detected. Please rub on a clear area and retry.** Uncertain
alignment remains a separate retry error and does not confirm a new mark.

**Show existing rub marks (OFF/ON)** controls green starting-mark overlays in
the before/after preview. It defaults to OFF. Toggling redraws the saved preview
immediately without capturing another image or changing the baseline masks or
detection. Confirmed new marks remain magenta when this switch is OFF, and the
live acid ROI remains available. The fixed **Rub spot** photo is separate.

The pH ON/OFF selection still applies. Once the new rub is confirmed, its stone
box and mask are fixed. If pH is ON, fresh-paper/reaction verification must pass
first; OFF explicitly skips it. In Automatic mode, **AUTO_STONE_SETTLE** then
waits for two seconds of quiet images in the confirmed stone region before
starting the baseline. Remove paper/hands during this handoff. Manual mode waits
for **Ready for stone test**. Both modes reuse the saved region without another
stone inference or image-alignment requirement. Keep the stone
in the same position from new-rub confirmation until the test finishes. Apply
acid only at **NOW APPLY ACID**, over the magenta mark; spreading beyond it is
allowed. The detected acid box must overlap at least 50% of the smaller of the
box and saved mask areas. Its centre must stay inside the stone, but may be
outside the mask and padded streak area. A larger acid box containing the
mark is accepted. Three consistent detections at confidence >= 0.60 are still
required. Detections elsewhere without sufficient overlap, including gaps
between new marks inside their shared bounding box, do not confirm acid
application. Matching uses the detected box rather than a segmented acid outline.

Temporary hand occlusion and missing or partial live stone predictions do not
abort Ready/capture or erase the confirmed box and mask. The acid flow does not
re-check live stone positions or wait for a marker before classification. The
operator keeps the stone still after confirmation. Camera loss, stale frames and
changed camera resolution still require a retry.

Each completed test saves before/after photos, old/new normalized masks,
`new_rub_roi.png`, reference details and mask acceptance settings in its
capture directory. After completion, **Next jewel paused** waits for cleaning;
it does not immediately capture another before reference. The existing acid
detector checks the whole oriented stone at confidence 0.60. A valid, uncovered
surface, at least three acid-free observations, and two seconds of observed
image stability are required before the next reference is saved automatically.
Acid, hand/cloth coverage, missing surface observations and movement keep the
next jewel blocked without erasing the previous jewel's result. Only fresh
frames count; a camera/processing gap over 1.5 seconds restarts this check.
The accepted clear photo supplies both the stone mask and the next before
reference, including any remaining old marks. The operator must dry the stone;
the detector's absence of acid is not a measurement of dryness. Manual retake
cannot bypass the cleanup wait. This handoff also runs with Automatic tests OFF.
Acid checks stop once the reference is ready. Cancel, Reset, changed zones and camera reconnection invalidate
in-flight snapshot results. Native snapshot work finishes safely on shutdown.

The rubbed-mask model loads lazily and is reused on before/after captures. The
shared INT8 model is serialized during cleanup and before/after surface captures; Ready
and acid capture reuse the confirmed geometry without further stone inference. Continuous jewellery
tracking still uses its single full-frame pass at confidence 0.15. The supplied
snapshot rubbed-mask model retains its separate 0.50 threshold. No second camera
reader is opened and no model is retrained.

## Earlier camera and gated flow

The main worker, ResNet identity model and rubbing watcher load on this PC.
The integrated temporal acid worker starts after a confirmed new-rub mask,
triggered by automatic settling or the manual **Start acid test** button. A USB camera became available
during the final check: the live GUI received 2560x1440 frames and reached
LOCKED tracking at about 7.5 FPS. Startup and shutdown both completed cleanly.
The earlier setup had no camera connected. Full physical rubbing and acid
reaction verification still needs an operator test.

Use the top bar to select USB/network/custom camera sources. Redraw ROI/stone
zones if the view differs from the saved `zones.json` view.

Rub any untested jewel first. With **Automatic tests ON**, return that jewel to
the bench and keep it and the stone settled for two seconds. The after image
must confirm a new rub. With pH ON, show fresh yellow paper, then add acid to
that same paper until the pink reaction is verified. With pH OFF, paper is
skipped. Remove paper/hands; after two seconds of stone settling, the baseline
starts automatically. Apply acid only at **NOW APPLY ACID** and keep still for
observation. Automatic tests OFF retains the Start/Ready buttons. The same
jewel's row receives **22K GOLD**, **NON-GOLD**, or **INCONCLUSIVE**, with
confidence. Only one jewel cycle runs at a time, while identity guarding continues.
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
`PH_REACTION`, `PH_VERIFIED`, then `AUTO_STONE_SETTLE` (automatic mode) or
`WAIT_FOR_STONE_READY` (manual mode) before the stone baseline. The Ready button
is enabled only for manual readiness; automatic mode advances after settling. Model-loading
errors remain visible for retry, and their tracebacks go to `jewel_guard.log`.
The active application's CPU loading check is recorded in
`diagnostics/ph_gate_loading_20261022/model_check.json` (no physical acid test).
