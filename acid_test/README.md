# Local acid-test module

This folder contains a local copy of the temporal acid lab code and its active
models. Jewel Tracker needs no imports or model files from `puritytesting`.

- `app.py`: copied standalone lab UI, adapted to local package imports/models.
  Run separately with `../.venv/bin/python app.py`, or from the tracker root:
  `.venv/bin/python -m acid_test.app`. It uses Qt6 and its own camera.
- `runtime.py`: `TemporalMobileNet` inference extracted from the original
  `app.py`, shared by the integrated tracker and the standalone lab UI.
- `sequence.py`: the baseline, pouring buffer, stable detection, and timed
  observation logic adapted from `app.py`, with no GUI dependency.
- `integration.py`: Qt5 background capture/classification adapter. It reads
  the tracker's existing camera grabber and never opens another camera.
- `onnx_detector.py`, `analyzer.py`, `analyzer_v2.py`, `storage.py`, `flow.py`:
  copied detection, analysis, storage and geometry helpers.
- `models/acid_detector.onnx`: copy of `acidbestmodel_hailo_ready.onnx`.
- `model_training/artifacts/temporal_shared_mobilenet_v3_small_20261021_144911.onnx`:
  the active classifier trained on 21 October 2026 and copied from the lab.
  Its PyTorch checkpoint, report, split, and dataset index are included.
  The previous `temporal_mobilenet_v3_20260912_170149.onnx` and companion
  artifacts remain available for rollback.
- `model_training/`: copied architecture, dataset preparation and training
  code for future development. The original lab's dataset is not copied.

The main tracker imports the Qt5 adapter and shared inference code. Do not
import the standalone Qt6 UI into the Qt5 tracker process or open it against
the same camera concurrently.

## Model reuse and acid preview

The tracker window owns one `AcidModelCache`. The OpenVINO pH-paper model loads and warms up on the first Start acid test
click after rubbing is confirmed. The temporal acid models load only after
the pH reaction is verified.
Later jewels, retries, resets, and camera reconnects in the same window reuse
those model sessions, but each attempt starts a new pH-paper verification. There is no background acid prediction while idle or rubbing:
pH inference runs only during the pH check, live acid detection runs only in WAIT_FOR_ACID, and the temporal classifier
runs once after observation. CPU sessions use two threads each with idle
spinning disabled. Restart the tracker to load newly replaced/retrained models.

The tracker selects the newest supported ONNX artifact by modification time;
copying today's artifact preserved its training timestamp. Its SHA-256 is
`868d1d8d1bfcae2f9c2a675df0b4161eba7c93a20119ed9868a81d104989b473`.
The three classes remain `gold_22k`, `non_gold`, and `invalid`, in that order.
Preprocessing, gold/general confidence thresholds (0.85/0.65), the acid
detector, observation timing, camera controls, and jewel/rubbing logic were
not changed for this update. The existing runtime already supports this
shared MobileNet architecture, so no Python source changes were required.

Deployment verification compares the lab's inference methods and the actual
tracker model cache on all 14 recorded validation sequences. See
[`../diagnostics/acid_model_update_20261021/verification.json`](../diagnostics/acid_model_update_20261021/verification.json)
for per-sequence results and model selection checks. This checks integration
and prediction parity; it is not an independent accuracy or physical acid
test. The training report marks this model as a development candidate.

The **pH / Acid crop** column first shows the detected pH paper, then the
detected acid box during observation,
updated only for its assigned jewel. Its last image remains next to the result;
the full-resolution crop is saved as `acid_preview.png` with the sequence.
Token checks reject previews from cancelled tests or old batches. Thumbnails
are bounded to 224 pixels and update one cell without rebuilding the table.

## Operator sequence

1. Prepare a clean, dry stone or fresh reaction area. Place jewels/stone and
   let the existing tracker assign identities. Keep hands clear while the
   automatic before-rubbing reference records old marks without recommending a
   target. Use **Capture / retake rubbing reference** if
   requested; clean the stone when no usable clear space remains.
2. Wait for reference guidance, then rub any untested jewel. The existing segmented gold-centroid movement
   logic assigns rubbing to that jewel; numeric order is unrestricted.
3. When Jn is confirmed RUBBED, further rubbing credit is paused. With
   **Automatic tests ON** (default), return that same jewel to the bench outside
   the stone; two seconds of settled jewel position and stone image starts the
   after-photo check. Box size/aspect-ratio changes are ignored. Brief misses
   pause the timer; clear movement restarts it. With Automatic tests OFF, use
   **Start acid test for Jn**. Select duration/batch before the test starts.
   Only a confirmed new mark enables acid verification; RETAKE retains the old
   reference. The accepted new mark becomes this jewel's saved acid ROI.
4. With **pH verification ON**, show unused yellow pH paper in the existing camera view. OpenVINO OBB
   detects the strip; the supplied color SOP requires 8 fresh-yellow detections,
   then 2 sustained pink-reaction detections on the same strip. Add acid to
   the paper while keeping it visible. Pre-colored paper, obscured paper and
   a missing strip cannot complete this check. Repeat for every jewel and retry.
   The check verifies an indicator reaction; it does not identify the reagent.
   With pH OFF this stage is skipped and recorded as `SKIPPED_BY_OPERATOR`.
   Hiding the stone during the pH stage does not restart verification.
5. After **pH reaction verified** (or pH OFF), remove paper/hands. Automatic
   mode enters AUTO_STONE_SETTLE and starts the two-second baseline after two
   seconds of quiet images. Manual mode waits for **Ready for stone test**.
   The confirmed new-rub box/mask stays fixed; no new stone-model detection or
   alignment is required at Ready or during acid capture. Keep the stone in
   position. Apply acid over the magenta mark at **NOW APPLY ACID**;
   the liquid may spread beyond the mark. Detection must repeat three times
   at IoU >= 0.45 and confidence >= 0.60, with its centre inside the stone.
   The detected acid box must overlap at least 50% of the smaller box/mask
   area. Its centre can be outside the new-rub mask and padded streak area:
   a larger acid box covering the new mark is accepted. Acid elsewhere
   without sufficient overlap is ignored and the prompt asks for the correct
   location. This check uses the detector box, not a segmented liquid outline.
6. Keep the stone/camera still during observation (5 seconds by default).
   The table receives **22K GOLD**, **NON-GOLD**, or **INCONCLUSIVE** plus
   confidence. Clean and dry the stone. **Next jewel paused** checks for acid
   across the whole oriented stone, using the existing detector at 0.60.
   At least three acid-free observations, an uncovered surface and two seconds
   of stillness trigger the next before-rubbing reference automatically.
   Wiping, hand/cloth coverage and missing surface observations keep waiting;
   the completed result remains saved. The accepted clear photo also records
   remaining old marks. Dryness remains the operator's responsibility.
   Manual retake cannot skip this wait; it applies with Automatic tests ON or
   OFF. Rub the next jewel only after the reference is ready.
7. The existing DONE behavior additionally waits for the jewel to rest off
   the stone. DONE means the process finished, including an inconclusive
   classification; it does not mean that the material was classified gold.

Capture defaults match the original lab: baseline/observation at 5 FPS, a
15 FPS pouring buffer of up to 15 seconds (JPEG quality 95), and a reaction
crop padded by 45% per side. Capture rates are targets and depend on CPU and
camera throughput. The classifier compares the baseline median, the last
three pouring frames' median, and the last five observation frames' median.
The copied inference applies RGB letterboxing to 224x224 with fill 24,
ImageNet normalization, and thresholds 0.85 for gold and 0.65 generally.
`runtime.py` is shared with the standalone UI; integrated capture behavior
is in `sequence.py` and `integration.py`.

Cancel or an error keeps the same rubbed jewel and saved target for retry.
Automatic relaunch is paused after cancellation or an acid error; use Start or
switch Automatic tests OFF/ON to resume. A failed photo check waits for movement
before an automatic retry. No test is automatically repeated on a static scene.
Reset/disconnect cancel the attempt; batch/test tokens reject late results.
Camera resolution changes and stale frames still fail the attempt. Temporary
missing or partial live stone boxes cannot abort a confirmed target's capture.

There are separate 120-second pH and acid-detection timeouts, and a three-second
camera-frame timeout. With pH ON, no stone baseline/detection/classification
starts before the reaction is verified. Each automatic attempt then settles the
stone; each manual attempt requires its own Ready click. The result manifest
records `automatic_test` along with the pH setting and verification evidence.

## Saved captures and future training

`captures/<batch>_Jn_<test>/manifest.json` stores the jewel identity, acid
metadata, model/detector hashes, detector/reaction geometry, probabilities,
and capture timing. `ph_verification` records the indicator result, model/weight
hashes, color metrics, and paths to `ph_baseline.png`, `ph_reaction.png`, and
`ph_reaction_mask.png`. The three subfolders contain lossless cropped frames.
Captures are committed as one directory only after saving succeeds.

These records use `jewel_acid_prediction_v1` with `actual_label: null`.
Predictions are not verified training labels. The copied original training
pipeline expects verified `acid_temporal_v2` samples, so an explicit review
and conversion step is required before using integrated captures for training.
The standalone lab retains its manual ground-truth labeling workflow.

The model distinguishes the known 22K reference class, non-gold and invalid;
it does not estimate numerical purity or other karats.
