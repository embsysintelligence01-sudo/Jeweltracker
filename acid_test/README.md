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

The tracker window owns one `AcidModelCache`. Models load and warm up with
synthetic frames on the first Start acid test click after rubbing is confirmed.
Later jewels, retries, resets, and camera reconnects in the same window reuse
those sessions. There is no background acid prediction while idle or rubbing:
live acid detection runs only in WAIT_FOR_ACID, and the temporal classifier
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

The **Acid crop** column shows the detected acid box during observation,
updated only for its assigned jewel. Its last image remains next to the result;
the full-resolution crop is saved as `acid_preview.png` with the sequence.
Token checks reject previews from cancelled tests or old batches. Thumbnails
are bounded to 224 pixels and update one cell without rebuilding the table.

## Operator sequence

1. Prepare a clean, dry stone or fresh reaction area. Place jewels/stone and
   let the existing tracker assign identities.
2. Rub any untested jewel. The existing segmented gold-centroid movement
   logic assigns rubbing to that jewel; numeric order is unrestricted.
3. When Jn is confirmed RUBBED, further rubbing credit is paused. Remove the
   jewel and hands and press **Start acid test for Jn**. Select the observation
   duration and optionally enter the acid grade/batch before pressing Start.
4. Wait through the two-second baseline. Apply acid only at **NOW APPLY ACID**.
   Detection must repeat three times at IoU >= 0.45, confidence >= 0.60, inside
   the stone and the selected jewel's padded rubbing region. Acid elsewhere
   on the stone is ignored.
5. Keep the stone/camera still during observation (5 seconds by default).
   The table receives **22K GOLD**, **NON-GOLD**, or **INCONCLUSIVE** plus
   confidence. Any next untested jewel may then be rubbed.
6. The existing DONE behavior additionally waits for the jewel to rest off
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

Cancel or a camera/detection error keeps the same rubbed jewel selected for
retry. Prepare its streak appropriately before retrying. Stone movement or
camera resolution changes fail the capture. Reset or camera disconnect
cancels a running test; unique batch/test tokens reject late results. A new
batch must establish rubbing again. There is a 120-second acid-detection
wait timeout and a three-second camera-frame timeout.

## Saved captures and future training

`captures/<batch>_Jn_<test>/manifest.json` stores the jewel identity, acid
metadata, model/detector hashes, detector/reaction geometry, probabilities,
and capture timing. The three subfolders contain lossless cropped frames.
Captures are committed as one directory only after saving succeeds.

These records use `jewel_acid_prediction_v1` with `actual_label: null`.
Predictions are not verified training labels. The copied original training
pipeline expects verified `acid_temporal_v2` samples, so an explicit review
and conversion step is required before using integrated captures for training.
The standalone lab retains its manual ground-truth labeling workflow.

The model distinguishes the known 22K reference class, non-gold and invalid;
it does not estimate numerical purity or other karats.
