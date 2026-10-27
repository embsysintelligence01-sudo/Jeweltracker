# Portable new-rub detection

Copy this entire `new_rub_detection` folder beside your main application's Python
entry point. Both ONNX models are included in `models/`; this bundle does not
depend on the original project's files or its parent-directory model location.
Python 3.10 or newer is required.

| File | Used for |
| --- | --- |
| `flow.py` | `RubCheck`: reusable before/after comparison without zone recommendations, with strict old-mark exclusion. |
| `core.py` | Stone ONNX inference; mark merging; exclusion of old marks; new-mark locations; drawing overlays. Functions are extracted from the working `App.py`. |
| `geometry.py` | Four stone corners, 512×512 normalization, feature/image alignment, and projecting masks back onto the photo. |
| `rubbing_zones.py` | Z1–Z4 clear-space analysis, target selection, instructions, and blue target drawing. |
| `photo_worker.py` | Rubbed-mask ONNX decoding. Exposes `create_session()` and `detect_mask()`, and still runs as a subprocess. |
| `live_camera.py` | Latest-frame camera reader for a responsive preview. Optional if your application already captures frames. |
| `App.py` | Complete working PyQt demo, including camera settings, worker lifecycle, buttons, state, and file saving. |
| `example_pair.py` | Camera-free example that runs both models and the reusable flow on two saved photos. Also exposes `infer_photo()`. |
| `models/newstone1.onnx` | Stone instance-segmentation weights. |
| `models/rubbedmask.onnx` | Existing/new rubbed-mark segmentation weights. |
| `tests/` | Copied regression tests and tests for the reusable adapter, including actual image rotation. |

## Install and run

Run these commands from the directory **containing** `new_rub_detection`:

```bash
python -m pip install -r new_rub_detection/requirements-core.txt
python -m new_rub_detection.example_pair before.png after.png --output-dir rub_result
```

For the complete camera/GUI demo, install the additional PyQt dependency:

```bash
python -m pip install -r new_rub_detection/requirements.txt
python -m new_rub_detection.App --camera 0
```

Add `--show-grid` to display the optional live grid. Captured photos always show
the grid and before-photo recommendation. A USB camera and desktop display are
needed for this demo. The reusable flow does not need a camera, PyQt, a GPU,
Hailo, PyTorch, or an AprilTag.

See [INTEGRATION.md](INTEGRATION.md) for the host-application code and result fields.
The original application and its model files have not been changed by packaging.

## Detection flow

1. Infer the stone mask and four corners from the captured before frame. Infer
   its rubbed mask once.
2. Normalize the stone to 512×512, merge old marks, and save the baseline.
3. Prefer an empty zone with a usable patch. If every zone has old marks, select
   a clear patch in a marked zone. Show the blue target circle. Request cleaning
   only if no usable patch remains; five marks or 6% overall coverage do not
   automatically reject a stone.
4. Infer both masks for the after frame and align the stone to the baseline.
5. Exclude old marked regions, suppress pixels with little image change, and
   check the remaining new-mask coverage.
6. Return new rubbing, its mask, its reference/frame coordinates, and the zone.
   Rubbing in another zone is accepted. Uncertain alignment/detection returns
   `RETAKE` and no confirmed new mask or locations.

The baseline mask is called `clean_mask` in the copied demo, but it can contain
old marks. Zone labels refer to the saved before photo, including after rotation.
A distinctive stone texture is needed for reliable alignment; a plain or
symmetric surface can require a retake.

The defaults preserve the working app's pixel thresholds. Targets have radius
24 on the 512-pixel reference, with margins of 12 from stone edges and 16 around
old marks. New-region matching tolerates a 25×25 neighborhood of old masks,
requires image change of at least 12 intensity units, and accepts new coverage
between 0.1% and 15%. These are image-relative parameters, not physical dimensions
or material PASS/FAIL criteria.

## Verification

```bash
python -B -m unittest discover -s new_rub_detection/tests -v
```

`TESTED_VERSIONS.txt` records the environment used to check the bundle.
`MANIFEST.sha256` records checksums for the portable source, documentation,
tests, and model files. No virtual environments, prior session photos, or logs
are included.
