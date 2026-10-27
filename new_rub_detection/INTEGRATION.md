# Add this flow to your main application

The integrated `jewel_tracker_5.py` uses `rub_preparation.RubCycle`: it shares
the live INT8 stone model, derives an oriented quad from the segmentation mask,
and uses image registration to compare before/after captures without AprilTags.
After a new rub is confirmed, Ready and acid capture reuse its fixed box/mask. See `../README-linux.md` for that flow. The examples below describe
the standalone marker-free adapter, which still accepts an ONNX stone model.

## Any Python GUI, API, or existing camera application

Place the folder beside your application:

```text
your_application/
    main.py
    new_rub_detection/
        flow.py
        core.py
        ...
        models/
            newstone1.onnx
            rubbedmask.onnx
```

The following code initializes the models once and exposes functions you can
call from your two capture handlers. It uses NumPy BGR images, like OpenCV.

```python
from pathlib import Path

import new_rub_detection
from new_rub_detection.core import OnnxStoneModel
from new_rub_detection.example_pair import infer_photo
from new_rub_detection.flow import RubCheck
from new_rub_detection.photo_worker import create_session

model_dir = Path(new_rub_detection.__file__).resolve().parent / "models"
stone_model = OnnxStoneModel(model_dir / "newstone1.onnx")
rub_session = create_session(model_dir / "rubbedmask.onnx")
check = RubCheck()


def capture_before(frame):
    stone_mask, rub_mask, quad = infer_photo(frame, stone_model, rub_session)
    return check.before(frame, stone_mask, rub_mask, quad)


def capture_after(frame):
    stone_mask, rub_mask, quad = infer_photo(frame, stone_model, rub_session)
    return check.after(frame, stone_mask, rub_mask, quad)


def new_test():
    check.reset()


def shutdown():
    stone_model.close()
```

Run capture handlers in your application's background worker, using a copied
captured frame. Serialize calls to one `RubCheck` instance; use one instance per
stone test. Process the **same frame** for stone inference, rubbed inference, and
comparison. Keep the preview independent from these calls. Handle `RuntimeError`
and `ValueError` in your worker and show their messages instead of treating them
as a successful result.

Wire the returned result to your own UI:

- `result["message"]`: display as the operator instruction/result.
- `result["photo"]`: annotated full-size BGR photo; convert to RGB if your UI
  requires RGB.
- `result["preview"]`: normalized stone preview or diagnostic mask; it can be
  a BGR image or grayscale mask.
- Enable the after button only after the before result is `READY`.
- The reference captures starting marks without recommending a zone.
- On `CLEAN_THE_STONE`, ask the operator to clean the stone and capture a new
  reference. Keep rubbing/acid flow blocked until the before result is `READY`.
- Confirm a new mark only on `NEW_RUB`; `RETAKE` has no confirmed new mask.

### Input contract

`before(frame, stone_mask, rubbed_mask, quad)` and `after(...)` accept:

| Argument | Format |
| --- | --- |
| `frame` | `uint8`, shape `(height, width, 3)`, BGR. |
| `stone_mask` | `uint8`, shape `(height, width)`, 0 background and 255 stone. |
| `rubbed_mask` | `uint8`, same full-frame dimensions, 0 background and 255 model-detected marks. |
| `quad` | Four `float32` corners in perimeter order: top-left, top-right, bottom-right, bottom-left of the current frame's stone. |

If your main application already supplies these inputs, use `RubCheck` directly;
you can omit `infer_photo`, the model loading, camera reader, and PyQt demo.
Use raw detected corners for captures, rather than smoothed display coordinates.

### Result fields

Every before result contains `status`, `message`, `photo`, `preview`,
`preview_old_mask`, `old_mask`, `old_mark_count`, and `coverage`.

- `status`: `READY` or `CLEAN_THE_STONE`; invalid inputs raise a validation error.
- The original clear-space test runs internally using a 12-pixel edge margin,
  a 16-pixel old-mark margin, and room for a 24-pixel-radius rub in the normalized
  reference. Its target is never displayed or returned as a recommendation.
  `CLEAN_THE_STONE` clears any previous usable reference.
- `preview`: plain normalized 512×512 reference, with no grid or recommendation.
- `preview_old_mask`: normalized starting-mark mask, available for an optional
  development overlay. Hiding the overlay must not change `old_mask` or detection.

Every after result contains `status`, `message`, `photo`, `preview`, `new_mask`,
and `locations`. After alignment it also contains `alignment`, `new_mark_count`,
and `coverage`.

- `status`: `NEW_RUB` or `RETAKE`.
- `new_mask`: confirmed 0/255 mask in the 512×512 before reference, or `None`.
- `locations`: list of `(reference_point, current_frame_point)` pairs, ordered
  by region area; empty on a retake. Each point is `(x, y)` pixels.
- `coverage`: fraction of the valid stone region covered by the new mask,
  not a percentage and not the old-mark coverage.
- `frame_to_reference`: available after successful alignment; the 3x3 transform
  from the current full frame to the saved 512x512 before reference. The tracker
  uses it to project `new_mask` into a full-frame acid ROI.
- Images may be `None` if no valid before reference or alignment exists.
- `preview_old_mask`: after alignment, old marks in reference coordinates for
  optional overlay. A successful `preview` shows the new mark in magenta.

The adapter has no automatic file saving. Save images with `cv2.imwrite()` in
your own session directory. Keep the instance between before/after calls; reset
it for a new stone. New marks can appear anywhere on the stone. A connected
region touching the 12-pixel margin around an old mark is excluded completely.
An unchanged stone or rubbing over old marks returns `RETAKE` with
"No new rub detected". Alignment errors are reported separately. This implements
new-rub detection, not acid detection or material grading.

## Reuse the existing PyQt camera worker

If the host already uses PyQt6, the copied demo worker includes model loading,
capture commands, comparison, recommendations, and session-file saving:

```python
from pathlib import Path
from PyQt6.QtCore import QTimer
from new_rub_detection.App import CameraWorker

worker = CameraWorker(camera=0, width=1280, height=720,
                      output_dir=Path("sessions"), show_grid=False)
worker.photo_ready.connect(your_photo_handler)  # receives (BGR photo, crop/mask)
worker.failed.connect(your_error_handler)       # receives error text
worker.start()

timer = QTimer(your_window)
timer.setInterval(33)

def refresh_preview():
    preview = worker.preview()
    if preview is not None:
        frame, info = preview
        your_preview_handler(frame, info)

timer.timeout.connect(refresh_preview)
timer.start()

# In button handlers:
# worker.commands.put("before")
# worker.commands.put("after")
# worker.commands.put("reset")

# On window/application shutdown:
# timer.stop()
# worker.requestInterruption()
# worker.wait()
```

Replace the `your_*` names with your UI handlers; this snippet assumes an active
`QApplication`. Keep the worker and timer as attributes of your window. The
`info` dictionary includes `guidance`, `stage`, `busy`, `available`, and the
prediction/status text. Follow the button-enable rules in `App.Window.show_frame`.
Do not open a second camera reader for the same USB device; choose the worker or
your existing frame source.

`RubCheck` and `CameraWorker` are alternative integrations: the worker retains
the original demo recommendations. The tracker uses the adapter, which does
not recommend zones and strictly excludes regions touching old marks.
