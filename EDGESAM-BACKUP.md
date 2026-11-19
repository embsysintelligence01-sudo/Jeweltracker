# Recorded-video recovery

The camera trigger, live rubbing/contact gate, live acid detector and reaction
capture remain unchanged. `classification_crop.py` and the temporal model's
224x224 preprocessing are unchanged. Classification still uses `pad_bbox(.45)`.

## Flow and ownership

1. A tracked pickup starts a cycle. The independent FrameJournal records original
   camera images with source timestamps even if the live rubbing overlap misses.
2. Capture closure queues the existing background finalizer. Its private
   `backup_status.json` says PENDING; this is not a completed SOP result.
3. The original contact, new-mark and acid-model recovery runs first. If either
   location is still missing, `backup_recovery.recover` sends only the finished
   cycle's frame paths, known poses, reference and evidence to a subprocess.
4. The child reuses the existing gold contact observer and recover_contact to
   inspect missed-overlap recordings with its own model. Contact colour,
   segmentation and back-and-forth motion checks still apply. It never fabricates
   a live trigger or jewel identity. The original live gate remains required live.
5. It starts with up to four sparse inspection frames, compares normalized stone
   images, and first tries the original rubbed-mark detector/RubCheck. Only an
   unresolved candidate is sent to EdgeSAM with a padded box, positive point and
   negative points. Three distinct clear observations and a contact match are
   required. Missing contact/motion remains unverified.
6. A post-rub baseline is compared with later images for acid. The original acid
   detector is tried first. EdgeSAM can refine an unsupported visual candidate.
   The existing AcidApplicationVerifier checks newness, applicator/glare/skin and
   persistence, requiring three clear frames in this backup path. A bounded binary
   search narrows the first-clear-patch time when sparse sampling found it late.
   This timestamp is a visual observation, not a proven chemical pouring time.
7. Verified acid coordinates and recorded times enter the existing classification
   export path. Original crops, 45% padding, baseline/application/reaction
   summaries, 224x224 preprocessing and classifier weights remain unchanged.
   Missing/occluded/incomplete reaction footage never produces a prediction.
8. Final publication occurs after backup resolves. Unresolved, expired, wrong
   location and timed-out cases receive REVIEW_REQUIRED. Model predictions and
   SOP verification remain separate. The PDF shows the evidence source and
   comparison; magenta marks rub and a yellow outline marks backup acid.

The persistent child uses CPUExecutionProvider, nice 15, one CPU affinity,
non-spinning ONNX sessions and one OpenCV thread. It owns all additional model
instances and caches four image embeddings. The existing finalizer waits for the
child; the camera and next cycle do not. The existing bounded finalization spool
and FrameJournal pins protect pending recordings. Native child inference is
terminated at its deadline. OS priority limits reduce contention; actual live FPS
and latency must still be measured under production load.

The clear-frame budget excludes contact-model scanning and the already saved
reference. Defaults allow 4 initial inspection candidates and 16 total candidate
samples; the contact scan is separately bounded at 12. The worker expands only
for missing stages/persistence/onset, not through every video frame. Unknown
stone orientation, sparse poses, blur or insufficient samples can cause review.
Recorded tracked corners use the same limited-motion assumption as live; no
unproven rotation after occlusion is corrected automatically.

## Installation and configuration

Use the existing SLIC environment, from this directory:

```bash
.venv-slic/bin/python -m pip install -r requirements-edgesam.txt
.venv-slic/bin/python install_edgesam.py
```

The installer downloads a pinned upstream ONNX encoder/decoder pair (~37 MiB),
verifies their upstream LFS SHA256s and writes `weights/edgesam/models.json`.
The models are already downloaded in this workspace. No model download happens
at camera startup. Missing or incompatible EdgeSAM files leave ordinary recovery
available and unresolved cases unverified.

Source/export protocol:
https://github.com/chongzhou96/EdgeSAM/blob/master/scripts/export_onnx_model.py
and https://huggingface.co/spaces/chongzhou/EdgeSAM . Review the upstream model
license for your deployment. This integration uses the upstream 1024 encoder,
three-input prompt decoder and zero-logit mask threshold, not standard SAM's
six-input decoder.

`sop_config.json` contains `edgesam_backup`:

| Setting | Default | Meaning |
| --- | --- | --- |
| enabled | true | Additional recorded-only recovery |
| encoder / decoder | weights/edgesam/*.onnx | Absolute paths also accepted |
| threads / max_cpus / nice | 1 / 1 / 15 | Child CPU limits |
| timeout_seconds | 45 | Hard child request deadline, including model startup |
| max_queue_seconds | 120 | Expired queued backup remains REVIEW_REQUIRED |
| initial_frames / max_frames | 4 / 16 | Inspection sampling budgets |
| contact_frames | 12 | Separate ordinary contact recovery budget |
| box_padding | .20 | EdgeSAM prompt box only; NEVER classifier padding |
| min_score | .85 | Decoder mask quality threshold plus seed/area checks |
| min_sharpness | 8 | Laplacian variance threshold on normalized stone |
| max_pose_age | .4 | Maximum seconds from a recorded known pose |
| embedding_cache_size | 4 | Bounded encoder cache |

Set `JEWEL_EDGESAM_ENABLED=0` to run the original recovery path alone. Environment
settings take precedence over JSON. Restart the application after changing config.
There is no assumption that an EdgeSAM quality score is calibrated SOP certainty.

## Step-by-step testing

1. Run the saved-video fixtures (they create and decode real MJPEG files):

   ```bash
   QT_QPA_PLATFORM=offscreen .venv-slic/bin/python -m unittest discover -s tests -p test_edgesam_backup.py -v
   ```

   These deterministic tests use a prompt-refiner double to test temporal,
   location, missing-model and timeout decisions independently of model accuracy.
   The subprocess test also checks process reuse, CPU affinity and priority.

2. Run existing flow/report regressions with the extension disabled:

   ```bash
   JEWEL_EDGESAM_ENABLED=0 QT_QPA_PLATFORM=offscreen .venv-slic/bin/python -m unittest discover -s tests
   ```

3. Start the tracker normally with backup enabled. Record a complete operation,
   including clear stone before rubbing, clear post-rub stone, acid and five
   seconds of unobstructed reaction. Begin the next jewel normally. Watch for
   `backup_pending` internally; no completed result is emitted for that cycle
   until its finalizer resolves. Do not delete raw evidence while it is pending.

4. Before automatic retention cleanup, inspect the cycle's `backup_request.json`,
   `backup_status.json`, `backup_source_*.png`, `backup_rub_mask.png` and
   `backup_acid_mask.png`. The worker's stderr is in `backup_worker.log`.
   Successful ordinary recovery can mean no EdgeSAM files/calls were needed.
   The final PDF embeds the recovery provenance and both marked locations.
   Normal configured retention may remove raw files after embedding the PDF.

5. Replay a retained request without altering the original session:

   ```bash
   .venv-slic/bin/python backup_worker.py --request /path/to/backup_request.json --output /tmp/jewel-backup-replay
   ```

   Output must be new/empty. A request can use saved video entries instead of
   JPEG entries: `{ "video": "/absolute/operation.avi", "frame_index": 12,
   "timestamp": 1234.567, "file": "video:12" }`. Use timestamps from the
   recorder's sidecar and known stone quads in `poses`; never infer event times
   from a compressed video's nominal FPS. Keep reference image, contact masks
   and source frame paths available. Recovery refuses missing geometry/reference.

6. Review `diagnostics/backup_performance.jsonl` (retained across session cleanup),
   or the copy `backup_performance.jsonl` in the session directory. Each cycle logs
   video decoding, encoder/decoder inference, rubbing verification, acid
   verification and classification seconds, along with counts/cache hits.
   `existing_recovery_seconds` and `total_backup_seconds` provide wider context.
   Stage timings overlap (verification includes calls within that stage) and
   should not be summed. Zero means a stage did not run. Archive logs separately
   before simple-output retention removes session intermediates.

Validate location masks manually on representative labelled recordings before
relying on this recovery for appraisal. Include reflections, moving applicators,
hands, old marks, acid away from the rub, stone movement and incomplete reaction.
No field-accuracy claim is made by the deterministic fixtures or model smoke test.

## Validation in this workspace

- Broad suite: 403 tests, 402 passed, one excluded existing failure:
  `test_full_frame_coordinates_are_not_offset_by_stone_origin` expects a full
  frame although the original `_detect` uses a padded stone crop. This failure
  was reproduced using the original detector function. That behavior was not
  changed to satisfy the outdated assertion.
- Eleven new tests cover decoded saved video, missed overlap, wrong location,
  missing models, transient change, pending next-cycle operation, complete
  classification crops, worker reuse/priority, timestamps and hard timeout.
- Real ONNX subprocess smoke on the deterministic saved-video fixture:
  11 decoded images, 2 encoder calls, 2 decoder calls; ~3.97 seconds total,
  ~1.93 seconds ONNX inference. Recorded first clear acid patch: 106.0 seconds;
  persistent confirmation: 106.4 seconds. This is not a field-accuracy benchmark.
- Separate inference on an existing 1280x720 acid recording confirmed that two
  decoder calls reuse one encoder embedding (one cache hit).
- Full real-model fixture timings/model hashes are saved in
  `diagnostics/edgesam_real_model_smoke.json`. Its temporary image paths belong
  to the smoke fixture and are not long-term evidence assets.
