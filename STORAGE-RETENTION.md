# Production recording retention

The active recorder still writes original-size JPEGs at quality 95 with the existing 15 FPS target and 45-frame nonblocking queue. Tracking, mask verification, acid decisions, model sessions and live inference scheduling are unchanged. Storage maintenance runs on the evidence writer; archive encoding is exclusively post-session, one low-priority encoder thread by default.

`evidence_retention.json` controls storage:

- `idle_buffer_seconds: 20`: rolling unneeded camera history. A five-second pre-roll floor is protected by the monitor's processing watermark; if the monitor falls behind, pruning waits rather than losing evidence. All test intervals from four seconds before pickup through their end remain pinned until finalization. Explicit reader leases protect export windows. Active or slow background work can therefore exceed the idle limit intentionally.
- `disk_warning_free_gib: 3`: storage warnings on the existing status channel. Active required evidence is never sacrificed for space. If the filesystem cannot accept writes, errors remain visible; this cannot guarantee recording on a full disk.
- `target_test_bytes: 2000000`: target per jewel/test attempt, including its videos/images and a proportional share of common PDFs/JSON. Each test has a separate size result; small tests cannot hide oversized tests.
- `archive_codec: "h264"`, `h264_preset: "veryfast"`, `encoder_threads: 1`: sequential low-priority libx264 profiles adapt CRF (22/25/28), full-frame resolution (up to 1600 then 1280 long edge) and frame rate (up to 15 then 10 FPS). Source clips with lower resolution/FPS are not enlarged. No trimming or spatial cropping is used. A native-resolution CRF 18 fallback is attempted if no adaptive profile passes. All profiles are fully decoded and compared with the rescaled source using SSIM. `archive_min_long_edge: 1280` and `archive_min_ssim: 0.95` define the quality floor. SSIM is a compression check, not a guarantee that every diagnostic detail is visible. If essential original evidence or acceptable videos exceed the budget, retain them and record a size exception. Without FFmpeg/libx264, automatic shutdown preserves validated MJPEG as a fallback; the historical H.264 execution command requires FFmpeg. Encoder/quality failures preserve raw evidence.
- `require_server_ack: true`: default fail-closed production policy. Local archive creation is automatic; raw test frames are retained until the central server's authenticated receipt is available.
- `cleanup_raw_after_archive: true`: after validation/ack, remove redundant raw `frames/`, original catalogued AVI exports, and duplicate classifier sequence folders only when their summary images are preserved. Partial sequences without summaries remain intact.
- `bank_min_retention_days: 0`: minimum final archive retention; the configured archive TTL cannot be shorter. `original_min_retention_days: 0`: optional minimum retention of original raw evidence after the last test ends, in addition to acknowledgment requirements.
- `retain_frame_catalogue: false`: remove redundant `frames.jsonl` only after its SHA-256 is recorded and original clip-frame timestamps, event times and adaptive frame mappings are archived in sidecars and `source_timestamps.json`. Set true if bank policy requires the complete camera catalogue.
- `archive_retention_days: 90`: whole archived sessions become eligible for expiration after this period, with acknowledgment required under the current policy. Expiration is performed by the cleanup command, intended for a scheduled system service/cron job; the GUI does not purge historic sessions.

To enable H.264 on Ubuntu if FFmpeg is absent:

```bash
sudo apt install ffmpeg
```

FFmpeg/libx264 is now tested with real session media. This workstation uses an isolated executable at `~/.local/share/jeweltracker/ffmpeg/ffmpeg`, extracted from imageio-ffmpeg 0.6.0 with its license notices. No existing Python environment packages were changed. Production Ubuntu can use the distribution FFmpeg above. `JEWELTRACKER_FFMPEG=/absolute/path/to/ffmpeg` overrides executable discovery.

## Session shutdown and atomic commit

After the recorder, photo/model/recovery/finalization workers and report writer have finished, the monitor writes `session_closed.json`. It holds a recording lock throughout this process. Failed test finalizers, incomplete required exports or missing test manifests prevent cleanup.

Archive generation writes an immutable staged directory, decodes each video to check frame count/playability, checks playback duration against source timestamps, preserves referenced original images and classifier inputs, rewrites JSON references, and renders a final PDF with links to final video paths. Acid candidate video content is deduplicated. The selected video timestamps remain authoritative for irregular camera sampling. Incomplete/partial/corrupted sessions are skipped; no cleanup is authorized for them. Complete pickup-to-return and acid application/reaction intervals are checked before migration. The final PDF is parser-checked with `pdfinfo` when installed; otherwise its complete PDF envelope and cross-reference table are checked.

Successful stages are renamed atomically into `archives/<id>/`. SHA-256 checksums cover every archived asset, final PDF, session JSON, per-test manifests, sidecars and compact timestamp metadata. Exact duplicate image/video bytes are stored once. Selected original JPEG/PNG evidence is retained byte-for-byte; reconstructed legacy classifier summaries are explicitly labelled. Canonical `report.pdf`, `session.json` and original per-test manifests are published, then `archive_commit.json` is committed last. Publication can resume after a crash. All reports share a filesystem lock, and a stale report request cannot restore deleted raw paths after commit.

Raw cleanup requires an intact committed archive and the configured acknowledgment. Raw directories are atomically renamed into a cleanup tomb before removal. A durable authorization bound to the new archive hash permits retry even after partial removal of a superseded archive tomb. Interrupted cleanup is idempotent and revalidates checksums/ack before retrying. Expiration uses a durable deletion ledger outside the expired session so an interrupted authorized deletion can resume.

Classifier summary PNGs saved in new sessions are the exact three pre-normalization RGB summary inputs derived from the live arrays. Saving those audit images cannot change the classifier's result on an I/O error; archive cleanup then fails closed. For legacy sessions, complete saved JPEG sequences can be reduced to clearly labelled reconstructed summaries; those are not claimed to reproduce the original in-memory inputs exactly.

## Central-server acknowledgment

No network upload or server endpoint is assumed. Upload the committed archive and `archive_manifest.json` using the deployment's existing transport. The server must verify/store it before returning a receipt. Receipt fields:

```json
{
  "session_id": "20261031_120000_example",
  "archive_manifest_sha256": "SHA256_OF_ARCHIVE_MANIFEST_JSON",
  "server_receipt_id": "SERVER_DURABLE_STORAGE_RECEIPT",
  "acknowledged_at": "2026-10-31T12:30:00+05:30",
  "signature": "HMAC_SHA256_HEX"
}
```

The signature is HMAC-SHA256 of the other fields serialized with sorted keys and separators `(',', ':')`, using the shared secret supplied via `JEWELTRACKER_ACK_HMAC_KEY` (or the configured environment name). Keep that secret outside source control. Missing keys, invalid signatures or a different archive hash prevent raw cleanup. The receipt authenticates durable server acknowledgment; merely creating an unsigned local JSON file is insufficient.

Explicitly offline installations may set `require_server_ack: false`. This permits cleanup after local validation alone and should be an intentional deployment choice.

## Commands

Run from `/home/embsys/Desktop/jewel_tracker/jewel_tracker`, using the existing SLIC environment. The historical tool visits **every** nonhidden directory under `captures/sessions/`, including sessions without modern shutdown markers. It checks final manifests, PDF and clip interval coverage, and skips missing/incomplete/corrupt/active sessions. Legacy sessions without recording locks are conservatively postponed while any Jeweltracker live app is running.

Read-only inspection (target-based estimates; no files created in session folders):

```bash
.venv-slic/bin/python historical_cleanup.py captures/sessions --dry-run --report /tmp/jewel-storage-dry-run.json
```

Add `--verify-videos` to fully decode all candidate source clips during inspection. Without it, dry-run eligibility is structural only. Execution always performs full decode, duration, timestamp and quality verification:

```bash
.venv-slic/bin/python historical_cleanup.py captures/sessions --execute --resume --report /tmp/jewel-storage-execute.json
```

`--resume` safely continues interrupted publication/cleanup, and is accepted only with `--execute`. Already committed older archives are repacked into a new immutable generation; the prior generation remains until the **new** generation is verified and acknowledged. The final report lists previous/new bytes, space recovered, optimized/skipped sessions, pending acknowledgment and per-test size exceptions. Estimates are not promises of savings.

The same compact archive builder runs automatically for new completed sessions after writer/recovery/classification/report workers have drained. It does not run model inference. Incomplete sessions retain raw evidence. Scanning all historical sessions is an explicit maintenance operation using the command above; do it while the live app is closed for lowest contention. Encoding uses one CPU thread and nice priority 10; keep maintenance outside business hours on the mini PC.

The default bank policy **does not authorize raw deletion without a signed server receipt**. Execution can create optimized archives while originals await acknowledgment, temporarily increasing usage. With low free space, migrate one session at a time by passing its full session directory instead of the sessions parent. Do not disable server acknowledgment solely to force savings. No upload endpoint is configured by this project.

Import a server receipt and apply eligible cleanup for its session:

```bash
.venv-slic/bin/python evidence_archive.py captures/sessions/SESSION_ID --ack-file /path/to/server_receipt.json --apply
```

Templates for a low-priority 30-minute cleanup service are provided in `deploy/jewel-evidence-cleanup.service` and `.timer`; adjust the user and application path before installing them. They were not installed or enabled automatically. Put the acknowledgment key in the optional `/etc/jeweltracker/evidence.env`, protected from other users.

Run the following periodically to retry authorized raw cleanup and expire acknowledged archives after the retention period:

```bash
.venv-slic/bin/python evidence_archive.py captures/sessions --apply
```

`--config /path/to/policy.json` selects another policy. Never use an offline cleanup policy unintentionally on installations requiring server acknowledgment.

## Measured storage and validation

A physical disposable copy of J2 from `20261030_113057_1898992b`, including 271 required raw frames and original exports, decreased from **139,351,370 bytes to 3,237,205 bytes**, recovering **136,114,165 bytes (97.68%)** in approximately **21 seconds** on this workstation. Both H.264 videos fully decoded at 1280x720/10 FPS, with 7.9 and 6.9 seconds playback. Original source timestamps, selected original images, three classifier summaries and valid rewritten references were retained. This test exceeded the 2 MB target and correctly reported a size exception: essential images plus videos and common report/metadata were preserved. It is not a throughput guarantee for an i5-9500T or a claim that every test fits 2 MB.

The all-session read-only scan found **165 folders, 51 structurally eligible and 114 skipped**, occupying approximately **141.19 GB logical bytes / 132.17 GiB allocated**. Existing historical originals were not converted or deleted during implementation: central acknowledgment remains required. See `diagnostics/storage_budget_20261031/historical-dry-run.json`, `measurement.json`, `measure_session_copy.py` and regression logs. The prior MJPEG measurement is retained under `diagnostics/storage_retention_20261031/` for comparison.
