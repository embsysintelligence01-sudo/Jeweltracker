# Production recording retention

The active recorder still writes original-size JPEGs at quality 95 with the existing 15 FPS target and 45-frame nonblocking queue. Tracking, mask verification, acid decisions, model sessions and live inference scheduling are unchanged. Storage maintenance runs on the evidence writer; archive encoding is exclusively post-session, one low-priority encoder thread by default.

`evidence_retention.json` controls storage:

- `idle_buffer_seconds: 20`: rolling unneeded camera history. A five-second pre-roll floor is protected by the monitor's processing watermark; if the monitor falls behind, pruning waits rather than losing evidence. All test intervals from four seconds before pickup through their end remain pinned until finalization. Explicit reader leases protect export windows. Active or slow background work can therefore exceed the idle limit intentionally.
- `disk_warning_free_gib: 3`: storage warnings on the existing status channel. Active required evidence is never sacrificed for space. If the filesystem cannot accept writes, errors remain visible; this cannot guarantee recording on a full disk.
- `archive_codec: "h264"`, `h264_crf: 18`, `h264_preset: "veryfast"`, `encoder_threads: 1`: final videos use FFmpeg/libx264 MP4 with faststart when available. Odd dimensions are padded, never shrunk. Without libx264/FFmpeg, the existing validated MJPEG AVI is retained as the supported fallback. Encoding failure with an available encoder retains raw evidence; it is not silently treated as success.
- `require_server_ack: true`: default fail-closed production policy. Local archive creation is automatic; raw test frames are retained until the central server's authenticated receipt is available.
- `cleanup_raw_after_archive: true`: after validation/ack, remove redundant raw `frames/`, original catalogued AVI exports, and duplicate classifier sequence folders only when their summary images are preserved. Partial sequences without summaries remain intact.
- `archive_retention_days: 90`: whole archived sessions become eligible for expiration after this period, with acknowledgment required under the current policy. Expiration is performed by the cleanup command, intended for a scheduled system service/cron job; the GUI does not purge historic sessions.

To enable H.264 on Ubuntu if FFmpeg is absent:

```bash
sudo apt install ffmpeg
```

No package or environment was changed by this implementation. The current host used the MJPEG fallback in real media verification; H.264 command construction is unit-tested, but actual H.264 encoding still needs validation on a host with FFmpeg/libx264.

## Session shutdown and atomic commit

After the recorder, photo/model/recovery/finalization workers and report writer have finished, the monitor writes `session_closed.json`. It holds a recording lock throughout this process. Failed test finalizers, incomplete required exports or missing test manifests prevent cleanup.

Archive generation writes an immutable staged directory, decodes each video to check frame count/playability, checks playback duration against source timestamps, preserves referenced original images and classifier inputs, rewrites JSON references, and renders a final PDF with links to final video paths. Acid candidate video content is deduplicated. The selected video timestamps remain authoritative for irregular camera sampling. Partial tests retain their declared partial durations; they are not represented as complete five-second reactions.

Successful stages are renamed atomically into `archives/<id>/`. SHA-256 checksums cover every archived asset, final PDF, session JSON, per-test manifests, sidecars and retained frame catalogue. Canonical `report.pdf`, `session.json` and original per-test manifests are published, then `archive_commit.json` is committed last. Publication can resume after a crash. All reports share a filesystem lock, and a stale report request cannot restore deleted raw paths after commit.

Raw cleanup requires an intact committed archive and the configured acknowledgment. Raw directories are atomically renamed into a cleanup tomb before removal. Interrupted cleanup is idempotent and revalidates checksums/ack before retrying. Expiration uses a durable deletion ledger outside the expired session so an interrupted authorized deletion can resume.

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

Run from the application directory, using the existing SLIC environment. Listing/cleanup defaults to **dry-run**:

```bash
.venv-slic/bin/python evidence_archive.py captures/sessions
```

This does not assume old sessions are finalized. Pre-retention sessions without a validated archive commit are listed as ineligible and are not deleted. To archive one verified closed historical session first:

```bash
.venv-slic/bin/python evidence_archive.py captures/sessions/SESSION_ID --archive-session --confirm-closed
```

The operator's `--confirm-closed` assertion is necessary only for legacy sessions that lack a quiescent shutdown marker. A live recording lock still blocks archival.

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

## Measured storage

An isolated physical copy of J2 from `20261030_113057_1898992b`, including 271 required raw camera frames and its original exports, decreased from **139,351,370 bytes (132.90 MiB)** to **36,521,551 bytes (34.83 MiB)**: **73.79% reduction**. Both 2560x1440 videos decoded successfully with the expected 95/82 frames and approximately 7.94/6.95 seconds playback, selected originals and three classifier summaries remained available, and final references validated. This measurement used the MJPEG fallback and explicitly disabled server-ack gating only for the disposable copy. It is not a prediction of savings across all sessions or an H.264 compression benchmark.

Existing historical sessions were not migrated or deleted. Their pre-change usage was approximately 132.17 GiB on disk; cleanup is intentionally separate from deploying this code. See `diagnostics/storage_retention_20261031/` for measurements, reproducible copy-only validation and regression logs.
