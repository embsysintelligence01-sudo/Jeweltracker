# Live scheduling change — 21 October 2026

`jewel_tracker_5.py` now sends predictions to one `InferenceDispatcher` in
`live_pipeline.py`. The queue has at most two pending requests. Pending bench
requests take priority; the existing rubbing watcher still limits checks to
10 per second. Each caller finishes processing its observation before asking
for another. Mask decoding and tracking can overlap with the next model call.

The GUI reads the newest camera frame on a 40 ms timer, without acquiring the
model or tracking locks. It applies the most recent drawing overlay, which
expires after 0.3 seconds. The preview is resized before conversion to Qt;
mouse coordinates still refer to the original camera resolution. Camera frames
used by inference are read-only references, avoiding copies while polling.

Frames/results older than 0.5 seconds are discarded. A skipped/expired result
does not become an empty detection or a missing-jewel observation. Resetting
the batch or changing zones invalidates in-flight observations. Disconnect
waits for native work and closes the inference scheduler.

## Existing process

The model files, confidence/overlap thresholds, motion thresholds, confirmation
times, and existing tuning values are unchanged. Bench guarding remains active
during rubbing; rubbing focus remains disabled. The decision methods for
identity assignment, ROI exit, missing/return alerts, rubbing credit, acid-test
ownership/results, and DONE are unchanged. No model has been retrained.

## Validation

Regression tests cover priority and serialization, bounded pending requests,
expired requests/results, batch reset, preview independence from model locks,
preview coordinate mapping, ROI exit and return under the same ID, removal
during rubbing, hand occlusion, and the existing temporal acid workflow.
All 34 tests passed. A source comparison also verified that the protected
business decision methods and all existing tuning values are unchanged.

Replay measurements use the actual gold model and latest saved 2560x1440
capture as a simulated 20 FPS source. Stone motion gating is forced open to
exercise simultaneous bench/stone inference. The GUI runs offscreen. These
measurements verify scheduling and display latency, not physical rubbing
accuracy. See `live_pipeline_benchmark.json` for the recorded results.

| Stone-processing phase | Before | After |
|---|---:|---:|
| Preview FPS | 8.00 | 20.00 |
| Mean displayed-frame age | 152.6 ms | 31.2 ms |
| Bench observations per second | 8.00 | 7.67 |
| Stone checks per second | 7.83 | 7.50 |
| Largest bench observation gap in this replay | 157.1 ms | 190.3 ms |

The replay demonstrates a faster preview, not a faster neural network.
Inference rates vary with CPU load. The original ROI follower still requires
observations less than 0.3 seconds apart; sustained overload can affect it.
Physical jewel removal, hand occlusion, rubbing, and acid tests still require
operator verification on the camera.

Run the checks with:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## Backup

The original script is preserved as
`jewel_tracker_5.py.before_live_scheduling.bak`, alongside the README backup.
`run_tracker.sh`, model weights, and camera/zone settings were not edited.
