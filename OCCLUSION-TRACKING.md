# Automatic pickup and rubbing evidence

The production entry point remains `jewel_tracker_5.py`. Use the existing SLIC environment for testing:

```bash
cd /home/embsys/Desktop/jewel_tracker/jewel_tracker
.venv-slic/bin/python jewel_tracker_5.py
```

`run_tracker.sh` currently selects `.venv`, which has no OpenCV SLIC support. This change does not alter that launcher or install anything into either environment.

At jewellery lock, each ID keeps its original home bbox and appearance, a current settled home, and its last reliable detection. `pickup_evidence.HomeAnchor` compares 32x32 patches against the saved clear home and surrounding background. It does not run a model. Skin colour inside the jewel's own segmentation is excluded from the existing hand-contact cue; hand-contact results are reused within each update.

A changed/covered home becomes `OCCLUDED_HOME`. Partial detections and bbox shifts under coverage cannot move the ID. The continuous recorder pins a provisional interval beginning 1.5 seconds before the first observed coverage/change. No SOP test, acid capture, or rubbing confirmation is created by this provisional event.

If the home becomes clear with its jewel still present, cancel the pickup candidate. A small nudge near home can also cancel after two fresh, clear, full-size appearance matches. If clear background is observed at the old home in successive inference frames, departure is confirmed. A reliable observation of the same jewel away from home can also support departure. Cached SLIC/gallery identity remains available; no new model or identity inference call is introduced. A unique-lost-jewel fallback requires all other jewels visibly at home, compatible size and foreground appearance, and an off-stone detection. Multiple covered homes remain ambiguous rather than choosing an ID by list order.

Only a supported departure becomes a test cycle. The provisional footage and original before-stone reference are carried into that cycle. The two-frame jewel/stone overlap gate is removed from live contact observation, the monitor and recorded recovery. Contact, stone movement and model segmentation remain supporting evidence; weak bounding-box or motion evidence is explicitly unverified. Recovery keeps its existing inference/read limit and prioritizes recorded frames near observed stone activity.

Return is accepted at any settled location inside the tracking bed and outside the stone. Two fresh detections, compatible identity/appearance and clear surrounding background support return. A returned jewel can occupy its original home, a slightly shifted position, or a different area. The original home bbox remains immutable; the settled current home can update. The monitor also requires a clear stable stone within its fixed placement ROI. A missing owner or clear stone alone cannot advance the cycle.

After supported return, the existing new-rub check runs even without live contact. The existing bounded mask decision completes as detected/not detected before baseline. A covered or moving stone cannot start baseline merely because the decision deadline elapsed. Acid detection, reaction timing, classifier crop/prediction and PDF outcomes retain their existing logic. No manual Retry/Ready controls are added.

If a false/unresolved pickup included stone activity, its clip is retained as **UNVERIFIED_PICKUP**, separately from test results. Clip writing uses the existing background executor with at most two pending immediate exports; remaining windows flush at shutdown. Final PDF exceptions link the video. These videos use the same post-session archive compression, hashes, timestamps and acknowledgment retention as other evidence. Cancellation releases unnecessary frame pins and full-frame memory. Unverified pickup does not confirm rubbing or advance acid testing.

## Camera checks

1. Cover J1 briefly and leave it in place: provisional recording should cancel; no acid cycle.
2. Partially cover a small earring and expose only a fragment: J1's saved bbox should not jump onto J2.
3. Pick up/rub quickly with missing live segmentation: retain the early clip; on supported return, run the new-rub check.
4. Return the jewel to the original spot, slightly shifted, and then a different bed area: all supported returns should work.
5. Cover both jewels simultaneously: preserve evidence without guessing identity.
6. Keep the jewel held while exposing the stone: no baseline/acid transition until the jewel's return is supported.
7. Start J2: mere home coverage must not stop J1's acid search; confirmed/debounced J2 departure retains the existing handoff behavior.

Image-only occlusion checks are conservative. Very poor contrast, reflections, gloves resembling the bed, or indistinguishable overlapping jewels can leave ownership unverified. The application preserves that uncertainty rather than fabricating a pickup or a chemical result.

## Verification

```bash
QT_QPA_PLATFORM=offscreen .venv-slic/bin/python -m unittest discover -s tests -v
```

Logs and the reproducible tracking-only benchmark are in `diagnostics/occlusion_pickup_20261031/`. On this Intel i5-9500T, a two-jewel synthetic benchmark measured approximately 0.6 ms added median tracking work per update. This excludes model inference and camera capture; actual camera FPS still needs validation under the operator's lighting and hand movements. Capture scheduling, inference sessions and model selection are unchanged.
