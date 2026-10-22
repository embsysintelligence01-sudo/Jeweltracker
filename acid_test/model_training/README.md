# Temporal Model Training

This isolated development pipeline trains one classifier sample per complete
acid-test session. It does not require a manually selected streak.

Each model receives three automatically prepared RGB images: the baseline
median, the last three pouring-transition frames, and the last five observation
frames. It predicts `gold_22k`, `non_gold`, or `invalid`.

`curation.json` changes training labels only. It never edits the original
dataset manifests or frames. Current splits are development-only because the
available sessions were collected under closely related conditions. Newly
captured sessions after candidate training must be kept as the untouched test.

Run:

```bash
cd /home/pi/Desktop/puritytesting/acid_temporal_lab/model_training
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 train.py
```

Controlled architecture trials can reuse a frozen dataset index and split via
`--index`, `--split`, and `--architecture`. Supported experiments are shared
MobileNetV3-Small, early-fusion MobileNetV3-Small, shared ResNet-18, and shared
EfficientNet-B0.

## Frozen architecture comparison (2026-09-12)

All candidates below used the same original 52-session index and the same
38/14 development split. They were then compared on the same 29 later sessions.
Those 29 sessions are now a development comparison set, not an untouched test.

| Candidate | Later sessions | False gold | ONNX size | Median CPU latency |
| --- | ---: | ---: | ---: | ---: |
| Shared MobileNetV3-Small | **26/29 (89.7%)** | **0** | 6.37 MiB | 16.1 ms |
| Shared EfficientNet-B0 | 25/29 (86.2%) | **0** | 21.54 MiB | 108.7 ms |
| Early-fusion MobileNetV3-Small | 25/29 (86.2%) | 1 | 4.12 MiB | 7.7 ms |
| Shared ResNet-18 | 17/29 (58.6%) | **0** | 45.14 MiB | 151.1 ms |

The existing shared MobileNetV3-Small remains the candidate. Do not integrate
it yet. First collect a new locked physical test set that is not used for model
selection or threshold changes: at least 20 22k, 20 non-gold, and 10 glare or
otherwise invalid sessions, with ordinary variation in rubbing, acid amount,
position, and lighting. The acceptance gate is zero false-gold predictions,
all invalid sessions rejected, and at least 90% correct overall.

Artifacts are written under `model_training/artifacts/`. A generated ONNX file
is a development candidate and must not be copied into the live application
until it passes the future locked physical-session test.

Evaluate the frozen candidate on its development split:

```bash
python3 evaluate_candidate.py
```

After collecting new sessions, evaluate only those sessions by supplying the
candidate-training cutoff timestamp:

```bash
python3 evaluate_candidate.py --after 2026-09-12T16:41:38.978+05:30
```
