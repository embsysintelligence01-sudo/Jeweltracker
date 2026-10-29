# Separate SLIC test environment

Run from this directory:

```bash
./run_tracker_slic.sh
```

This runs `jewel_tracker_5.py` with `.venv-slic`. The normal `./run_tracker.sh`
continues to use `.venv`. The environments have separate package files; the
application code, camera settings, zones and output locations are shared.
Run one application instance at a time.

The test environment contains private copies of the working dependencies,
with `opencv-python` replaced by `opencv-contrib-python==5.0.0.93`, plus
`scikit-learn==1.7.2` and its dependencies. NumPy and model runtime versions
were preserved. SLIC is available through
`cv2.ximgproc.createSuperpixelSLIC`; the update's identity matcher uses it.
This environment setup does not integrate that matcher into the live tracker.

`pip check` reports that Ultralytics requires `opencv-python` by distribution
name. The contrib distribution supplies the same `cv2` import plus SLIC, but
does not satisfy that package name. Do not install both OpenCV distributions
to silence this warning: they share the same files. After dependency changes,
check that standard OpenCV has not been installed alongside contrib.

Validation logs and package snapshots are in
`diagnostics/slic_environment_20261028/`.

Initial checks on 2026-10-28 passed: 206 application regression tests,
6 identity-matcher tests, offscreen PyQt window creation, MP4 decoding,
jewellery ONNX inference and a small gallery using the existing ResNet weights.
The isolated SLIC signature check measured approximately 5.8 ms median on
one jewellery crop; this is not an integrated live-tracking performance result.
The live SLIC tracking integration and physical-camera validation remain separate.

Both environments completed two jewellery cycles and PDF generation on the
reference video. On one sequential replay pair, published worker observations
were approximately 7.4/s in the original environment and 5.7/s in the test
environment. These include reused observations, not just model inference;
one pair is not a stable benchmark. The test environment is therefore not
the default launcher. Real-time replay sampling produced different acid
predictions, including one INVALID result in the test environment. Repeating
the classifier in both environments on identical saved frames produced exactly
the same probabilities. See `verification.json` in the diagnostic directory.

To stop using this environment, close the test application and return to
`./run_tracker.sh`. To remove it, delete only `.venv-slic` from this directory:

```bash
rm -rf -- .venv-slic
```

The original `.venv` and application files are unaffected by that removal.
