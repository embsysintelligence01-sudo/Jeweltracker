# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Jewel Guard (jewel_tracker.py)
# Build:  powershell -ExecutionPolicy Bypass -File build_jewel_tracker.ps1   (add -Clean after spec/package changes)
#
# Produces a onedir build in dist\jewel_tracker\ . Models (*.pt, *.onnx, weights\), settings (camera.json,
# zones.json), the classifier data (classification\) and captures\ / jewel_guard.log live NEXT TO the .exe
# (SCRIPT_DIR in jewel_tracker.py); build_jewel_tracker.ps1 copies them there after the build.
#
# Built with the installed Python, like prototype_app (its torch is the CUDA build: the app itself runs
# on the CPU, the CUDA DLLs just ride along).

import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

datas, binaries, hiddenimports = [], [], []

# ultralytics reads installed-package metadata at import time (and check_requirements() when it loads the
# ONNX models reads onnxruntime's), so the dist-info folders must be bundled (see prototype_app's spec).
for _meta_pkg in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "opencv-contrib-python",
                  "pillow", "onnxruntime"):
    try:
        datas += copy_metadata(_meta_pkg, recursive=True)
    except Exception as _e:
        print(f"[spec] skip copy_metadata({_meta_pkg}): {_e}")

# torch and ultralytics: collected by pyi_hooks\ (filesystem walk instead of collect_submodules, as in
# prototype_app). matplotlib/polars: ultralytics imports them lazily; onnxruntime: its DLLs + capi.
for _pkg in ("torchvision", "cv2", "onnxruntime", "matplotlib", "polars", "_polars_runtime_32", "certifi"):
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as _e:
        print(f"[spec] skip collect_all({_pkg}): {_e}")

hiddenimports += ["jewel_capture", "jewel_classifier", "siglip_vision", "scipy.optimize", "requests"]

a = Analysis(
    ['jewel_tracker.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['pyi_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    # unused heavy packages installed in the shared Python (as in prototype_app's vision_app.spec, but
    # onnxruntime is kept: the stone, rubbing, acid and SigLIP2 models run on it)
    excludes=['paddle', 'paddleocr', 'tkinter', 'pytest',
              'PySide2', 'PySide6', 'PyQt6', 'shiboken2', 'shiboken6',
              'tensorflow', 'keras', 'openvino', 'transformers', 'boto3', 'botocore', 'bitsandbytes',
              'coremltools', 'jax', 'flax', 'wandb', 'comet_ml', 'clearml', 'mlflow', 'ray',
              'tritonclient', 'torchdata', 'clip', 'mobileclip', 'timm', 'albumentations', 'thop',
              'flask', 'streamlit', 'roboflow', 'onnxslim', 'sentry_sdk', 'dvclive', 'neptune',
              'hub_sdk', 'super_gradients', 'nncf', 'pymongo', 'faiss', 'IPython', 'jupyter',
              'notebook', 'sklearn', 'pandas.io.formats.style'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='jewel_tracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI app - no console window (events and crashes go to jewel_guard.log)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='jewel_guard.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='jewel_tracker',
)
