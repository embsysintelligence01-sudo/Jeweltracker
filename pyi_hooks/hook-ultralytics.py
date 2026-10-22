# Local override for ultralytics (YOLOv8 detection/segmentation).
#
# WHY NOT THE STOCK collect_all / collect_submodules BEHAVIOUR
# collect_submodules("ultralytics") IMPORTS every ultralytics submodule to
# enumerate them -- including optional export/integration backends (TensorFlow,
# Keras, ONNX, OpenVINO, transformers, boto3/S3 logging, bitsandbytes, ...).
# In this shared environment those are installed for unrelated projects, so the
# walk drags their entire dependency trees onto the module graph, nesting deep
# enough to blow PyInstaller's recursion limit (RecursionError during Analysis).
#
# WHY A DATA-FILES-ONLY HOOK IS NOT ENOUGH EITHER
# ultralytics exposes its model classes LAZILY: `from ultralytics import YOLO`
# hits ultralytics/__init__.py's __getattr__, which does
# `importlib.import_module("ultralytics.models")` at runtime. Static bytecode
# analysis cannot see that dynamic import, so ultralytics.models (and the rest
# of the package that YOLO().train()/.predict() reach, plus the dynamic imports
# in ultralytics/nn/tasks.py) would be MISSING from the build. The app would
# then silently fall into its `except` branch (_YOLO_AVAILABLE = False) and the
# whole YOLO/segmentation pipeline would be dead in the frozen exe.
#
# THE FIX
# Enumerate every ultralytics submodule by WALKING THE FILESYSTEM (never
# importing them), exactly like pyi_hooks/hook-torch.py. That gives complete
# coverage of the lazily/dynamically imported submodules without triggering the
# import-time dependency explosion. The heavy optional backends the ultralytics
# source references (tensorflow, transformers, onnx, ...) are pruned by the
# `excludes` list in vision_app.spec, so static analysis of these submodules
# does not descend into them.
#
# Also collect the bundled non-.py data (cfg/*.yaml default configs, dataset
# templates, assets) that ultralytics reads by path at runtime.

import os

from PyInstaller.utils.hooks import collect_data_files, get_package_paths

datas = collect_data_files("ultralytics", excludes=["**/*.py"])


def _ultralytics_submodules():
    """Every ultralytics.* submodule name, found by walking the package
    directory on disk (no imports -> no dependency-tree explosion)."""
    pkg_base, pkg_dir = get_package_paths("ultralytics")
    mods = set()
    for dirpath, _dirnames, filenames in os.walk(pkg_dir):
        if "__pycache__" in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), pkg_base)
            mod = rel[:-3].replace(os.sep, ".").replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            mods.add(mod)
    return sorted(mods)


hiddenimports = _ultralytics_submodules()

# Some submodules gate on optional backends that are excluded from the build;
# don't warn when the graph can't fully resolve them.
warn_on_missing_hiddenimports = False
