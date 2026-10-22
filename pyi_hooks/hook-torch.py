# Local override for the stock pyinstaller-hooks-contrib hook-torch.py.
#
# WHY THIS EXISTS
# The stock hook (and collect_all) call collect_submodules("torch"), which
# recursively IMPORTS every torch submodule inside PyInstaller's "isolated"
# child-process sandbox to enumerate them. With this torch build (2.7.0+cu126)
# that walk crashes the sandboxed child with a native abort in torch_python.dll
# (exit code 3221226505) partway through -- so collect_submodules("torch") is
# unusable here and the whole build aborts.
#
# We still NEED the full submodule list, though: parts of torch (e.g.
# torch._dynamo.polyfills.loader) dynamically import their submodules at
# runtime via importlib.import_module(). PyInstaller's static bytecode analysis
# cannot see those dynamic imports, so without an explicit list the frozen app
# dies at startup with e.g. "ModuleNotFoundError: No module named
# 'torch._dynamo.polyfills.fx'".
#
# THE FIX
# Enumerate every torch submodule by WALKING THE FILESYSTEM (never importing
# them), which gives complete coverage without triggering the import-time
# crash. Static graph analysis of these modules only parses bytecode, so it
# does not execute torch and does not abort.
#
# module_collection_mode = "pyz+py" is also REQUIRED (the stock hook sets it):
# at import time torch runs inspect.getsource() on some of its own modules
# (e.g. torch/fx/experimental/_config.py, to parse "# @compile_ignored"
# comments). That only works if the actual .py SOURCE is on disk, not just
# frozen bytecode -- so collect torch as both bytecode (pyz) AND source (py).
# Without it the frozen app crashes with: OSError: could not get source code.

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    get_package_paths,
)

module_collection_mode = "pyz+py"

# Some enumerated submodules are conditionally present (platform-specific,
# optional-dependency gated). Don't warn when the graph can't resolve them --
# the stock torch hook suppresses these warnings the same way.
warn_on_missing_hiddenimports = False

datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h", "**/*.hpp", "**/*.cuh", "**/*.lib",
        "**/*.cpp", "**/*.pyi", "**/*.cmake",
    ],
)
binaries = collect_dynamic_libs("torch")


def _torch_submodules():
    """Every importable torch.* submodule name, found by walking the package
    directory on disk (no imports -> no native crash)."""
    pkg_base, pkg_dir = get_package_paths("torch")
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


hiddenimports = _torch_submodules()
