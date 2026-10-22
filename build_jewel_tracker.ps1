# Build Jewel Guard (jewel_tracker.py) into a standalone Windows .exe (onedir).
# Usage:  powershell -ExecutionPolicy Bypass -File build_jewel_tracker.ps1
#   Incremental (fast, default):  ...build_jewel_tracker.ps1
#   Full clean rebuild:           ...build_jewel_tracker.ps1 -Clean
#
# Uses the installed Python (the one that runs jewel_tracker.py) and its PyInstaller, like prototype_app's
# build_vision_app.ps1. Use -Clean only when the build acts stale (after editing jewel_tracker.spec,
# upgrading packages, or a corrupted cache).
param([switch]$Clean)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$cleanArg = if ($Clean) { "--clean" } else { $null }
Write-Host "==> Running PyInstaller$(if ($Clean) { ' (clean)' } else { ' (incremental)' })..."
python -m PyInstaller jewel_tracker.spec --noconfirm $cleanArg
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

$dest = Join-Path $PSScriptRoot "dist\jewel_tracker"

# PyQt5 ships older VC++ runtime DLLs; Windows can resolve torch's c10.dll against them and fail with
# WinError 1114 (same workaround as prototype_app's build_vision_app.ps1).
Write-Host "==> Removing stale PyQt5-bundled CRT DLLs (torch/PyQt5 DLL conflict workaround)..."
$qtBin = Join-Path $dest "_internal\PyQt5\Qt5\bin"
foreach ($name in @("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")) {
    $f = Join-Path $qtBin $name
    if (Test-Path $f) { Remove-Item $f -Force }
}

Write-Host "==> Copying runtime data files next to the .exe ($dest)..."
# models: nivagar_gold_v3.pt (jewels), newstone1.onnx (stone), rubbing_gold.onnx (rubbing), acid.onnx (acid),
# optional bestnewacid_mobilenet_v3.pt (acid false-positive filter); settings: camera.json, zones.json
Get-ChildItem "$PSScriptRoot\*.pt", "$PSScriptRoot\*.onnx", "$PSScriptRoot\camera.json", "$PSScriptRoot\zones.json" -File |
    ForEach-Object { Copy-Item $_.FullName -Destination $dest -Force }
# weights\ = SigLIP2 vision encoder; classification\ = jewel type text embeddings + correction gallery
foreach ($d in @("weights", "classification")) {
    $src = Join-Path $PSScriptRoot $d
    if (Test-Path $src) { Copy-Item $src -Destination $dest -Recurse -Force }
}
Copy-Item (Join-Path $PSScriptRoot "setup_direct_camera.ps1") -Destination $dest -Force

Write-Host "==> Done. Launch:  $dest\jewel_tracker.exe"
