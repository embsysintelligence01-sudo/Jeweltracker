"""Native V4L2 controls for the Dell USB camera; no model or image processing."""
import re
import subprocess
import time


# Verified on the WB3023 at 2560x1440 MJPG, 30 FPS under this bench's lighting.
LINUX_AF_DEFAULT = {"enabled": True, "mains_hz": 50, "shutter_denom": 50,
                    "brightness": 96, "gain": 0}
CONTROLS = ("power_line_frequency", "auto_exposure", "exposure_time_absolute",
            "exposure_dynamic_framerate", "brightness", "gain")


def _ctl(device, option):
    try:
        result = subprocess.run(["v4l2-ctl", "-d", device, option],
                                capture_output=True, text=True, timeout=5, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Install v4l-utils to apply the Linux camera profile") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError((exc.stderr or exc.stdout or str(exc)).strip()) from exc
    return result.stdout


def read_controls(device):
    output = _ctl(device, "--get-ctrl=" + ",".join(CONTROLS))
    values = {key: int(value) for key, value in
              re.findall(r"^\s*(\w+):\s*(-?\d+)", output, re.MULTILINE)}
    missing = set(CONTROLS) - values.keys()
    if missing:
        raise RuntimeError("Camera controls unavailable: " + ", ".join(sorted(missing)))
    return values


def _set(device, key, value):
    _ctl(device, f"--set-ctrl={key}={value}")


def apply_linux_antiflicker(cap, af, state, index=0):
    """Apply on the capture thread, verify native units, retain controls for Off.

    V4L2 exposure is in 100 microsecond units (200 = 20 ms). Do not use
    DirectShow's logarithmic exposure codes on Linux. WB/focus are untouched.
    """
    device = f"/dev/video{int(index)}"
    original = state.get("linux_original")
    if not af.get("enabled", True) and original is None:
        return "Anti-flicker off: Linux camera settings left unchanged"
    # Some UVC cameras need samples flowing before they accept controls.
    for _ in range(5):
        cap.grab()
    if original is None:
        original = read_controls(device)
        state["linux_original"] = original
    enabled = bool(af.get("enabled", True))
    if enabled:
        hz = int(af.get("mains_hz", 50))
        if hz not in (50, 60):
            raise ValueError("Mains frequency must be 50 or 60 Hz")
        denom = max(8, min(200, int(af.get("shutter_denom", 50))))
        wanted = {"power_line_frequency": 1 if hz == 50 else 2,
                  "auto_exposure": 1,
                  "exposure_time_absolute": round(10000 / denom),
                  "exposure_dynamic_framerate": 0,
                  "brightness": int(af.get("brightness", 96)),
                  "gain": int(af.get("gain", 0))}
    else:
        wanted = dict(original)
    # Set exposure with manual mode active; restore the original mode last on Off.
    _set(device, "auto_exposure", 1)
    for key in CONTROLS:
        if key != "auto_exposure":
            _set(device, key, wanted[key])
    if wanted["auto_exposure"] != 1:
        _set(device, "auto_exposure", wanted["auto_exposure"])
    for _ in range(5):
        cap.grab()
    time.sleep(0.12)
    actual = read_controls(device)
    mismatch = [f"{key}: asked {value}, read {actual[key]}" for key, value in wanted.items()
                if actual[key] != value and not
                (key == "exposure_time_absolute" and wanted["auto_exposure"] != 1)]
    if mismatch:
        raise RuntimeError("Linux anti-flicker NOT verified: " + "; ".join(mismatch))
    if not enabled:
        return "Anti-flicker off: original Linux camera controls restored and verified"
    return (f"Anti-flicker verified (V4L2): {hz} Hz, "
            f"{actual['exposure_time_absolute'] / 10:.1f} ms manual exposure | "
            f"brightness {actual['brightness']}, gain {actual['gain']}")
