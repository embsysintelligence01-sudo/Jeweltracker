"""Run the portable flow on two saved photos, without PyQt or a USB camera."""

import argparse
from pathlib import Path

import cv2

from .core import OnnxStoneModel, infer_stone
from .flow import RubCheck
from .geometry import stone_quad
from .photo_worker import create_session, detect_mask


def infer_photo(frame, stone_model, rub_session):
    """Use the working app's area and ambiguity checks before accepting corners."""
    detections = [d for d in infer_stone(stone_model, frame)
                  if (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) >= 0.02 * frame.shape[0] * frame.shape[1]]
    if not detections:
        raise RuntimeError("RETAKE - stone not found")
    area = lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1])
    detections.sort(key=lambda d: area(d) * d[1], reverse=True)
    if len(detections) > 1 and area(detections[1]) > 0.6 * area(detections[0]):
        raise RuntimeError("RETAKE - multiple possible stones")
    bbox, _, stone_mask = detections[0]
    quad = stone_quad(stone_mask, bbox)
    if quad is None:
        raise RuntimeError("RETAKE - stone corners unclear")
    return stone_mask, detect_mask(frame, rub_session), quad


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("rub_result"))
    parser.add_argument("--stone-model", type=Path, default=root / "models" / "newstone1.onnx")
    parser.add_argument("--rub-model", type=Path, default=root / "models" / "rubbedmask.onnx")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    stone = OnnxStoneModel(args.stone_model)
    try:
        rubbed = create_session(args.rub_model)
        check = RubCheck()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for kind, path in (("before", args.before), ("after", args.after)):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Cannot read {path}")
            masks = infer_photo(image, stone, rubbed)
            result = getattr(check, kind)(image, *masks)
            print(f'{kind}: {result["status"]}: {result["message"]}')
            for key in ("photo", "preview", "new_mask"):
                if result.get(key) is not None:
                    target = args.output_dir / f"{kind}_{key}.png"
                    if not cv2.imwrite(str(target), result[key]):
                        raise OSError(f"Cannot save {target}")
            if kind == "before" and result["status"] != "READY":
                break
    finally:
        stone.close()


if __name__ == "__main__":
    main()
