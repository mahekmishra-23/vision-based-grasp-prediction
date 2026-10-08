"""Run locally after training: python webcam.py runs_subset_gpu/best.pt"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from methods import draw_grasp, predict_three
from model import load_checkpoint


def object_mask(frame, background):
    """Keep the largest region that changed since the empty-table image."""
    difference = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = (gray > 25).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return np.zeros(gray.shape, np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[largest, cv2.CC_STAT_AREA] < 100:
        return np.zeros(gray.shape, np.uint8)
    return (labels == largest).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Live two-finger grasp overlay")
    parser.add_argument("checkpoint", help="Path to best.pt from notebook training")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(
            f"Checkpoint not found: {checkpoint}. "
            "Use runs_more_data/best.pt or provide a valid checkpoint path."
        )

    model, size, device = load_checkpoint(checkpoint)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam")
    background = None
    mode = "h"
    modes = {"g": "geometry", "l": "learned", "h": "hybrid", "r": "robust"}
    print("Press b for empty background; g=geometry, l=learning, h=hybrid, r=robust, q=quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            if background is None:
                shown = frame.copy()
                cv2.putText(shown, "Press b to capture empty table", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                mask = object_mask(frame, background)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictions = predict_three(model, size, device, rgb, mask,
                                            include_robust=(mode == "r"))
                grasp = predictions[modes[mode]]
                shown = draw_grasp(rgb, mask, grasp, modes[mode],
                                   show_robustness=(mode == "r"))
                cv2.putText(shown, "g geometry | l learning | h hybrid | r robust",
                            (10, shown.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Grasp demo", shown)
            key = cv2.waitKey(1) & 255
            if key == ord("b"):
                background = frame.copy()
            elif key in (ord("g"), ord("l"), ord("h"), ord("r")):
                mode = chr(key)
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
