"""Read Jacquard scenes and turn grasp annotations into CNN targets."""

from dataclasses import dataclass
from pathlib import Path
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Grasp:
    x: float                 # center column in the original image
    y: float                 # center row in the original image
    angle: float             # opening direction, radians in image coordinates
    width: float             # gripper opening in pixels
    height: float            # rectangle thickness used for display/evaluation
    score: float = 0.0
    hybrid_score: float | None = None
    robustness_score: float | None = None
    worst_case_score: float | None = None
    final_score: float | None = None


def find_images(folder):
    """Only include RGB images with a matching mask and grasp text file."""
    images = []
    for rgb_path in sorted(Path(folder).rglob("*_RGB.png")):
        prefix = rgb_path.name[:-8]
        if (rgb_path.with_name(prefix + "_mask.png").exists()
                and rgb_path.with_name(prefix + "_grasps.txt").exists()):
            images.append(rgb_path)
    return images


def split_by_object(images, seed=42):
    """All views from one object folder stay in the same split."""
    objects = sorted({path.parent for path in images})
    if len(objects) < 3:
        raise ValueError("At least 3 object folders are needed")
    random.Random(seed).shuffle(objects)
    n = max(1, round(0.2 * len(objects)))
    groups = {
        "test": set(objects[:n]),
        "val": set(objects[n:2*n]),
        "train": set(objects[2*n:]),
    }
    return {name: [path for path in images if path.parent in folders]
            for name, folders in groups.items()}


def choose(images, maximum, seed=42):
    """Repeatable limit for a quick run on a large subset."""
    if maximum is None or len(images) <= maximum:
        return images
    return sorted(random.Random(seed).sample(images, maximum))


def read_scene(rgb_path):
    """Return RGB array, binary mask, and every annotated grasp."""
    rgb_path = Path(rgb_path)
    prefix = rgb_path.name[:-8]
    bgr = cv2.imread(str(rgb_path))
    mask_image = cv2.imread(str(rgb_path.with_name(prefix + "_mask.png")), 0)
    if bgr is None or mask_image is None or bgr.shape[:2] != mask_image.shape:
        raise ValueError(f"Missing or mismatched image/mask: {rgb_path}")
    grasps = []
    text_path = rgb_path.with_name(prefix + "_grasps.txt")
    for line in text_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split(";")]
        if len(values) != 5 or not all(map(math.isfinite, values)):
            continue
        x, y, degrees, opening, jaw_size = values
        if opening > 0 and jaw_size > 0:
            # Jacquard's angle is mirrored in displayed image coordinates.
            grasps.append(Grasp(x, y, math.radians(-degrees), opening, jaw_size))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, (mask_image > 0).astype(np.uint8), grasps


def make_input(rgb, mask, size):
    """Four input channels: RGB scaled to 0..1, plus the binary mask."""
    small_rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA) / 255.0
    small_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    channels = np.dstack((small_rgb, small_mask)).astype(np.float32)
    return channels.transpose(2, 0, 1)  # PyTorch expects channels first


def flip_scene(rgb, mask, grasps):
    """Random training-only flips, with grasp centers and angles flipped too."""
    height, width = mask.shape
    if random.random() < 0.5:
        rgb, mask = cv2.flip(rgb, 1), cv2.flip(mask, 1)
        grasps = [Grasp(width - 1 - g.x, g.y, -g.angle, g.width, g.height)
                  for g in grasps]
    if random.random() < 0.5:
        rgb, mask = cv2.flip(rgb, 0), cv2.flip(mask, 0)
        grasps = [Grasp(g.x, height - 1 - g.y, -g.angle, g.width, g.height)
                  for g in grasps]
    return rgb, mask, grasps


def make_targets(grasps, original_shape, size):
    """Quality peaks, cos(2θ), sin(2θ), width, and valid regression pixels."""
    height, width = original_shape
    quality = np.zeros((size, size), np.float32)
    regression = np.zeros((3, size, size), np.float32)
    valid = np.zeros((size, size), np.float32)
    sigma = max(1.5, size / 55)
    radius = math.ceil(3 * sigma)

    # Jacquard may repeat a grasp location for several jaw sizes.
    unique = {}
    for grasp in grasps:
        unique.setdefault((round(grasp.x), round(grasp.y), round(grasp.angle, 2)), grasp)

    for grasp in list(unique.values())[:150]:
        cx, cy = grasp.x * size / width, grasp.y * size / height
        x0, x1 = max(0, round(cx) - radius), min(size, round(cx) + radius + 1)
        y0, y1 = max(0, round(cy) - radius), min(size, round(cy) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        peak = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        replace = peak > quality[y0:y1, x0:x1]
        quality[y0:y1, x0:x1][replace] = peak[replace]
        regression[0, y0:y1, x0:x1][replace] = math.cos(2 * grasp.angle)
        regression[1, y0:y1, x0:x1][replace] = math.sin(2 * grasp.angle)
        regression[2, y0:y1, x0:x1][replace] = min(1, grasp.width / width)
        valid[y0:y1, x0:x1][replace] = 1
    return quality, regression, valid


class JacquardDataset(Dataset):
    """Open an image only when a training batch requests it."""

    def __init__(self, image_paths, size, augment=False):
        self.image_paths = image_paths
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        rgb, mask, grasps = read_scene(self.image_paths[index])
        if self.augment:
            rgb, mask, grasps = flip_scene(rgb, mask, grasps)
        image = make_input(rgb, mask, self.size)
        quality, regression, valid = make_targets(grasps, mask.shape, self.size)
        return (torch.from_numpy(image), torch.from_numpy(quality[None]),
                torch.from_numpy(regression), torch.from_numpy(valid[None]))
