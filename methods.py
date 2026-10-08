"""Four grasp-selection modes, a matching metric, and drawing helpers."""

import math

import cv2
import numpy as np
import torch

from data import Grasp, make_input

RECTANGLE_HEIGHT_FRACTION = 0.50  # Approximate jaw-size/opening ratio in training labels.
HYBRID_CNN_WEIGHT = 0.90  # Validation favored trusting the CNN more than PCA geometry.


def geometry_grasp(mask):
    """Cross the short PCA axis at the mask centroid."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    points = np.column_stack((xs, ys)).astype(float)
    center = points.mean(0)
    _, vectors = np.linalg.eigh(np.cov(points.T))
    direction = vectors[:, 0]  # eigenvector of the smallest spread
    angle = math.atan2(direction[1], direction[0])
    projection = (points - center) @ direction
    width = (np.percentile(projection, 95) - np.percentile(projection, 5)) * 1.12
    return Grasp(float(center[0]), float(center[1]), angle,
                 max(5, float(width)), max(8, float(width) * RECTANGLE_HEIGHT_FRACTION))


def cnn_candidates(model, size, device, rgb, mask, with_quality=False):
    """Read the strongest local peaks from the CNN's output maps."""
    height, width = mask.shape
    image = torch.from_numpy(make_input(rgb, mask, size))[None].to(device)
    with torch.no_grad():
        output = model(image)[0].cpu().numpy()
    quality = 1 / (1 + np.exp(-np.clip(output[0], -30, 30)))
    small_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    quality = cv2.GaussianBlur(quality * small_mask, (5, 5), 0)
    peaks = quality >= cv2.dilate(quality, np.ones((7, 7), np.uint8)) - 1e-7
    rows, cols = np.nonzero(peaks & (small_mask > 0))
    order = np.argsort(quality[rows, cols])[::-1][:30]
    candidates = []
    for index in order:
        row, col = int(rows[index]), int(cols[index])
        angle = 0.5 * math.atan2(math.tanh(float(output[2, row, col])),
                                  math.tanh(float(output[1, row, col])))
        opening = width / (1 + math.exp(-float(np.clip(output[3, row, col], -30, 30))))
        candidates.append(Grasp((col + 0.5) * width / size,
                                (row + 0.5) * height / size,
                                angle, max(5, opening), max(8, RECTANGLE_HEIGHT_FRACTION * opening),
                                float(quality[row, col])))
    if with_quality:
        return candidates, quality
    return candidates


def local_width(mask, grasp):
    """Ray-cast through the mask along the proposed opening direction."""
    height, width = mask.shape
    dx, dy = math.cos(grasp.angle), math.sin(grasp.angle)
    span = 0
    for sign in (-1, 1):
        for step in range(1, max(height, width)):
            x = round(grasp.x + sign * dx * step)
            y = round(grasp.y + sign * dy * step)
            if x < 0 or x >= width or y < 0 or y >= height or mask[y, x] == 0:
                break
            span += 1
    return span


def geometry_score(mask, grasp, center, main_angle, clearance):
    """Score center safety, centrality, opening fit, and PCA angle fit."""
    height, width = mask.shape
    x, y = round(grasp.x), round(grasp.y)
    if x < 0 or x >= width or y < 0 or y >= height or mask[y, x] == 0:
        return 0.0
    scale = max(10, math.sqrt(float(mask.sum()) / math.pi))
    center_score = math.exp(-math.hypot(grasp.x - center[0], grasp.y - center[1]) / scale)
    clearance_score = min(1, float(clearance[y, x]) / max(3, 0.3 * scale))
    span = local_width(mask, grasp)
    relative_opening = (grasp.width - span) / max(span, 1)
    width_score = math.exp(-max(0, abs(relative_opening - 0.12) - 0.15) * 3)
    angle_score = abs(math.cos(grasp.angle - main_angle))
    return 0.30 * clearance_score + 0.25 * center_score + 0.30 * width_score + 0.15 * angle_score


def hybrid_score(grasp, quality, mask, center, main_angle, clearance):
    """CNN quality plus a modest geometry check, sampled at this grasp center."""
    height, width = mask.shape
    row = int(grasp.y * quality.shape[0] / height)
    col = int(grasp.x * quality.shape[1] / width)
    cnn_score = 0.0
    if 0 <= row < quality.shape[0] and 0 <= col < quality.shape[1]:
        cnn_score = float(quality[row, col])
    return (HYBRID_CNN_WEIGHT * cnn_score
            + (1 - HYBRID_CNN_WEIGHT) * geometry_score(
                mask, grasp, center, main_angle, clearance))


def create_perturbations(grasp):
    """Eight small errors: position, angle, and opening, one at a time."""
    changes = [
        (10, 0, 0, 1), (-10, 0, 0, 1),
        (0, 10, 0, 1), (0, -10, 0, 1),
        (0, 0, 5, 1), (0, 0, -5, 1),
        (0, 0, 0, 1.1), (0, 0, 0, 0.9),
    ]
    perturbations = []
    for dx, dy, degrees, width_factor in changes:
        perturbations.append(Grasp(
            grasp.x + dx, grasp.y + dy,
            grasp.angle + math.radians(degrees),
            grasp.width * width_factor, grasp.height,
        ))
    return perturbations


def robustness_scores(grasp, quality, mask, center, main_angle, clearance):
    """Average and minimum hybrid score over the eight small errors."""
    scores = []
    for changed_grasp in create_perturbations(grasp):
        scores.append(hybrid_score(changed_grasp, quality, mask,
                                   center, main_angle, clearance))
    return sum(scores) / len(scores), min(scores)


def robust_select(candidates, quality, mask, center, main_angle, clearance):
    """Pick the candidate with the best normal/perturbed score mixture."""
    best = None
    for candidate in candidates:
        candidate.hybrid_score = hybrid_score(candidate, quality, mask,
                                              center, main_angle, clearance)
        candidate.robustness_score, candidate.worst_case_score = robustness_scores(
            candidate, quality, mask, center, main_angle, clearance)
        candidate.final_score = (0.7 * candidate.hybrid_score
                                 + 0.3 * candidate.robustness_score)
        if best is None or candidate.final_score > best.final_score:
            best = candidate
    return best


def predict_three(model, size, device, rgb, mask, include_robust=False):
    """Return the three original modes, plus robust mode when requested."""
    baseline = geometry_grasp(mask)
    if baseline is None:
        result = {"geometry": None, "learned": None, "hybrid": None}
        if include_robust:
            result["robust"] = None
        return result
    candidates, quality = cnn_candidates(model, size, device, rgb, mask, with_quality=True)
    learned = candidates[0] if candidates else None
    if not candidates:
        result = {"geometry": baseline, "learned": None, "hybrid": baseline}
        if include_robust:
            result["robust"] = None
        return result
    ys, xs = np.nonzero(mask)
    center = (float(xs.mean()), float(ys.mean()))
    clearance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    def combined_score(grasp):
        return hybrid_score(grasp, quality, mask, center, baseline.angle, clearance)
    chosen = max(candidates, key=combined_score)
    hybrid = Grasp(chosen.x, chosen.y, chosen.angle, chosen.width, chosen.height,
                   combined_score(chosen))
    result = {"geometry": baseline, "learned": learned, "hybrid": hybrid}
    if include_robust:
        result["robust"] = robust_select(candidates, quality, mask, center,
                                          baseline.angle, clearance)
        hybrid.hybrid_score = chosen.hybrid_score
        hybrid.robustness_score = chosen.robustness_score
        hybrid.worst_case_score = chosen.worst_case_score
        hybrid.final_score = chosen.final_score
    return result


def corners(grasp):
    return cv2.boxPoints(((grasp.x, grasp.y), (grasp.width, grasp.height),
                          math.degrees(grasp.angle))).astype(np.float32)


def angle_error(a, b):
    """Smallest difference, treating 0° and 180° as equivalent."""
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def rectangle_iou(a, b):
    """Intersection divided by union for two rotated grasp rectangles."""
    first = ((a.x, a.y), (a.width, a.height), math.degrees(a.angle))
    second = ((b.x, b.y), (b.width, b.height), math.degrees(b.angle))
    _, intersection = cv2.rotatedRectangleIntersection(first, second)
    area = abs(cv2.contourArea(intersection)) if intersection is not None else 0
    union = a.width * a.height + b.width * b.height - area
    return area / union if union > 0 else 0


def matches_annotation(prediction, annotations):
    """Standard 2D match: angle <= 30° and rectangle IoU >= 0.25."""
    return prediction is not None and any(
        angle_error(prediction.angle, truth.angle) <= math.radians(30)
        and rectangle_iou(prediction, truth) >= 0.25
        for truth in annotations
    )


def draw_grasp(rgb, mask, grasp, label, show_robustness=False):
    """Tint the object green and draw a yellow grasp rectangle/red center."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr[mask > 0] = (0.55 * bgr[mask > 0] + 0.45 * np.array([40, 160, 40])).astype(np.uint8)
    if grasp is not None:
        cv2.polylines(bgr, [corners(grasp).astype(np.int32)], True, (0, 230, 255), 2)
        cv2.circle(bgr, (round(grasp.x), round(grasp.y)), 5, (0, 0, 255), -1)
        label += f" | x={grasp.x:.0f} y={grasp.y:.0f} angle={math.degrees(grasp.angle):.0f} width={grasp.width:.0f}"
    if show_robustness and grasp is not None and grasp.final_score is not None:
        lines = [
            "ROBUST HYBRID",
            f"x={grasp.x:.0f} y={grasp.y:.0f} angle={math.degrees(grasp.angle):.1f} width={grasp.width:.0f}",
            f"Hybrid={grasp.hybrid_score:.2f} Robustness={grasp.robustness_score:.2f}",
            f"Worst={grasp.worst_case_score:.2f} Final={grasp.final_score:.2f}",
        ]
        if grasp.worst_case_score >= 0.6:
            lines.append("Robust to the tested small errors")
        cv2.rectangle(bgr, (0, 0), (min(420, bgr.shape[1]), 27 * len(lines) + 8), (0, 0, 0), -1)
        for row, line in enumerate(lines):
            cv2.putText(bgr, line, (8, 22 + 27 * row), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(bgr, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr
