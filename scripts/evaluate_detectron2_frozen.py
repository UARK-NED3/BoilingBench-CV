"""Evaluate a frozen Detectron2 Mask R-CNN checkpoint on a canonical split.

Run this script from an environment that provides numpy, OpenCV, torch, and
Detectron2. It intentionally does not train or tune the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def build_predictor(weights: str, threshold: float, device: str, anchor_profile: str, num_classes: int) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    if anchor_profile == "flow":
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = 640, 900
    return DefaultPredictor(cfg)


def save_panel(path: Path, frame: np.ndarray, prediction: np.ndarray, target: np.ndarray, title: str) -> None:
    """Save raw, ground-truth, prediction, and pixel-error panes for one image."""
    gt_union = np.any(target, axis=0) if len(target) else np.zeros(frame.shape[:2], dtype=bool)
    pred_union = np.any(prediction, axis=0) if len(prediction) else np.zeros(frame.shape[:2], dtype=bool)
    gt = frame.copy(); gt[gt_union] = (0.35 * gt[gt_union] + 0.65 * np.array([0, 220, 0])).astype(np.uint8)
    pred = frame.copy(); pred[pred_union] = (0.35 * pred[pred_union] + 0.65 * np.array([0, 0, 220])).astype(np.uint8)
    error = frame.copy()
    error[np.logical_and(gt_union, pred_union)] = (0, 220, 0)  # true positive: green
    error[np.logical_and(~gt_union, pred_union)] = (0, 0, 255) # false positive: red
    error[np.logical_and(gt_union, ~pred_union)] = (255, 0, 0) # false negative: blue
    panes = [(frame, "raw"), (gt, "ground truth"), (pred, "prediction"), (error, "TP green / FP red / FN blue")]
    rendered = []
    for pane, label in panes:
        cv2.putText(pane, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pane, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(pane)
    grid = cv2.vconcat([cv2.hconcat(rendered[:2]), cv2.hconcat(rendered[2:])])
    cv2.putText(grid, title, (8, grid.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(grid, title, (8, grid.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def rasterize_gt(annotations: list[dict], height: int, width: int) -> np.ndarray:
    masks = []
    for annotation in annotations:
        points = np.asarray(annotation["segmentation"][0], dtype=np.int32).reshape(-1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask.astype(bool))
    return np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool)


def match_counts(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int]:
    pairs = []
    for p in range(len(pred)):
        for t in range(len(target)):
            union = np.logical_or(pred[p], target[t]).sum()
            score = 0.0 if union == 0 else np.logical_and(pred[p], target[t]).sum() / union
            if score >= threshold:
                pairs.append((score, p, t))
    used_p, used_t, tp = set(), set(), 0
    for _score, p, t in sorted(pairs, reverse=True):
        if p not in used_p and t not in used_t:
            used_p.add(p); used_t.add(t); tp += 1
    return tp, len(pred) - tp, len(target) - tp


def summarize(rows: list[dict]) -> dict:
    tp = sum(row["pixel_tp"] for row in rows)
    fp = sum(row["pixel_fp"] for row in rows)
    fn = sum(row["pixel_fn"] for row in rows)
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    instance_tp = sum(row["instance_tp"] for row in rows)
    instance_fp = sum(row["instance_fp"] for row in rows)
    instance_fn = sum(row["instance_fn"] for row in rows)
    return {
        "images": len(rows), "pixel_iou_micro": iou, "pixel_precision_micro": precision,
        "pixel_recall_micro": recall, "pixel_f1_micro": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "instance_precision_iou50": instance_tp / (instance_tp + instance_fp) if instance_tp + instance_fp else 1.0,
        "instance_recall_iou50": instance_tp / (instance_tp + instance_fn) if instance_tp + instance_fn else 1.0,
        "count_mae": float(np.mean([abs(r["pred_count"] - r["gt_count"]) for r in rows])),
        "projected_area_fraction_mae": float(np.mean([abs(r["pred_area_fraction"] - r["gt_area_fraction"]) for r in rows])),
        "median_inference_seconds": float(np.median([r["inference_seconds"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-checkpoint screening evaluation on one canonical split.")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--anchor-profile", choices=("default", "flow"), default="default")
    parser.add_argument("--num-classes", type=int, default=1, help="Foreground classes expected by checkpoint; all are merged for class-agnostic bubble scoring.")
    parser.add_argument("--panel-image-ids", default="", help="Comma-separated canonical image IDs for visual panels.")
    parser.add_argument("--panels-dir", type=Path)
    parser.add_argument("--only-image-ids", default="", help="Optional comma-separated image IDs; intended for panel-only runs.")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    wanted = {int(row["image_id"]) for row in csv.DictReader(args.split.open(newline="", encoding="utf-8")) if row["split"] == "test"}
    images = [image for image in coco["images"] if image["id"] in wanted]
    if args.limit is not None:
        images = images[: args.limit]
    only_ids = {int(value) for value in args.only_image_ids.split(",") if value.strip()}
    if only_ids:
        images = [image for image in images if image["id"] in only_ids]
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["image_id"] in wanted and not annotation.get("ignore", 0):
            by_image[annotation["image_id"]].append(annotation)
    panel_ids = {int(value) for value in args.panel_image_ids.split(",") if value.strip()}
    predictor = build_predictor(args.weights, args.score_threshold, args.device, args.anchor_profile, args.num_classes)
    rows = []
    for image in images:
        frame = cv2.imread(str(args.data_root / image["file_name"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read {image['file_name']}")
        start = time.perf_counter()
        instances = predictor(frame)["instances"].to("cpu")
        elapsed = time.perf_counter() - start
        predicted = instances.pred_masks.numpy() if instances.has("pred_masks") else np.zeros((0, image["height"], image["width"]), dtype=bool)
        target = rasterize_gt(by_image[image["id"]], image["height"], image["width"])
        pred_union = np.any(predicted, axis=0) if len(predicted) else np.zeros((image["height"], image["width"]), dtype=bool)
        target_union = np.any(target, axis=0) if len(target) else np.zeros_like(pred_union)
        pixel_tp = int(np.logical_and(pred_union, target_union).sum())
        pixel_fp = int(np.logical_and(pred_union, ~target_union).sum())
        pixel_fn = int(np.logical_and(~pred_union, target_union).sum())
        instance_tp, instance_fp, instance_fn = match_counts(predicted, target)
        if args.panels_dir and image["id"] in panel_ids:
            save_panel(args.panels_dir / f"{image['id']:04d}_{Path(image['file_name']).stem}.png", frame, predicted, target,
                       f"image {image['id']} | {image['extra']['regime']} | pred={len(predicted)}, gt={len(target)}")
        rows.append({"image_id": image["id"], "regime": image["extra"]["regime"], "pixel_tp": pixel_tp,
                     "pixel_fp": pixel_fp, "pixel_fn": pixel_fn, "instance_tp": instance_tp,
                     "instance_fp": instance_fp, "instance_fn": instance_fn, "pred_count": int(len(predicted)),
                     "gt_count": int(len(target)), "pred_area_fraction": float(pred_union.mean()),
                     "gt_area_fraction": float(target_union.mean()), "inference_seconds": elapsed})
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["regime"]].append(row)
    report = {"protocol": "frozen-checkpoint screening; no benchmark-label access or tuning",
              "model": {"weights": args.weights, "score_threshold": args.score_threshold,
                        "anchor_profile": args.anchor_profile, "num_classes": args.num_classes, "device": args.device},
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
              "overall": summarize(rows), "by_regime": {name: summarize(items) for name, items in sorted(grouped.items())}, "per_image": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_regime": report["by_regime"]}, indent=2))


if __name__ == "__main__":
    main()
