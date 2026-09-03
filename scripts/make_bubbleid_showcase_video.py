"""Create a labeled BubbleID-base segmentation showcase from source videos.

This is a qualitative, frozen-checkpoint demonstration.  It is deliberately
separate from benchmark scoring: no ground truth is drawn and no benchmark
annotations influence the model or its displayed predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


DEFAULT_CLIPS = (
    ("FCu-H2O | water on fluorinated copper | 100 W", "FCu-H2O/100W.mp4", 30),
    ("PCu-H2O | water on plain copper | 10 W", "PCu-H2O/10W.mp4", 0),
    ("PSi-HFE | HFE-7100 on polished silicon | 6 W", "PSi-HFE/6W.mp4", 0),
    ("SSi-HFE | HFE-7100 on smooth silicon | 6 W", "SSi-HFE/6W.mp4", 0),
)


def predictor(weights: Path, threshold: float, device: str) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(weights)
    # BubbleID-base was trained with two foreground categories.  For display,
    # both are rendered as class-agnostic predicted bubble instances.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = 640, 900
    return DefaultPredictor(cfg)


def add_text(image: np.ndarray, text: str, pos: tuple[int, int], scale: float = 0.72) -> None:
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def overlay_instances(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    output = frame.copy()
    palette = ((32, 196, 255), (255, 152, 32), (193, 71, 255), (74, 220, 121), (47, 116, 255))
    for index, mask in enumerate(masks):
        color = np.asarray(palette[index % len(palette)], dtype=np.uint8)
        output[mask] = (0.42 * output[mask] + 0.58 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, tuple(int(value) for value in color), 2, cv2.LINE_AA)
    return output


def render_page(left: np.ndarray, right: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    display_size = (930, 523)
    y = 285
    canvas[y : y + display_size[1], 20 : 20 + display_size[0]] = cv2.resize(left, display_size, interpolation=cv2.INTER_AREA)
    canvas[y : y + display_size[1], 970 : 970 + display_size[0]] = cv2.resize(right, display_size, interpolation=cv2.INTER_AREA)
    add_text(canvas, title, (30, 62), 0.92)
    add_text(canvas, subtitle, (30, 101), 0.58)
    add_text(canvas, "Raw camera frame", (20, 255), 0.68)
    add_text(canvas, "BubbleID-base predicted instance masks", (970, 255), 0.68)
    add_text(canvas, "Qualitative frozen-checkpoint demonstration; colors distinguish predicted instances only.", (30, 1035), 0.52)
    return canvas


def write_title(writer: cv2.VideoWriter, title: str, subtitle: str, frames: int) -> None:
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    add_text(canvas, "BubbleID-base: cross-regime segmentation showcase", (90, 430), 1.1)
    add_text(canvas, title, (90, 500), 0.9)
    add_text(canvas, subtitle, (90, 548), 0.62)
    add_text(canvas, "Frozen pretrained checkpoint; no benchmark-label access or tuning.", (90, 640), 0.6)
    for _ in range(frames):
        writer.write(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=42)
    parser.add_argument("--stride", type=int, default=3, help="Source frames skipped between displayed frames.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.frames_per_clip < 1 or args.stride < 1 or args.fps <= 0:
        raise ValueError("frames-per-clip, stride, and fps must be positive.")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video output: {args.output}")
    model = predictor(args.weights, args.score_threshold, args.device)
    records: list[dict] = []
    try:
        for clip_index, (label, relative_path, start_frame) in enumerate(DEFAULT_CLIPS, start=1):
            path = args.video_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
            write_title(writer, f"Segment {clip_index}/4 — {label}", f"Source: {relative_path}; start frame: {start_frame}", int(args.fps))
            capture = cv2.VideoCapture(str(path))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            shown = 0
            source_frames: list[int] = []
            prediction_counts: list[int] = []
            while shown < args.frames_per_clip:
                ok, frame = capture.read()
                if not ok:
                    break
                source_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                instances = model(frame)["instances"].to("cpu")
                masks = instances.pred_masks.numpy() if instances.has("pred_masks") else np.zeros((0, *frame.shape[:2]), dtype=bool)
                count_text = f"predicted bubbles: {len(masks)}"
                # Detectron2's default maximum number of detections per image
                # is 100 for this BubbleID configuration.
                if len(masks) >= 100:
                    count_text += " (model display limit reached)"
                page = render_page(frame, overlay_instances(frame, masks), label, f"Source frame {source_frame} | {count_text}")
                writer.write(page)
                shown += 1
                source_frames.append(source_frame)
                prediction_counts.append(int(len(masks)))
                for _ in range(args.stride - 1):
                    if not capture.grab():
                        break
            capture.release()
            if shown == 0:
                raise RuntimeError(f"No frames read from {path}")
            records.append({"label": label, "source_video": str(path), "start_frame": start_frame,
                            "displayed_source_frames": source_frames, "predicted_instance_counts": prediction_counts})
    finally:
        writer.release()
    args.summary.write_text(json.dumps({"purpose": "qualitative frozen-checkpoint BubbleID-base demonstration",
                                        "weights": str(args.weights), "score_threshold": args.score_threshold,
                                        "device": args.device, "fps": args.fps, "frames_per_clip": args.frames_per_clip,
                                        "stride": args.stride, "clips": records}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "segments": len(records), "displayed_frames": sum(len(x["displayed_source_frames"]) for x in records)}, indent=2))


if __name__ == "__main__":
    main()
