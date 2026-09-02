"""Read collaborator contour annotations without modifying their source files."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from .geometry import polygon_area, polygon_bbox, validate_polygon

REGIMES = ("FCu-H2O", "PCu-H2O", "PSi-HFE", "SSi-HFE")


def jpeg_size(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions without an image-processing dependency."""
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG: {path}")
        while True:
            prefix = handle.read(1)
            while prefix == b"\xff":
                marker = handle.read(1)
                if marker not in (b"\xff", b"\x00"):
                    break
            else:
                marker = prefix
            if not marker:
                raise ValueError(f"JPEG dimensions not found: {path}")
            code = marker[0]
            if code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                handle.read(2)
                _precision, height, width = struct.unpack(">BHH", handle.read(5))
                return width, height
            if code in {0xD8, 0xD9}:
                continue
            length = struct.unpack(">H", handle.read(2))[0]
            handle.seek(length - 2, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source_video(filename: str, regime: str) -> tuple[str, int]:
    stem = Path(filename).stem
    prefix = f"{regime}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected image name for {regime}: {filename}")
    power, frame = stem[len(prefix):].rsplit("_", 1)
    return f"{power}.mp4", int(frame)


def build_poolboiling_coco(root: Path, output: Path, hash_images: bool = False) -> dict[str, Any]:
    """Create derived COCO-style records and a source manifest from contour JSON.

    Hashing every network-hosted image is optional because it can take much
    longer than the conversion itself. A release candidate must use it.
    """
    # Preserve the caller's mapped-drive path. Resolving it to a UNC path can
    # make repeated image opens stall on some institutional file servers.
    root, output = Path(root), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coco: dict[str, Any] = {
        "info": {"description": "BoilingBench-CV derived PoolBoiling contours", "version": "0.1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "bubble", "supercategory": "vapor"}],
        "images": [], "annotations": [],
    }
    manifest_rows: list[dict[str, Any]] = []
    image_id, annotation_id = 1, 1
    for regime in REGIMES:
        annotation_dir = root / regime / "annotatedBubbles"
        json_files = sorted(annotation_dir.glob("*-BubbleContours.json"))
        if len(json_files) != 1:
            raise FileNotFoundError(f"Expected one contour JSON in {annotation_dir}; found {len(json_files)}")
        source_json = json_files[0]
        records = json.loads(source_json.read_text(encoding="utf-8"))
        # The supplied annotated images are uniform in geometry within each
        # regime. Read one representative file here; a later release audit can
        # re-check every source image alongside full content hashing.
        first_record = records[sorted(records)[0]]
        regime_width, regime_height = jpeg_size(annotation_dir / first_record["FileName"])
        for source_record, record in sorted(records.items()):
            source_name = record["FileName"]
            source_image = annotation_dir / source_name
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            width, height = regime_width, regime_height
            source_video, frame_index = parse_source_video(source_name, regime)
            relative_name = source_image.relative_to(root).as_posix()
            coco["images"].append({
                "id": image_id, "file_name": relative_name, "width": width, "height": height,
                "extra": {"regime": regime, "source_video": source_video, "frame_index": frame_index,
                          "source_record": source_record,
                          "source_annotation": source_json.relative_to(root).as_posix()},
            })
            bubble_count, image_flags = 0, []
            for source_bubble, bubble in sorted(record["Bubbles"].items()):
                xs, ys = bubble["x_coordinate"], bubble["y_coordinate"]
                if not isinstance(xs, list) or not isinstance(ys, list):
                    points, flags = [], ["non_array_coordinates"]
                elif len(xs) != len(ys):
                    points, flags = [], ["mismatched_xy_length"]
                else:
                    points = [value for pair in zip(xs, ys) for value in pair]
                    flags = validate_polygon(points, width, height)
                image_flags.extend(flags)
                coco["annotations"].append({
                    "id": annotation_id, "image_id": image_id, "category_id": 1,
                    "segmentation": [points],
                    "bbox": polygon_bbox(points) if points else [0, 0, 0, 0],
                    "area": polygon_area(points), "iscrowd": 0,
                    "ignore": int(bool(flags)),
                    "extra": {"source_bubble": source_bubble, "validation_flags": flags},
                })
                annotation_id += 1
                bubble_count += 1
            manifest_rows.append({
                "image_id": image_id, "relative_file_name": relative_name, "source_path": str(source_image),
                "sha256": sha256(source_image) if hash_images else "", "regime": regime,
                "fluid": "H2O" if regime.endswith("H2O") else "HFE-7100",
                "surface": regime.split("-")[0], "source_video": source_video,
                "frame_index": frame_index, "source_record": source_record,
                "source_annotation": str(source_json), "bubble_count": bubble_count,
                "image_validation_flags": ";".join(sorted(set(image_flags))),
            })
            image_id += 1
    (output / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {"images": len(coco["images"]), "annotations": len(coco["annotations"]), "output": str(output)}
