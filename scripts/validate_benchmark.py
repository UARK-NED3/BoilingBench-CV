from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from boilingbench_cv.geometry import validate_polygon


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical annotations and split leakage rules.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_by_id = {image["id"]: image for image in coco["images"]}
    flags = Counter()
    for annotation in coco["annotations"]:
        image = image_by_id[annotation["image_id"]]
        flags.update(validate_polygon(annotation["segmentation"][0], image["width"], image["height"]))
    split_reports = {}
    for path in sorted((args.annotations.parent / "splits").glob("*.csv")):
        groups: dict[str, set[str]] = defaultdict(set)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                groups[row["group_id"]].add(row["split"])
        leaked = sorted(group for group, partitions in groups.items() if len(partitions) > 1)
        split_reports[path.name] = {"sha256": _digest(path), "groups": len(groups), "leaked_groups": leaked}
    report = {"annotation_sha256": _digest(args.annotations), "images": len(coco["images"]),
              "annotations": len(coco["annotations"]), "geometry_flags": dict(flags), "splits": split_reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(value["leaked_groups"] for value in split_reports.values()):
        raise SystemExit("Split leakage detected")


if __name__ == "__main__":
    main()
