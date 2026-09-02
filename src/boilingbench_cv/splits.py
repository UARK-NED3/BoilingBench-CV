"""Leakage-safe split generation for source-video grouped data."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def _stable_bucket(group: str) -> int:
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 10


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "group_id"])
        writer.writeheader()
        writer.writerows(rows)


def build_splits(manifest_csv: Path, output: Path) -> dict[str, int]:
    """Build deterministic partitions without splitting a source video across sets."""
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['regime']}::{row['source_video']}"] .append(row)

    pooled = []
    for group, members in sorted(grouped.items()):
        bucket = _stable_bucket(group)
        split = "test" if bucket >= 8 else "val" if bucket == 7 else "train"
        pooled.extend({"image_id": row["image_id"], "split": split, "group_id": group} for row in members)
    _write(output / "pooled_grouped.csv", pooled)

    for name, train_fluid, test_fluid in (("water_to_hfe", "H2O", "HFE-7100"), ("hfe_to_water", "HFE-7100", "H2O")):
        rows_out = []
        for row in rows:
            group = f"{row['regime']}::{row['source_video']}"
            if row["fluid"] == test_fluid:
                split = "test"
            elif row["fluid"] == train_fluid:
                split = "val" if _stable_bucket(group) == 7 else "train"
            else:
                continue
            rows_out.append({"image_id": row["image_id"], "split": split, "group_id": group})
        _write(output / f"{name}.csv", rows_out)

    for held_out in sorted({row["regime"] for row in rows}):
        rows_out = []
        for row in rows:
            group = f"{row['regime']}::{row['source_video']}"
            split = "test" if row["regime"] == held_out else "val" if _stable_bucket(group) == 7 else "train"
            rows_out.append({"image_id": row["image_id"], "split": split, "group_id": group})
        _write(output / f"leave_{held_out}.csv", rows_out)
    return {"images": len(rows), "groups": len(grouped)}
