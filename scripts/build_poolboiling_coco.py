from __future__ import annotations

import argparse
from pathlib import Path

from boilingbench_cv.poolboiling import build_poolboiling_coco
from boilingbench_cv.splits import build_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical records from collaborator bubble contours.")
    parser.add_argument("--root", required=True, type=Path, help="Authorized PoolBoilingDatasets root.")
    parser.add_argument("--output", required=True, type=Path, help="Ignored directory for derived annotations.")
    args = parser.parse_args()
    summary = build_poolboiling_coco(args.root, args.output)
    summary.update(build_splits(args.output / "manifest.csv", args.output / "splits"))
    print(summary)


if __name__ == "__main__":
    main()
