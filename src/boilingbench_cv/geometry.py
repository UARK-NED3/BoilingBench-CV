"""Geometry helpers operating in source-image pixel coordinates."""

from __future__ import annotations

from collections.abc import Sequence


def validate_polygon(points: Sequence[float], width: int, height: int) -> list[str]:
    """Return validation flags without changing the supplied polygon."""
    flags: list[str] = []
    if len(points) < 6 or len(points) % 2:
        return ["invalid_coordinate_count"]
    xs, ys = points[0::2], points[1::2]
    if any(x < 0 or x >= width for x in xs) or any(y < 0 or y >= height for y in ys):
        flags.append("out_of_bounds")
    if polygon_area(points) <= 0:
        flags.append("zero_area")
    return flags


def polygon_area(points: Sequence[float]) -> float:
    """Return absolute shoelace area in square pixels."""
    if len(points) < 6 or len(points) % 2:
        return 0.0
    xy = list(zip(points[0::2], points[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]))) / 2


def polygon_bbox(points: Sequence[float]) -> list[float]:
    """Return the COCO [x, y, width, height] bounding box."""
    xs, ys = points[0::2], points[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
