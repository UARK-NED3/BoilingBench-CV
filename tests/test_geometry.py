from boilingbench_cv.geometry import polygon_area, polygon_bbox, validate_polygon


def test_square_geometry() -> None:
    points = [1, 1, 5, 1, 5, 5, 1, 5]
    assert polygon_area(points) == 16
    assert polygon_bbox(points) == [1, 1, 4, 4]
    assert validate_polygon(points, 10, 10) == []


def test_invalid_geometry_is_flagged_without_mutation() -> None:
    assert "out_of_bounds" in validate_polygon([0, 0, 11, 0, 0, 1], 10, 10)
    assert validate_polygon([0, 0, 1], 10, 10) == ["invalid_coordinate_count"]
