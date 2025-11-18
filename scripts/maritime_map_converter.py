"""Utilities to convert maritime segment lists into SMART map HeteroData.

The converter expects segment dictionaries shaped like::
    {
        "polyline_id": "water_1231",
        "shape_type": "water",
        "shape_id": 1231,
        "start_x": 357912.43937938113,
        "start_y": 3462367.9400085662,
        "end_x": 358036.3697113833,
        "end_y": 3462233.259432094,
    }

It groups segments by ``polyline_id``, orders them by geometric continuity,
computes headings, and fills the ``map_point``/``map_polygon`` fields expected by
``TokenProcessor.preprocess``.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import torch
from torch_geometric.data import HeteroData

Segment = MutableMapping[str, object]


def group_segments_by_polyline(segments: Iterable[Segment]) -> Dict[str, List[Segment]]:
    """Group raw segments by ``polyline_id``.

    Args:
        segments: Iterable of raw segment dictionaries.

    Returns:
        Mapping from ``polyline_id`` to a list of segments preserving the
        original order.
    """

    grouped: Dict[str, List[Segment]] = {}
    for seg in segments:
        pid = str(seg["polyline_id"])
        grouped.setdefault(pid, []).append(seg)
    return grouped


def sort_segments_by_continuity(segments: Sequence[Segment]) -> List[Segment]:
    """Sort segments so each segment's start is closest to the previous end.

    This is a greedy ordering that works for well-formed polylines where each
    segment connects to the next. If the segments are already ordered, they are
    returned unchanged.
    """

    if len(segments) <= 1:
        return list(segments)

    ordered: List[Segment] = [segments[0]]
    remaining = list(segments[1:])

    while remaining:
        last_end = (ordered[-1]["end_x"], ordered[-1]["end_y"])
        next_idx = min(
            range(len(remaining)),
            key=lambda i: (remaining[i]["start_x"] - last_end[0]) ** 2
            + (remaining[i]["start_y"] - last_end[1]) ** 2,
        )
        ordered.append(remaining.pop(next_idx))
    return ordered


def points_from_ordered_segments(segments: Sequence[Segment]) -> torch.Tensor:
    """Construct an ordered point tensor ``[[x, y], ...]`` from sorted segments."""

    if not segments:
        return torch.empty((0, 2), dtype=torch.float)

    points = [(segments[0]["start_x"], segments[0]["start_y"])]
    for seg in segments:
        points.append((seg["end_x"], seg["end_y"]))
    return torch.tensor(points, dtype=torch.float)


def compute_headings(points: torch.Tensor) -> torch.Tensor:
    """Compute per-point heading (radians) along a polyline.

    For the final point, reuse the previous segment's heading. If only a single
    point exists, heading defaults to ``0.0``.
    """

    if points.numel() == 0:
        return torch.empty((0,), dtype=torch.float)

    headings: List[float] = []
    for i in range(points.size(0)):
        if i + 1 < points.size(0):
            dx = points[i + 1, 0] - points[i, 0]
            dy = points[i + 1, 1] - points[i, 1]
            headings.append(math.atan2(dy, dx))
        else:
            headings.append(headings[-1] if headings else 0.0)
    return torch.tensor(headings, dtype=torch.float)


def convert_segments_to_heterodata(
    segments: Iterable[Segment],
    type_map: Mapping[str, int] | None = None,
    default_type: int = 0,
) -> HeteroData:
    """Convert maritime segments into ``HeteroData`` for ``tokenize_map``.

    Args:
        segments: Iterable of segment dictionaries with start/end coordinates
            and ``polyline_id``/``shape_type`` keys.
        type_map: Optional mapping from ``shape_type`` to integer type id used
            for both points and polygons.
        default_type: Fallback type id when ``shape_type`` is unknown.

    Returns:
        A ``HeteroData`` object with ``map_point``/``map_polygon`` and
        ``('map_point', 'to', 'map_polygon')`` edge_index populated.
    """

    type_map = type_map or {}
    grouped = group_segments_by_polyline(segments)

    data = HeteroData()
    positions: List[torch.Tensor] = []
    orientations: List[torch.Tensor] = []
    point_types: List[torch.Tensor] = []
    edge_rows: List[torch.Tensor] = []
    edge_cols: List[torch.Tensor] = []
    polygon_types: List[int] = []
    polygon_lights: List[int] = []
    point_offset = 0

    for poly_idx, pid in enumerate(sorted(grouped.keys())):
        ordered_segs = sort_segments_by_continuity(grouped[pid])
        pts_xy = points_from_ordered_segments(ordered_segs)
        pts = torch.cat([pts_xy, torch.zeros((pts_xy.size(0), 1))], dim=1)
        headings = compute_headings(pts_xy)

        shape_type = str(ordered_segs[0].get("shape_type", "")) if ordered_segs else ""
        type_id = type_map.get(shape_type, default_type)

        positions.append(pts)
        orientations.append(headings)
        point_types.append(torch.full((pts.size(0),), type_id, dtype=torch.long))

        edge_rows.append(torch.arange(point_offset, point_offset + pts.size(0), dtype=torch.long))
        edge_cols.append(torch.full((pts.size(0),), poly_idx, dtype=torch.long))
        point_offset += pts.size(0)

        polygon_types.append(type_id)
        polygon_lights.append(0)

    if positions:
        data["map_point"].position = torch.cat(positions)
        data["map_point"].orientation = torch.cat(orientations)
        data["map_point"].type = torch.cat(point_types)
        data["map_point", "to", "map_polygon"].edge_index = torch.stack(
            [torch.cat(edge_rows), torch.cat(edge_cols)]
        )
    else:
        data["map_point"].position = torch.empty((0, 3), dtype=torch.float)
        data["map_point"].orientation = torch.empty((0,), dtype=torch.float)
        data["map_point"].type = torch.empty((0,), dtype=torch.long)
        data["map_point", "to", "map_polygon"].edge_index = torch.empty((2, 0), dtype=torch.long)

    data["map_polygon"].type = torch.tensor(polygon_types, dtype=torch.long)
    data["map_polygon"].light_type = torch.tensor(polygon_lights, dtype=torch.long)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert maritime start/end segments JSON into SMART HeteroData fields.",
    )
    parser.add_argument(
        "input",
        help="Path to a JSON file containing a list of segment dictionaries.",
    )
    parser.add_argument(
        "--type-map",
        nargs="*",
        default=[],
        help="Optional mappings like water=1 channel=0 berth=3. Unknown types use default-type.",
    )
    parser.add_argument(
        "--default-type",
        type=int,
        default=0,
        help="Fallback type id when shape_type is not found in the type map.",
    )
    parser.add_argument(
        "--tokenize-map",
        action="store_true",
        help="Run TokenProcessor.tokenize_map to build pt_token/map_save for the converted map.",
    )
    parser.add_argument(
        "--token-size",
        type=int,
        default=2048,
        help="Token size for TokenProcessor (needs matching agent token file, default 2048).",
    )
    parser.add_argument(
        "--save",
        help="Optional path to torch.save() the converted (or tokenized) map data.",
    )
    return parser.parse_args()


def parse_type_map(raw: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in raw:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        mapping[key] = int(value)
    return mapping


def main() -> None:
    args = parse_args()
    type_map = parse_type_map(args.type_map)

    with open(args.input, "r", encoding="utf-8") as f:
        segments = json.load(f)

    data = convert_segments_to_heterodata(segments, type_map=type_map, default_type=args.default_type)

    print("map_point.position:", data["map_point"].position)
    print("map_point.orientation:", data["map_point"].orientation)
    print("map_point.type:", data["map_point"].type)
    print("map_polygon.type:", data["map_polygon"].type)
    print("map_polygon.light_type:", data["map_polygon"].light_type)
    print("edge_index:", data["map_point", "to", "map_polygon"].edge_index)

    if args.tokenize_map:
        from smart.datasets.preprocess import TokenProcessor

        token_processor = TokenProcessor(token_size=args.token_size)
        map_data = {
            "map_point": data["map_point"],
            "map_polygon": data["map_polygon"],
            ("map_point", "to", "map_polygon"): data["map_point", "to", "map_polygon"],
        }
        tokenized = token_processor.tokenize_map(map_data)
        pt_token = tokenized["pt_token"]
        map_save = tokenized["map_save"]

        print("pt_token keys:", {k: v.shape if hasattr(v, "shape") else v for k, v in pt_token.items()})
        print(
            "map_save shapes:",
            {k: v.shape if hasattr(v, "shape") else v for k, v in map_save.items()},
        )

        if args.save:
            torch.save(tokenized, args.save)
            print(f"Tokenized map saved to {args.save}")
    elif args.save:
        torch.save(data, args.save)
        print(f"Converted map saved to {args.save}")


if __name__ == "__main__":
    main()
