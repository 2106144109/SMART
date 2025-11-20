"""Inspect a processed maritime sample for anchor metadata.

This helper prints whether a sample contains the latitude/longitude anchors
(`origin_lat/origin_lon` or `ref_lat/ref_lon/ref_theta`) that are required by
``visualize_scenes_folium.py``. It is intended to quickly troubleshoot cases
where Folium visualizations fall back to the default inland center because the
preprocessed data lacks these fields.
"""

import argparse
import pprint
from typing import Any, Mapping

import torch


def _maybe_get(container: Mapping[str, Any], key: str) -> Any:
    """Return ``container[key]`` if it exists, otherwise ``None``.

    ``dict.get`` would silently return ``None`` even if the key exists but its
    value is ``None``. We only return ``None`` when the key is genuinely
    missing so callers can tell the difference between "missing" and "present
    but empty".
    """

    if key in container:
        return container[key]
    return None


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def inspect_sample(path: str) -> None:
    data = torch.load(path, map_location="cpu")

    _print_section("Top-level object")
    print(f"type: {type(data)}")
    if isinstance(data, Mapping):
        print(f"keys: {sorted(data.keys())}")
    else:
        print("(not a Mapping; cannot inspect metadata fields)")
        return

    metadata = _maybe_get(data, "metadata")
    _print_section("metadata")
    if metadata is None:
        print("<missing 'metadata' in sample>")
    else:
        print(f"type: {type(metadata)}")
        if isinstance(metadata, Mapping):
            print(f"keys: {sorted(metadata.keys())}")
            print(
                "origin_lat/lon:",
                _maybe_get(metadata, "origin_lat"),
                _maybe_get(metadata, "origin_lon"),
            )
        else:
            pprint.pprint(metadata)

    scene_info = _maybe_get(data, "scene_info") or _maybe_get(data, "scene_metadata")
    _print_section("scene_info / scene_metadata")
    if scene_info is None:
        print("<missing 'scene_info' and 'scene_metadata'>")
    else:
        print(f"type: {type(scene_info)}")
        if isinstance(scene_info, Mapping):
            print(f"keys: {sorted(scene_info.keys())}")
            print(
                "ref_lat/lon/theta:",
                _maybe_get(scene_info, "ref_lat"),
                _maybe_get(scene_info, "ref_lon"),
                _maybe_get(scene_info, "ref_theta"),
            )
        else:
            pprint.pprint(scene_info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a processed maritime .pt sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_sample(args.path)


if __name__ == "__main__":
    main()
