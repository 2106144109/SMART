"""Inspect a processed maritime sample for anchor metadata.

This helper prints whether a sample contains the latitude/longitude anchors
(`origin_lat/origin_lon` or `ref_lat/ref_lon/ref_theta`) that are required by
``visualize_scenes_folium.py``. It is intended to quickly troubleshoot cases
where Folium visualizations fall back to the default inland center because the
preprocessed data lacks these fields.
"""

import argparse
import pprint
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _sorted_keys(container: Mapping[Any, Any]) -> list[str]:
    """Return keys sorted safely even when they have different types."""

    return sorted(container.keys(), key=repr)


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


def _find_anchor_candidates(obj: Any, target_keys: set[str], path: str = "$") -> list[tuple[str, Any]]:
    """Return a list of (path, value) pairs for matching anchor keys.

    The search is best-effort and inspects mappings and sequences recursively,
    ignoring other container types. Paths are printed in a JSONPath-like
    format (e.g., ``$.metadata.origin_lat`` or ``$.agent[0].ref_lat``).
    """

    results: list[tuple[str, Any]] = []

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_path = f"{path}.{key}"
            if isinstance(key, str) and key in target_keys:
                results.append((key_path, value))
            results.extend(_find_anchor_candidates(value, target_keys, key_path))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for idx, value in enumerate(obj):
            idx_path = f"{path}[{idx}]"
            results.extend(_find_anchor_candidates(value, target_keys, idx_path))

    return results


def _find_geoish_keys(obj: Any, path: str = "$") -> list[tuple[str, Any]]:
    """Return (path, value) pairs for keys that *look* geo-related.

    This is a looser heuristic than :func:`_find_anchor_candidates` and tries
    to surface anything with ``lat``, ``lon`` or ``theta`` in the key name
    (case-insensitive). Tuple keys are stringified so that graph-like metadata
    is still inspectable.
    """

    results: list[tuple[str, Any]] = []

    def key_is_geoish(key: Any) -> bool:
        if isinstance(key, str):
            haystack = key
        else:
            haystack = repr(key)
        haystack = haystack.lower()
        return any(token in haystack for token in ("lat", "lon", "theta"))

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_path = f"{path}.{key}"
            if key_is_geoish(key):
                results.append((key_path, value))
            results.extend(_find_geoish_keys(value, key_path))
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for idx, value in enumerate(obj):
            idx_path = f"{path}[{idx}]"
            results.extend(_find_geoish_keys(value, idx_path))

    return results


def inspect_sample(path: str) -> None:
    data = torch.load(path, map_location="cpu")

    _print_section("Top-level object")
    print(f"type: {type(data)}")
    if isinstance(data, Mapping):
        print(f"keys: {_sorted_keys(data)}")
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
            print(f"keys: {_sorted_keys(metadata)}")
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
            print(f"keys: {_sorted_keys(scene_info)}")
            print(
                "ref_lat/lon/theta:",
                _maybe_get(scene_info, "ref_lat"),
                _maybe_get(scene_info, "ref_lon"),
                _maybe_get(scene_info, "ref_theta"),
            )
        else:
            pprint.pprint(scene_info)

    _print_section("Search for anchor candidates anywhere in sample")
    anchor_candidates = _find_anchor_candidates(
        data,
        {"origin_lat", "origin_lon", "ref_lat", "ref_lon", "ref_theta"},
    )
    if not anchor_candidates:
        print("<no anchor-like keys found>")
    else:
        for candidate_path, value in anchor_candidates:
            print(f"{candidate_path}: {value}")

    _print_section("Search for geo-related keys (fuzzy)")
    geoish_candidates = _find_geoish_keys(data)
    if not geoish_candidates:
        print("<no geo-ish keys found>")
    else:
        for candidate_path, value in geoish_candidates:
            print(f"{candidate_path}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a processed maritime .pt sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_sample(args.path)


if __name__ == "__main__":
    main()
