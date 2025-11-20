"""Inspect a processed maritime sample for anchor metadata.

This helper prints whether a sample contains the latitude/longitude anchors
(`origin_lat/origin_lon` or `ref_lat/ref_lon/ref_theta`) that are required by
``visualize_scenes_folium.py``. It is intended to quickly troubleshoot cases
where Folium visualizations fall back to the default inland center because the
preprocessed data lacks these fields.

It also looks inside common map stores (`map_save`/`pt_token`) and PyG
``HeteroData`` attributes in case metadata was nested instead of living at the
top level.
"""

import argparse
import pprint
from typing import Any, Mapping, MutableMapping

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


def _maybe_get_attr(obj: Any, key: str) -> Any:
    """Try ``getattr`` first, then mapping lookup."""

    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    return None


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def inspect_sample(path: str) -> None:
    data = torch.load(path, map_location="cpu")

    _print_section("Top-level object")
    print(f"type: {type(data)}")
    if isinstance(data, Mapping):
        print(f"keys: {sorted(data.keys())}")
    elif hasattr(data, "keys"):
        try:
            print(f"keys: {sorted(data.keys())}")
        except Exception:  # pragma: no cover - best-effort introspection
            print("(object exposes keys() but listing failed)")
    else:
        print("(not a Mapping; trying attribute lookup only)")

    # --- anchor metadata ---
    metadata = _maybe_get_attr(data, "metadata")
    _print_section("metadata (origin_lat/origin_lon)")
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

    scene_info = _maybe_get_attr(data, "scene_info") or _maybe_get_attr(data, "scene_metadata")
    _print_section("scene_info / scene_metadata (ref_lat/ref_lon/ref_theta)")
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

    # --- map stores: occasionally contain origin offsets ---
    map_save = _maybe_get_attr(data, "map_save")
    _print_section("map_save (origin / traj_pos hints)")
    if map_save is None:
        print("<missing 'map_save'>")
    elif isinstance(map_save, Mapping):
        keys = sorted(map_save.keys())
        print(f"keys: {keys}")
        origin_candidates = {k: map_save[k] for k in keys if "origin" in str(k)}
        if origin_candidates:
            print("origin-like entries:")
            pprint.pprint(origin_candidates)
        traj_pos = map_save.get("traj_pos")
        if traj_pos is not None:
            try:
                import numpy as np

                arr = traj_pos if isinstance(traj_pos, np.ndarray) else traj_pos.detach().cpu().numpy()
                print(f"traj_pos shape: {arr.shape}")
            except Exception as exc:  # pragma: no cover - best-effort introspection
                print(f"traj_pos present but shape check failed: {exc}")
    else:
        print(f"type: {type(map_save)} (not a Mapping)")

    pt_token = _maybe_get_attr(data, "pt_token")
    _print_section("pt_token (map tokenization)")
    if pt_token is None:
        print("<missing 'pt_token'>")
    else:
        print(f"type: {type(pt_token)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a processed maritime .pt sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_sample(args.path)


if __name__ == "__main__":
    main()
