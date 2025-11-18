"""CLI to validate converted/tokenized maritime map .pt files."""

from __future__ import annotations

import argparse
from typing import List, Mapping

import torch
from torch_geometric.data import HeteroData


CheckList = List[str]


# Utility helpers ------------------------------------------------------------

def _get_from_data(data: object, key: object):
    if isinstance(data, Mapping):
        return data.get(key)
    if isinstance(data, HeteroData):
        return data[key]
    return None


def _tensor_desc(t: torch.Tensor | None) -> str:
    if t is None:
        return "missing"
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"


def _has_nan_or_inf(t: torch.Tensor | None) -> bool:
    if t is None:
        return False
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()


# Core checks ----------------------------------------------------------------

def check_map_point(data: object, errors: CheckList, warnings: CheckList) -> int:
    mp = _get_from_data(data, "map_point")
    if mp is None:
        errors.append("missing map_point")
        return 0

    pos = getattr(mp, "position", None)
    ori = getattr(mp, "orientation", None)
    typ = getattr(mp, "type", None)
    if pos is None:
        errors.append("map_point.position is missing")
        return 0

    num_points = pos.shape[0]
    if ori is None:
        warnings.append("map_point.orientation is missing")
    elif ori.shape[0] != num_points:
        errors.append(
            f"map_point.orientation length {ori.shape[0]} does not match position {num_points}"
        )
    if typ is None:
        warnings.append("map_point.type is missing")
    elif typ.shape[0] != num_points:
        errors.append(f"map_point.type length {typ.shape[0]} does not match position {num_points}")

    if _has_nan_or_inf(pos) or _has_nan_or_inf(ori) or _has_nan_or_inf(typ):
        errors.append("map_point contains NaN/Inf values")

    return num_points


def check_map_polygon(data: object, errors: CheckList, warnings: CheckList) -> int:
    mp = _get_from_data(data, "map_polygon")
    if mp is None:
        errors.append("missing map_polygon")
        return 0

    typ = getattr(mp, "type", None)
    light = getattr(mp, "light_type", None)
    num_poly = typ.shape[0] if typ is not None else 0

    if typ is None:
        errors.append("map_polygon.type is missing")
    if light is None:
        warnings.append("map_polygon.light_type is missing (default 0 recommended)")
    elif typ is not None and light.shape[0] != num_poly:
        errors.append(
            f"map_polygon.light_type length {light.shape[0]} does not match type {num_poly}"
        )

    if _has_nan_or_inf(typ) or _has_nan_or_inf(light):
        errors.append("map_polygon contains NaN/Inf values")

    return num_poly


def check_edges(data: object, num_points: int, num_polygons: int, errors: CheckList) -> None:
    edge = _get_from_data(data, ("map_point", "to", "map_polygon"))
    edge_index = getattr(edge, "edge_index", None)
    if edge_index is None:
        errors.append("edge_index for ('map_point','to','map_polygon') is missing")
        return
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        errors.append(f"edge_index must be shape [2, E], got {tuple(edge_index.shape)}")
        return

    if edge_index.numel() == 0:
        errors.append("edge_index is empty")
        return

    if edge_index[0].max().item() >= num_points:
        errors.append("edge_index point indices exceed map_point count")
    if edge_index[1].max().item() >= num_polygons:
        errors.append("edge_index polygon indices exceed map_polygon count")

    if _has_nan_or_inf(edge_index):
        errors.append("edge_index contains NaN/Inf values")


def check_tokens(data: Mapping[str, object], errors: CheckList, warnings: CheckList) -> None:
    pt_token = data.get("pt_token") if isinstance(data, Mapping) else None
    map_save = data.get("map_save") if isinstance(data, Mapping) else None

    if pt_token is None and map_save is None:
        warnings.append("pt_token/map_save not found (raw converted map assumed)")
        return

    if pt_token is None or map_save is None:
        warnings.append("pt_token or map_save missing: tokenized map incomplete")
        return

    # Basic consistency between map_save and pt_token
    traj_pos = map_save.get("traj_pos") if isinstance(map_save, Mapping) else None
    pl_idx = map_save.get("pl_idx_list") if isinstance(map_save, Mapping) else None
    traj_theta = map_save.get("traj_theta") if isinstance(map_save, Mapping) else None

    if traj_pos is None or traj_theta is None or pl_idx is None:
        errors.append("map_save is missing traj_pos/traj_theta/pl_idx_list")
        return

    num_rows = traj_pos.shape[0]
    if traj_theta.shape[0] != num_rows:
        errors.append("traj_theta length does not match traj_pos rows")
    if pl_idx.shape[0] != num_rows:
        errors.append("pl_idx_list length does not match traj_pos rows")

    tt_num_nodes = pt_token.get("num_nodes") if isinstance(pt_token, Mapping) else None
    if tt_num_nodes is not None and tt_num_nodes != num_rows:
        errors.append(f"pt_token.num_nodes={tt_num_nodes} differs from map_save rows={num_rows}")

    if _has_nan_or_inf(traj_pos) or _has_nan_or_inf(traj_theta) or _has_nan_or_inf(pl_idx):
        errors.append("map_save contains NaN/Inf values")


# CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a converted/tokenized maritime map .pt file")
    parser.add_argument("path", help="Path to the converted_map.pt or tokenized map file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = torch.load(args.path, map_location="cpu")

    errors: CheckList = []
    warnings: CheckList = []

    num_points = check_map_point(loaded, errors, warnings)
    num_polygons = check_map_polygon(loaded, errors, warnings)
    check_edges(loaded, num_points, num_polygons, errors)

    if isinstance(loaded, Mapping):
        check_tokens(loaded, errors, warnings)

    print(f"Loaded: {args.path}")
    # Print map_point details
    mp = _get_from_data(loaded, 'map_point')
    mp_pos = getattr(mp, "position", None)
    mp_ori = getattr(mp, "orientation", None)
    mp_typ = getattr(mp, "type", None)
    print(f"map_point.position: {_tensor_desc(mp_pos)}")
    print(f"map_point.orientation: {_tensor_desc(mp_ori)}")
    print(f"map_point.type: {_tensor_desc(mp_typ)}")

    # Print map_polygon details
    mpg = _get_from_data(loaded, 'map_polygon')
    mpg_typ = getattr(mpg, "type", None)
    mpg_light = getattr(mpg, "light_type", None)
    print(f"map_polygon.type: {_tensor_desc(mpg_typ)}")
    print(f"map_polygon.light_type: {_tensor_desc(mpg_light)}")

    # Print edge_index details
    edge = _get_from_data(loaded, ('map_point', 'to', 'map_polygon'))
    edge_index = getattr(edge, "edge_index", None)
    print(f"edge_index: {_tensor_desc(edge_index)}")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"- {err}")
    if warnings:
        print("\nWarnings:")
        for warn in warnings:
            print(f"- {warn}")

    if errors:
        raise SystemExit(1)
    print("\nValidation passed.")


if __name__ == "__main__":
    main()
