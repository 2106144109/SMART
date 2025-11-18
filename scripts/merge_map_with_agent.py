"""
Merge a converted maritime map with agent data and run TokenProcessor.preprocess.

This is a convenience CLI so you can smoke-test map tokens together with a real
scenario sample. It will:
1) load a map .pt produced by maritime_map_converter.py (tokenized or not)
2) load an agent scenario .pt
3) ensure map tokens exist (tokenize if missing)
4) merge map + agent data, run preprocess, and optionally save the processed
   output for downstream training/eval.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import torch
from torch_geometric.data import HeteroData

from smart.datasets.preprocess import TokenProcessor


def _as_dict(data: Any) -> Dict[Any, Any]:
    """Return a mutable mapping view for HeteroData or plain dict inputs."""
    if isinstance(data, HeteroData):
        return data
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unsupported data type: {type(data)}")


def _require_map_keys(data: Dict[Any, Any]) -> None:
    missing = [k for k in ["map_point", "map_polygon", ("map_point", "to", "map_polygon")] if k not in data]
    if missing:
        raise KeyError(f"Map data is missing required keys: {missing}")


def _maybe_tokenize_map(map_data: Dict[Any, Any], token_size: int) -> Dict[Any, Any]:
    if "pt_token" in map_data and "map_save" in map_data:
        return map_data

    _require_map_keys(map_data)
    tp = TokenProcessor(token_size=token_size)
    minimal_map = {
        "map_point": map_data["map_point"],
        "map_polygon": map_data["map_polygon"],
        ("map_point", "to", "map_polygon"): map_data[("map_point", "to", "map_polygon")],
    }
    tokenized = tp.tokenize_map(minimal_map)
    map_with_tokens = copy.deepcopy(map_data)
    map_with_tokens.update({"pt_token": tokenized["pt_token"], "map_save": tokenized["map_save"]})
    return map_with_tokens


def _merge_map_into_agent(agent_data: Dict[Any, Any], map_data: Dict[Any, Any]) -> Dict[Any, Any]:
    merged = copy.deepcopy(agent_data)
    merged["map_point"] = map_data["map_point"]
    merged["map_polygon"] = map_data["map_polygon"]
    merged[("map_point", "to", "map_polygon")] = map_data[("map_point", "to", "map_polygon")]

    for key in ["pt_token", "map_save"]:
        if key in map_data:
            merged[key] = map_data[key]
    return merged


def _print_summary(processed: Dict[Any, Any]) -> None:
    pt_count = processed["map_point"]["position"].shape[0]
    poly_count = processed["map_polygon"]["type"].shape[0]
    token_rows = processed["pt_token"].get("num_nodes", 0) if "pt_token" in processed else 0
    agent_count = processed["agent"]["position"].shape[0] if "agent" in processed else 0
    print(f"map_point: {pt_count}, map_polygon: {poly_count}, pt_token rows: {token_rows}, agents: {agent_count}")


def main():
    parser = argparse.ArgumentParser(description="Merge map .pt with agent data and run preprocess")
    parser.add_argument("map_path", type=Path, help="Path to map .pt from maritime_map_converter.py")
    parser.add_argument("agent_path", type=Path, help="Path to agent scenario .pt containing agent/edge fields")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save preprocessed output")
    parser.add_argument("--token-size", type=int, default=2048, help="Token size for TokenProcessor/tokenize_map")
    args = parser.parse_args()

    map_raw = _as_dict(torch.load(args.map_path, map_location="cpu"))
    agent_raw = _as_dict(torch.load(args.agent_path, map_location="cpu"))

    map_with_tokens = _maybe_tokenize_map(map_raw, token_size=args.token_size)
    merged = _merge_map_into_agent(agent_raw, map_with_tokens)

    tp = TokenProcessor(token_size=args.token_size)
    processed = tp.preprocess(merged)
    _print_summary(processed)

    if args.output:
        torch.save(processed, args.output)
        print(f"Saved preprocessed sample to {args.output}")


if __name__ == "__main__":
    main()
