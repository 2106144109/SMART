"""
Merge a converted maritime map with agent data and run TokenProcessor.preprocess.

This is a convenience CLI so you can smoke-test map tokens together with one or
more real scenario samples. It will:
1) load a map .pt produced by maritime_map_converter.py (tokenized or not)
2) load one or many agent scenario .pt files
3) ensure map tokens exist (tokenize if missing)
4) merge map + agent data, run preprocess, and optionally save the processed
   output for downstream training/eval.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import sys

import sys

import torch
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart.datasets.preprocess import TokenProcessor


def _as_dict(data: Any) -> Dict[Any, Any]:
    """Return a mutable mapping view for HeteroData or plain dict inputs."""

    if isinstance(data, (list, tuple)):
        if len(data) == 1:
            return _as_dict(data[0])
        raise TypeError(
            "Unsupported data type: non-empty list/tuple. "
            "Please provide a single map sample per .pt file."
            "Please provide a single scenario per .pt file."
        )

    if isinstance(data, HeteroData):
        return data
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unsupported data type: {type(data)}")


def _normalize_agent_scenarios(data: Any) -> List[Dict[Any, Any]]:
    """Coerce single or batched agent inputs into a list of dict-like samples."""

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise TypeError("Empty list/tuple is not a valid agent scenario")
        return [_as_dict(item) for item in data]

    return [_as_dict(data)]


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


def _require_agent_keys(data: Dict[Any, Any], label: str) -> None:
    if "agent" not in data:
        raise KeyError(f"{label} is missing required 'agent' data")

    required = ["position", "velocity", "heading", "valid_mask", "type", "category"]
    missing = [k for k in required if k not in data["agent"]]
    if missing:
        raise KeyError(f"{label} agent data missing keys: {missing}")


def _print_summary(processed: Dict[Any, Any]) -> None:
    pt_count = processed["map_point"]["position"].shape[0]
    poly_count = processed["map_polygon"]["type"].shape[0]
    token_rows = processed["pt_token"].get("num_nodes", 0) if "pt_token" in processed else 0
    agent_count = processed["agent"]["position"].shape[0] if "agent" in processed else 0
    print(f"map_point: {pt_count}, map_polygon: {poly_count}, pt_token rows: {token_rows}, agents: {agent_count}")


def main():
    parser = argparse.ArgumentParser(description="Merge map .pt with agent data and run preprocess")
    parser.add_argument("map_path", type=Path, help="Path to map .pt from maritime_map_converter.py")
    parser.add_argument(
        "agent_path",
        type=Path,
        help="Path to agent scenario .pt containing agent/edge fields or a directory of .pt files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to save preprocessed output. For a single agent file this can be a file path; "
            "for a directory of agent files this must be a directory. If an agent file contains multiple "
            "scenarios, use a directory so each scenario can be written separately."
            "for a directory of agent files this must be a directory."
        ),
    )
    parser.add_argument("--token-size", type=int, default=2048, help="Token size for TokenProcessor/tokenize_map")
    args = parser.parse_args()

    map_raw = _as_dict(torch.load(args.map_path, map_location="cpu"))
    map_with_tokens = _maybe_tokenize_map(map_raw, token_size=args.token_size)
    tp = TokenProcessor(token_size=args.token_size)

    def _process_agent(agent_file: Path, output_path: Path | None) -> None:
        agent_raw = torch.load(agent_file, map_location="cpu")
        scenarios = _normalize_agent_scenarios(agent_raw)
        has_multiple = len(scenarios) > 1

        output_is_dir = output_path and (
            output_path.is_dir() or (has_multiple and output_path.suffix == "")
        )
        if has_multiple and output_path and not output_is_dir:
            raise ValueError(
                f"{agent_file} contains {len(scenarios)} scenarios; "
                "please use --output as a directory to save each sample."
            )

        for idx, scenario in enumerate(scenarios):
            merged = _merge_map_into_agent(scenario, map_with_tokens)
            _require_agent_keys(merged, label=f"{agent_file}[{idx}]" if has_multiple else str(agent_file))
            processed = tp.preprocess(merged)
            label = f"{agent_file}[{idx}]" if has_multiple else f"{agent_file}"
            print(f"{label}: ", end="")
            _print_summary(processed)

            if output_path:
                if output_is_dir:
                    output_path.mkdir(parents=True, exist_ok=True)
                    suffix = f"_{idx}" if has_multiple else ""
                    target_file = output_path / f"{agent_file.stem}{suffix}{agent_file.suffix}"
                else:
                    target_file = output_path
                torch.save(processed, target_file)
                print(f"Saved preprocessed sample to {target_file}")
        agent_raw = _as_dict(torch.load(agent_file, map_location="cpu"))
        merged = _merge_map_into_agent(agent_raw, map_with_tokens)
        processed = tp.preprocess(merged)
        print(f"{agent_file}: ", end="")
        _print_summary(processed)
        if output_path:
            torch.save(processed, output_path)
            print(f"Saved preprocessed sample to {output_path}")

    if args.agent_path.is_dir():
        if args.output and args.output.exists() and not args.output.is_dir():
            raise ValueError("--output must be a directory when agent_path is a directory")

        output_dir = args.output
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        agent_files = sorted(p for p in args.agent_path.iterdir() if p.is_file() and p.suffix == ".pt")
        if not agent_files:
            raise FileNotFoundError(f"No .pt files found in {args.agent_path}")

        for agent_file in agent_files:
            _process_agent(agent_file, output_dir)
            target = output_dir / agent_file.name if output_dir else None
            _process_agent(agent_file, target)
    else:
        _process_agent(args.agent_path, args.output)


if __name__ == "__main__":
    main()
