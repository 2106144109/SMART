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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _get_map_store(data: Dict[Any, Any], key: Any):
    if isinstance(data, dict):
        return data.get(key)
    try:
        return data[key]
    except Exception:
        return None


def _set_map_store(data: Dict[Any, Any], key: Any, value: Any) -> None:
    if isinstance(data, dict):
        data[key] = value
    else:
        try:
            data[key] = value
        except Exception:
            setattr(data, key, value)


def _maybe_get_plain_value(data: Any, key: Any):
    if isinstance(data, dict):
        return data.get(key)
    try:
        return data[key]
    except Exception:
        return getattr(data, key, None)


def _extract_scene_origin(agent_data: Dict[Any, Any]) -> Optional[torch.Tensor]:
    agent = agent_data.get("agent") if isinstance(agent_data, dict) else _get_map_store(agent_data, "agent")
    if not agent or "scene_origin" not in agent:
        return None

    origin = agent["scene_origin"]
    if origin is None:
        return None
    if not isinstance(origin, torch.Tensor):
        origin = torch.as_tensor(origin, dtype=torch.float32)
    if origin.numel() < 2:
        return None
    return origin.detach().clone().to(torch.float32).flatten()[:2]


def _shift_xy(tensor_like: Any, origin_xy: torch.Tensor) -> Any:
    if tensor_like is None:
        return tensor_like
    if not isinstance(tensor_like, torch.Tensor):
        tensor = torch.as_tensor(tensor_like, dtype=torch.float32)
    else:
        tensor = tensor_like.clone()
    if tensor.size(-1) < 2:
        return tensor_like
    origin = origin_xy.to(dtype=tensor.dtype, device=tensor.device)
    tensor[..., 0] -= origin[0]
    tensor[..., 1] -= origin[1]
    return tensor


def _align_map_coordinates(map_data: Dict[Any, Any], origin_xy: Optional[torch.Tensor]) -> Dict[Any, Any]:
    map_copy = copy.deepcopy(map_data)
    if origin_xy is None:
        return map_copy

    map_point = _get_map_store(map_copy, "map_point")
    if map_point is not None:
        pos = _get_map_store(map_point, "position")
        if pos is not None:
            _set_map_store(map_point, "position", _shift_xy(pos, origin_xy))

    map_save = _get_map_store(map_copy, "map_save")
    if map_save is not None:
        traj_pos = _get_map_store(map_save, "traj_pos")
        if traj_pos is not None:
            _set_map_store(map_save, "traj_pos", _shift_xy(traj_pos, origin_xy))

    return map_copy


def _merge_map_into_agent(agent_data: Dict[Any, Any], map_data: Dict[Any, Any]) -> Dict[Any, Any]:
    merged = copy.deepcopy(agent_data)
    origin_xy = _extract_scene_origin(merged)
    aligned_map = _align_map_coordinates(map_data, origin_xy)

    merged["map_point"] = aligned_map["map_point"]
    merged["map_polygon"] = aligned_map["map_polygon"]
    merged[("map_point", "to", "map_polygon")] = aligned_map[("map_point", "to", "map_polygon")]

    for key in ["pt_token", "map_save"]:
        if key in aligned_map:
            merged[key] = aligned_map[key]
    city_val = _maybe_get_plain_value(aligned_map, "city")
    if city_val is None:
        city_val = _maybe_get_plain_value(map_data, "city")
    merged["city"] = city_val if city_val is not None else "unknown"
    return merged


def _coerce_agent_struct(data: Dict[Any, Any], label: str) -> Dict[Any, Any]:
    """Best-effort compatibility layer for legacy/flattened agent samples.

    - If no 'agent' dict but top-level agent-like keys exist, wrap them under 'agent'.
    - Fill missing keys with reasonable defaults inferred from position/valid_mask.
    - Normalize dtypes/shapes expected by TokenProcessor.
    """
    out = copy.deepcopy(data)

    # Wrap top-level fields into 'agent' if needed (only for plain dicts)
    if "agent" not in out and isinstance(out, dict):
        top_keys = [
            "position",
            "velocity",
            "heading",
            "valid_mask",
            "type",
            "category",
            "num_nodes",
        ]
        if any(k in out for k in top_keys):
            agent = {}
            for k in top_keys:
                if k in out:
                    # safe pop to avoid KeyError on exotic mappings
                    val = out.pop(k, None)
                    if val is not None:
                        agent[k] = val
    if isinstance(out, HeteroData):
        node_types = getattr(out, "node_types", [])
        ordered_nt = (["agent"] if "agent" in node_types else []) + [nt for nt in node_types if nt != "agent"]
        for nt in ordered_nt:
            try:
                store = out[nt]
                has_vm = "valid_mask" in store
                has_x = "x" in store
                has_txy = "target_xy" in store
                has_pos = "position" in store
                has_type = "type" in store
                if not (has_vm and (has_x or has_txy or has_pos)):
                    continue

                # Determine N, T
                N = T = None
                if has_vm and hasattr(store["valid_mask"], "dim") and store["valid_mask"].dim() == 2:
                    N, T = store["valid_mask"].shape
                elif has_x and hasattr(store["x"], "dim") and store["x"].dim() == 3:
                    N, T = store["x"].shape[:2]
                elif has_txy and hasattr(store["target_xy"], "dim") and store["target_xy"].dim() == 3:
                    N, T = store["target_xy"].shape[:2]
                elif has_pos and hasattr(store["position"], "dim") and store["position"].dim() == 3:
                    N, T = store["position"].shape[:2]
                else:
                    if "num_nodes" in store:
                        N = int(store["num_nodes"]) if not isinstance(store["num_nodes"], int) else store["num_nodes"]
                    T = T or 1

                agent_dict: Dict[str, Any] = {}
                # position
                pos = None
                if has_x and store["x"].dim() == 3 and store["x"].shape[-1] >= 2:
                    pos2 = store["x"][..., :2]
                    pos = torch.cat([pos2, pos2.new_zeros(pos2.shape[0], pos2.shape[1], 1)], dim=-1)
                elif has_pos and store["position"].dim() == 3:
                    pos = store["position"]
                elif has_txy and store["target_xy"].dim() == 3:
                    pos2 = store["target_xy"]
                    pos = torch.cat([pos2, pos2.new_zeros(pos2.shape[0], pos2.shape[1], 1)], dim=-1)
                elif N is not None and T is not None:
                    pos = torch.zeros(N, T, 3)
                if pos is not None:
                    agent_dict["position"] = pos

                # valid_mask
                if has_vm:
                    vm = store["valid_mask"]
                    vm = vm.to(torch.bool) if vm.dtype is not torch.bool else vm
                    agent_dict["valid_mask"] = vm
                elif N is not None and T is not None:
                    agent_dict["valid_mask"] = torch.zeros(N, T, dtype=torch.bool)

                # heading
                if "heading" in store and hasattr(store["heading"], "dim"):
                    hd = store["heading"]
                    if hd.dim() == 3 and hd.shape[-1] == 1:
                        hd = hd.squeeze(-1)
                else:
                    if pos is not None:
                        hd = torch.zeros(pos.shape[0], pos.shape[1])
                    elif N is not None and T is not None:
                        hd = torch.zeros(N, T)
                    else:
                        hd = None
                if hd is not None:
                    agent_dict["heading"] = hd

                # velocity
                if "velocity" in store and hasattr(store["velocity"], "dim"):
                    vel = store["velocity"]
                    if vel.dim() == 2 and pos is not None:
                        vel = torch.zeros(pos.shape[0], pos.shape[1], 2)
                else:
                    if pos is not None:
                        vel = torch.zeros(pos.shape[0], pos.shape[1], 2)
                    elif N is not None and T is not None:
                        vel = torch.zeros(N, T, 2)
                    else:
                        vel = None
                if vel is not None:
                    agent_dict["velocity"] = vel

                # type/category
                if has_type:
                    typ = store["type"]
                    typ = typ.to(torch.uint8) if typ.dtype is not torch.uint8 else typ
                    agent_dict["type"] = typ
                elif N is not None:
                    agent_dict["type"] = torch.zeros(N, dtype=torch.uint8)
                baseN = agent_dict.get("type", torch.zeros(N or 0, dtype=torch.uint8)).shape[0]
                cat = store["category"] if "category" in store else torch.zeros(baseN, dtype=torch.uint8)
                if hasattr(cat, "dtype") and cat.dtype is not torch.uint8:
                    cat = cat.to(torch.uint8)
                agent_dict["category"] = cat

                # num_nodes
                if "num_nodes" in store:
                    try:
                        agent_dict["num_nodes"] = int(store["num_nodes"]) if not isinstance(store["num_nodes"], int) else store["num_nodes"]
                    except Exception:
                        agent_dict["num_nodes"] = store["num_nodes"]
                elif pos is not None:
                    agent_dict["num_nodes"] = pos.shape[0]

                # Build a plain dict, preserve any map fields
                def _to_plain(v: Any) -> Any:
                    try:
                        keys = list(v._mapping.keys())
                        return {k: v[k] for k in keys}
                    except Exception:
                        return v
                out_plain: Dict[str, Any] = {}
                for k in ["map_point", "map_polygon", ("map_point", "to", "map_polygon"), "pt_token", "map_save", "city"]:
                    if k in out:
                        out_plain[k] = _to_plain(out[k])
                out_plain["agent"] = agent_dict
                return out_plain
            except Exception:
                continue

    if "agent" not in out:
        return out

    agent = out["agent"]

    pos = agent.get("position", None)
    # Ensure position has 3 channels (x,y,z)
    if pos is not None and pos.dim() == 3 and pos.shape[-1] == 2:
        agent["position"] = torch.cat([pos, pos.new_zeros(pos.shape[0], pos.shape[1], 1)], dim=-1)
        pos = agent["position"]

    # valid_mask
    if "valid_mask" not in agent and pos is not None:
        N, T = pos.shape[0], pos.shape[1]
        agent["valid_mask"] = torch.zeros(N, T, dtype=torch.bool)
    if "valid_mask" in agent and agent["valid_mask"].dtype is not torch.bool:
        agent["valid_mask"] = agent["valid_mask"].to(torch.bool)

    # heading
    if "heading" not in agent and pos is not None:
        N, T = pos.shape[0], pos.shape[1]
        agent["heading"] = torch.zeros(N, T, dtype=torch.float32)
    if "heading" in agent and agent["heading"].dim() == 3 and agent["heading"].shape[-1] == 1:
        agent["heading"] = agent["heading"].squeeze(-1)

    # velocity (TokenProcessor can handle [...,2] and will extend)
    if "velocity" not in agent and pos is not None:
        N, T = pos.shape[0], pos.shape[1]
        agent["velocity"] = torch.zeros(N, T, 2, dtype=torch.float32)

    # type/category
    if "type" not in agent and pos is not None:
        agent["type"] = torch.zeros(pos.shape[0], dtype=torch.uint8)
    elif "type" in agent and agent["type"].dtype is not torch.uint8:
        agent["type"] = agent["type"].to(torch.uint8)

    if "category" not in agent and pos is not None:
        agent["category"] = torch.zeros(pos.shape[0], dtype=torch.uint8)

    # num_nodes
    if "num_nodes" not in agent and pos is not None:
        agent["num_nodes"] = pos.shape[0]

    return out


def _require_agent_keys(data: Dict[Any, Any], label: str) -> None:
    if "agent" not in data:
        raise KeyError(f"{label} is missing required 'agent' data")

    required = ["position", "velocity", "heading", "valid_mask", "type", "category"]
    missing = [k for k in required if k not in data["agent"]]
    if missing:
        raise KeyError(f"{label} agent data missing keys: {missing}")


def _print_summary(processed: Dict[Any, Any]) -> None:
    print(_format_summary(processed))


def _format_summary(processed: Dict[Any, Any]) -> str:
    pt_count = processed["map_point"]["position"].shape[0]
    poly_count = processed["map_polygon"]["type"].shape[0]
    token_rows = processed["pt_token"].get("num_nodes", 0) if "pt_token" in processed else 0
    agent_count = processed["agent"]["position"].shape[0] if "agent" in processed else 0
    return f"map_point: {pt_count}, map_polygon: {poly_count}, pt_token rows: {token_rows}, agents: {agent_count}"


def _process_agent_file(
    agent_file: Path, output_path: Path | None, map_with_tokens: Dict[Any, Any], token_size: int
) -> List[str]:
    messages: List[str] = []
    tp = TokenProcessor(token_size=token_size)

    agent_raw = torch.load(agent_file, map_location="cpu")
    scenarios = _normalize_agent_scenarios(agent_raw)
    has_multiple = len(scenarios) > 1

    output_is_dir = output_path and (output_path.is_dir() or (has_multiple and output_path.suffix == ""))
    if has_multiple and output_path and not output_is_dir:
        raise ValueError(
            f"{agent_file} contains {len(scenarios)} scenarios; "
            "please use --output as a directory to save each sample."
        )

    for idx, scenario in enumerate(scenarios):
        label = f"{agent_file}[{idx}]" if has_multiple else str(agent_file)
        scenario_norm = _coerce_agent_struct(scenario, label=label)
        merged = _merge_map_into_agent(scenario_norm, map_with_tokens)
        _require_agent_keys(merged, label=label)
        processed = tp.preprocess(merged)
        summary_text = _format_summary(processed)
        messages.append(f"{label}: {summary_text}")

        if output_path:
            if output_is_dir:
                output_path.mkdir(parents=True, exist_ok=True)
                suffix = f"_{idx}" if has_multiple else ""
                target_file = output_path / f"{agent_file.stem}{suffix}{agent_file.suffix}"
            else:
                target_file = output_path
            torch.save(processed, target_file)
            messages.append(f"Saved preprocessed sample to {target_file}")

    return messages


# Backward compatibility for older call sites that referenced `_process_agent`.
def _process_agent(
    agent_file: Path, output_path: Path | None, map_with_tokens: Dict[Any, Any], token_size: int
) -> List[str]:
    return _process_agent_file(agent_file, output_path, map_with_tokens, token_size)


def main():
    parser = argparse.ArgumentParser(description="Merge map .pt with agent data and run preprocess")
    parser.add_argument("map_path", type=Path, help="Path to map .pt from maritime_map_converter.py")
    parser.add_argument(
        "agent_path",
        type=Path,
        help=(
            "Path to agent scenario .pt, a directory of .pt files, or a directory containing "
            "train/test/val subdirectories of .pt files"
        ),
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers when processing directories of agent files. "
            "Values >1 will fork subprocesses to speed up preprocessing."
        ),
    )
    args = parser.parse_args()

    map_raw = _as_dict(torch.load(args.map_path, map_location="cpu"))
    map_with_tokens = _maybe_tokenize_map(map_raw, token_size=args.token_size)
    if args.agent_path.is_dir():
        if args.output and args.output.exists() and not args.output.is_dir():
            raise ValueError("--output must be a directory when agent_path is a directory")

        output_dir = args.output
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        split_dirs = [
            p for p in sorted(args.agent_path.iterdir()) if p.is_dir() and p.name in {"train", "test", "val"}
        ]

        if split_dirs:
            for split_dir in split_dirs:
                agent_files = sorted(p for p in split_dir.iterdir() if p.is_file() and p.suffix == ".pt")
                if not agent_files:
                    raise FileNotFoundError(f"No .pt files found in {split_dir}")

                split_output = output_dir / split_dir.name if output_dir else None
                if split_output:
                    split_output.mkdir(parents=True, exist_ok=True)

                tasks = [(agent_file, split_output) for agent_file in agent_files]
                _run_tasks(tasks, args.num_workers, map_with_tokens, args.token_size)
                for agent_file in agent_files:
                    _process_agent(agent_file, split_output)
        else:
            agent_files = sorted(p for p in args.agent_path.iterdir() if p.is_file() and p.suffix == ".pt")
            if not agent_files:
                raise FileNotFoundError(f"No .pt files found in {args.agent_path}")

            tasks = [(agent_file, output_dir) for agent_file in agent_files]
            _run_tasks(tasks, args.num_workers, map_with_tokens, args.token_size)
            for agent_file in agent_files:
                _process_agent(agent_file, output_dir)
    else:
        for line in _process_agent_file(args.agent_path, args.output, map_with_tokens, args.token_size):
            print(line)


def _run_tasks(
    tasks: List[tuple[Path, Path | None]],
    num_workers: int,
    map_with_tokens: Dict[Any, Any],
    token_size: int,
) -> None:
    if num_workers <= 1:
        for task in tasks:
            for line in _process_agent_file(*task, map_with_tokens=map_with_tokens, token_size=token_size):
                print(line)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(
                _process_agent_file,
                task[0],
                task[1],
                map_with_tokens,
                token_size,
            ): task[0]
            for task in tasks
        }
        for future in as_completed(future_map):
            task_file = future_map[future]
            try:
                for line in future.result():
                    print(line)
            except Exception as exc:
                raise RuntimeError(f"Failed processing {task_file}: {exc}") from exc


if __name__ == "__main__":
    main()
