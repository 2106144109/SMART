import argparse
import json
import os
from typing import List

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from smart.datasets.maritime_dataset import MaritimeDataset
from smart.datasets.scalable_dataset import MultiDataset
from smart.model import SMART
from smart.transforms import MaritimeTargetBuilder, WaymoTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging


def compute_ade_fde(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    valid_mask: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    """
    pred_traj: [N, T, 2]
    gt_traj:   [N, T, 2]
    valid_mask: [N, T] bool
    Returns per-agent ADE and FDE (NaN for agents without any valid future step).
    """
    mask = valid_mask.bool()
    diff = pred_traj - gt_traj
    dist = torch.linalg.norm(diff, dim=-1)  # [N, T]

    lengths = mask.sum(dim=-1)  # [N]
    ade = torch.full((pred_traj.shape[0],), float("nan"), dtype=torch.float64)
    fde = torch.full((pred_traj.shape[0],), float("nan"), dtype=torch.float64)

    non_empty = lengths > 0
    if non_empty.any():
        masked_dist = dist[non_empty] * mask[non_empty]
        ade_vals = masked_dist.sum(dim=-1) / lengths[non_empty]
        ade[non_empty] = ade_vals.to(torch.float64)

        last_indices = (lengths[non_empty] - 1).unsqueeze(-1)
        last_indices = last_indices.clamp_min(0)
        fde_vals = dist[non_empty].gather(1, last_indices).squeeze(-1)
        fde[non_empty] = fde_vals.to(torch.float64)

    return ade, fde


def main():
    parser = argparse.ArgumentParser(description="Compute average ADE/FDE on validation or test set.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output_json", type=str, default="", help="Optional path to dump statistics as JSON.")
    args = parser.parse_args()

    config = load_config_act(args.config)
    data_cfg = config.Dataset

    dataset_classes = {
        "scalable": MultiDataset,
        "maritime": MaritimeDataset,
    }
    transform_classes = {
        "scalable": WaymoTargetBuilder,
        "maritime": MaritimeTargetBuilder,
    }

    dataset_class = dataset_classes[data_cfg.dataset]
    transform_class = transform_classes[data_cfg.dataset]

    if args.split == "test":
        raw_dir = data_cfg.test_raw_dir
        processed_dir = data_cfg.test_processed_dir
    else:
        raw_dir = data_cfg.val_raw_dir
        processed_dir = data_cfg.val_processed_dir

    dataset = dataset_class(
        root=data_cfg.root,
        split=args.split,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        transform=transform_class(config.Model.num_historical_steps, config.Model.decoder.num_future_steps),
    )

    batch_size = getattr(data_cfg, f"{args.split}_batch_size", getattr(data_cfg, "batch_size", 1))
    num_workers = getattr(data_cfg, "num_workers", 0)
    pin_memory = getattr(data_cfg, "pin_memory", False)
    persistent_workers = getattr(data_cfg, "persistent_workers", False)
    if num_workers == 0:
        persistent_workers = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    logger = Logging().log(level="INFO")
    logger.info(f"Loaded {args.split} dataset with {len(dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SMART(config.Model)
    model.load_params_from_file(filename=args.ckpt, logger=logger)
    model = model.to(device)
    model.eval()

    all_ade: List[float] = []
    all_fde: List[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {args.split}"):
            batch = batch.to(device)
            pred = model.inference(batch)
            pred_traj = pred["pred_traj"].detach().cpu()
            gt_traj = pred["gt"].detach().cpu()
            valid_mask = pred["valid_mask"].detach().cpu()
            
            # Extract only position coordinates (first 2 dims) from GT
            gt_pos = gt_traj[..., :2]
            
            ade, fde = compute_ade_fde(pred_traj, gt_pos, valid_mask)
            ade = ade[~torch.isnan(ade)]
            fde = fde[~torch.isnan(fde)]

            all_ade.extend(ade.tolist())
            all_fde.extend(fde.tolist())

    if len(all_ade) == 0 or len(all_fde) == 0:
        print("No valid agents found to compute ADE/FDE.")
        return

    mean_ade = float(torch.tensor(all_ade).mean())
    mean_fde = float(torch.tensor(all_fde).mean())

    print(f"Average ADE ({args.split}): {mean_ade:.4f} m")
    print(f"Average FDE ({args.split}): {mean_fde:.4f} m")

    if args.output_json:
        payload = {
            "split": args.split,
            "num_agents": len(all_ade),
            "mean_ade": mean_ade,
            "mean_fde": mean_fde,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()

