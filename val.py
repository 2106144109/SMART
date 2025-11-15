
from argparse import ArgumentParser
import pytorch_lightning as pl
import os
import torch
from torch_geometric.loader import DataLoader
from smart.datasets.scalable_dataset import MultiDataset
from smart.datasets.maritime_dataset import MaritimeDataset
from smart.model import SMART
from smart.transforms import WaymoTargetBuilder, MaritimeTargetBuilder
from smart.utils.config import load_config_act
from smart.utils.log import Logging

if __name__ == '__main__':
    pl.seed_everything(2, workers=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/validation/validation_scalable.yaml")
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="")
    args = parser.parse_args()
    # 合理的默认线程设置（若用户未显式设置）
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    # 启用 TF32 / CUDNN benchmark 提升推理/验证速度
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    config = load_config_act(args.config)

    data_config = config.Dataset
    
    # 根据数据集类型选择不同的Dataset和Transform
    dataset_classes = {
        "scalable": MultiDataset,
        "maritime": MaritimeDataset,
    }
    
    transform_classes = {
        "scalable": WaymoTargetBuilder,
        "maritime": MaritimeTargetBuilder,
    }
    
    dataset_class = dataset_classes[data_config.dataset]
    transform_class = transform_classes[data_config.dataset]
    
    val_dataset = dataset_class(
        root=data_config.root,
        split='val',
        raw_dir=data_config.val_raw_dir,
        processed_dir=data_config.val_processed_dir,
        transform=transform_class(config.Model.num_historical_steps, config.Model.decoder.num_future_steps)
    )
    batch_size = getattr(data_config, 'val_batch_size', getattr(data_config, 'batch_size', 1))
    num_workers = getattr(data_config, 'num_workers', 0)
    pin_memory = getattr(data_config, 'pin_memory', False)
    persistent_workers = getattr(data_config, 'persistent_workers', False)
    if num_workers == 0:
        persistent_workers = False

    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    Predictor = SMART
    if args.pretrain_ckpt == "":
        model = Predictor(config.Model)
    else:
        logger = Logging().log(level='DEBUG')
        model = Predictor(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt,
                                    logger=logger)

    trainer_config = config.Trainer
    # DDP 策略配置可与训练保持一致
    strategy = 'ddp'
    try:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(
            # 更安全的默认，避免 reducer 断言；可在配置中覆盖
            find_unused_parameters=getattr(trainer_config, 'find_unused_parameters', True),
            gradient_as_bucket_view=getattr(trainer_config, 'gradient_as_bucket_view', True),
            static_graph=getattr(trainer_config, 'static_graph', False)
        )
    except Exception:
        pass

    precision_cfg = getattr(trainer_config, 'precision', None)
    if precision_cfg is None and hasattr(pl, 'Trainer'):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision_cfg = 'bf16-mixed'
    if precision_cfg is None:
        precision_cfg = '16-mixed'

    trainer = pl.Trainer(accelerator=trainer_config.accelerator,
                         devices=trainer_config.devices,
                         strategy=strategy,
                         num_sanity_val_steps=0,
                         precision=precision_cfg,
                         log_every_n_steps=getattr(trainer_config, 'log_every_n_steps', 50))
    trainer.validate(model, dataloader)
