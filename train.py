
from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from smart.utils.config import load_config_act
from smart.datamodules import MultiDataModule
from smart.model import SMART
from smart.utils.log import Logging

# Ampere(A40) 建议：启用 TF32/高精度矩阵，提升 GEMM 吞吐
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = ArgumentParser()
    Predictor_hash = {"smart": SMART, }
    parser.add_argument('--config', type=str, default='configs/train/train_scalable.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="")
    args = parser.parse_args()
    # 合理的默认线程设置，避免过度线程竞争（若用户未显式设置）
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    config = load_config_act(args.config)
    Predictor = Predictor_hash[config.Model.predictor]
    # DDP 策略：可从配置覆盖，默认为更快的设置（假设无未用参数，静态图）
    trainer_config = config.Trainer
    # 为避免 DDP reducer 断言（动态图/未用参数），默认更安全：find_unused_parameters=True, static_graph=False
    ddp_find_unused = getattr(trainer_config, 'find_unused_parameters', True)
    ddp_bucket_view = getattr(trainer_config, 'gradient_as_bucket_view', True)
    ddp_static_graph = getattr(trainer_config, 'static_graph', False)
    strategy = DDPStrategy(
        find_unused_parameters=ddp_find_unused,
        gradient_as_bucket_view=ddp_bucket_view,
        static_graph=ddp_static_graph
    )
    Data_config = config.Dataset
    datamodule = MultiDataModule(**vars(Data_config))

    if args.pretrain_ckpt == "":
        model = Predictor(config.Model)
    else:
        logger = Logging().log(level='DEBUG')
        model = Predictor(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt,
                                    logger=logger)

    # 可选：PyTorch 2.x 编译加速（设置 USE_TORCH_COMPILE=1 开启）
    use_compile_cfg = getattr(config.Trainer, 'use_torch_compile', None)
    compile_mode_cfg = getattr(config.Trainer, 'torch_compile_mode', 'max-autotune')
    if ((os.getenv('USE_TORCH_COMPILE', '0') == '1') or bool(use_compile_cfg)) and hasattr(torch, 'compile'):
        compile_mode = os.getenv('TORCH_COMPILE_MODE', compile_mode_cfg)
        model = torch.compile(model, mode=compile_mode)
    # Checkpoint/验证频率可配置
    ckpt_monitor = getattr(trainer_config, 'ckpt_monitor', 'val_cls_acc')
    ckpt_mode = getattr(trainer_config, 'ckpt_mode', 'max')
    ckpt_every = getattr(trainer_config, 'ckpt_every_n_epochs', 1)
    save_top_k = getattr(trainer_config, 'save_top_k', 5)
    save_last = getattr(trainer_config, 'save_last', False)
    model_checkpoint = ModelCheckpoint(dirpath=args.save_ckpt_path,
                                       filename="{epoch:02d}",
                                       monitor=ckpt_monitor,
                                       every_n_epochs=ckpt_every,
                                       save_top_k=save_top_k,
                                       save_last=save_last,
                                       mode=ckpt_mode)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # 自动选择更优精度：若 GPU 支持 bfloat16，且未固定 precision，则优先 bf16-mixed
    precision_cfg = getattr(trainer_config, 'precision', None)
    if precision_cfg is None and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision_cfg = 'bf16-mixed'
    if precision_cfg is None:
        precision_cfg = '16-mixed'

    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=strategy,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        num_nodes=trainer_config.num_nodes,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=trainer_config.max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=0.5,
        precision=precision_cfg,
        check_val_every_n_epoch=getattr(trainer_config, 'check_val_every_n_epoch', 1),
        log_every_n_steps=getattr(trainer_config, 'log_every_n_steps', 50)
    )
    if args.ckpt_path == "":
        trainer.fit(model,
                    datamodule)
    else:
        trainer.fit(model,
                    datamodule,
                    ckpt_path=args.ckpt_path)
