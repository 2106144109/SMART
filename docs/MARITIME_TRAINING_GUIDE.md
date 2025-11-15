# 🚢 海上场景SMART模型训练指南

## 📋 当前进度总结

### ✅ 已完成的工作

1. **数据预处理** ✅
   - `maritime_scene_preprocessor.py` - 完整的海上场景预处理器
   - `compute_global_norm.py` - 全局归一化统计脚本

2. **数据集准备** ✅
   - 训练集: 209,183 个样本 (15GB)
   - 验证集: 26,023 个样本
   - 测试集: 26,032 个样本
   - 位置: `data/maritime_windows_v1/`

3. **代码适配** ✅
   - `smart/datasets/maritime_dataset.py` - 海上数据集加载器
   - `smart/transforms/maritime_target_builder.py` - 数据transform
   - `configs/train/train_maritime.yaml` - 训练配置
   - 已注册到 DataModule 系统

---

## 🚀 下一步：开始训练

### 步骤 1: 测试数据加载 (必须!)

```bash
cd /home/mahexing/SMART-main
python test_maritime_data.py
```

**预期输出:**
```
✅ 数据集大小: 209183
✅ 样本加载成功!
✅ Batch加载成功!
🎉 所有测试通过! 可以开始训练了!
```

如果测试失败，请查看错误信息并修复。

---

### 步骤 2: 小规模测试训练 (推荐)

先用少量数据测试训练流程是否正常：

```bash
# 创建一个小数据集测试目录
mkdir -p data/maritime_test/train
mkdir -p data/maritime_test/val

# 复制100个训练样本
ls data/maritime_windows_v1/train/*.pt | head -100 | xargs -I {} cp {} data/maritime_test/train/

# 复制10个验证样本
ls data/maritime_windows_v1/val/*.pt | head -10 | xargs -I {} cp {} data/maritime_test/val/
```

修改配置文件进行测试：

```yaml
# configs/train/train_maritime_test.yaml
Dataset:
  train_raw_dir: ["data/maritime_test/train"]
  val_raw_dir: ["data/maritime_test/val"]
  train_batch_size: 2
  num_workers: 2

Trainer:
  max_epochs: 2  # 只训练2轮测试
  devices: 1
```

运行测试训练：

```bash
python train.py --config configs/train/train_maritime_test.yaml \
                --save_ckpt_path logs/test_run
```

---

### 步骤 3: 完整训练

如果测试通过，开始完整训练：

```bash
python train.py --config configs/train/train_maritime.yaml \
                --save_ckpt_path logs/maritime_checkpoints
```

**训练参数说明:**
- `--config`: 配置文件路径
- `--save_ckpt_path`: 检查点保存目录
- `--pretrain_ckpt`: (可选) 预训练模型路径
- `--ckpt_path`: (可选) 继续训练的检查点

---

### 步骤 4: 监控训练

训练过程中可以监控以下指标：

1. **训练损失** (train_loss)
2. **验证准确率** (val_cls_acc) - Token分类准确率
3. **ADE/FDE** - 轨迹预测误差

使用 TensorBoard 可视化（如果启用）：
```bash
tensorboard --logdir logs/
```

---

## ⚙️ 配置调优建议

### 根据GPU内存调整

| GPU内存 | batch_size | accumulate_grad_batches | 有效batch |
|---------|-----------|------------------------|-----------|
| 8GB     | 1         | 8                      | 8         |
| 12GB    | 2         | 4                      | 8         |
| 16GB    | 4         | 4                      | 16        |
| 24GB    | 8         | 2                      | 16        |

修改 `train_maritime.yaml`:
```yaml
Dataset:
  train_batch_size: 4  # 根据GPU调整

Trainer:
  accumulate_grad_batches: 4  # 梯度累积
```

### 加速训练

1. **使用混合精度训练:**
```yaml
Trainer:
  precision: 16  # 从32改为16
```

2. **增加workers:**
```yaml
Dataset:
  num_workers: 8  # 增加数据加载并行度
```

3. **多GPU训练:**
```yaml
Trainer:
  devices: 2  # 使用2个GPU
  strategy: ddp  # 分布式数据并行
```

---

## 🐛 常见问题排查

### 问题 1: CUDA Out of Memory

**解决方案:**
- 减小 `batch_size`
- 减小 `hidden_dim` (128 → 64)
- 减小 `num_agent_layers` (6 → 4)
- 启用混合精度 (`precision: 16`)

### 问题 2: DataLoader 很慢

**解决方案:**
- 增加 `num_workers` (4 → 8)
- 启用 `pin_memory: True`
- 确保数据在SSD上

### 问题 3: 训练不收敛

**可能原因:**
- 学习率太大/太小
- 数据归一化问题
- 模型配置不合适

**排查步骤:**
1. 检查数据: 打印batch查看数值范围
2. 降低学习率: `lr: 0.0001`
3. 增加warmup: `warmup_steps: 2000`
4. 检查损失曲线: 是否nan/爆炸

### 问题 4: 找不到模块

**解决方案:**
```bash
# 确保在正确的环境
conda activate SMART

# 安装缺失的依赖
pip install -r requirements.txt

# 添加到Python路径
export PYTHONPATH=/home/mahexing/SMART-main:$PYTHONPATH
```

---

## 📊 数据统计回顾

你的海上数据集特征范围（局部坐标，相对于T_h-1）:

| 特征 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| x (m) | -441.5 | 574.8 | [-10204, 9201] |
| y (m) | 136.0 | 759.8 | [-11845, 10122] |
| vx (m/s) | -1.91 | 3.85 | [-302, 306] |
| vy (m/s) | 0.62 | 3.85 | [-341, 354] |
| ax (m/s²) | 0.0002 | 0.21 | [-29.1, 24.1] |
| ay (m/s²) | 0.0004 | 0.18 | [-25.6, 26.4] |
| theta (rad) | -0.019 | 0.64 | [-π, π] |
| omega (rad/s) | -0.00003 | 0.035 | [-0.31, 0.31] |

**注意:** 速度有异常值（>300 m/s），可能需要在训练前过滤。

---

## 📈 预期训练时间

基于配置估算（单GPU V100）:

- 样本数: 209,183
- Batch size: 4
- 每epoch步数: ~52,000
- 每步时间: ~1秒
- **每epoch时间: ~14小时**
- **50 epochs总时间: ~30天**

**建议:**
- 使用多GPU训练（2-4卡）可缩短至 7-15天
- 先训练10-20 epochs观察效果
- 考虑减少数据或增加stride

---

## 🎯 验证和测试

训练完成后：

```bash
# 验证模型
python val.py --config configs/train/train_maritime.yaml \
              --pretrain_ckpt logs/maritime_checkpoints/epoch_XX.ckpt

# 生成预测
python inference.py --config configs/inference/inference_maritime.yaml \
                    --ckpt logs/maritime_checkpoints/best.ckpt
```

---

## 📝 下一步改进方向

1. **数据清洗**: 过滤速度异常值
2. **特征工程**: 添加船只类型、尺寸等信息
3. **地图信息**: 如有港口边界、航道信息可添加
4. **Token优化**: 用海上数据重新训练轨迹token
5. **多任务学习**: 同时预测碰撞风险

---

## 📞 需要帮助？

如果遇到问题：
1. 检查日志文件: `logs/`
2. 运行测试脚本: `test_maritime_data.py`
3. 查看配置文件: `train_maritime.yaml`
4. 检查数据: 随机加载几个样本查看

祝训练顺利！🚀

