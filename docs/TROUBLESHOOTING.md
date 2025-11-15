# 🔧 问题诊断和修复指南

## 问题描述

```
ValueError: 数据中缺少 'agent' 节点
```

## 🔍 诊断步骤

### 步骤1: 检查数据实际结构

在你的环境中运行：

```bash
conda activate AIS_Data_Processing
cd /home/mahexing/SMART-main
python debug_data_structure.py
```

**预期输出应该包含:**
```
节点类型: ['agent']
```

**如果看到其他节点名（如 'ship', 'vessel'），说明数据格式不匹配。**

---

### 步骤2: 检查数据是如何生成的

问题可能的原因：

#### 原因1: 数据生成脚本版本不匹配

你的数据可能是用早期版本的 `maritime_scene_preprocessor.py` 生成的，那时候可能使用了不同的节点名称。

**解决方案:** 重新生成数据

```bash
# 找到你的数据生成脚本
# 确保使用最新的 maritime_scene_preprocessor.py
```

#### 原因2: 数据是用其他方法生成的

如果你的数据是用不同的脚本生成的，需要：
- 检查生成脚本中使用的节点名称
- 修改 `maritime_target_builder.py` 以适配实际的节点名

---

## 🛠️ 临时修复方案

如果你确认数据结构正确，只是节点名不同，可以修改transform代码：

### 修改 `smart/transforms/maritime_target_builder.py`

找到第46行:
```python
if 'agent' not in data:
    raise ValueError("数据中缺少 'agent' 节点")
```

替换为:
```python
# 支持多种节点名称
node_name = None
for possible_name in ['agent', 'ship', 'vessel', 'ships']:
    if possible_name in data:
        node_name = possible_name
        break

if node_name is None:
    # 打印实际的节点类型帮助调试
    actual_nodes = list(data.node_types) if hasattr(data, 'node_types') else []
    raise ValueError(f"数据中缺少船只节点。找到的节点: {actual_nodes}")

# 如果节点名不是 'agent'，重命名它
if node_name != 'agent':
    # 复制数据到 'agent' 节点
    data['agent'].update(data[node_name])
    # 如果有边，也需要更新
    # (这部分比较复杂，暂时跳过)
```

---

## 💡 完整诊断脚本

运行这个会告诉你确切的问题：

```python
# full_diagnosis.py
import torch
import os

data_dir = "data/maritime_windows_v1/train"
files = os.listdir(data_dir)[:5]  # 检查前5个文件

print("检查前5个数据文件:")
print("=" * 70)

for fname in files:
    fpath = os.path.join(data_dir, fname)
    try:
        data = torch.load(fpath, map_location='cpu', weights_only=False)
        
        if hasattr(data, 'node_types'):
            nodes = list(data.node_types)
            has_agent = 'agent' in nodes
            status = "✅" if has_agent else "❌"
            print(f"{status} {fname[:50]:50s} nodes={nodes}")
        else:
            print(f"⚠️  {fname[:50]:50s} 不是HeteroData")
    except Exception as e:
        print(f"❌ {fname[:50]:50s} 加载失败: {e}")

print("=" * 70)
```

---

## 🎯 推荐解决方案

### 方案A: 如果数据文件都没有'agent'节点

说明数据生成方式和预期不符，需要**重新生成数据**。

检查你之前是如何生成 `maritime_windows_v1/` 数据的：
- 使用的是哪个脚本？
- 是否使用了最新版本的 `maritime_scene_preprocessor.py`？

### 方案B: 如果数据有其他节点名

修改 `maritime_target_builder.py` 和 `maritime_dataset.py` 以支持实际的节点名。

### 方案C: 如果数据格式完全不同

可能需要：
1. 创建一个数据转换脚本
2. 或者修改整个数据加载流程

---

## 📞 下一步

1. **先运行** `python debug_data_structure.py`
2. **把输出发给我**，我会告诉你具体怎么修复
3. 根据诊断结果选择对应的修复方案

---

## ⚡ 快速检查命令

```bash
# 在你的环境中运行
conda activate AIS_Data_Processing
cd /home/mahexing/SMART-main

# 检查单个文件
python debug_data_structure.py

# 或者快速检查
python -c "
import torch
data = torch.load('data/maritime_windows_v1/train/scene_POS_OK_2024-07-01_Waigaoqiao_Port_processed_batches_idx0_pid3620855_part0000.pt', weights_only=False)
print('节点类型:', list(data.node_types) if hasattr(data, 'node_types') else '不是HeteroData')
"
```

把结果发给我！

