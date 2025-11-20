SMART 代码库总结文档
项目简介
目标：面向多场景（Waymo 车流、Maritime 海上）运动预测的“Token化轨迹”模型与训练评估工具链。
核心思想：将短时间段轨迹片段离散成“运动词典（token）”，通过“下一token分类”逐步生成未来轨迹，兼顾交互关系（agent-agent）与地图约束（map-agent）。
关键特性：
多数据集统一接口：scalable（Waymo）、maritime（海上 AIS 窗口数据）。
可选地图模块：海上无地图（num_map_layers=0），城市场景启用地图解码器。
Lightning/PL + PyG：易于扩展的训练/验证脚手架与图神经网络算子。
仓库结构总览
入口脚本
train.py
：训练入口（PL Trainer、DDP、混合精度、torch.compile）。
val.py
：验证入口（按配置加载数据集与模型，Trainer.validate）。
evaluate_val_metrics.py
：离线计算 ADE/FDE。
配置
configs/train/train_maritime.yaml
：海上配置（30s间隔，5历史+16未来，输入8维）。
configs/train/train_scalable.yaml
、
configs/validation/validation_scalable.yaml
：Waymo/Scalable 配置示例。
数据与流程
smart/datasets/
：
MultiDataset
（Waymo），
MaritimeDataset
（海上 .pt 列表/单样本读取）。
smart/datamodules/scalable_datamodule.py
：PL DataModule（拼装 Dataset、DataLoader、Transform）。
smart/transforms/
：
WaymoTargetBuilder
、
MaritimeTargetBuilder
（构造 token 必需字段）。
pipeline/
：原始数据预处理与词典生成脚本（Waymo 转换、海上场景预处理、词典生成等）。
模型与组件
smart/model/smart.py
：LightningModule，封装 
SMARTDecoder
 与优化器/日志逻辑。
smart/modules/
：
SMARTDecoder
（协调 map/agent 两路）、
SMARTAgentDecoder
、
SMARTMapDecoder
。
smart/layers/
：注意力层、傅里叶/MLP嵌入、通用 MLP。
度量与可视化
smart/metrics/
：
minADE
、
minFDE
、
TokenCls
 等。
visualize_*：基于 Folium 的场景与预测可视化，支持动画与批量导出 HTML。
词典
smart/tokens/
：
maritime_tokens_no_norm.pkl
、
cluster_frame_5_2048.pkl
、
map_traj_token5.pkl
。*_
数据与预处理（pipeline）
海上预处理：
pipeline/maritime_scene_preprocessor.py
将 AIS 数据重采样到 30s，计算特征：x,y,vx,vy,ax,ay,theta,omega（米/秒/弧度/弧度每秒）。
窗口化（滑动窗口：T_h=5, T_f=16），将全局坐标转换到参考帧（历史末帧）局部坐标，构建 HeteroData。
提供构图边（基于影响矩阵或距离阈值）。
Waymo 预处理：
pipeline/data_preprocess.py
 等
解析 TFRecord，解码 agent 与地图要素，构成可训练数据（兼容 
MultiDataset
 使用）。
数据集与 DataModule
MaritimeDataset
直接读取目录中的 .pt 文件，支持“一个文件含多样本（list）”与单样本两种格式。
内置 LRU 文件缓存，减少 I/O。
返回 HeteroData，并按需应用 
MaritimeTargetBuilder
。
MultiDataset（Waymo）
读取 pickled 数据，TokenProcessor 进行预处理（token 化准备）。
MultiDataModule
根据配置选择数据集和变换。
海上场景 transform：
MaritimeTargetBuilder(num_historical_steps=5, num_future_steps=16, token_size=2048)
。
DataLoader 参数从 yaml 注入（num_workers、pin_memory、prefetch_factor、persistent_workers 自动兜底）。
Transform（关键字段约定）
WaymoTargetBuilder
标注 ego、筛选训练对象、构建类别与掩码。
MaritimeTargetBuilder
校验/裁剪时间步，使 T = T_h + T_f。
添加模型所需字段到 data['agent']：
token_pos：[N,T,8] 完整8维特征（匹配 Model.input_dim=8）。
token_heading：[N,T] 航向角。
agent_valid_mask：[N,T] 有效步掩码（无则置 True）。
type：[N]（海上默认 0=ship/vehicle）。
category：[N]（默认 3=需要预测）。
token_idx：[N,T] 与运动词典匹配得到的 token 索引（窗口匹配，shift=1）。
token_velocity：[N,T,8] 差分/30s 得速度与角速度（theta 用环绕差分）。
shape：[N,T,3] 船舶尺寸占位（L,W,H）。
海上无地图：自动创建空的 pt_token 节点，避免模型访问失败。
若词典大小与配置 token_size 不一致会打印警告，但流程可继续。
模型架构
SMART（LightningModule）
读取 Model 配置：input_dim/hidden_dim/num_heads/head_dim/dropout 等。
通过 
SMARTDecoder
 组合两路解码器：
地图解码器 
SMARTMapDecoder
（当 num_map_layers>0 且非 maritime 时启用）。
Agent 解码器 
SMARTAgentDecoder
（始终启用）。
词典加载：
海上优先 
maritime_tokens_no_norm.pkl
，Waymo 默认 
cluster_frame_5_2048.pkl
。
地图 token（Waymo）默认 
map_traj_token5.pkl
，maritime 场景提供占位结构。
训练目标：
以“下一 token 分类”为主（CrossEntropy + label smoothing）。
记录指标：train_loss、cls_loss、val_cls_acc、val_loss。
可选开启推理阶段 ADE/FDE 度量（需 self.inference_token=True）。
优化器与调度：AdamW + 线性 warmup + 余弦退火（按 warmup_steps/total_steps）。
SMARTDecoder
has_map 控制是否实例化地图解码器（maritime=无地图）。
forward
 将 map_enc 与 agent_enc 合并返回。
SMARTAgentDecoder（核心）
输入：
agent.token_pos（8维）/token_heading/agent_valid_mask/type/token_idx/shape 等。
编码：
类别与形状嵌入、傅里叶位置嵌入（速度/相对几何）。
注意力层堆叠：时间边（t→t+Δt）、a2a 半径图、可选地图到 agent 融合（maritime 不触发）。
shift：海上 shift=1；其他数据集 shift=5。
训练输出：
next_token_prob（logits）、next_token_idx（top-k）、next_token_idx_gt、next_token_eval_mask。
推理输出（关键给评估/可视化）：
pred_traj：[N, T_fut, 2]
gt：[N, T_fut, D]（评估脚本取前2维）
valid_mask：[N, T_fut]
以及 pos_a/head_a/next_token_idx/pred_prob 等调试信息
训练与验证流程
训练（train.py）
读取 yaml（
smart.utils.config.load_config_act
 → EasyDict）。
实例化 
MultiDataModule
 与 
SMART
。
DDP 策略默认更安全（find_unused_parameters=True 等，配置可覆盖）。
精度自动选择：若 GPU 支持 bf16 优先 bf16-mixed，否则 16-mixed。
回调：ModelCheckpoint（监控 val_cls_acc，top-k 保存可配置），LearningRateMonitor。
可选 torch.compile（环境变量或配置开关）。
验证（val.py）
依据 Dataset.dataset 选择 
MultiDataset
 或 
MaritimeDataset
，并匹配 Transform。
Trainer.validate，混合精度同训练策略，DDP 同步。
离线评估（evaluate_val_metrics.py）
调用 
model.inference
，使用 pred_traj 与 gt[..., :2]、valid_mask 计算 ADE/FDE 平均值。
度量（metrics）
TokenCls：下一 token top-1 准确率（训练/验证的主要监控 val_cls_acc）。
minADE/minFDE：提供最优轨迹 ADE/FDE 版本与简化评估实现（训练时默认未开启推理计算，可在 
SMART
 中打开 self.inference_token 使用）。
工具函数 
smart/metrics/utils.py
 支持 top-k 选择、valid 过滤等。
可视化
场景 GT 可视化：
visualize_scenes_folium.py
将历史（实线）与未来 GT（绿色虚线）绘制到 Folium 地图。
采用“窗口参考锚点”（默认），支持坐标交换/翻转与旋回到全局东-北系。
常用运行命令：

```
# 可选：提供归一化统计以还原真实坐标
FOLIUM_NORM_STATS=pipeline/maritime_scene_preprocessor/stats.json \
python visualize_scenes_folium.py \
  --config configs/train/train_maritime.yaml \
  --split val \
  --num_scenes 5 \
  --output_dir folium_maps
```

- 默认读取验证集并按文件分桶抽样 5 个场景，输出到 `folium_maps` 目录（包含索引页）。
- 若数据样本包含 `map_save`/`pt_token`，脚本会将合并后的地图线段叠加在船舶轨迹上，颜色按 `pt_token.pl_type` 区分。
- `FOLIUM_USE_REF_ANCHOR=0` 可关闭参考锚点，保留原始局部坐标轴；`FOLIUM_SAMPLE_MODE=uniform/random` 可更改抽样策略。
预测可视化：
visualize_predictions_folium.py
加载模型与权重，推理得到 pred_traj，与 GT 对比。
输出每场景 HTML，支持动画与 token 选择 JSON 导出。
提供分桶/等距/随机抽样多种采样策略与索引页生成。
词典可视化：
visualize_token_vocabulary.py
将 token 代表轨迹的位移向量从公共原点发射，展示方向与模长分布。
配置要点（train_maritime.yaml）
时间设置（30s）：
num_historical_steps=5（2.5 分钟），decoder.num_future_steps=16（8 分钟），总 21 步。
海上 decoder.time_span=300（影响时间边构建范围）。
Dataset
dataset: "maritime"，raw 目录指向修复后的 data/maritime_windows_v1/{train,val,test}。
DataLoader：num_workers/pin_memory/persistent_workers/prefetch_factor 可按机器调整。
Model
input_dim=8（[x,y,vx,vy,ax,ay,theta,omega]）
decoder.num_map_layers=0、a2a_radius=1000、pl2pl_radius/pl2a_radius=0（无地图）。
token_path: "smart/tokens/maritime_tokens_no_norm.pkl"，token_size=2048。
num_heads=8, head_dim=16, hidden_dim=128, num_freq_bands=64。
Trainer
devices=2、precision=16-mixed、accumulate_grad_batches=4 等。
运行示例
训练（海上）
python train.py --config configs/train/train_maritime.yaml --save_ckpt_path logs/maritime_checkpoints
验证（Waymo/Scalable）
python val.py --config configs/validation/validation_scalable.yaml --ckpt_path path/to/ckpt
离线 ADE/FDE
python evaluate_val_metrics.py --config configs/train/train_maritime.yaml --ckpt logs/maritime_checkpoints/epoch=XX.ckpt --split val --output_json logs/val_metrics.json
可视化预测
python visualize_predictions_folium.py --config configs/train/train_maritime.yaml --pretrain_ckpt logs/maritime_checkpoints/epoch=XX.ckpt --split test --num_scenes 5 --output_dir folium_pred_maps
可视化 GT
python visualize_scenes_folium.py --config configs/train/train_maritime.yaml --split val --num_scenes 5 --output_dir folium_maps
  - `--split` 选择数据集划分：`val` 使用验证集（默认），`test` 使用测试集；对应路径取自配置文件中的 `val_raw_dir/test_raw_dir` 与 `val_processed_dir/test_processed_dir`
关键张量与接口速查
Transform 输出（海上）必备字段
agent.token_pos：[N,T,8]
agent.token_heading：[N,T]
agent.agent_valid_mask：[N,T]
agent.type：[N]（0=ship）
agent.category：[N]（3=预测对象）
agent.token_idx：[N,T]（token 库匹配）
agent.token_velocity：[N,T,8]
agent.shape：[N,T,3]
模型推理输出（用于评估/可视化）
pred_traj：[N,T_fut,2]
gt：[N,T_fut,D]（评估取前2维）
valid_mask：[N,T_fut]
next_token_idx、pred_prob（可选诊断）
扩展与自定义
新数据集：实现 Dataset 与相应 TargetBuilder（按上述字段约定产出 HeteroData 即可）。
词典定制：更新 smart/tokens/*.pkl，结构包含 {'token','traj','token_all','metadata'}。Maritime 匹配过程支持字典键（优先 ship）。
交互半径与层数：decoder.a2a_radius/num_agent_layers 可按密度/场景调优。
训练指标：默认监控 val_cls_acc，也可开启 SMART.inference_token=True 在验证计算 minADE/minFDE。*
依赖与环境（非穷尽）
PyTorch、PyTorch Lightning、torch-geometric、torch-cluster
numpy、pandas、tqdm、easydict、yaml
folium、matplotlib（词典图）
Waymo（可选，使用其数据时）
常见注意事项
词典大小不一致：
MaritimeTargetBuilder
 会提示告警，不影响运行，但建议统一 token_size 与词典文件。
num_workers=0：脚本会自动将 persistent_workers=False，避免 DataLoader 报错。
海上无地图：num_map_layers 设 0；模型内部自动提供地图 token 占位，不参与计算。
评估脚本键名：
model.inference
 返回的 pred_traj/gt/valid_mask 与评估/可视化脚本一致。
TF32/bf16/torch.compile：训练/验证默认开启 TF32；如 GPU 支持，优先使用 bf16-mixed；可通过环境变量启用 torch.compile。
产出物
训练日志与权重：logs/*/epoch=XX.ckpt
可视化 HTML：folium_maps/（GT）、folium_pred_maps/（预测）
指标 JSON：evaluate_val_metrics.py --output_json 指定路径*
推荐行动
[跑通海上任务] 按 
train_maritime.yaml
 启动训练与验证，产出 ckpt、可视化 HTML。
[评估与对比] 使用 
evaluate_val_metrics.py
 统计 ADE/FDE，与可视化定性检查对照。
[调参与扩展] 依据数据密度与 FPS，调整 a2a_radius/num_agent_layers/time_span，或更换 token 词典。