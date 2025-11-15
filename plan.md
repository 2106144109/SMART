<!-- 12a35f75-639c-475b-a04c-2cc8e448e2d0 b2562507-18e6-42aa-8705-de4dae87d605 -->
# 轨迹预测劣化根因排查计划

## 阶段A｜数据链路体检（不依赖模型）

- 目的：确认数据坐标系、锚点、朝向与时间步是否健康，排除“数据/可视化映射”导致的假问题。
- 操作：

1) 打开参考锚点+禁用轴推断，仅采样打印指标，不生成HTML（200均匀样本）：

  ```bash
  cd /home/mahexing/SMART-main
  FOLIUM_USE_REF_ANCHOR=1 FOLIUM_DISABLE_AUTO_AXIS=1 FOLIUM_SAMPLE_MODE=uniform \
  python visualize_folium.py --num_scenes 200 --output_dir folium_maps --no_save_map
  ```

2) 期望：日志出现“使用窗口参考锚点”；不同场景来自不同分片（分桶抽样或均匀抽样）。若失败，多为数据缺失 `scene_info` 或轴推断干扰。

- 参考代码：
```82:92:/home/mahexing/SMART-main/smart/datamodules/scalable_datamodule.py
def train_dataloader(self):
    loader_kwargs = dict(
        batch_size=self.train_batch_size,
        shuffle=self.shuffle,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        persistent_workers=self.persistent_workers,
    )
    if self.num_workers > 0:
        loader_kwargs['prefetch_factor'] = self.prefetch_factor
    return DataLoader(self.train_dataset, **loader_kwargs)
```


## 阶段B｜模型前向一致性（Lightning 评估口径）

- 目的：确认分类头与推理口径（ADE/FDE）的一致性，排除“训练OK、推理异常”的情况。
- 操作：

1) 开启推理度量，直接跑 Lightning 验证：

  ```bash
  cd /home/mahexing/SMART-main
  CKPT="$(ls -t logs/maritime_ft0deg_ckpts/*.ckpt 2>/dev/null | head -n1)"; \
  [ -z "$CKPT" ] && CKPT="$(ls -t logs/maritime_checkpoints/*.ckpt | head -n1)"; \
  python eval_test.py --config configs/train/train_maritime.yaml --pretrain_ckpt "$CKPT" --split test
  ```

2) 记录：`val_cls_acc / val_loss / val_minADE / val_minFDE`。若分类高而 ADE/FDE 弱，说明“几何/解码阶段”更可疑。

- 参考代码：
```199:226:/home/mahexing/SMART-main/smart/model/smart.py
self.log('val_cls_acc', self.TokenCls, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
...
if self.inference_token:
    pred = self.inference(data)
    ...
    self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
    self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
```


## 阶段C｜解码器几何/采样消融（快速定位主因）

- 目的：用“不改代码”的环境变量开关，对旋转角来源/符号/偏置、贪心/采样、方向先验进行网格化或小消融，判断是否对指标有决定性影响。
- 操作1：200样本网格（不保存地图），验证“差分角 × 符号 × 偏置”的影响：
```bash
cd /home/mahexing/SMART-main
CKPT="$(ls -t logs/maritime_ft0deg_ckpts/*.ckpt 2>/dev/null | head -n1)"; [ -z "$CKPT" ] && CKPT="$(ls -t logs/maritime_checkpoints/*.ckpt | head -n1)";
for sign in pos neg; do
  for deg in 0 90 -90 180; do
    printf "sign=%s deg=%s  " "$sign" "$deg";
    FOLIUM_USE_REF_ANCHOR=1 FOLIUM_DISABLE_AUTO_AXIS=1 FOLIUM_SAMPLE_MODE=uniform \
    DECODER_USE_DIFF_ANGLE=1 DECODER_ROT_SIGN=$sign DECODER_ROT_OFFSET_DEG=$deg \
    python visualize_predictions_folium.py --config configs/train/train_maritime.yaml \
      --pretrain_ckpt "$CKPT" --split test --num_scenes 200 --no_save_map \
    | awk -F '[:= m,]+' '/Scene .*ADE=/{ade+=$5; fde+=$7; n++} /DirCos/{d+=$NF} END{printf "ADE=%.1f FDE=%.1f DirCos=%.3f\n", ade/n, fde/n, d/n}'
  done
done
```

- 判定：若某一组合（常见为“差分角+R(θ)正号+0°”）显著提升 DirCos/ADE/FDE → 问题主要在“解码器几何（旋转号/偏置/角源）”。
- 操作2：方向先验与贪心（抑制模式坍缩）：
```bash
cd /home/mahexing/SMART-main
FOLIUM_USE_REF_ANCHOR=1 FOLIUM_DISABLE_AUTO_AXIS=1 FOLIUM_SAMPLE_MODE=uniform \
DECODER_USE_DIFF_ANGLE=1 DECODER_ROT_SIGN=pos DECODER_ROT_OFFSET_DEG=0 \
DECODER_DIR_PRIOR=1 DECODER_DIR_WEIGHT=0.3 DECODER_GREEDY=1 \
python visualize_predictions_folium.py --config configs/train/train_maritime.yaml \
  --pretrain_ckpt "$CKPT" --split test --num_scenes 200 --no_save_map
```

- 判定：若开启先验/贪心后“PathLen(Pred) 接近 GT、DirCos 上升、token 不再单一”，则原问题含“采样或排序坍缩”。
- 操作3：单场景深入日志（验证是否“重复同一 token”）：
```bash
cd /home/mahexing/SMART-main
DECODER_DEBUG=1 FOLIUM_USE_REF_ANCHOR=1 FOLIUM_DISABLE_AUTO_AXIS=1 \
DECODER_USE_DIFF_ANGLE=1 DECODER_ROT_SIGN=pos DECODER_ROT_OFFSET_DEG=0 \
python visualize_predictions_folium.py --config configs/train/train_maritime.yaml \
  --pretrain_ckpt "$CKPT" --split test --num_scenes 1 --no_save_map
```

- 判定：若 `picked=` 在 t=0..15 基本恒定，且 `cos(prev,cur)≈1`、`step_len` 近似常数 → 模式坍缩；反之更多是几何错位。
- 参考代码：
```540:555:/home/mahexing/SMART-main/smart/modules/agent_decoder.py
_dbg = os.getenv('DECODER_DEBUG', '0') == '1'
if _dbg and pos_a.shape[0] > 0:
    ...
    print(f"[DECODER_DEBUG] t={t:02d} picked={picked} top2={top2} cos(prev,cur)={cos_prev:.3f} step_len={step_len:.2f}")
```


## 阶段D｜归因规则（观察→根因）

- A) 分类高(≈0.5)、ADE/FDE弱，且“旋转网格”对指标有强影响 → 解码器几何为主因。
- B) 分类一般、ADE/FDE随训练同步改善 → 模型能力/训练不足为主。
- C) 关闭/开启参考锚点导致可视化位置明显错位 → 数据坐标/映射问题。
- D) `picked` 恒定、PathLen(Pred)≈固定且显著小于 GT → 采样/排序坍缩或步长尺度不匹配。

## 阶段E｜可选加严（需要轻量改动，若你同意再执行）

- E1) “CV/CA 基线”加入 `visualize_predictions_folium.py`：基于最后两步速度/加速度外推，报告 baseline ADE/FDE 作为难度下界。
- E2) “Teacher-Forcing 探针”：在 `SMART.inference` 增加 `force_tokens` 路径，用 GT token 滚动，若 ADE/FDE 显著改善 → 分类头为主因；若仍差 → 几何/写回为主因。
- E3) “步长先验+反重复小惩罚”：对连续命中相同 token 施小幅降权，并对与参考步长差距大的候选扣分，进一步破坍缩。

---

如你确认以上方案，我将按阶段A→C顺序执行并汇总结论；若需要E阶段探针，我会再征求你同意后做最小改动。

### To-dos

- [ ] 运行200样本数据体检（参考锚点+禁用轴，no_save_map）
- [ ] 跑Lightning评估并记录val_cls_acc/val_minADE/val_minFDE
- [ ] 做差分角×符号×偏置网格（200样本），汇总DirCos/ADE/FDE
- [ ] 开启方向先验与贪心（200样本），观察是否破坍缩
- [ ] 单场景开启DECODER_DEBUG检查token重复与步长稳定性
- [ ] 依据观测套用归因规则，明确主因与次因
- [ ] 如同意，添加Teacher-Forcing探针（轻量改动）


