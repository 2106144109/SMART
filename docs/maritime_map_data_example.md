# 海事地图数据示例（SMART 所需字段）

下表和示例展示了 **在调用 `TokenProcessor.preprocess` 之前**，为海事场景补充地图信息时需要构造的 `HeteroData` 关键字段。若配置中开启了地图分支（`decoder.num_map_layers>0` 且 `dataset!='maritime'`），这些字段会被 `tokenize_map` 读取并拆分为 polyline token，用于 `SMARTMapDecoder` 的地图建模。

## 必备节点与属性

| 节点/边 | 关键键 | 说明 |
| --- | --- | --- |
| `map_point` 节点 | `position` (形状 `[N_pt, 3]`，xy 必填，z 可为 0)<br>`orientation` (朝向弧度，形状 `[N_pt]`，会自动 wrap 到 `[-pi, pi]`)<br>`type` (整型 ID，使用 0~16 的嵌入表) | 每个点属于某条多边形/多段线。`type` 用于区分车道中心/车道边界/人行横道/停止线等类别。`tokenize_map` 会按 `type` 对点进行分组、插值并生成 polyline 片段。|
| `map_polygon` 节点 | `type` (整型 ID，嵌入大小为 4)<br>`light_type` (整型 ID，嵌入大小为 4) | `map_point` 通过边连接到对应的 polygon。`light_type` 在 `SMARTMapDecoder` 中参与嵌入，与 token → polygon 的映射配合使用。|
| 边 `('map_point','to','map_polygon')` | `edge_index` (形状 `[2, E]` 的长整型张量，第一行是 point 索引，第二行是 polygon 索引) | 指明每个点属于哪条 polygon/线。`tokenize_map` 会遍历每个 polygon，将其点序列插值为定长 polyline token 并保存 `pl_idx_list` 等信息。|

> 这些字段的使用逻辑可见 `tokenize_map`：将 `map_point` 的位置、朝向、类型与其所属 `map_polygon` 绑定，分段插值生成 `pt_token`/`map_save`，供地图解码器使用。【F:smart/datasets/preprocess.py†L403-L468】<br>地图解码器随后读取 `pt_token`、`map_polygon.light_type`、`('pt_token','to','map_polygon')` 等字段进行嵌入和图构建。【F:smart/modules/map_decoder.py†L22-L139】

## 示例：用你提供的海事几何生成输入

下面的伪代码展示如何把“航道中心线、岸线、跨越区/警戒区、靠泊线”映射到模型所需字段，并构造一个最小可用的 `HeteroData`（以 Python 字典形式示意）。

```python
import torch
from torch_geometric.data import HeteroData

data = HeteroData()

# 1) 将所有线段/多边形离散为点序列（顺序很重要，需沿线排序）。
channel_centerline_pts = torch.tensor([  # 对应 lane_centerlines
    [0.0, 0.0, 0.0],
    [40.0, 5.0, 0.0],
    [80.0, 12.0, 0.0],
])
bank_boundary_pts = torch.tensor([      # 对应 road_boundaries / 岸线
    [[-5.0, -10.0, 0.0], [30.0, -12.0, 0.0], [70.0, -8.0, 0.0]],
    [[5.0, 15.0, 0.0], [30.0, 18.0, 0.0], [75.0, 16.0, 0.0]],
])
berth_line_pts = torch.tensor([         # 对应 stop_lines / 靠泊线
    [[50.0, 2.0, 0.0], [50.0, 10.0, 0.0]],
])

# 2) 展平点并为每条线赋一个 polygon 索引。
map_point_positions = torch.cat([
    channel_centerline_pts,
    bank_boundary_pts.view(-1, 3),
    berth_line_pts.view(-1, 3),
], dim=0)

# 3) 设置点的朝向（与折线方向一致），简单示例直接填 0。
map_point_orient = torch.zeros(map_point_positions.size(0))

# 4) 为不同要素指定类型 ID（可按需要自定义约定）。
#    例：0=航道中心线, 1=岸线, 2=警戒/跨越区边界, 3=靠泊线。
map_point_type = torch.cat([
    torch.zeros(channel_centerline_pts.size(0), dtype=torch.long),
    torch.ones(bank_boundary_pts.numel() // 3, dtype=torch.long),
    torch.full((berth_line_pts.numel() // 3,), 3, dtype=torch.long),
])

# 5) 为每条线创建 polygon 节点，并建立 point→polygon 边。
#    下例共有 1 条中心线 + 2 条岸线 + 1 条靠泊线 = 4 个 polygon。
num_polygons = 4
map_polygon_type = torch.tensor([0, 1, 1, 3], dtype=torch.long)  # 与业务含义对应
map_polygon_light = torch.zeros(num_polygons, dtype=torch.long)   # 海事可默认 0=无灯控
# edge_index: 第一行 point 索引，第二行对应的 polygon 索引
edge_rows = []
edge_cols = []
start = 0
for poly_idx, pts in enumerate([
    channel_centerline_pts,
    bank_boundary_pts[0],
    bank_boundary_pts[1],
    berth_line_pts[0],
]):
    end = start + pts.size(0)
    edge_rows.append(torch.arange(start, end, dtype=torch.long))
    edge_cols.append(torch.full((pts.size(0),), poly_idx, dtype=torch.long))
    start = end
edge_index = torch.stack([torch.cat(edge_rows), torch.cat(edge_cols)])

# 6) 写入 HeteroData（即 preprocess 入口要求的字段）。
data['map_point'].position = map_point_positions
data['map_point'].orientation = map_point_orient
data['map_point'].type = map_point_type
data['map_polygon'].type = map_polygon_type
data['map_polygon'].light_type = map_polygon_light
data['map_point', 'to', 'map_polygon'].edge_index = edge_index

# 7) 可选：如果你有航道/禁航区多边形，照样展开成点并追加到 map_point，
#    再增加对应的 polygon 节点和 edge_index。
```

运行 `TokenProcessor.preprocess(data)` 时，它会：
1. 基于 `map_point` → `map_polygon` 的拓扑，将每条线插值为长度统一的 polyline token，写入 `pt_token` 与 `map_save`；
2. 使用 `map_polygon.light_type` 补充 token 类别嵌入；
3. 生成 `('pt_token','to','map_polygon')` 的边，供地图解码器进行半径图构建与下一步 token 预测。【F:smart/datasets/preprocess.py†L403-L468】【F:smart/modules/map_decoder.py†L96-L139】

如需贴合你最初的 JSON 结构（`lane_centerlines`/`road_boundaries`/`crosswalks`/`stop_lines`），只需将每条几何拆成有序点列，分别赋不同的 `type` ID，并确保 `edge_index` 把点连到正确的 polygon 索引即可。

## 将“起点-终点”分段数据转为 HeteroData（基于你提供的示例）

你的数据是按段记录的，每条折线由多个 `{start_x, start_y, end_x, end_y}` 片段组成。下面的步骤与示例代码展示如何：
1. 按 `polyline_id` / `shape_id` 聚合同一条折线的片段。
2. 依据几何连续性排序片段（让上一个片段的 `end` 尽量衔接下一个片段的 `start`）。
3. 把排序后的片段拼成有序点列：`[start0, end0, end1, ...]`。
4. 为每条折线指定一个 `map_polygon` 节点，分配 `type`（如 `shape_type=='water'` → `type=1` 代表航道边界/水域边界）。
5. 生成 point → polygon 的 `edge_index`，并填充 `orientation`（相邻点方向角）。

```python
import math
import torch
from torch_geometric.data import HeteroData

# ====== 示例输入：多段水域边界 ======
raw_segments = [
    {
        "polyline_id": "water_1231",
        "shape_type": "water",
        "shape_id": 1231,
        "start_x": 357912.43937938113,
        "start_y": 3462367.9400085662,
        "end_x": 358036.3697113833,
        "end_y": 3462233.259432094,
    },
    {
        "polyline_id": "water_1231",
        "shape_type": "water",
        "shape_id": 1231,
        "start_x": 358036.3697113833,
        "start_y": 3462233.259432094,
        "end_x": 358150.93269944505,
        "end_y": 3462081.199864556,
    },
    {
        "polyline_id": "water_1231",
        "shape_type": "water",
        "shape_id": 1231,
        "start_x": 358150.93269944505,
        "start_y": 3462081.199864556,
        "end_x": 358250.72811640496,
        "end_y": 3461908.423217092,
    },
    {
        "polyline_id": "water_1231",
        "shape_type": "water",
        "shape_id": 1231,
        "start_x": 358250.72811640496,
        "start_y": 3461908.423217092,
        "end_x": 358377.7982863577,
        "end_y": 3461748.56933738,
    },
]

# ====== 1) 按 polyline_id 聚合并排序片段 ======
segments_by_poly = {}
for seg in raw_segments:
    segments_by_poly.setdefault(seg["polyline_id"], []).append(seg)

def sort_segments(segments):
    """简单按几何连续性排序：从第一个片段出发，逐步寻找 end≈start 的下一个。"""
    ordered = [segments[0]]
    remaining = segments[1:]
    while remaining:
        last_end = (ordered[-1]["end_x"], ordered[-1]["end_y"])
        # 用距离最小的 start 作为下一个片段
        next_idx = min(
            range(len(remaining)),
            key=lambda i: (remaining[i]["start_x"] - last_end[0]) ** 2 + (remaining[i]["start_y"] - last_end[1]) ** 2,
        )
        ordered.append(remaining.pop(next_idx))
    return ordered

ordered_polylines = {k: sort_segments(v) for k, v in segments_by_poly.items()}

# ====== 2) 拼接为点列 (start0, end0, end1, ...) ======
poly_points = {}
for pid, segs in ordered_polylines.items():
    pts = [(segs[0]["start_x"], segs[0]["start_y"])]
    for seg in segs:
        pts.append((seg["end_x"], seg["end_y"]))
    poly_points[pid] = torch.tensor(pts, dtype=torch.float)

# ====== 3) 计算朝向（相邻点的 atan2），最后一个点可重复前一角度 ======
def compute_heading(points):
    headings = []
    for i in range(len(points)):
        if i + 1 < len(points):
            dx = points[i + 1, 0] - points[i, 0]
            dy = points[i + 1, 1] - points[i, 1]
            headings.append(math.atan2(dy, dx))
        else:
            headings.append(headings[-1])
    return torch.tensor(headings, dtype=torch.float)

poly_headings = {pid: compute_heading(pts) for pid, pts in poly_points.items()}

# ====== 4) 映射 shape_type -> type ID（可自定义约定） ======
TYPE_MAP = {"channel": 0, "water": 1, "restricted": 2, "berth": 3}

# ====== 5) 组装 HeteroData 字段 ======
data = HeteroData()
positions = []
orientations = []
types = []
edge_rows = []
edge_cols = []
poly_types = []
poly_light = []
point_offset = 0

for poly_idx, (pid, pts) in enumerate(poly_points.items()):
    positions.append(torch.cat([pts, torch.zeros(len(pts), 1)], dim=1))  # 填 z=0
    orientations.append(poly_headings[pid])
    types.append(torch.full((len(pts),), TYPE_MAP.get("water", 1), dtype=torch.long))

    edge_rows.append(torch.arange(point_offset, point_offset + len(pts)))
    edge_cols.append(torch.full((len(pts),), poly_idx, dtype=torch.long))
    point_offset += len(pts)

    poly_types.append(TYPE_MAP.get("water", 1))
    poly_light.append(0)

data['map_point'].position = torch.cat(positions)
data['map_point'].orientation = torch.cat(orientations)
data['map_point'].type = torch.cat(types)
data['map_polygon'].type = torch.tensor(poly_types, dtype=torch.long)
data['map_polygon'].light_type = torch.tensor(poly_light, dtype=torch.long)
data['map_point', 'to', 'map_polygon'].edge_index = torch.stack([
    torch.cat(edge_rows), torch.cat(edge_cols)
])
```

得到的 `data` 已包含 `tokenize_map` 需要的字段：
* 每个点的坐标/朝向/类型：`data['map_point'].position`、`orientation`、`type`；
* 每条折线对应的 polygon 类型（`map_polygon.type`）与点到 polygon 的连接关系（`edge_index`）。
将其传入 `TokenProcessor.preprocess` 后，模型即可把你的水域折线转换为 polyline token 并参与地图建模。

### 命令行快速使用

仓库新增了 `scripts/maritime_map_converter.py`，可直接把上述 JSON 段列表转成 `HeteroData` 并打印关键字段：

```bash
python scripts/maritime_map_converter.py your_segments.json \
  --type-map channel=0 water=1 restricted=2 berth=3 \
  --default-type 0
```

`--type-map` 可根据你的 `shape_type` 自定义类别 ID；未知类型会使用 `--default-type`。

若需直接生成 `pt_token`/`map_save` 供地图分支使用，可加上 `--tokenize-map`（默认使用仓库提供的 2048 agent token），并用 `--save` 将结果保存：

```bash
python scripts/maritime_map_converter.py your_segments.json \
  --type-map channel=0 water=1 restricted=2 berth=3 \
  --tokenize-map --save converted_map.pt
```

* `converted_map.pt` 会包含 `map_point`/`map_polygon` 以及 `pt_token` 等字段，可直接被下游 dataloader 读取。
* 如需更换 token 大小（取决于 `smart/tokens/cluster_frame_5_*.pkl` 是否存在），使用 `--token-size` 覆盖默认值。

### 验证生成的 `.pt` 文件

使用 `scripts/validate_map_pt.py` 可快速检查关键字段是否齐全、尺寸是否对齐、是否含 NaN/Inf：

```bash
python scripts/validate_map_pt.py converted_map.pt
```

常见检查包括：

* `map_point.position`/`orientation`/`type` 数量一致且无 NaN。
* `map_polygon.type` 与 `light_type` 长度一致。
* `('map_point','to','map_polygon')` 的 `edge_index` 未越界。
* 若文件已包含 `pt_token`/`map_save`，它们的行数与 `num_nodes` 对齐。

### 生成 token 后的下一步

1. **在预处理/评估阶段加载 `converted_map.pt`**：
   ```python
   import torch
   from smart.datasets.preprocess import TokenProcessor

   # 加载 CLI 保存的 tokenized map（含 map_point / map_polygon / pt_token / map_save）
   map_data = torch.load("converted_map.pt")

   # 若你只保存了未 token 化的 HeteroData，可用 TokenProcessor 重新构建 pt_token
   if "pt_token" not in map_data:
       tp = TokenProcessor(token_size=2048)
       map_data = tp.tokenize_map({
           "map_point": map_data["map_point"],
           "map_polygon": map_data["map_polygon"],
           ("map_point", "to", "map_polygon"): map_data[("map_point", "to", "map_polygon")],
       })
   ```

2. **与 agent 数据合并后调用 `TokenProcessor.preprocess`**：
   将 `map_data` 放入场景 `HeteroData` 中（键名保持 `map_point`、`map_polygon`、`('map_point','to','map_polygon')` 以及 `pt_token`/`map_save`），再补齐
   `agent` 节点和对应的边。随后调用：
   ```python
   scene_data = {**map_data, **agent_data, **agent_edges}
   tp = TokenProcessor(token_size=2048)
   processed = tp.preprocess(scene_data)
   ```
   `processed` 即可直接送入数据加载器/推理流程。

3. **确认配置启用地图分支**：在训练或推理的配置中，将 `decoder.num_map_layers` 设为大于 0 的值，并确保数据集名称不是 `maritime`（否则地图分支会
   被跳过）。必要时调整 `pl2pl_radius`、`pl2a_radius` 以控制地图-代理交互范围。

通过以上步骤，你即可把起终点段列表转换为模型可用的 polyline token，并在下游流程中参与预测。
