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
