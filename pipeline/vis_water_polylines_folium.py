#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis_water_polylines_folium.py

功能：
    1. 读取 build_osm_water_polylines.py 生成的 water_polylines_*.json
    2. 将 UTM(例如 EPSG:32651) 坐标转换为经纬度(WGS84, EPSG:4326)
    3. 用 folium 在真实地图上画出 polyline，并保存为 HTML
"""

import argparse
import json
from collections import defaultdict

import folium
from pyproj import Transformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="用 folium 可视化 water_polylines_*.json"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="water_polylines_*.json 的路径",
    )
    parser.add_argument(
        "--html_out",
        type=str,
        default="water_polylines_map.html",
        help="输出 HTML 地图路径",
    )
    parser.add_argument(
        "--src_epsg",
        type=int,
        default=32651,  # 生成 polyline 时用的 UTM EPSG
        help="JSON 中坐标的 EPSG（默认 32651，对应 UTM 51N）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 读取 JSON
    with open(args.json_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    if not isinstance(nodes, list) or len(nodes) == 0:
        print("JSON 内容看起来不对，应该是一个非空的 list。")
        return

    print(f"读取到 {len(nodes)} 个 polyline 节点")

    # 2. 按 polyline_id 分组，把线段拼成折线
    polylines_xy = defaultdict(list)

    for node in nodes:
        pid = node["polyline_id"]
        sx = node["start_x"]
        sy = node["start_y"]
        ex = node["end_x"]
        ey = node["end_y"]

        if len(polylines_xy[pid]) == 0:
            polylines_xy[pid].append((sx, sy))
        polylines_xy[pid].append((ex, ey))

    print(f"一共得到 {len(polylines_xy)} 条 polyline")

    # 3. 坐标转换：UTM(32651) -> WGS84(4326)
    transformer = Transformer.from_crs(
        f"EPSG:{args.src_epsg}", "EPSG:4326", always_xy=True
    )

    polylines_latlon = []
    all_lats = []
    all_lons = []

    for pid, pts in polylines_xy.items():
        ll = []
        for x, y in pts:
            lon, lat = transformer.transform(x, y)  # (x,y) -> (lon,lat)
            ll.append((lat, lon))                   # folium 用 (lat, lon)
            all_lats.append(lat)
            all_lons.append(lon)
        polylines_latlon.append(ll)

    # 4. 选一个中心点（所有点的平均值）
    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
    else:
        # 兜底：用上海附近的中心点
        center_lat = 31.33
        center_lon = 121.55

    print(f"地图中心点: lat={center_lat:.6f}, lon={center_lon:.6f}")

    # 5. 创建 folium 地图并绘制
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    for ll in polylines_latlon:
        if len(ll) < 2:
            continue
        folium.PolyLine(
            locations=ll,
            weight=2,
            color="blue",
            opacity=0.8,
        ).add_to(m)

    # 6. 保存为 HTML
    m.save(args.html_out)
    print(f"已保存 folium 地图到: {args.html_out}")
    print("在浏览器中打开该 HTML 文件即可查看。")


if __name__ == "__main__":
    main()
