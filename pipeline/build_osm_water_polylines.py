#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_osm_water_polylines.py

功能：
    1. 从 OSM 的水域面 shapefile 中（例如 gis_osm_water_a_free_1.shp）
       裁剪出指定经纬度范围内的水面 polygon（海+河+湖）。
    2. 将每个水面 polygon 的边界转为等距采样的 polyline。
    3. 生成 AISFuser 风格的 polyline 节点 JSON，可在图构建时作为 "water/landscape" 约束使用。

依赖：
    pip install geopandas shapely pyproj numpy

示例使用：
    python build_osm_water_polylines.py \
      --water_shp /home/mahexing/SMART-main/data/shanghai-251114-free/gis_osm_water_a_free_1.shp \
      --output_json /home/mahexing/SMART-main/data/water_polylines_1215_1216_3128_3138.json \
      --min_lon 121.5 --max_lon 121.6 \
      --min_lat 31.28 --max_lat 31.38 \
      --step_m 200
"""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box, LineString, MultiLineString


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 OSM 水域面 shapefile 中生成 AISFuser 风格的 water polylines"
    )

    parser.add_argument(
        "--water_shp",
        type=str,
        required=True,
        help="OSM 水域面 shapefile 路径，例如 gis_osm_water_a_free_1.shp",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="water_polylines.json",
        help="输出 polyline 节点 JSON 文件路径",
    )

    # 经纬度范围
    parser.add_argument("--min_lon", type=float, default=121.5, help="经度最小值")
    parser.add_argument("--max_lon", type=float, default=121.6, help="经度最大值")
    parser.add_argument("--min_lat", type=float, default=31.28, help="纬度最小值")
    parser.add_argument("--max_lat", type=float, default=31.38, help="纬度最大值")

    # 目标投影坐标系（米制）
    parser.add_argument(
        "--target_epsg",
        type=int,
        default=32651,  # 上海附近 UTM 51N
        help="投影坐标系 EPSG，例如 32651 表示 UTM zone 51N",
    )

    # 等距采样间隔（米）
    parser.add_argument(
        "--step_m",
        type=float,
        default=200.0,
        help="沿边界线的采样间距（米），例如 200 表示每 200 米一个点",
    )

    return parser.parse_args()


def densify_line(line: LineString, step: float) -> LineString:
    """
    对一条 LineString 按固定间距 step（米）做等距重采样。
    line: 已经在米制投影坐标系下的 LineString
    step: 采样间距（米）
    """
    length = line.length
    if length <= step or length == 0:
        return line

    num = int(length // step) + 1
    distances = np.linspace(0, length, num)
    points = [line.interpolate(d) for d in distances]
    return LineString(points)


def main():
    setup_logger()
    args = parse_args()

    water_path = Path(args.water_shp)
    if not water_path.exists():
        logging.error(f"水域 shapefile 不存在: {water_path}")
        return

    logging.info(f"读取 OSM 水域数据: {water_path}")
    water = gpd.read_file(str(water_path))

    # CRS 处理
    if water.crs is None:
        logging.warning("输入数据没有 CRS，假定为 WGS84 (EPSG:4326)，如不正确请手动修改代码！")
        water.set_crs(epsg=4326, inplace=True)
    elif water.crs.to_epsg() != 4326:
        logging.info(f"将数据 CRS 从 {water.crs} 转为 EPSG:4326 (WGS84)")
        water = water.to_crs(epsg=4326)

    # 构造 bbox
    bbox_geom = box(args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")

    logging.info(
        f"使用 bbox 裁剪: lon[{args.min_lon}, {args.max_lon}], "
        f"lat[{args.min_lat}, {args.max_lat}]"
    )
    water_clip = gpd.clip(water, bbox_gdf)

    if water_clip.empty:
        logging.warning("裁剪后没有任何水域 polygon，可能该区域 OSM 没标水面或 bbox 太小。")
        return

    logging.info(f"裁剪后水域 polygon 数量: {len(water_clip)}")

    # 投影到目标米制坐标系
    target_crs = f"EPSG:{args.target_epsg}"
    logging.info(f"将数据投影到 {target_crs} 以便按米重采样")
    water_clip_utm = water_clip.to_crs(target_crs)

    # 生成边界并重采样
    water_clip_utm["boundary"] = water_clip_utm.geometry.boundary

    logging.info(f"开始对边界进行等距重采样，步长 = {args.step_m} 米")
    dense_boundaries = []
    for idx, geom in enumerate(water_clip_utm["boundary"]):
        if geom is None:
            dense_boundaries.append(None)
            continue

        if isinstance(geom, LineString):
            dense_boundaries.append(densify_line(geom, args.step_m))
        elif isinstance(geom, MultiLineString):
            lines = [densify_line(ls, args.step_m) for ls in geom.geoms if isinstance(ls, LineString)]
            if len(lines) == 0:
                dense_boundaries.append(None)
            elif len(lines) == 1:
                dense_boundaries.append(lines[0])
            else:
                dense_boundaries.append(MultiLineString(lines))
        else:
            logging.warning(f"索引 {idx} 的边界几何类型不支持: {type(geom)}")
            dense_boundaries.append(None)

    water_clip_utm["boundary_dense"] = dense_boundaries

    # 构造 AISFuser 风格 polyline 节点列表
    water_nodes = []
    logging.info("开始构建 water polyline 节点列表（AISFuser 风格）")

    for idx, row in water_clip_utm.iterrows():
        geom = row["boundary_dense"]
        if geom is None:
            continue

        centroid = geom.centroid
        shape_id = int(idx)

        def process_line(line: LineString, polyline_suffix: str = ""):
            coords = list(line.coords)
            if len(coords) < 2:
                return

            polyline_id = f"water_{shape_id}{polyline_suffix}"

            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]

                node_feature = {
                    "polyline_id": polyline_id,
                    "shape_type": "water",   # 你也可以改成 "landscape"
                    "shape_id": shape_id,
                    # UTM 坐标（米）
                    "start_x": float(start[0]),
                    "start_y": float(start[1]),
                    "end_x": float(end[0]),
                    "end_y": float(end[1]),
                    "centroid_x": float(centroid.x),
                    "centroid_y": float(centroid.y),
                }
                water_nodes.append(node_feature)

        if isinstance(geom, LineString):
            process_line(geom)
        elif isinstance(geom, MultiLineString):
            for j, ls in enumerate(geom.geoms):
                if isinstance(ls, LineString):
                    process_line(ls, polyline_suffix=f"_{j}")

    logging.info(f"共生成 water polyline 节点数: {len(water_nodes)}")

    # 导出 JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(water_nodes, f, ensure_ascii=False, indent=2)

    logging.info(f"已导出 JSON 至: {output_path.resolve()}")
    logging.info("OSM 水域（water）部分处理完毕。可以与 landscape / ship lane / waypoint 一起构建图。")


if __name__ == "__main__":
    main()
