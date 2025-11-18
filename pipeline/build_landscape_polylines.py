#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_landscape_polylines.py

功能：
    1. 从全球/区域陆地 shapefile 中裁剪出指定海域范围的陆地/岛屿。
    2. 将多边形边界转为等距采样的 polyline。
    3. 生成 AISFuser 风格的 "landscape" polyline 节点列表，并导出为 JSON。

使用前你需要：
    - 安装依赖：
        pip install geopandas shapely pyproj numpy

    - 修改脚本中的 INPUT_SHP 为你的陆地 shapefile 路径。

默认参数：
    - 海域范围：经度 [121.5, 121.6]，纬度 [31.28, 31.38]
    - 投影坐标系：EPSG:32651（上海附近的 UTM 51N，可根据需要修改）
    - 采样间距：每 500 米一个点
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
        description="从陆地 shapefile 中提取指定海域的 landscape polylines（AISFuser 风格）"
    )

    # 你可以通过命令行传参，也可以直接改默认值
    parser.add_argument(
        "--input_shp",
        type=str,
        default="PATH/TO/your_land.shp",  # TODO: 改成你的陆地 shapefile 路径
        help="陆地 / 海岸线 shapefile 的路径",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="landscape_polylines_1215_1216_3128_3138.json",
        help="导出的 JSON 文件路径",
    )

    # 你的海域范围（经纬度）
    parser.add_argument("--min_lon", type=float, default=121.5, help="经度最小值")
    parser.add_argument("--max_lon", type=float, default=121.6, help="经度最大值")
    parser.add_argument("--min_lat", type=float, default=31.28, help="纬度最小值")
    parser.add_argument("--max_lat", type=float, default=31.38, help="纬度最大值")

    # 目标投影坐标系（米制）
    parser.add_argument(
        "--target_epsg",
        type=int,
        default=32651,  # 上海附近区域 UTM 51N
        help="投影坐标系 EPSG，例如 32651 表示 UTM zone 51N",
    )

    # 等距采样间隔（米）
    parser.add_argument(
        "--step_m",
        type=float,
        default=500.0,
        help="沿边界线的采样间距（米），例如 500 表示每 500 米一个点",
    )

    args = parser.parse_args()
    return args


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

    input_path = Path(args.input_shp)
    if not input_path.exists():
        logging.error(f"输入 shapefile 不存在: {input_path}")
        return

    logging.info(f"读取陆地数据: {input_path}")
    land = gpd.read_file(str(input_path))

    # 确保有 CRS
    if land.crs is None:
        logging.warning("输入数据没有 CRS，假定为 WGS84 (EPSG:4326)，如不正确请手动修改代码！")
        land.set_crs(epsg=4326, inplace=True)

    # 转为 WGS84 方便按经纬度裁剪
    if land.crs.to_epsg() != 4326:
        logging.info(f"将陆地数据 CRS 从 {land.crs} 转为 EPSG:4326 (WGS84)")
        land = land.to_crs(epsg=4326)

    # 构造裁剪 bbox（经纬度）
    bbox_geom = box(args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")

    logging.info(
        f"使用 bbox 裁剪: "
        f"lon[{args.min_lon}, {args.max_lon}], lat[{args.min_lat}, {args.max_lat}]"
    )
    # 使用 clip，而不是 overlay，通常更快
    land_clip = gpd.clip(land, bbox_gdf)

    if land_clip.empty:
        logging.warning("裁剪后没有任何陆地几何体，可能数据源不覆盖该区域。")
        return

    logging.info(f"裁剪后陆地几何数量: {len(land_clip)}")

    # 投影到目标米制坐标系
    target_crs = f"EPSG:{args.target_epsg}"
    logging.info(f"将数据投影到 {target_crs} 以便按米重采样")
    land_clip_utm = land_clip.to_crs(target_crs)

    # 生成边界并重采样
    land_clip_utm["boundary"] = land_clip_utm.geometry.boundary

    logging.info(f"开始对边界进行等距重采样，步长 = {args.step_m} 米")
    dense_boundaries = []
    for idx, geom in enumerate(land_clip_utm["boundary"]):
        if geom is None:
            dense_boundaries.append(None)
            continue

        # 可能是 LineString 或 MultiLineString
        if isinstance(geom, LineString):
            dense_boundaries.append(densify_line(geom, args.step_m))
        elif isinstance(geom, MultiLineString):
            # 对每条子线重采样，然后合并
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

    land_clip_utm["boundary_dense"] = dense_boundaries

    # 构造 AISFuser 风格的 polyline 节点列表
    land_nodes = []
    logging.info("开始构建 polyline 节点列表（AISFuser 风格）")

    for idx, row in land_clip_utm.iterrows():
        geom = row["boundary_dense"]
        if geom is None:
            continue

        centroid = geom.centroid
        shape_id = int(idx)

        def process_line(line: LineString, polyline_suffix: str = ""):
            coords = list(line.coords)
            if len(coords) < 2:
                return

            polyline_id = f"land_{shape_id}{polyline_suffix}"

            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]

                node_feature = {
                    "polyline_id": polyline_id,
                    "shape_type": "landscape",
                    "shape_id": shape_id,
                    # UTM 坐标（米）
                    "start_x": float(start[0]),
                    "start_y": float(start[1]),
                    "end_x": float(end[0]),
                    "end_y": float(end[1]),
                    "centroid_x": float(centroid.x),
                    "centroid_y": float(centroid.y),
                    # 你还可以在这里加更多属性，如 land 的某些字段
                }
                land_nodes.append(node_feature)

        if isinstance(geom, LineString):
            process_line(geom)
        elif isinstance(geom, MultiLineString):
            for j, ls in enumerate(geom.geoms):
                if isinstance(ls, LineString):
                    process_line(ls, polyline_suffix=f"_{j}")

    logging.info(f"共生成 landscape polyline 节点数: {len(land_nodes)}")

    # 导出为 JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(land_nodes, f, ensure_ascii=False, indent=2)

    logging.info(f"已导出 JSON 至: {output_path.resolve()}")
    logging.info("地形（landscape）部分处理完毕。你可以将这些节点与航道/waypoint polyline 一起构建图。")


if __name__ == "__main__":
    main()
