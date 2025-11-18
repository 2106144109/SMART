#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis_osm_water_folium.py

功能：
    1. 读取 OSM 提供的水域面 shapefile（例如 gis_osm_water_a_free.shp）
    2. 按给定经纬度 bbox 裁剪
    3. 把水域 polygon 画在 folium + OpenStreetMap 底图上
    4. 保存为 HTML，浏览器打开即可查看“河面 + 海面”边界

注意：
    - 需要安装 geopandas、folium：
        pip install geopandas folium
"""

import argparse

import geopandas as gpd
from shapely.geometry import box
import folium


def parse_args():
    parser = argparse.ArgumentParser(
        description="用 folium 可视化 OSM 水域多边形（河流+湖泊+海面）"
    )
    parser.add_argument(
        "--water_shp",
        type=str,
        required=True,
        help="OSM 水域面 shapefile 路径，例如 gis_osm_water_a_free.shp",
    )
    parser.add_argument(
        "--html_out",
        type=str,
        default="osm_water_map.html",
        help="输出 HTML 地图路径",
    )

    # 默认用你这块区域的 bbox
    parser.add_argument("--min_lon", type=float, default=121.5, help="经度最小值")
    parser.add_argument("--max_lon", type=float, default=121.6, help="经度最大值")
    parser.add_argument("--min_lat", type=float, default=31.28, help="纬度最小值")
    parser.add_argument("--max_lat", type=float, default=31.38, help="纬度最大值")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"读取 OSM 水域 shapefile: {args.water_shp}")
    water = gpd.read_file(args.water_shp)

    # 确保在 WGS84，经纬度
    if water.crs is None:
        print("输入数据没有 CRS，假定为 WGS84 (EPSG:4326)，若不正确请手动修改。")
        water.set_crs(epsg=4326, inplace=True)
    elif water.crs.to_epsg() != 4326:
        print(f"将数据从 {water.crs} 转为 EPSG:4326 (WGS84)")
        water = water.to_crs(epsg=4326)

    # bbox 裁剪
    bbox_geom = box(args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")

    print(
        f"使用 bbox 裁剪: lon[{args.min_lon}, {args.max_lon}], "
        f"lat[{args.min_lat}, {args.max_lat}]"
    )
    water_clip = gpd.clip(water, bbox_gdf)

    if water_clip.empty:
        print("裁剪后没有任何水域 polygon，可能这块区域的 OSM 没标水面，或者 bbox 太小。")
        return

    print(f"裁剪后水域 polygon 数量: {len(water_clip)}")

    # 地图中心用 bbox 中心
    center_lat = (args.min_lat + args.max_lat) / 2.0
    center_lon = (args.min_lon + args.max_lon) / 2.0

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=13,
                   tiles="OpenStreetMap")

    # 叠加水域 polygon
    folium.GeoJson(
        water_clip.__geo_interface__,
        name="OSM water areas",
        style_function=lambda feature: {
            "fillColor": "#66ccff",
            "color": "#0033aa",
            "weight": 1,
            "fillOpacity": 0.6,
        },
    ).add_to(m)

    # 画 bbox 边框（红色）
    bbox_latlon = [
        (args.min_lat, args.min_lon),
        (args.min_lat, args.max_lon),
        (args.max_lat, args.max_lon),
        (args.max_lat, args.min_lon),
    ]
    folium.PolyLine(
        locations=bbox_latlon + [bbox_latlon[0]],
        color="red",
        weight=1,
        opacity=0.7,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save(args.html_out)
    print(f"已保存 folium 地图到: {args.html_out}")
    print("在浏览器中打开该 HTML 文件即可查看。")


if __name__ == "__main__":
    main()
