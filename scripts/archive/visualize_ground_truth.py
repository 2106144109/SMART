#!/usr/bin/env python3
"""
只可视化真实数据（Ground Truth），不使用模型预测
用于验证数据本身是否正确
"""

import torch
import folium
from folium import plugins
import os
import numpy as np
import math

def meters_to_lat_lon(x_meters, y_meters, center_lat, center_lon):
    """将米制坐标转换为经纬度"""
    lat_per_meter = 1.0 / 111000.0
    lon_per_meter = 1.0 / (111000.0 * np.cos(np.radians(center_lat)))
    
    lat = float(center_lat + y_meters * lat_per_meter)
    lon = float(center_lon + x_meters * lon_per_meter)
    
    return lat, lon

def visualize_ground_truth(data_file, output_path, center_lat=30.0, center_lon=122.0):
    """
    可视化真实数据
    
    Args:
        data_file: .pt数据文件路径
        output_path: 输出HTML路径
        center_lat: 地图中心纬度
        center_lon: 地图中心经度
    """
    
    print(f"\n📂 加载数据: {data_file}")
    data = torch.load(data_file)
    
    # 处理数据格式（可能是列表或HeteroData）
    if isinstance(data, list):
        print(f"  数据是列表，长度: {len(data)}")
        if len(data) > 0:
            data = data[0]  # 取第一个元素
            print(f"  取第一个元素: {type(data)}")
    
    # 提取数据
    features = data['agent']['x'].cpu().numpy()  # [N_agents, T_total, 8]
    positions = features[:, :, :2]  # [N_agents, T_total, 2]
    headings = features[:, :, 6]     # [N_agents, T_total]
    
    if 'valid_mask' in data['agent']:
        valid_mask = data['agent']['valid_mask'].cpu().numpy()
    else:
        valid_mask = np.ones(positions.shape[:2], dtype=bool)
    
    num_agents = positions.shape[0]
    num_historical = 5
    num_future = 16
    total_timesteps = positions.shape[1]
    
    print(f"  ✓ 船只数量: {num_agents}")
    print(f"  ✓ 总时间步: {total_timesteps}")
    print(f"  ✓ 历史步数: {num_historical}")
    print(f"  ✓ 未来步数: {num_future}")
    
    # 坐标统计
    print(f"\n📊 坐标统计（原始数据）:")
    print(f"  全局范围: X=[{positions[:,:,0].min():.1f}, {positions[:,:,0].max():.1f}] 米")
    print(f"           Y=[{positions[:,:,1].min():.1f}, {positions[:,:,1].max():.1f}] 米")
    
    # 分析每个时间步
    for t in [0, 4, 10, 20]:
        if t < total_timesteps:
            pos_t = positions[:, t, :]
            print(f"  t={t:2d}: X=[{pos_t[:,0].min():>8.1f}, {pos_t[:,0].max():>8.1f}], "
                  f"Y=[{pos_t[:,1].min():>8.1f}, {pos_t[:,1].max():>8.1f}]")
    
    # 创建地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
    
    all_coords = []
    
    # 为每艘船添加轨迹
    for agent_id in range(num_agents):
        color = colors[agent_id % len(colors)]
        
        # === 历史轨迹 ===
        hist_positions = positions[agent_id, :num_historical, :]
        hist_valid = valid_mask[agent_id, :num_historical]
        valid_hist = hist_positions[hist_valid]
        
        if len(valid_hist) > 0:
            hist_coords = []
            for pos in valid_hist:
                lat, lon = meters_to_lat_lon(pos[0], pos[1], center_lat, center_lon)
                hist_coords.append([lat, lon])
            
            all_coords.extend(hist_coords)
            
            # 绘制历史轨迹（粗实线）
            folium.PolyLine(
                hist_coords,
                color=color,
                weight=4,
                opacity=0.9,
                popup=f'船 {agent_id} - 历史轨迹 (真实数据)',
                tooltip=f'船 {agent_id} 历史'
            ).add_to(m)
            
            # 起点标记
            folium.CircleMarker(
                hist_coords[0],
                radius=8,
                popup=f'船 {agent_id} 起点 (t=0)',
                tooltip=f'船 {agent_id} 起点',
                color=color,
                fill=True,
                fillColor='white',
                fillOpacity=1
            ).add_to(m)
            
            # T_h-1 标记（历史结束帧）
            folium.CircleMarker(
                hist_coords[-1],
                radius=10,
                popup=f'船 {agent_id} T_h-1 (历史结束帧)',
                tooltip=f'船 {agent_id} T_h-1',
                color=color,
                fill=True,
                fillColor='yellow',
                fillOpacity=0.8
            ).add_to(m)
        
        # === 未来真实轨迹 ===
        future_positions = positions[agent_id, num_historical:num_historical+num_future, :]
        future_valid = valid_mask[agent_id, num_historical:num_historical+num_future]
        valid_future = future_positions[future_valid]
        
        if len(valid_future) > 0:
            future_coords = []
            for pos in valid_future:
                lat, lon = meters_to_lat_lon(pos[0], pos[1], center_lat, center_lon)
                future_coords.append([lat, lon])
            
            all_coords.extend(future_coords)
            
            # 绘制未来轨迹（虚线）
            folium.PolyLine(
                future_coords,
                color=color,
                weight=3,
                opacity=0.6,
                dash_array='10, 5',
                popup=f'船 {agent_id} - 未来轨迹 (真实数据)',
                tooltip=f'船 {agent_id} 未来'
            ).add_to(m)
            
            # 终点标记
            folium.CircleMarker(
                future_coords[-1],
                radius=8,
                popup=f'船 {agent_id} 终点',
                tooltip=f'船 {agent_id} 终点',
                color=color,
                fill=True,
                fillColor='red',
                fillOpacity=1
            ).add_to(m)
            
            # 添加方向箭头（每3个点）
            for i in range(0, len(future_coords), 3):
                if i < len(headings[agent_id, num_historical:num_historical+num_future]):
                    theta = float(headings[agent_id, num_historical + i])
                    if np.isfinite(theta):
                        heading_deg = float(np.degrees(theta))
                        folium.RegularPolygonMarker(
                            location=future_coords[i],
                            fill_color=color,
                            number_of_sides=3,
                            radius=6,
                            rotation=heading_deg,
                            popup=f'方向: {heading_deg:.1f}°',
                            opacity=0.5
                        ).add_to(m)
    
    # 自动调整地图范围
    if len(all_coords) > 0:
        lats = [coord[0] for coord in all_coords]
        lons = [coord[1] for coord in all_coords]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        m.fit_bounds(bounds, padding=[50, 50])
        
        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        print(f"\n🗺️  地图范围:")
        print(f"  纬度: {min(lats):.6f} ~ {max(lats):.6f} (跨度: {lat_span:.6f}°)")
        print(f"  经度: {min(lons):.6f} ~ {max(lons):.6f} (跨度: {lon_span:.6f}°)")
    
    # 添加标题
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 550px; 
                background-color: white; border: 2px solid grey; 
                z-index: 9999; font-size: 14px; padding: 10px;
                border-radius: 5px; opacity: 0.95;">
    <h3 style="margin:0; color:#2c3e50;">🗺️ Ground Truth Data Visualization</h3>
    <p style="margin:5px 0;">
    <b>只显示真实数据，不使用模型预测</b><br>
    <span style="color:blue;">━━━</span> 历史轨迹 (5步 × 30s = 2.5分钟)<br>
    <span style="color:blue;">- - -</span> 未来轨迹 (16步 × 30s = 8分钟)<br>
    ⚪ 起点 | 🟡 T_h-1 (历史结束) | 🔴 终点
    </p>
    <p style="margin:5px 0; font-size:12px; background-color:#d1ecf1; padding:5px;">
    <b>📍 坐标系统：</b> 原始数据，未经过模型处理
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # 添加信息框
    info_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; 
                background-color: white; border: 2px solid grey; 
                z-index: 9999; font-size: 12px; padding: 10px;
                border-radius: 5px; opacity: 0.9;">
    <p style="margin:0;"><b>📊 场景信息:</b></p>
    <p style="margin:5px 0;">船只数量: {num_agents}</p>
    <p style="margin:5px 0;">数据来源: {os.path.basename(data_file)}</p>
    <p style="margin:5px 0; font-size:10px;">
    如果轨迹正常分散，说明数据没问题。<br>
    如果轨迹挤在一起，说明坐标系统有问题。
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(info_html))
    
    # 添加工具
    plugins.MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)
    plugins.Fullscreen(position='topright').add_to(m)
    plugins.MousePosition().add_to(m)
    
    # 保存地图
    m.save(output_path)
    print(f"\n✅ 保存成功: {output_path}")

def main():
    print("="*80)
    print(" " * 20 + "🗺️  Ground Truth 数据可视化")
    print("="*80)
    print("\n目的：验证原始数据是否正确，不使用模型预测")
    
    # 配置
    test_dir = 'data/maritime_windows_30s_no_norm/test'
    output_dir = 'folium_maps'
    center_lat = 30.0
    center_lon = 122.0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取第一个测试文件
    import glob
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.pt')))
    
    if not test_files:
        print(f"❌ 错误: 找不到测试文件在 {test_dir}")
        return
    
    # 可视化第一个场景
    test_file = test_files[0]
    output_path = os.path.join(output_dir, 'ground_truth_scene_000.html')
    
    visualize_ground_truth(test_file, output_path, center_lat, center_lon)
    
    print(f"\n" + "="*80)
    print("✅ Ground Truth 可视化完成！")
    print("="*80)
    print(f"\n🌐 打开方式:")
    print(f"  file://{os.path.abspath(output_path)}")
    print(f"\n💡 判断标准:")
    print(f"  ✓ 如果轨迹正常分散 → 数据没问题，问题在模型或可视化脚本")
    print(f"  ✗ 如果轨迹挤在一起 → 数据的坐标系统有问题")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

