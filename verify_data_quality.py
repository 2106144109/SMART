
#!/usr/bin/env python3
"""
验证生成的maritime数据质量
基于HeteroData结构进行详细分析
"""

import torch
import numpy as np
from glob import glob
from pathlib import Path
import random

def analyze_sample(data, sample_idx=0):
    """详细分析单个HeteroData样本"""
    print(f"\n{'='*80}")
    print(f"样本 #{sample_idx}")
    print(f"{'='*80}")
    
    # 1. 基本信息
    print(f"\n【数据类型】 HeteroData")
    print(f"【Node类型】 {data.node_types}")
    print(f"【Edge类型】 {data.edge_types}")
    
    # 2. Metadata
    if hasattr(data, 'metadata') and isinstance(data.metadata, dict):
        print(f"\n【场景元数据】")
        for k, v in data.metadata.items():
            print(f"  {k:20s}: {v}")
    
    # 3. Agent数据
    agent_data = data['agent']
    x = agent_data.x.numpy()
    
    num_agents, num_steps, num_features = x.shape
    print(f"\n【Agent特征矩阵】")
    print(f"  形状: {x.shape}")
    print(f"  -> 船只数量: {num_agents}")
    print(f"  -> 时间步数: {num_steps} (应为 5历史 + 16未来 = 21)")
    print(f"  -> 特征维度: {num_features} (应为 8)")
    
    # 特征名称
    feature_names = ['x(m)', 'y(m)', 'vx(m/s)', 'vy(m/s)', 
                     'ax(m/s²)', 'ay(m/s²)', 'theta(rad)', 'omega(rad/s)']
    
    # 4. 验证T_h-1时刻（历史最后一步，index=4）
    hist_last_step = 4  # 第5步（index=4）是历史最后一步
    av_index = agent_data.av_index.item() if hasattr(agent_data.av_index, 'item') else agent_data.av_index
    
    print(f"\n【关键验证】T_h-1时刻（第{hist_last_step+1}步，index={hist_last_step}）")
    print(f"  参考船索引（av_index）= {av_index}")
    print(f"  参考船（Agent {av_index}）的状态：")
    ref_state = x[av_index, hist_last_step, :]
    
    check_passed = True
    for i, (name, val) in enumerate(zip(feature_names, ref_state)):
        if i < 2:  # 位置
            status = "✅" if abs(val) < 1e-3 else "❌"
            if abs(val) >= 1e-3:
                check_passed = False
            print(f"    {name:12s} = {val:10.6f}  {status}")
        elif i == 6:  # 航向
            status = "✅" if abs(val) < 1e-6 else "❌"
            if abs(val) >= 1e-6:
                check_passed = False
            print(f"    {name:12s} = {val:10.6f}  {status}")
        else:
            print(f"    {name:12s} = {val:10.6f}")
    
    if check_passed:
        print(f"\n  ✅ 局部坐标系验证通过！")
    else:
        print(f"\n  ❌ 局部坐标系验证失败！")
    
    # 5. 统计所有Agent在所有时刻的特征范围
    print(f"\n【特征统计】（所有船只 × 所有时刻）")
    print(f"  {'特征':<12s} {'最小值':>10s} {'平均值':>10s} {'最大值':>10s} {'标准差':>10s}")
    print(f"  {'-'*56}")
    
    for i, name in enumerate(feature_names):
        feat_vals = x[:, :, i]
        print(f"  {name:<12s} {feat_vals.min():10.3f} {feat_vals.mean():10.3f} "
              f"{feat_vals.max():10.3f} {feat_vals.std():10.3f}")
    
    # 6. 速度合理性检查
    print(f"\n【速度合理性检查】")
    vx = x[:, :, 2]
    vy = x[:, :, 3]
    speed = np.sqrt(vx**2 + vy**2)
    
    print(f"  速度幅值 (m/s):")
    print(f"    平均速度: {speed.mean():.2f} m/s ({speed.mean()/0.514444:.2f} 节)")
    print(f"    中位速度: {np.median(speed):.2f} m/s ({np.median(speed)/0.514444:.2f} 节)")
    print(f"    95%分位: {np.percentile(speed, 95):.2f} m/s ({np.percentile(speed, 95)/0.514444:.2f} 节)")
    print(f"    最大速度: {speed.max():.2f} m/s ({speed.max()/0.514444:.2f} 节)")
    
    # 典型港区船速：5-15节 = 2.6-7.7 m/s
    speed_check = "✅" if 2.0 < speed.mean() < 8.0 else "⚠️"
    print(f"  {speed_check} 平均速度{'合理' if speed_check == '✅' else '异常'}（港区典型：2-8 m/s 或 4-15节）")
    
    # 7. 加速度合理性检查
    print(f"\n【加速度合理性检查】")
    ax = x[:, :, 4]
    ay = x[:, :, 5]
    accel = np.sqrt(ax**2 + ay**2)
    
    print(f"  加速度幅值 (m/s²):")
    print(f"    平均值: {accel.mean():.4f} m/s²")
    print(f"    中位数: {np.median(accel):.4f} m/s²")
    print(f"    95%分位: {np.percentile(accel, 95):.4f} m/s²")
    print(f"    最大值: {accel.max():.4f} m/s²")
    
    accel_check = "✅" if accel.mean() < 0.5 else "⚠️"
    print(f"  {accel_check} 加速度{'合理' if accel_check == '✅' else '偏大'}（船舶加速通常 < 0.5 m/s²）")
    
    # 8. 航向角检查
    print(f"\n【航向角检查】")
    theta = x[:, :, 6]
    print(f"  航向范围: [{theta.min():.3f}, {theta.max():.3f}] rad")
    print(f"  期望范围: [-π, π] = [-3.142, 3.142] rad")
    
    theta_check = "✅" if -np.pi <= theta.min() and theta.max() <= np.pi else "⚠️"
    print(f"  {theta_check} 航向角{'已正确wrap' if theta_check == '✅' else '超出范围'}")
    
    # 9. 有效性掩码检查
    print(f"\n【有效性掩码】")
    valid_mask = agent_data.valid_mask.numpy()
    is_hist_mask = agent_data.is_history_mask.numpy()
    
    print(f"  valid_mask 形状: {valid_mask.shape}")
    print(f"  is_history_mask 形状: {is_hist_mask.shape}")
    
    # 检查每个agent的有效步数
    valid_counts = valid_mask.sum(axis=1)
    hist_counts = is_hist_mask.sum(axis=1)
    
    print(f"  有效步数范围: [{valid_counts.min()}, {valid_counts.max()}]")
    print(f"  历史步数（应全为5）: {np.unique(hist_counts)}")
    
    # 10. 目标轨迹
    print(f"\n【目标轨迹】")
    target_xy = agent_data.target_xy.numpy()
    print(f"  target_xy 形状: {target_xy.shape} (应为 [N, 16, 2])")
    print(f"  目标位置范围:")
    print(f"    x: [{target_xy[:,:,0].min():.2f}, {target_xy[:,:,0].max():.2f}] m")
    print(f"    y: [{target_xy[:,:,1].min():.2f}, {target_xy[:,:,1].max():.2f}] m")
    
    # 11. MMSI信息
    print(f"\n【船只标识】")
    mmsi = agent_data.mmsi.numpy()
    print(f"  MMSI数量: {len(mmsi)}")
    print(f"  前5个MMSI: {mmsi[:5].tolist()}")
    
    # 12. 图结构
    print(f"\n【图结构】")
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        print(f"  {edge_type}: {edge_index.shape[1]} 条边")
    
    return {
        'num_agents': num_agents,
        'speed_mean': speed.mean(),
        'speed_95p': np.percentile(speed, 95),
        'accel_mean': accel.mean(),
        'check_passed': check_passed
    }


def main():
    """主函数"""
    data_root = Path("/home/mahexing/SMART-main/data/maritime_windows_30s_no_norm")
    
    print(f"\n{'#'*80}")
    print(f"# Maritime数据质量验证")
    print(f"{'#'*80}\n")
    
    # 统计各分片
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        if split_dir.exists():
            files = list(split_dir.glob("*.pt"))
            total_size = sum(f.stat().st_size for f in files)
            print(f"【{split:5s}】 {len(files):6d} 文件, 总大小: {total_size/1e9:.2f} GB")
    
    # 随机抽取训练集样本
    train_dir = data_root / "train"
    train_files = sorted(train_dir.glob("*.pt"))
    
    print(f"\n从训练集随机抽取 5 个文件进行详细分析...")
    
    random.seed(42)
    sample_files = random.sample(train_files, min(5, len(train_files)))
    
    total_samples = 0
    total_files = 0
    all_stats = []
    
    for file_idx, file_path in enumerate(sample_files, 1):
        print(f"\n" + "#"*80)
        print(f"# 文件 #{file_idx}: {file_path.name}")
        print(f"# 文件大小: {file_path.stat().st_size / 1024:.1f} KB")
        print("#"*80)
        
        try:
            # 加载文件
            data_list = torch.load(file_path, weights_only=False)
            print(f"\n包含 {len(data_list)} 个样本")
            total_samples += len(data_list)
            total_files += 1
            
            # 分析第一个样本
            if len(data_list) > 0:
                stats = analyze_sample(data_list[0], sample_idx=file_idx)
                all_stats.append(stats)
        
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 全局统计
    print(f"\n{'#'*80}")
    print(f"# 全局统计汇总")
    print(f"{'#'*80}\n")
    
    print(f"检查文件数: {total_files}")
    print(f"检查样本数: {total_samples}")
    
    if all_stats:
        passed_count = sum(1 for s in all_stats if s['check_passed'])
        print(f"\n【局部坐标系验证】")
        print(f"  通过率: {passed_count}/{len(all_stats)} ({100*passed_count/len(all_stats):.1f}%)")
        
        print(f"\n【速度统计】")
        speeds = [s['speed_mean'] for s in all_stats]
        print(f"  平均速度: {np.mean(speeds):.2f} ± {np.std(speeds):.2f} m/s")
        print(f"  范围: [{np.min(speeds):.2f}, {np.max(speeds):.2f}] m/s")
        
        print(f"\n【加速度统计】")
        accels = [s['accel_mean'] for s in all_stats]
        print(f"  平均加速度: {np.mean(accels):.4f} ± {np.std(accels):.4f} m/s²")
        print(f"  范围: [{np.min(accels):.4f}, {np.max(accels):.4f}] m/s²")
    
    print(f"\n{'='*80}")
    print(f"✅ 验证完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

