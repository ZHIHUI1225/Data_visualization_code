"""
创建三种方法的箱线图对比可视化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_boxplot_comparison(data: Dict, output_path: Path):
    """
    创建箱线图展示三种方法的轨迹长度对比
    """
    # 收集有效数据
    dubins_lengths = []
    bezier_lengths = []
    maps_lengths = []
    case_names = []
    
    for case, case_data in sorted(data.items()):
        dubins = case_data.get('dubins', {})
        bezier = case_data.get('bezier', {})
        maps = case_data.get('maps', {})
        
        # 只包含所有三种方法都有数据的案例
        dubins_len = dubins.get('length', 0)
        bezier_len = bezier.get('length', 0)
        maps_len = maps.get('length', 0)
        
        # 检查dubins是否失败
        dubins_failed = 'FAILED' in dubins.get('source', '')
        
        if not dubins_failed and dubins_len > 0 and bezier_len > 0 and maps_len > 0:
            dubins_lengths.append(dubins_len)
            bezier_lengths.append(bezier_len)
            maps_lengths.append(maps_len)
            case_names.append(case)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：箱线图
    box_data = [dubins_lengths, bezier_lengths, maps_lengths]
    labels = ['A* + Dubins', 'A* + Bezier', 'Proposed\n(MAPS)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bp = ax1.boxplot(box_data, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True,
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # 设置箱体颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('Trajectory Length (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Trajectory Length Comparison (Box Plot)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加图例
    ax1.legend([bp['medians'][0], bp['means'][0]], 
              ['Median', 'Mean'], 
              loc='upper right')
    
    # 右图：分组柱状图
    x = np.arange(len(case_names))
    width = 0.25
    
    bars1 = ax2.bar(x - width, [dubins_lengths[i] for i in range(len(case_names))], 
                    width, label='A* + Dubins', color=colors[0], alpha=0.8)
    bars2 = ax2.bar(x, [bezier_lengths[i] for i in range(len(case_names))], 
                    width, label='A* + Bezier', color=colors[1], alpha=0.8)
    bars3 = ax2.bar(x + width, [maps_lengths[i] for i in range(len(case_names))], 
                    width, label='Proposed (MAPS)', color=colors[2], alpha=0.8)
    
    ax2.set_xlabel('Test Cases', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Trajectory Length (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Trajectory Length by Test Case', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(case_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"箱线图已保存: {output_path}")
    plt.close()


def create_success_rate_chart(data: Dict, output_path: Path):
    """
    创建成功率和碰撞率对比图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 统计成功率
    total_cases = len(data)
    
    dubins_success = sum(1 for d in data.values() 
                         if 'dubins' in d and 'FAILED' not in d['dubins'].get('source', '') 
                         and d['dubins'].get('length', 0) > 0)
    bezier_success = sum(1 for d in data.values() 
                         if 'bezier' in d and d['bezier'].get('length', 0) > 0)
    maps_success = sum(1 for d in data.values() 
                       if 'maps' in d and d['maps'].get('length', 0) > 0)
    
    # 统计碰撞率
    dubins_collision = sum(1 for d in data.values() 
                           if 'dubins' in d and d['dubins'].get('has_collision') is True)
    bezier_collision = sum(1 for d in data.values() 
                           if 'bezier' in d and d['bezier'].get('has_collision') is True)
    maps_collision = sum(1 for d in data.values() 
                         if 'maps' in d and d['maps'].get('has_collision') is True)
    
    # 左图：成功率
    methods = ['A* + Dubins', 'A* + Bezier', 'Proposed']
    success_rates = [dubins_success/total_cases*100, 
                     bezier_success/total_cases*100,
                     maps_success/total_cases*100]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax1.bar(methods, success_rates, color=colors, alpha=0.8)
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Planning Success Rate', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 右图：碰撞率
    collision_rates = [dubins_collision/total_cases*100,
                       bezier_collision/total_cases*100,
                       maps_collision/total_cases*100]
    
    bars2 = ax2.bar(methods, collision_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('Collision Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Trajectory Collision Rate', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(collision_rates) + 10])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, rate in zip(bars2, collision_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"成功率/碰撞率图已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("生成额外的可视化图表")
    print("=" * 70)
    
    # 收集数据
    print("\n收集数据...")
    data = collect_all_metrics()
    
    if not data:
        print("未找到任何数据!")
        return
    
    # 输出目录
    output_dir = Path(r'd:\Data_visualization_code\result')
    
    # 创建箱线图
    print("\n创建箱线图...")
    create_boxplot_comparison(data, output_dir / 'three_methods_boxplot.png')
    
    # 创建成功率/碰撞率图
    print("\n创建成功率/碰撞率对比图...")
    create_success_rate_chart(data, output_dir / 'three_methods_success_collision.png')
    
    print("\n" + "=" * 70)
    print("可视化完成!")
    print("=" * 70)


if __name__ == "__main__":
    from compare_three_methods import collect_all_metrics
    from typing import Dict
    main()
