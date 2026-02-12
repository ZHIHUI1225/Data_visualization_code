"""
轨迹对比分析脚本
对比 RRT_Results, Waypoint_bezier 和 MAPS 三种方法的轨迹指标：
- 轨迹长度 (Trajectory Length)
- 最小转弯半径 (Minimum Radius)
- 碰撞点比例 (Collision Ratio)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_rrt_metrics(rrt_path: Path) -> Optional[Dict]:
    """
    从RRT_Results结果文件中提取指标
    文件路径: RRT_Results/{CASE}/bezier_trajectory_{CASE}.json

    Returns:
        dict with metrics or None
    """
    try:
        with open(rrt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trajectory = data.get('trajectory', {})

        total_distance = trajectory.get('total_distance', 0)
        max_curvature = trajectory.get('max_curvature', 0)

        # 最小半径 = 1/最大曲率
        min_radius = 1.0 / max_curvature if max_curvature > 1e-6 else float('inf')

        # 碰撞比例
        collision_info = data.get('collision_info', {})
        collision_ratio = collision_info.get('collision_ratio', 0)

        return {
            'length': total_distance,
            'min_radius': min_radius,
            'collision_ratio': collision_ratio,
            'source': str(rrt_path)
        }
    except Exception as e:
        print(f"  [WARN] Error reading RRT file {rrt_path}: {e}")
        return None


def extract_waypoint_bezier_metrics(bezier_path: Path) -> Optional[Dict]:
    """
    从Waypoint_bezier结果文件中提取指标
    文件路径: Waypoint_bezier/{CASE}/bezier_curves_summary.json
              或 Waypoint_bezier/{CASE}/bezier_trajectory_*.json

    Returns:
        dict with metrics or None
    """
    try:
        with open(bezier_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否是summary文件
        if 'sequences' in data:
            sequences = data.get('sequences', [])
            if not sequences:
                return None
            first_seq = sequences[0]
            trajectory = first_seq.get('trajectory', {})
        else:
            # 单个轨迹文件
            trajectory = data.get('trajectory', {})

        total_distance = trajectory.get('total_distance', 0)
        max_curvature = trajectory.get('max_curvature', 0)

        # 最小半径 = 1/最大曲率
        min_radius = 1.0 / max_curvature if max_curvature > 1e-6 else float('inf')

        # 碰撞比例（从summary或单个文件获取）
        collision_ratio = 0
        if 'sequences' in data:
            sequences = data.get('sequences', [])
            if sequences:
                first_seq = sequences[0]
                collision_info = first_seq.get('collision_info', {})
                collision_ratio = collision_info.get('collision_ratio', 0)
        else:
            collision_info = data.get('collision_info', {})
            collision_ratio = collision_info.get('collision_ratio', 0)

        return {
            'length': total_distance,
            'min_radius': min_radius,
            'collision_ratio': collision_ratio,
            'source': str(bezier_path)
        }
    except Exception as e:
        print(f"  [WARN] Error reading Waypoint_bezier file {bezier_path}: {e}")
        return None


def extract_maps_metrics(maps_path: Path) -> Optional[Dict]:
    """
    从MAPS结果文件中提取指标
    文件路径: MAPS/{CASE}/.../complete_trajectory_parameters_{case}.json

    Returns:
        dict with metrics or None
    """
    try:
        with open(maps_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 轨迹长度：sum(l数组)
        l_array = data.get('l', [])
        total_length = sum(l_array)

        # 最小半径：min(abs(r0数组))
        r0_array = data.get('r0', [])
        min_radius = min([abs(r) for r in r0_array if abs(r) > 1e-6]) if r0_array else float('inf')

        return {
            'length': total_length,
            'min_radius': min_radius,
            'collision_ratio': 0,  # MAPS没有碰撞检测
            'source': str(maps_path)
        }
    except Exception as e:
        print(f"  [WARN] Error reading MAPS file {maps_path}: {e}")
        return None


def find_maps_trajectory_file(case_name: str, maps_base: Path) -> Optional[Path]:
    """
    查找MAPS目录下的轨迹文件
    路径模式多样：
    - MAP1/MAP1/map1/complete_trajectory_parameters_map1.json
    - MAP2/map2/map2/complete_trajectory_parameters_map2.json
    - warehouse/complete_trajectory_parameters_warehouse.json
    - maze/complete_trajectory_parameters_maze.json
    """
    case_lower = case_name.lower()
    case_dir = maps_base / case_name

    if not case_dir.exists():
        return None

    # 使用glob递归搜索
    pattern = f"**/complete_trajectory_parameters_{case_lower}.json"
    matches = list(case_dir.glob(pattern))

    if matches:
        return matches[0]

    # 也尝试不带下划线的名称
    pattern2 = f"**/complete_trajectory_parameters_*.json"
    matches2 = list(case_dir.glob(pattern2))

    if matches2:
        return matches2[0]

    return None


def find_waypoint_bezier_file(case_name: str, waypoint_base: Path) -> Optional[Path]:
    """
    查找Waypoint_bezier目录下的轨迹文件
    """
    case_dir = waypoint_base / case_name

    if not case_dir.exists():
        return None

    # 优先查找summary文件
    summary_file = case_dir / "bezier_curves_summary.json"
    if summary_file.exists():
        return summary_file

    # 否则查找单个轨迹文件
    traj_files = list(case_dir.glob("bezier_trajectory_*.json"))
    if traj_files:
        return traj_files[0]

    return None


def collect_all_metrics() -> Dict:
    """
    收集所有案例的指标数据

    Returns:
        dict: {case_name: {'rrt': metrics, 'waypoint_bezier': metrics, 'maps': metrics}}
    """
    base_dir = Path(r'd:\Data_visualization_code\result')
    rrt_base = base_dir / 'RRT_Results'
    waypoint_base = base_dir / 'Waypoint_bezier'
    maps_base = base_dir / 'MAPS'

    results = {}

    # 获取所有案例名称（合并三个目录的案例）
    all_cases = set()

    if rrt_base.exists():
        all_cases.update([d.name for d in rrt_base.iterdir() if d.is_dir()])
    if waypoint_base.exists():
        all_cases.update([d.name for d in waypoint_base.iterdir() if d.is_dir()])
    if maps_base.exists():
        all_cases.update([d.name for d in maps_base.iterdir() if d.is_dir()])

    # 过滤掉不需要的目录
    exclude_dirs = {'LOOP1', 'LOOP2', 'MAP5_2', 'bezier_results', '__pycache__'}
    all_cases = [c for c in all_cases if c not in exclude_dirs and not c.startswith('warehouse_')]
    all_cases = sorted(all_cases)

    print(f"\n发现 {len(all_cases)} 个案例: {all_cases}\n")

    for case in all_cases:
        print(f"处理案例: {case}")

        case_results = {}

        # 1. RRT_Results
        rrt_file = rrt_base / case / f"bezier_trajectory_{case}.json"
        if rrt_file.exists():
            rrt_metrics = extract_rrt_metrics(rrt_file)
            if rrt_metrics:
                case_results['rrt'] = rrt_metrics
                print(f"  [RRT] length={rrt_metrics['length']:.4f}m")
        else:
            print(f"  [RRT] 文件未找到")

        # 2. Waypoint_bezier
        waypoint_file = find_waypoint_bezier_file(case, waypoint_base)
        if waypoint_file:
            waypoint_metrics = extract_waypoint_bezier_metrics(waypoint_file)
            if waypoint_metrics:
                case_results['waypoint_bezier'] = waypoint_metrics
                print(f"  [Waypoint] length={waypoint_metrics['length']:.4f}m")
        else:
            print(f"  [Waypoint] 文件未找到")

        # 3. MAPS
        maps_file = find_maps_trajectory_file(case, maps_base)
        if maps_file:
            maps_metrics = extract_maps_metrics(maps_file)
            if maps_metrics:
                case_results['maps'] = maps_metrics
                print(f"  [MAPS] length={maps_metrics['length']:.4f}m")
        else:
            print(f"  [MAPS] 文件未找到")

        if case_results:
            results[case] = case_results

    return results


def create_comparison_table(data: Dict, output_path: Path):
    """
    创建对比表格
    """
    if not data:
        print("没有数据可显示")
        return

    cases = sorted(data.keys())

    # 表头
    headers = [
        'Case',
        'RRT\nLength(m)', 'Waypoint\nLength(m)', 'Proposed\nLength(m)',
        'RRT\nMin R(m)', 'Waypoint\nMin R(m)', 'Proposed\nMin R(m)',
        'RRT\nCollision%', 'Waypoint\nCollision%'
    ]

    table_data = []
    radius_values = []  # 存储每行的半径值，用于高亮小于0.2的单元格
    collision_values = []  # 存储每行的碰撞比例，用于高亮非零值

    for case in cases:
        case_data = data[case]

        rrt = case_data.get('rrt', {})
        waypoint = case_data.get('waypoint_bezier', {})
        maps = case_data.get('maps', {})

        row = [case]

        # 轨迹长度
        row.append(f"{rrt.get('length', 0):.4f}" if rrt.get('length', 0) > 0 else "-")
        row.append(f"{waypoint.get('length', 0):.4f}" if waypoint.get('length', 0) > 0 else "-")
        row.append(f"{maps.get('length', 0):.4f}" if maps.get('length', 0) > 0 else "-")

        # 最小转弯半径
        rrt_r = rrt.get('min_radius', 0)
        waypoint_r = waypoint.get('min_radius', 0)
        maps_r = maps.get('min_radius', 0)

        row.append(f"{rrt_r:.4f}" if 0 < rrt_r < 100 else "-")
        row.append(f"{waypoint_r:.4f}" if 0 < waypoint_r < 100 else "-")
        row.append(f"{maps_r:.4f}" if 0 < maps_r < 100 else "-")

        # 存储半径值用于后续高亮
        radius_values.append([rrt_r, waypoint_r, maps_r])

        # 碰撞比例
        rrt_col = rrt.get('collision_ratio', 0)
        waypoint_col = waypoint.get('collision_ratio', 0)

        row.append(f"{rrt_col*100:.1f}%" if rrt else "-")
        row.append(f"{waypoint_col*100:.1f}%" if waypoint else "-")

        # 存储碰撞比例用于后续高亮
        collision_values.append([rrt_col, waypoint_col])

        table_data.append(row)

    # 创建图表
    fig, ax = plt.subplots(figsize=(18, len(cases) * 0.7 + 3))
    ax.axis('tight')
    ax.axis('off')

    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.08] + [0.1] * 8
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)  # 增大字体
    table.scale(1.2, 2.4)

    # 表头样式
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#2F5496')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # 数据行样式
    # 半径列索引: j=4 (RRT), j=5 (Waypoint), j=6 (MAPS)
    # 对应 radius_values 索引: 0, 1, 2
    radius_col_map = {4: 0, 5: 1, 6: 2}

    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]

            if j == 0:  # Case名称列
                cell.set_facecolor('#D9E2F3')
            elif j in [1, 4, 7]:  # RRT列
                cell.set_facecolor('#DAE3F3')
            elif j in [2, 5, 8]:  # Waypoint列
                cell.set_facecolor('#E2F0D9')
            elif j in [3, 6]:  # MAPS列
                cell.set_facecolor('#FCE4D6')

            # 半径小于0.2时标红（四舍五入到4位小数后比较，与显示一致）
            if j in radius_col_map:
                radius_idx = radius_col_map[j]
                r_val = radius_values[i - 1][radius_idx]
                r_rounded = round(r_val, 4)
                if 0 < r_rounded < 0.2:
                    cell.set_text_props(color='red', weight='bold', fontsize=11)

            # 碰撞比例非零时标红
            # 碰撞列索引: j=7 (RRT), j=8 (Waypoint)
            collision_col_map = {7: 0, 8: 1}
            if j in collision_col_map:
                col_idx = collision_col_map[j]
                col_val = collision_values[i - 1][col_idx]
                if col_val > 0:
                    cell.set_text_props(color='red', weight='bold', fontsize=11)

    plt.title('Trajectory Metrics Comparison\nRRT_Results vs Waypoint_bezier vs MAPS',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n表格已保存: {output_path}")
    plt.close()


def print_summary(data: Dict):
    """
    打印汇总统计
    """
    if not data:
        return

    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    rrt_count = sum(1 for d in data.values() if 'rrt' in d)
    waypoint_count = sum(1 for d in data.values() if 'waypoint_bezier' in d)
    maps_count = sum(1 for d in data.values() if 'maps' in d)

    print(f"\n数据覆盖情况:")
    print(f"  RRT_Results:     {rrt_count}/{len(data)} 案例")
    print(f"  Waypoint_bezier: {waypoint_count}/{len(data)} 案例")
    print(f"  MAPS:            {maps_count}/{len(data)} 案例")

    # 计算平均值
    rrt_lengths = [d['rrt']['length'] for d in data.values() if 'rrt' in d and d['rrt']['length'] > 0]
    waypoint_lengths = [d['waypoint_bezier']['length'] for d in data.values()
                        if 'waypoint_bezier' in d and d['waypoint_bezier']['length'] > 0]
    maps_lengths = [d['maps']['length'] for d in data.values() if 'maps' in d and d['maps']['length'] > 0]

    print(f"\n平均轨迹长度:")
    if rrt_lengths:
        print(f"  RRT_Results:     {np.mean(rrt_lengths):.4f} m")
    if waypoint_lengths:
        print(f"  Waypoint_bezier: {np.mean(waypoint_lengths):.4f} m")
    if maps_lengths:
        print(f"  MAPS:            {np.mean(maps_lengths):.4f} m")


def main():
    """主函数"""
    print("=" * 70)
    print("轨迹指标对比分析")
    print("数据源: RRT_Results, Waypoint_bezier, MAPS")
    print("=" * 70)

    # 收集数据
    print("\n收集各案例指标数据...")
    data = collect_all_metrics()

    if not data:
        print("\n未找到数据，请检查目录结构。")
        return

    # 输出目录
    output_dir = Path(r'd:\Data_visualization_code\result')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建对比表格
    print("\n创建对比表格...")
    create_comparison_table(data, output_dir / 'trajectory_comparison_table.png')

    # 打印汇总
    print_summary(data)

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()