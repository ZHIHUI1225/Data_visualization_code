"""
四种轨迹规划方法对比分析
对比 A*, Waypoints (Bezier), Dubins, 和 Proposed (MAPS) 四种方法的轨迹指标：
- 轨迹长度 (Trajectory Length)
- 最小转弯半径 (Minimum Radius)
- 是否有碰撞 (Has Collision)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
PIXEL_TO_METER = 0.0023
OBSTACLE_EXPANSION = 0.03  # 与规划器一致


def compute_curvature(path: np.ndarray) -> np.ndarray:
    """
    计算路径每点的曲率
    使用三点法: κ = 2*sin(α) / |AB|
    """
    if len(path) < 3:
        return np.zeros(len(path))
    
    curvatures = np.zeros(len(path))
    
    for i in range(1, len(path) - 1):
        p1 = path[i-1]
        p2 = path[i]
        p3 = path[i+1]
        
        # 向量 AB 和 BC
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算长度
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 1e-6 or len_v2 < 1e-6:
            curvatures[i] = 0
            continue
        
        # 计算叉积
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # 弦长
        chord_length = np.linalg.norm(p3 - p1)
        if chord_length < 1e-10:
            curvatures[i] = 0
            continue
        
        # sin(α) = |cross| / (len_v1 * len_v2)
        denom = len_v1 * len_v2
        if abs(denom) < 1e-10:
            curvatures[i] = 0
            continue
        
        sin_alpha = abs(cross) / denom
        
        # 曲率公式: κ = 2*sin(α) / |chord|
        curvatures[i] = 2 * sin_alpha / chord_length
        if cross < 0:
            curvatures[i] = -curvatures[i]
    
    # 端点曲率使用相邻点
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    
    return curvatures


def load_environment(env_file: Path, pixel_to_meter: float) -> Dict:
    """加载环境数据"""
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            env_data = json.load(f)
        
        # 构建障碍物多边形
        obstacles = []
        for poly_data in env_data['polygons']:
            vertices = np.array(poly_data['vertices']) * pixel_to_meter
            if len(vertices) >= 3:
                poly = Polygon(vertices)
                if poly.is_valid:
                    obstacles.append(poly)
        
        # 获取边界
        bounds = [b * pixel_to_meter for b in env_data['coord_bounds']]
        
        return {
            'polygons': obstacles,
            'bounds': bounds
        }
    except Exception as e:
        return None


def check_path_collision(path: np.ndarray, obstacles: List, bounds: List[float]) -> bool:
    """检查路径是否与障碍物或边界碰撞"""
    if len(path) == 0:
        return False
    
    # 合并障碍物
    if obstacles:
        obstacles_union = unary_union(obstacles)
        obstacles_expanded = obstacles_union.buffer(OBSTACLE_EXPANSION)
    else:
        obstacles_expanded = None
    
    safety_distance = OBSTACLE_EXPANSION
    
    for point in path:
        x, y = point[0], point[1]
        
        # 检查边界碰撞
        if (x < bounds[0] + safety_distance or x > bounds[1] - safety_distance or
            y < bounds[2] + safety_distance or y > bounds[3] - safety_distance):
            return True
        
        # 检查障碍物碰撞
        if obstacles_expanded is not None:
            p = Point(x, y)
            if obstacles_expanded.contains(p):
                return True
    
    return False


def extract_dubins_metrics(case_name: str, dubins_base: Path) -> Optional[Dict]:
    """从RRT_Dubins_Results的quality report中提取指标"""
    report_file = dubins_base / "trajectory_quality_report.json"
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                reports = json.load(f)
            
            for report in reports:
                if report.get('map_name') == case_name:
                    if report.get('status') == 'SUCCESS':
                        return {
                            'length': report.get('trajectory_length', 0),
                            'min_radius': report.get('min_curvature_radius', 0),
                            'has_collision': report.get('has_collision', False),
                            'source': f"quality_report:{case_name}"
                        }
                    else:
                        return {
                            'length': 0,
                            'min_radius': 0,
                            'has_collision': None,
                            'source': f"FAILED:{case_name}"
                        }
        except Exception as e:
            print(f"  [WARN] Error reading quality report: {e}")
    return None


def extract_astar_metrics(case_name: str, astar_base: Path, maps_base: Path) -> Optional[Dict]:
    """从Astar_Results中提取路径长度、最小转弯半径和碰撞检测"""
    astar_file = astar_base / case_name / f"astar_path_{case_name}.json"
    if not astar_file.exists():
        return None
    
    try:
        with open(astar_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        waypoints = data.get('waypoints', [])
        smooth_path = np.array(data.get('smooth_path', []))
        
        if len(waypoints) < 2:
            return None
        
        # 计算路径长度 (从 waypoints)
        total_length = 0
        for i in range(len(waypoints) - 1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i + 1])
            total_length += np.linalg.norm(p2 - p1)
        
        # 计算最小转弯半径 (从 smooth_path，如果存在)
        min_radius = None
        if len(smooth_path) > 2:
            curvatures = compute_curvature(smooth_path)
            max_curvature = float(np.max(np.abs(curvatures)))
            if max_curvature > 1e-6:
                min_radius = 1.0 / max_curvature
        
        # 碰撞检测 (从 smooth_path)
        has_collision = None
        if len(smooth_path) > 0:
            # 查找环境文件
            env_file = maps_base / case_name / f'environment_{case_name.lower()}.json'
            if not env_file.exists():
                # 尝试其他文件名
                alt_env_files = list((maps_base / case_name).glob('environment*.json'))
                if alt_env_files:
                    env_file = alt_env_files[0]
                else:
                    env_file = None
            
            if env_file and env_file.exists():
                env_data = load_environment(env_file, PIXEL_TO_METER)
                if env_data:
                    has_collision = check_path_collision(
                        smooth_path, 
                        env_data['polygons'], 
                        env_data['bounds']
                    )
        
        return {
            'length': total_length,
            'min_radius': min_radius,
            'has_collision': has_collision,
            'source': str(astar_file)
        }
    except Exception as e:
        print(f"  [WARN] Error reading Astar file: {e}")
        return None


def extract_waypoint_bezier_metrics(case_name: str, bezier_base: Path) -> Optional[Dict]:
    """从Waypoint_bezier的summary文件中提取指标"""
    case_dir = bezier_base / case_name
    if not case_dir.exists():
        return None
    
    summary_file = case_dir / "bezier_curves_summary.json"
    if not summary_file.exists():
        return None
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sequences = data.get('sequences', [])
        if not sequences:
            return None
        
        seq = sequences[0]
        trajectory = seq.get('trajectory', {})
        collision_info = seq.get('collision_info', {})
        
        total_distance = trajectory.get('total_distance', 0)
        max_curvature = trajectory.get('max_curvature', 0)
        min_radius = 1.0 / max_curvature if max_curvature > 1e-6 else float('inf')
        has_collision = collision_info.get('has_collision', False)
        
        return {
            'length': total_distance,
            'min_radius': min_radius,
            'has_collision': has_collision,
            'source': str(summary_file)
        }
    except Exception as e:
        print(f"  [WARN] Error reading Bezier file: {e}")
        return None


def find_maps_trajectory_file(case_name: str, maps_base: Path) -> Optional[Path]:
    """查找MAPS目录下的轨迹文件"""
    case_lower = case_name.lower()
    case_dir = maps_base / case_name
    
    if not case_dir.exists():
        return None
    
    pattern = f"**/complete_trajectory_parameters_{case_lower}.json"
    matches = list(case_dir.glob(pattern))
    
    if matches:
        return matches[0]
    
    pattern2 = f"**/complete_trajectory_parameters_*.json"
    matches2 = list(case_dir.glob(pattern2))
    
    if matches2:
        return matches2[0]
    
    return None


def extract_maps_metrics(case_name: str, maps_base: Path) -> Optional[Dict]:
    """从MAPS结果文件中提取指标"""
    maps_file = find_maps_trajectory_file(case_name, maps_base)
    if not maps_file:
        return None
    
    try:
        with open(maps_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        l_array = data.get('l', [])
        total_length = sum(l_array)
        
        r0_array = data.get('r0', [])
        min_radius = min([abs(r) for r in r0_array if abs(r) > 1e-6]) if r0_array else float('inf')
        
        return {
            'length': total_length,
            'min_radius': min_radius,
            'has_collision': False,
            'source': str(maps_file)
        }
    except Exception as e:
        print(f"  [WARN] Error reading MAPS file: {e}")
        return None


def collect_all_metrics() -> Dict:
    """收集所有案例的指标数据"""
    base_dir = Path(r'd:\Data_visualization_code\result')
    dubins_base = base_dir / 'RRT_Dubins_Results'
    astar_base = base_dir / 'Astar_Results'
    bezier_base = base_dir / 'Waypoint_bezier'
    maps_base = base_dir / 'MAPS'
    
    results = {}
    all_cases = set()
    
    if dubins_base.exists():
        all_cases.update([d.name for d in dubins_base.iterdir() if d.is_dir()])
    if astar_base.exists():
        all_cases.update([d.name for d in astar_base.iterdir() if d.is_dir()])
    if bezier_base.exists():
        all_cases.update([d.name for d in bezier_base.iterdir() if d.is_dir()])
    if maps_base.exists():
        all_cases.update([d.name for d in maps_base.iterdir() if d.is_dir()])
    
    exclude_dirs = {'MAP5_2', 'warehouse_N', '__pycache__', 
                    'separate_plots', 'velocity_plots', 'video', 'bezier_results'}
    all_cases = [c for c in all_cases if c not in exclude_dirs]
    
    # 只保留指定的案例
    selected_cases = ['MAP5', 'MAP2', 'MAP3', 'MAP4', 'bottleneck_expansion', 'maze']
    all_cases = [c for c in all_cases if c in selected_cases]
    all_cases = sorted(all_cases)
    
    print(f"\n发现 {len(all_cases)} 个案例: {all_cases}\n")
    
    for case in all_cases:
        print(f"处理案例: {case}")
        case_results = {}
        
        # A*
        astar_metrics = extract_astar_metrics(case, astar_base, maps_base)
        if astar_metrics:
            case_results['astar'] = astar_metrics
            min_r_str = f", min_r={astar_metrics['min_radius']:.4f}" if astar_metrics['min_radius'] else ""
            collision_str = f", collision={astar_metrics['has_collision']}" if astar_metrics['has_collision'] is not None else ""
            print(f"  ✓ A*: length={astar_metrics['length']:.4f}{min_r_str}{collision_str}")
        else:
            print(f"  ✗ A*: 无数据")
        
        # Waypoints (Bezier)
        bezier_metrics = extract_waypoint_bezier_metrics(case, bezier_base)
        if bezier_metrics:
            case_results['bezier'] = bezier_metrics
            print(f"  ✓ Waypoints: length={bezier_metrics['length']:.4f}")
        else:
            print(f"  ✗ Waypoints: 无数据")
        
        # Dubins
        dubins_metrics = extract_dubins_metrics(case, dubins_base)
        if dubins_metrics:
            case_results['dubins'] = dubins_metrics
            status = "FAILED" if 'FAILED' in dubins_metrics.get('source', '') else f"length={dubins_metrics['length']:.4f}"
            print(f"  ✓ Dubins: {status}")
        else:
            print(f"  ✗ Dubins: 无数据")
        
        # Proposed (MAPS)
        maps_metrics = extract_maps_metrics(case, maps_base)
        if maps_metrics:
            case_results['maps'] = maps_metrics
            print(f"  ✓ Proposed: length={maps_metrics['length']:.4f}")
        else:
            print(f"  ✗ Proposed: 无数据")
        
        if case_results:
            results[case] = case_results
    
    return results


def create_comparison_table(data: Dict, output_path: Path):
    """创建对比表格"""
    if not data:
        print("没有数据可显示")
        return
    
    cases = sorted(data.keys())
    
    # 表头
    headers = [
        'Case',
        'A*\nLength(m)', 'Waypoints\nLength(m)', 'Dubins\nLength(m)', 'Proposed\nLength(m)',
        'A*\nMin R(m)', 'Waypoints\nMin R(m)', 'Dubins\nMin R(m)', 'Proposed\nMin R(m)',
        'A*\nCollision', 'Waypoints\nCollision', 'Dubins\nCollision', 'Proposed\nCollision'
    ]
    
    table_data = []
    radius_values = []
    collision_values = []
    
    for case in cases:
        case_data = data[case]
        
        astar = case_data.get('astar', {})
        bezier = case_data.get('bezier', {})
        dubins = case_data.get('dubins', {})
        maps = case_data.get('maps', {})
        
        row = [case]
        
        # 轨迹长度
        astar_len = astar.get('length', 0)
        bezier_len = bezier.get('length', 0)
        dubins_len = dubins.get('length', 0)
        maps_len = maps.get('length', 0)
        
        dubins_failed = 'FAILED' in dubins.get('source', '')
        
        row.append(f"{astar_len:.4f}" if astar_len > 0 else "-")
        row.append(f"{bezier_len:.4f}" if bezier_len > 0 else "-")
        
        if dubins_failed:
            row.append("FAILED")
        elif dubins_len > 0:
            row.append(f"{dubins_len:.4f}")
        else:
            row.append("-")
        
        row.append(f"{maps_len:.4f}" if maps_len > 0 else "-")
        
        # 最小转弯半径
        astar_r = astar.get('min_radius')
        bezier_r = bezier.get('min_radius', 0)
        dubins_r = dubins.get('min_radius', 0)
        maps_r = maps.get('min_radius', 0)
        
        row.append(f"{astar_r:.4f}" if astar_r and astar_r < float('inf') else "-")
        row.append(f"{bezier_r:.4f}" if bezier_r and bezier_r < float('inf') else "-")
        
        if dubins_failed:
            row.append("FAILED")
        else:
            row.append(f"{dubins_r:.4f}" if dubins_r and dubins_r < float('inf') else "-")
        
        row.append(f"{maps_r:.4f}" if maps_r and maps_r < float('inf') else "-")
        
        radius_values.append([
            astar_r if astar_r and astar_r < float('inf') else None,
            bezier_r if bezier_r and bezier_r < float('inf') else None,
            dubins_r if dubins_r and dubins_r < float('inf') else None,
            maps_r if maps_r and maps_r < float('inf') else None
        ])
        
        # 碰撞状态
        astar_col = astar.get('has_collision')
        bezier_col = bezier.get('has_collision')
        dubins_col = dubins.get('has_collision')
        maps_col = maps.get('has_collision')
        
        row.append("Yes" if astar_col else ("No" if astar_col is False else "-"))
        row.append("Yes" if bezier_col else ("No" if bezier_col is False else "-"))
        
        if dubins_failed:
            row.append("FAILED")
        else:
            row.append("Yes" if dubins_col else ("No" if dubins_col is False else "-"))
        
        row.append("Yes" if maps_col else ("No" if maps_col is False else "-"))
        
        collision_values.append([astar_col, bezier_col, dubins_col, maps_col])
        
        table_data.append(row)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(22, len(cases) * 0.7 + 3))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.06] + [0.07]*4 + [0.07]*4 + [0.08]*4
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.5)
    
    # 表头样式
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 数据行样式
    # 半径列索引: j=5 (A*), j=6 (Waypoints), j=7 (Dubins), j=8 (Proposed)
    # 碰撞列索引: j=9 (A*), j=10 (Waypoints), j=11 (Dubins), j=12 (Proposed)
    radius_col_map = {5: 0, 6: 1, 7: 2, 8: 3}
    collision_col_map = {9: 0, 10: 1, 11: 2, 12: 3}
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            
            # 斑马条纹
            if i % 2 == 0:
                cell.set_facecolor('#E9EDF4')
            else:
                cell.set_facecolor('white')
            
            # 高亮小于0.2m的半径
            if j in radius_col_map:
                r_idx = radius_col_map[j]
                r_val = radius_values[i-1][r_idx]
                if r_val is not None and r_val < 0.2:
                    cell.set_facecolor('#FFC7CE')
                    cell.set_text_props(color='#9C0006', weight='bold')
            
            # 高亮有碰撞的情况
            if j in collision_col_map:
                col_idx = collision_col_map[j]
                has_col = collision_values[i-1][col_idx]
                if has_col is True:
                    cell.set_facecolor('#FFC7CE')
                    cell.set_text_props(color='#9C0006', weight='bold')
                elif has_col is False:
                    cell.set_facecolor('#C6EFCE')
                    cell.set_text_props(color='#006100')
    
    plt.title('Trajectory Planning Methods Comparison\n'
              'A* vs Waypoints (Bezier) vs Dubins vs Proposed Method (MAPS)',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n表格已保存: {output_path}")
    plt.close()


def print_summary(data: Dict):
    """打印汇总统计"""
    if not data:
        return
    
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    
    astar_count = sum(1 for d in data.values() if 'astar' in d)
    bezier_count = sum(1 for d in data.values() if 'bezier' in d)
    dubins_count = sum(1 for d in data.values() if 'dubins' in d)
    maps_count = sum(1 for d in data.values() if 'maps' in d)
    
    print(f"\n数据覆盖情况:")
    print(f"  A*:              {astar_count}/{len(data)} 案例")
    print(f"  Waypoints:       {bezier_count}/{len(data)} 案例")
    print(f"  Dubins:          {dubins_count}/{len(data)} 案例")
    print(f"  Proposed (MAPS): {maps_count}/{len(data)} 案例")
    
    # 计算平均值
    astar_lengths = [d['astar']['length'] for d in data.values() 
                     if 'astar' in d and d['astar']['length'] > 0]
    bezier_lengths = [d['bezier']['length'] for d in data.values() 
                      if 'bezier' in d and d['bezier']['length'] > 0]
    dubins_lengths = [d['dubins']['length'] for d in data.values() 
                      if 'dubins' in d and d['dubins']['length'] > 0 and 'FAILED' not in d['dubins'].get('source', '')]
    maps_lengths = [d['maps']['length'] for d in data.values() 
                    if 'maps' in d and d['maps']['length'] > 0]
    
    print(f"\n平均轨迹长度:")
    if astar_lengths:
        print(f"  A*:        {np.mean(astar_lengths):.4f} m")
    if bezier_lengths:
        print(f"  Waypoints: {np.mean(bezier_lengths):.4f} m")
    if dubins_lengths:
        print(f"  Dubins:    {np.mean(dubins_lengths):.4f} m")
    if maps_lengths:
        print(f"  Proposed:  {np.mean(maps_lengths):.4f} m")
    
    # 计算平均最小转弯半径
    astar_radii = [d['astar']['min_radius'] for d in data.values() 
                   if 'astar' in d and d['astar'].get('min_radius') and d['astar']['min_radius'] < float('inf')]
    bezier_radii = [d['bezier']['min_radius'] for d in data.values() 
                    if 'bezier' in d and d['bezier'].get('min_radius') and d['bezier']['min_radius'] < float('inf')]
    dubins_radii = [d['dubins']['min_radius'] for d in data.values() 
                    if 'dubins' in d and d['dubins'].get('min_radius') and d['dubins']['min_radius'] < float('inf') 
                    and 'FAILED' not in d['dubins'].get('source', '')]
    maps_radii = [d['maps']['min_radius'] for d in data.values() 
                  if 'maps' in d and d['maps'].get('min_radius') and d['maps']['min_radius'] < float('inf')]
    
    print(f"\n平均最小转弯半径:")
    if astar_radii:
        print(f"  A*:        {np.mean(astar_radii):.4f} m")
    else:
        print(f"  A*:        N/A")
    if bezier_radii:
        print(f"  Waypoints: {np.mean(bezier_radii):.4f} m")
    if dubins_radii:
        print(f"  Dubins:    {np.mean(dubins_radii):.4f} m")
    if maps_radii:
        print(f"  Proposed:  {np.mean(maps_radii):.4f} m")
    
    # 碰撞统计
    astar_collisions = sum(1 for d in data.values() 
                           if 'astar' in d and d['astar'].get('has_collision') is True)
    astar_collision_checked = sum(1 for d in data.values() 
                                  if 'astar' in d and d['astar'].get('has_collision') is not None)
    bezier_collisions = sum(1 for d in data.values() 
                            if 'bezier' in d and d['bezier'].get('has_collision') is True)
    dubins_collisions = sum(1 for d in data.values() 
                            if 'dubins' in d and d['dubins'].get('has_collision') is True)
    maps_collisions = sum(1 for d in data.values() 
                          if 'maps' in d and d['maps'].get('has_collision') is True)
    
    print(f"\n碰撞案例数:")
    print(f"  A*:        {astar_collisions}/{astar_collision_checked}")
    print(f"  Waypoints: {bezier_collisions}/{bezier_count}")
    print(f"  Dubins:    {dubins_collisions}/{dubins_count}")
    print(f"  Proposed:  {maps_collisions}/{maps_count}")


def main():
    """主函数"""
    print("=" * 70)
    print("四种轨迹规划方法对比分析")
    print("数据源: Astar_Results, Waypoint_bezier, RRT_Dubins_Results, MAPS")
    print("=" * 70)
    
    # 收集数据
    print("\n收集各案例指标数据...")
    data = collect_all_metrics()
    
    if not data:
        print("\n未找到任何数据!")
        return
    
    # 输出目录
    output_dir = Path(r'd:\Data_visualization_code\result')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建对比表格
    print("\n创建对比表格...")
    create_comparison_table(data, output_dir / 'four_methods_comparison_table.png')
    
    # 打印汇总
    print_summary(data)
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
