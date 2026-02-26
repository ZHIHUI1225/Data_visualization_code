"""
A* 轨迹可视化工具
从 Astar_Results 文件夹加载数据并绘制轨迹图

只绘制轨迹图，使用与现有脚本相同的风格和大小
数据结构简洁，代码直截了当 - Linus式设计
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from matplotlib.collections import LineCollection
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 工具函数
# ============================================================================

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
        
        # 计算多边形面积 (叉积)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # 计算点积
        dot = np.dot(v1, v2)
        
        # 计算角度
        denom = len_v1 * len_v2
        if abs(denom) < 1e-10:
            curvatures[i] = 0
            continue
        
        # 曲率公式: κ = 2*sin(α) / |chord|
        # 弦长
        chord_length = np.linalg.norm(p3 - p1)
        if chord_length < 1e-10:
            curvatures[i] = 0
            continue
        
        # sin(α) = |cross| / (len_v1 * len_v2)
        sin_alpha = abs(cross) / denom
        
        # 保留符号（正负表示左右转）
        curvatures[i] = 2 * sin_alpha / chord_length
        if cross < 0:
            curvatures[i] = -curvatures[i]
    
    # 端点曲率使用相邻点
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    
    return curvatures


def check_path_collision(path: np.ndarray, obstacles: List, bounds: List[float]) -> Tuple[bool, List[int]]:
    """
    检查路径是否与障碍物或边界碰撞
    
    Args:
        path: 路径 shape=(N, 2)
        obstacles: 障碍物多边形列表
        bounds: [x_min, x_max, y_min, y_max] 边界
        
    Returns:
        (是否有碰撞, 碰撞点索引列表)
    """
    # 合并障碍物
    if obstacles:
        obstacles_union = unary_union(obstacles)
        # 膨胀障碍物（增加安全边距）
        obstacles_expanded = obstacles_union.buffer(Config.OBSTACLE_EXPANSION)
    else:
        obstacles_expanded = None
    
    safety_distance = Config.OBSTACLE_EXPANSION
    collision_indices = []
    
    for i, point in enumerate(path):
        x, y = point[0], point[1]
        
        # 检查边界碰撞
        if (x < bounds[0] + safety_distance or x > bounds[1] - safety_distance or
            y < bounds[2] + safety_distance or y > bounds[3] - safety_distance):
            collision_indices.append(i)
            continue
        
        # 检查障碍物碰撞
        if obstacles_expanded is not None:
            p = Point(x, y)
            if obstacles_expanded.contains(p):
                collision_indices.append(i)
    
    has_collision = len(collision_indices) > 0
    return has_collision, collision_indices


# ============================================================================
# 核心配置
# ============================================================================

class Config:
    """配置参数"""
    PIXEL_TO_METER = 0.0023
    
    # 碰撞检测参数（与 astar_bezier_planner.py 保持一致）
    ROBOT_RADIUS = 0.0265  # e-puck2 半径 (米)
    SAFETY_MARGIN = 0.01   # 额外安全边距 (米)
    OBSTACLE_EXPANSION = 0.03  # 障碍物膨胀距离 (米) - 与规划器一致


# ============================================================================
# 数据加载
# ============================================================================

def load_environment(env_file: Path, pixel_to_meter: float) -> Dict:
    """
    加载环境数据
    
    Args:
        env_file: 环境JSON文件路径
        pixel_to_meter: 像素转米比例
        
    Returns:
        环境数据字典
    """
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
        'bounds': bounds,
        'env_data': env_data
    }


def load_astar_path(result_file: Path) -> Dict:
    """
    加载 A* 路径数据
    
    Args:
        result_file: A* 结果JSON文件路径
        
    Returns:
        路径数据字典
    """
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


# ============================================================================
# 可视化
# ============================================================================

def plot_astar_trajectory(path_data: Dict, env_data: Dict, save_path: Path):
    """
    绘制 A* 轨迹图（单一图表）
    
    Args:
        path_data: A* 路径数据
        env_data: 环境数据
        save_path: 保存路径
    """
    # 创建图表 - 使用与 rrt_results_visualizer 左侧子图相同的比例
    # 综合图 (14, 9), 左侧占 1.5/(1.5+1)=0.6 宽度, 约 8.4英寸宽
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 提取数据
    waypoints = np.array(path_data.get('waypoints', []))
    smooth_path = np.array(path_data.get('smooth_path', []))
    start_pose = path_data.get('start_pose', [0, 0, 0])
    goal_pose = path_data.get('goal_pose', [1, 1, 0])
    map_name = path_data.get('map_name', 'Unknown')
    
    # 绘制障碍物
    for poly in env_data['polygons']:
        x, y = poly.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)
    
    # 计算最小转弯半径（如果有平滑路径）
    min_radius = None
    if len(smooth_path) > 0:
        curvatures = compute_curvature(smooth_path)
        max_curvature = float(np.max(np.abs(curvatures)))
        if max_curvature > 1e-6:
            min_radius = 1.0 / max_curvature
    
    # 添加最小转弯半径信息到图例（如果有）
    if min_radius is not None:
        ax.plot([], [], ' ', label=f'Min R = {min_radius:.4f} m')
    
    # 检查碰撞
    has_collision = False
    collision_indices = []
    if len(smooth_path) > 0:
        has_collision, collision_indices = check_path_collision(smooth_path, env_data['polygons'], env_data['bounds'])
    
    # 绘制平滑路径（如果存在）
    if len(smooth_path) > 0:
        # 转换collision_indices为集合以便快速查找
        collision_set = set(collision_indices)
        
        # 分段绘制：根据是否包含碰撞点决定线宽
        i = 0
        while i < len(smooth_path) - 1:
            # 检查当前线段是否涉及碰撞点
            segment_has_collision = (i in collision_set) or ((i + 1) in collision_set)
            
            # 找到相同状态的连续线段
            j = i + 1
            while j < len(smooth_path) - 1:
                next_has_collision = (j in collision_set) or ((j + 1) in collision_set)
                if next_has_collision != segment_has_collision:
                    break
                j += 1
            
            # 绘制这一段
            linewidth = 5 if segment_has_collision else 3
            if i == 0:
                # 第一段添加图例
                if segment_has_collision:
                    ax.plot(smooth_path[i:j+1, 0], smooth_path[i:j+1, 1], '-', color='darkred',
                            linewidth=linewidth, label='A* Path (Collision)', zorder=3)
                else:
                    ax.plot(smooth_path[i:j+1, 0], smooth_path[i:j+1, 1], '-', color='darkred',
                            linewidth=linewidth, label='A* Path', zorder=3)
            else:
                ax.plot(smooth_path[i:j+1, 0], smooth_path[i:j+1, 1], '-', color='darkred',
                        linewidth=linewidth, zorder=3)
            
            i = j
        
        # 如果有碰撞，标记碰撞点
        if collision_indices:
            collision_points = smooth_path[collision_indices]
            ax.scatter(collision_points[:, 0], collision_points[:, 1],
                      c='red', s=20, alpha=0.6, marker='x',
                      label=f'Collision ({len(collision_indices)})', zorder=15)
    
    # 绘制航点（如果存在）
    if len(waypoints) > 0:
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'o--',
                color='orange', linewidth=2, markersize=6,
                label=f'Waypoints ({len(waypoints)})', zorder=4)
    
    # 起点和终点
    ax.plot(start_pose[0], start_pose[1], 'go', markersize=15,
            label='Start', zorder=10)
    ax.plot(goal_pose[0], goal_pose[1], 'r*', markersize=20,
            label='Goal', zorder=10)
    
    # 边界设置
    bounds = env_data['bounds']
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    
    ax.set_xlabel('X (m)', fontsize=16)
    ax.set_ylabel('Y (m)', fontsize=16)
    ax.set_title(f'A* Path - {map_name}', fontsize=18, fontweight='bold')
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', fontsize=12, 
              framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=16)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 保存: {save_path.name}")


# ============================================================================
# 主处理流程
# ============================================================================

def process_all_maps(astar_results_dir: Path, maps_dir: Path, output_dir: Optional[Path] = None):
    """
    处理所有地图的 A* 结果
    
    Args:
        astar_results_dir: Astar_Results 文件夹路径
        maps_dir: MAPS 文件夹路径
        output_dir: 输出文件夹（如果为None，则保存到astar_results_dir）
    """
    if output_dir is None:
        output_dir = astar_results_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有地图文件夹
    map_folders = [d for d in astar_results_dir.iterdir() if d.is_dir()]
    
    print(f"找到 {len(map_folders)} 个地图文件夹")
    
    success_count = 0
    fail_count = 0
    
    for map_folder in sorted(map_folders):
        map_name = map_folder.name
        print(f"\n处理地图: {map_name}")
        
        try:
            # 查找 A* 路径文件
            astar_files = list(map_folder.glob(f'astar_path_{map_name}.json'))
            if not astar_files:
                print(f"  [SKIP] 未找到 A* 路径文件")
                fail_count += 1
                continue
            
            astar_file = astar_files[0]
            
            # 查找环境文件
            env_file = maps_dir / map_name / f'environment_{map_name.lower()}.json'
            if not env_file.exists():
                # 尝试其他文件名
                alt_env_files = list((maps_dir / map_name).glob('environment*.json'))
                if alt_env_files:
                    env_file = alt_env_files[0]
                else:
                    print(f"  [SKIP] 未找到环境文件: {env_file}")
                    fail_count += 1
                    continue
            
            # 加载数据
            print(f"  加载路径: {astar_file.name}")
            print(f"  加载环境: {env_file.name}")
            
            path_data = load_astar_path(astar_file)
            env_data = load_environment(env_file, Config.PIXEL_TO_METER)
            
            # 绘制轨迹
            output_path = output_dir / map_name / f'astar_trajectory_viz_{map_name}.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plot_astar_trajectory(path_data, env_data, output_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print(f"\n" + "="*60)
    print(f"处理完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"="*60)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent / 'result'
    astar_results_dir = base_dir / 'Astar_Results'
    maps_dir = base_dir / 'MAPS'
    
    if not astar_results_dir.exists():
        print(f"[ERROR] Astar_Results 文件夹不存在: {astar_results_dir}")
        return
    
    if not maps_dir.exists():
        print(f"[ERROR] MAPS 文件夹不存在: {maps_dir}")
        return
    
    print("="*60)
    print("A* 轨迹可视化工具")
    print("="*60)
    
    process_all_maps(astar_results_dir, maps_dir)


if __name__ == '__main__':
    main()
