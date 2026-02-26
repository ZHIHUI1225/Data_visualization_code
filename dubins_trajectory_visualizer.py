"""
RRT-Dubins 轨迹可视化工具
从 RRT_Dubins_Results 文件夹加载数据并绘制轨迹图

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

def check_path_collision(path: np.ndarray, obstacles: List, bounds: List[float]) -> Tuple[bool, List[int]]:
    """
    检查路径是否与障碍物或边界碰撞
    
    注意：障碍物应该已经膨胀过（在规划阶段），这里不再重复膨胀
    
    Args:
        path: 路径 shape=(N, 2)
        obstacles: 障碍物多边形列表（应该是已膨胀的）
        bounds: [x_min, x_max, y_min, y_max] 边界
        
    Returns:
        (是否有碰撞, 碰撞点索引列表)
    """
    # 合并障碍物（使用已膨胀的障碍物，不再重复膨胀）
    if obstacles:
        obstacles_union = unary_union(obstacles)
    else:
        obstacles_union = None
    
    safety_distance = Config.OBSTACLE_EXPANSION
    collision_indices = []
    
    for i, point in enumerate(path):
        x, y = point[0], point[1]
        
        # 检查边界碰撞
        if (x < bounds[0] + safety_distance or x > bounds[1] - safety_distance or
            y < bounds[2] + safety_distance or y > bounds[3] - safety_distance):
            collision_indices.append(i)
            continue
        
        # 检查障碍物碰撞（使用已膨胀的障碍物）
        if obstacles_union is not None:
            p = Point(x, y)
            if obstacles_union.contains(p):
                collision_indices.append(i)
    
    has_collision = len(collision_indices) > 0
    return has_collision, collision_indices


# ============================================================================
# 核心配置
# ============================================================================

class Config:
    """配置参数"""
    PIXEL_TO_METER = 0.0023
    
    # 机器人运动学约束
    WHEEL_V_MAX = 0.035  # m/s 单轮最大线速度
    WHEELBASE = 0.053    # m 轮距
    OMEGA_MAX = 0.35     # rad/s 最大角速度
    
    # 碰撞检测参数
    ROBOT_RADIUS = 0.033  # m 机器人半径（e-puck2）
    SAFETY_MARGIN = 0.02  # m 安全边距
    OBSTACLE_EXPANSION = ROBOT_RADIUS + SAFETY_MARGIN


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
        环境数据字典（包含原始和膨胀后的障碍物）
    """
    with open(env_file, 'r', encoding='utf-8') as f:
        env_data = json.load(f)
    
    # 构建原始障碍物多边形
    obstacles_original = []
    for poly_data in env_data['polygons']:
        vertices = np.array(poly_data['vertices']) * pixel_to_meter
        if len(vertices) >= 3:
            poly = Polygon(vertices)
            if poly.is_valid:
                obstacles_original.append(poly)
    
    # 膨胀障碍物（与规划器保持一致）
    obstacles_expanded = []
    if obstacles_original:
        obstacles_union = unary_union(obstacles_original)
        obstacles_union_expanded = obstacles_union.buffer(Config.OBSTACLE_EXPANSION)
        
        # 将膨胀后的union转换为多边形列表
        if obstacles_union_expanded.geom_type == 'Polygon':
            obstacles_expanded = [obstacles_union_expanded]
        elif obstacles_union_expanded.geom_type == 'MultiPolygon':
            obstacles_expanded = list(obstacles_union_expanded.geoms)
    
    # 获取边界
    bounds = [b * pixel_to_meter for b in env_data['coord_bounds']]
    
    return {
        'polygons': obstacles_original,          # 原始障碍物（用于绘图）
        'polygons_expanded': obstacles_expanded, # 膨胀障碍物（用于碰撞检测）
        'bounds': bounds,
        'env_data': env_data
    }


def load_dubins_trajectory(trajectory_file: Path) -> Dict:
    """
    加载 Dubins 轨迹数据
    
    Args:
        trajectory_file: Dubins 轨迹JSON文件路径
        
    Returns:
        轨迹数据字典
    """
    with open(trajectory_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_dubins_path(path_file: Path) -> Dict:
    """
    加载 Dubins 路径数据（包含segments信息）
    
    Args:
        path_file: Dubins 路径JSON文件路径
        
    Returns:
        路径数据字典
    """
    with open(path_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


# ============================================================================
# 可视化
# ============================================================================

def plot_dubins_trajectory(trajectory_data: Dict, path_data: Optional[Dict], 
                          env_data: Dict, save_path: Path):
    """
    绘制 Dubins 轨迹图（单一图表）
    
    Args:
        trajectory_data: Dubins 轨迹数据
        path_data: Dubins 路径数据（可选，包含segments信息）
        env_data: 环境数据
        save_path: 保存路径
    """
    # 创建图表 - 使用与 rrt_results_visualizer 左侧子图相同的比例
    # 综合图 (14, 9), 左侧占 1.5/(1.5+1)=0.6 宽度, 约 8.4英寸宽
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 提取数据
    full_path = np.array(trajectory_data.get('full_path', []))
    start_pose = trajectory_data.get('start_pose', [0, 0, 0])
    goal_pose = trajectory_data.get('goal_pose', [1, 1, 0])
    map_name = trajectory_data.get('map_name', 'Unknown')
    
    # 从path_data获取额外信息（如果有）
    dubins_min_radius = None
    if path_data:
        dubins_min_radius = path_data.get('dubins_min_radius', None)
    
    # 绘制障碍物（使用原始障碍物）
    for poly in env_data['polygons']:
        x, y = poly.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)
    
    # 添加最小转弯半径信息到图例（如果有）
    if dubins_min_radius is not None:
        ax.plot([], [], ' ', label=f'Min R = {dubins_min_radius:.4f} m')
    
    # 绘制 Dubins 轨迹
    if len(full_path) > 0:
        # 提取 x, y 坐标
        if full_path.shape[1] >= 2:
            trajectory_xy = full_path[:, :2]
        else:
            trajectory_xy = full_path
        
        # 检查碰撞（使用膨胀后的障碍物）
        has_collision, collision_indices = check_path_collision(
            trajectory_xy, 
            env_data['polygons_expanded'],  # 使用膨胀后的障碍物
            env_data['bounds']
        )
        
        # 转换collision_indices为集合以便快速查找
        collision_set = set(collision_indices)
        
        # 分段绘制：根据是否包含碰撞点决定线宽
        i = 0
        while i < len(trajectory_xy) - 1:
            # 检查当前线段是否涉及碰撞点
            segment_has_collision = (i in collision_set) or ((i + 1) in collision_set)
            
            # 找到相同状态的连续线段
            j = i + 1
            while j < len(trajectory_xy) - 1:
                next_has_collision = (j in collision_set) or ((j + 1) in collision_set)
                if next_has_collision != segment_has_collision:
                    break
                j += 1
            
            # 绘制这一段
            linewidth = 5 if segment_has_collision else 3
            if i == 0:
                # 第一段添加图例
                if segment_has_collision:
                    ax.plot(trajectory_xy[i:j+1, 0], trajectory_xy[i:j+1, 1], '-', color='darkred',
                            linewidth=linewidth, label='Dubins Path (Collision)', zorder=3)
                else:
                    ax.plot(trajectory_xy[i:j+1, 0], trajectory_xy[i:j+1, 1], '-', color='darkred',
                            linewidth=linewidth, label='Dubins Path', zorder=3)
            else:
                ax.plot(trajectory_xy[i:j+1, 0], trajectory_xy[i:j+1, 1], '-', color='darkred',
                        linewidth=linewidth, zorder=3)
            
            i = j
        
        # 如果有碰撞，标记碰撞点
        if collision_indices:
            collision_points = trajectory_xy[collision_indices]
            ax.scatter(collision_points[:, 0], collision_points[:, 1],
                      c='red', s=20, alpha=0.6, marker='x',
                      label=f'Collision ({len(collision_indices)})', zorder=15)
    
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
    ax.set_title(f'RRT-Dubins Path - {map_name}', fontsize=18, fontweight='bold')
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

def process_all_maps(dubins_results_dir: Path, maps_dir: Path, output_dir: Optional[Path] = None):
    """
    处理所有地图的 RRT-Dubins 结果
    
    Args:
        dubins_results_dir: RRT_Dubins_Results 文件夹路径
        maps_dir: MAPS 文件夹路径
        output_dir: 输出文件夹（如果为None，则保存到dubins_results_dir）
    """
    if output_dir is None:
        output_dir = dubins_results_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有地图文件夹
    map_folders = [d for d in dubins_results_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
    
    print(f"找到 {len(map_folders)} 个地图文件夹")
    
    success_count = 0
    fail_count = 0
    
    for map_folder in sorted(map_folders):
        map_name = map_folder.name
        print(f"\n处理地图: {map_name}")
        
        try:
            # 查找 Dubins 轨迹文件
            trajectory_files = list(map_folder.glob(f'dubins_trajectory_{map_name}.json'))
            if not trajectory_files:
                print(f"  [SKIP] 未找到 Dubins 轨迹文件")
                fail_count += 1
                continue
            
            trajectory_file = trajectory_files[0]
            
            # 查找 Dubins 路径文件（可选）
            path_files = list(map_folder.glob(f'dubins_path_{map_name}.json'))
            path_data = None
            if path_files:
                path_data = load_dubins_path(path_files[0])
                print(f"  加载路径: {path_files[0].name}")
            
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
            print(f"  加载轨迹: {trajectory_file.name}")
            print(f"  加载环境: {env_file.name}")
            
            trajectory_data = load_dubins_trajectory(trajectory_file)
            env_data = load_environment(env_file, Config.PIXEL_TO_METER)
            
            # 绘制轨迹
            output_path = output_dir / map_name / f'dubins_trajectory_viz_{map_name}.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            plot_dubins_trajectory(trajectory_data, path_data, env_data, output_path)
            
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
    dubins_results_dir = base_dir / 'RRT_Dubins_Results'
    maps_dir = base_dir / 'MAPS'
    
    if not dubins_results_dir.exists():
        print(f"[ERROR] RRT_Dubins_Results 文件夹不存在: {dubins_results_dir}")
        return
    
    if not maps_dir.exists():
        print(f"[ERROR] MAPS 文件夹不存在: {maps_dir}")
        return
    
    print("="*60)
    print("RRT-Dubins 轨迹可视化工具")
    print("="*60)
    
    process_all_maps(dubins_results_dir, maps_dir)


if __name__ == '__main__':
    main()
