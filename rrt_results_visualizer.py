"""
RRT结果可视化工具
从RRT_Results_Tmech文件夹加载数据并绘制轨迹图

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
# 核心配置
# ============================================================================

class Config:
    """配置参数"""
    # 机器人运动学约束
    WHEEL_V_MAX = 0.035  # m/s 单轮最大线速度
    WHEELBASE = 0.053    # m 轮距
    OMEGA_MAX = 0.35     # rad/s 最大角速度

    # 碰撞检测参数
    ROBOT_RADIUS = 0.033  # m 机器人半径（e-puck2）
    SAFETY_MARGIN = 0.02  # m 安全边距
    OBSTACLE_EXPANSION = ROBOT_RADIUS + SAFETY_MARGIN  # 0.053m 障碍物膨胀距离


# ============================================================================
# 数据加载
# ============================================================================

def load_rrt_results(rrt_dir: Path) -> Dict:
    """
    从RRT结果目录加载数据

    Args:
        rrt_dir: RRT结果目录路径 (例如 RRT_Results_Tmech/MAP1)

    Returns:
        {
            'map_name': str,
            'waypoints': dict,
            'trajectory': dict
        }
    """
    map_name = rrt_dir.name

    # 加载航点文件
    waypoint_files = list(rrt_dir.glob('rrt_waypoints_*.json'))
    if not waypoint_files:
        raise FileNotFoundError(f"未找到航点文件在 {rrt_dir}")

    with open(waypoint_files[0], 'r', encoding='utf-8') as f:
        waypoints_data = json.load(f)

    # 加载贝塞尔轨迹文件
    trajectory_files = list(rrt_dir.glob('bezier_trajectory_*.json'))
    if not trajectory_files:
        raise FileNotFoundError(f"未找到轨迹文件在 {rrt_dir}")

    with open(trajectory_files[0], 'r', encoding='utf-8') as f:
        trajectory_data = json.load(f)

    return {
        'map_name': map_name,
        'waypoints': waypoints_data,
        'trajectory': trajectory_data
    }


def load_environment(base_dir: Path, map_name: str) -> Dict:
    """
    加载环境数据（障碍物和边界）

    Args:
        base_dir: 基础目录 (result文件夹)
        map_name: 地图名称 (例如 MAP1)

    Returns:
        环境数据字典
    """
    # 在MAPS文件夹中查找环境文件
    maps_dir = base_dir / 'MAPS' / map_name

    if not maps_dir.exists():
        # 尝试直接在result下查找
        maps_dir = base_dir / map_name

    if not maps_dir.exists():
        raise FileNotFoundError(f"未找到地图目录: {maps_dir}")

    # 查找环境文件
    env_files = list(maps_dir.glob('environment*.json')) + list(maps_dir.glob('Environment*.json'))

    if not env_files:
        raise FileNotFoundError(f"未找到环境文件在 {maps_dir}")

    with open(env_files[0], 'r', encoding='utf-8') as f:
        env_data = json.load(f)

    print(f"  加载环境文件: {env_files[0].name}")

    return env_data


# ============================================================================
# 碰撞检测
# ============================================================================

def prepare_collision_detection(env_data: Dict, pixel_to_meter: float) -> Dict:
    """
    准备碰撞检测所需的障碍物和边界数据

    Args:
        env_data: 环境数据
        pixel_to_meter: 像素转米比例

    Returns:
        {
            'obstacles_union': 膨胀后的障碍物联合体,
            'obstacles_original': 原始障碍物联合体,
            'bounds': [x_min, x_max, y_min, y_max] 边界
        }
    """
    # 构建障碍物多边形
    obstacles = []
    for poly_data in env_data['polygons']:
        vertices = np.array(poly_data['vertices']) * pixel_to_meter
        if len(vertices) >= 3:
            poly = Polygon(vertices)
            if poly.is_valid:
                obstacles.append(poly)

    # 合并所有障碍物
    if obstacles:
        obstacles_original = unary_union(obstacles)
        # 膨胀障碍物（增加安全边距）
        obstacles_expanded = obstacles_original.buffer(Config.OBSTACLE_EXPANSION)
    else:
        obstacles_original = None
        obstacles_expanded = None

    # 获取边界
    bounds = [b * pixel_to_meter for b in env_data['coord_bounds']]

    return {
        'obstacles_union': obstacles_expanded,
        'obstacles_original': obstacles_original,
        'bounds': bounds
    }


def check_path_collision(path: np.ndarray,
                         obstacles_union,
                         bounds: List[float]) -> Tuple[bool, List[int], Dict]:
    """
    检查路径是否与障碍物或边界碰撞

    Args:
        path: 路径 shape=(N, 2)
        obstacles_union: 膨胀后的障碍物联合体
        bounds: [x_min, x_max, y_min, y_max] 边界

    Returns:
        (是否碰撞, 碰撞点索引列表, 碰撞统计信息)
    """
    collision_indices = []
    boundary_collisions = 0
    obstacle_collisions = 0

    # 安全距离：使用0.05m（与障碍物膨胀距离一致）
    safety_distance = Config.OBSTACLE_EXPANSION

    for i, point in enumerate(path):
        x, y = point[0], point[1]

        # 检查边界碰撞
        if (x < bounds[0] + safety_distance or x > bounds[1] - safety_distance or
            y < bounds[2] + safety_distance or y > bounds[3] - safety_distance):
            collision_indices.append(i)
            boundary_collisions += 1
            continue

        # 检查障碍物碰撞（直接contains，因为obstacles_union已经膨胀过了）
        if obstacles_union is not None:
            p = Point(x, y)
            if obstacles_union.contains(p):
                collision_indices.append(i)
                obstacle_collisions += 1

    has_collision = len(collision_indices) > 0

    collision_info = {
        'has_collision': has_collision,
        'total_collision_points': len(collision_indices),
        'boundary_collisions': boundary_collisions,
        'obstacle_collisions': obstacle_collisions,
        'collision_ratio': len(collision_indices) / len(path) if len(path) > 0 else 0.0
    }

    return has_collision, collision_indices, collision_info


# ============================================================================
# 可视化
# ============================================================================

def plot_rrt_trajectory(data: Dict, env: Dict, save_path: str):
    """
    绘制RRT轨迹综合图：轨迹+速度剖面

    左侧：轨迹图（航点、贝塞尔曲线、障碍物、速度编码）
    右侧：线速度、角速度、轮子速度剖面

    Args:
        data: RRT结果数据
        env: 环境数据
        save_path: 保存路径
    """
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(14, 9))

    # 创建子图布局: 左侧轨迹图，右侧3个速度剖面
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.25)

    ax_traj = fig.add_subplot(gs[:, 0])    # 左侧：轨迹图
    ax_v = fig.add_subplot(gs[0, 1])       # 右上：线速度
    ax_omega = fig.add_subplot(gs[1, 1])   # 右中：角速度
    ax_wheel = fig.add_subplot(gs[2, 1])   # 右下：轮子速度

    # 提取数据
    waypoints_data = data['waypoints']
    trajectory_data = data['trajectory']
    map_name = data['map_name']

    waypoints = np.array(waypoints_data['waypoints'])
    start_pose = waypoints_data['start_pose']
    goal_pose = waypoints_data['goal_pose']

    # smooth_path 在顶层，不在 trajectory 嵌套对象中
    smooth_path = np.array(trajectory_data['smooth_path'])

    # velocities 等数据在 trajectory 嵌套对象中
    traj_nested = trajectory_data['trajectory']
    velocities = np.array(traj_nested['velocities'])
    wheel_velocities = np.array(traj_nested.get('wheel_velocities', []))
    timestamps = np.array(traj_nested['timestamps'])

    v_vals = velocities[:, 0]
    omega_vals = velocities[:, 1]

    # 碰撞信息 - 从JSON读取（用于终端显示）
    collision_info_json = trajectory_data.get('collision_info', {})

    # ==================== 左侧：轨迹图 ====================
    # 1. 绘制障碍物
    pixel_to_meter = env.get('pixel_to_meter_scale', 0.0023)
    for poly_data in env['polygons']:
        vertices = np.array(poly_data['vertices']) * pixel_to_meter
        poly = Polygon(vertices)
        x, y = poly.exterior.xy
        ax_traj.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)

    # 1.1 准备碰撞检测并重新计算碰撞点索引
    collision_data = prepare_collision_detection(env, pixel_to_meter)
    has_collision, collision_indices, collision_info = check_path_collision(
        smooth_path,
        collision_data['obstacles_union'],
        collision_data['bounds']
    )

    # 1.5 添加最小转弯半径信息到图例（放在第一位）
    # min_turn_radius 在嵌套的 trajectory 对象中，如果没有则从 max_curvature 计算
    min_radius = traj_nested.get('min_turn_radius', None)
    if min_radius is None:
        # 从 max_curvature 计算 min_turn_radius
        max_curvature = traj_nested.get('max_curvature', 0)
        if max_curvature > 1e-6:
            min_radius = 1.0 / max_curvature
        else:
            min_radius = float('inf')

    if min_radius != float('inf'):
        ax_traj.plot([], [], ' ', label=f'Min R = {min_radius:.4f} m')

    # 2. 绘制RRT航点序列（橙色标记）
    ax_traj.plot(waypoints[:, 0], waypoints[:, 1], 'o--',
                color='orange', linewidth=2, markersize=6,
                label=f'RRT Waypoints ({len(waypoints)})', zorder=4)

    # 3. 绘制贝塞尔曲线（速度编码）
    if smooth_path is not None and len(smooth_path) > 0:
        points = smooth_path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='jet', linewidths=3)
        lc.set_array(v_vals[:-1])
        lc.set_clim(0, Config.WHEEL_V_MAX)
        line = ax_traj.add_collection(lc)

        # 颜色条
        cbar = fig.colorbar(line, ax=ax_traj, shrink=0.6, pad=0.08)
        cbar.set_label('Velocity (m/s)', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # 标记碰撞点（如果有）
        if has_collision and collision_indices:
            collision_points = smooth_path[collision_indices]
            ax_traj.scatter(collision_points[:, 0], collision_points[:, 1],
                           c='red', s=20, alpha=0.6, marker='x',
                           label=f'Collision ({len(collision_indices)})', zorder=15)

    # 4. 起点终点
    ax_traj.plot(start_pose[0], start_pose[1], 'go', markersize=15,
                label='Start', zorder=10)
    ax_traj.plot(goal_pose[0], goal_pose[1], 'r*', markersize=20,
                label='Goal', zorder=10)

    # 5. 边界设置
    bounds = [b * pixel_to_meter for b in env['coord_bounds']]
    ax_traj.set_xlim(bounds[0], bounds[1])
    ax_traj.set_ylim(bounds[2], bounds[3])

    ax_traj.set_xlabel('X (m)', fontsize=16)
    ax_traj.set_ylabel('Y (m)', fontsize=16)
    ax_traj.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', fontsize=12, framealpha=0.9, ncol=2)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal')
    ax_traj.tick_params(labelsize=16)

    # ==================== 右上：线速度剖面 ====================
    ax_v.plot(timestamps, v_vals, 'b-', linewidth=2, label='v(t)')
    ax_v.axhline(y=Config.WHEEL_V_MAX, color='r', linestyle='--', linewidth=2,
                 label=f'V_MAX={Config.WHEEL_V_MAX}')
    ax_v.fill_between(timestamps, 0, v_vals, alpha=0.3)
    ax_v.set_xlabel('Time (s)', fontsize=14)
    ax_v.set_ylabel('v (m/s)', fontsize=14)
    ax_v.legend(loc='upper right', fontsize=12)
    ax_v.grid(True, alpha=0.3)
    ax_v.set_ylim(0, Config.WHEEL_V_MAX * 1.15)
    ax_v.tick_params(labelsize=14)

    # ==================== 右中：角速度剖面 ====================
    ax_omega.plot(timestamps, omega_vals, 'g-', linewidth=2, label='ω(t)')
    ax_omega.axhline(y=Config.OMEGA_MAX, color='r', linestyle='--', linewidth=2, label=f'±ω_MAX')
    ax_omega.axhline(y=-Config.OMEGA_MAX, color='r', linestyle='--', linewidth=2)
    ax_omega.fill_between(timestamps, 0, omega_vals, alpha=0.3, color='green')
    ax_omega.set_xlabel('Time (s)', fontsize=14)
    ax_omega.set_ylabel('ω (rad/s)', fontsize=14)
    ax_omega.legend(loc='upper right', fontsize=12)
    ax_omega.grid(True, alpha=0.3)
    ax_omega.set_ylim(-Config.OMEGA_MAX * 1.2, Config.OMEGA_MAX * 1.2)
    ax_omega.tick_params(labelsize=14)

    # ==================== 右下：轮子速度剖面 ====================
    if len(wheel_velocities) > 0:
        v_L = wheel_velocities[:, 0]
        v_R = wheel_velocities[:, 1]
        ax_wheel.plot(timestamps, v_L, 'b-', linewidth=2, label='v_L (left)')
        ax_wheel.plot(timestamps, v_R, 'r-', linewidth=2, label='v_R (right)')
        ax_wheel.axhline(y=Config.WHEEL_V_MAX, color='k', linestyle='--', linewidth=2,
                        label=f'±V_wheel_MAX')
        ax_wheel.axhline(y=-Config.WHEEL_V_MAX, color='k', linestyle='--', linewidth=2)
        ax_wheel.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_wheel.set_xlabel('Time (s)', fontsize=14)
    ax_wheel.set_ylabel('v_wheel (m/s)', fontsize=14)
    ax_wheel.legend(loc='upper right', fontsize=12)
    ax_wheel.grid(True, alpha=0.3)
    ax_wheel.set_ylim(-Config.WHEEL_V_MAX * 1.2, Config.WHEEL_V_MAX * 1.2)
    ax_wheel.tick_params(labelsize=14)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] RRT轨迹图保存: {save_path}")


# ============================================================================
# 主处理流程
# ============================================================================

def process_rrt_results(rrt_base_dir: Path, result_base_dir: Path, output_dir: Path):
    """
    处理RRT_Results_Tmech目录下的所有案例

    Args:
        rrt_base_dir: RRT结果基础目录 (RRT_Results_Tmech)
        result_base_dir: result目录
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print(f"处理RRT结果: {rrt_base_dir}")
    print(f"{'='*70}\n")

    # 查找所有案例目录
    case_dirs = [d for d in rrt_base_dir.iterdir() if d.is_dir()]

    if not case_dirs:
        print(f"未找到任何案例目录在 {rrt_base_dir}")
        return

    print(f"发现 {len(case_dirs)} 个案例:")
    for case_dir in sorted(case_dirs):
        print(f"  - {case_dir.name}")
    print()

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个案例
    success_count = 0
    failed_count = 0

    for case_dir in sorted(case_dirs):
        case_name = case_dir.name
        print(f"\n处理案例: {case_name}")
        print(f"{'='*50}")

        try:
            # 1. 加载RRT结果
            print("  [1/3] 加载RRT结果数据...")
            rrt_data = load_rrt_results(case_dir)

            # 2. 加载环境数据
            print("  [2/3] 加载环境数据...")
            env_data = load_environment(result_base_dir, case_name)

            # 3. 绘制可视化
            print("  [3/3] 绘制可视化...")
            output_filename = f"rrt_trajectory_{case_name}.png"
            output_path = output_dir / output_filename

            plot_rrt_trajectory(rrt_data, env_data, str(output_path))

            # 显示统计信息
            trajectory_data = rrt_data['trajectory']
            traj_nested = trajectory_data['trajectory']
            collision_info = trajectory_data.get('collision_info', {})

            # 计算 min_turn_radius（如果不存在则从 max_curvature 计算）
            min_turn_radius = traj_nested.get('min_turn_radius', None)
            if min_turn_radius is None:
                max_curvature = traj_nested.get('max_curvature', 0)
                if max_curvature > 1e-6:
                    min_turn_radius = 1.0 / max_curvature
                else:
                    min_turn_radius = float('inf')

            print(f"\n  统计信息:")
            print(f"    完成时间: {traj_nested.get('total_time', 0):.2f} s")
            print(f"    总距离: {traj_nested.get('total_distance', 0):.4f} m")
            print(f"    最大速度: v={traj_nested.get('max_v', 0):.4f} m/s, ω={traj_nested.get('max_omega', 0):.4f} rad/s")
            print(f"    最小转弯半径: R={min_turn_radius:.4f} m" if min_turn_radius != float('inf') else "    最小转弯半径: R=inf m")

            # 显示碰撞信息
            if collision_info.get('has_collision', False):
                print(f"\n  ⚠ 碰撞警告:")
                print(f"    碰撞点数: {collision_info.get('total_collision_points', 0)}/{len(rrt_data['trajectory']['smooth_path'])} ({collision_info.get('collision_ratio', 0):.1%})")
                print(f"    边界碰撞: {collision_info.get('boundary_collisions', 0)} 点")
                print(f"    障碍物碰撞: {collision_info.get('obstacle_collisions', 0)} 点")
            else:
                print(f"    碰撞检测: ✓ 无碰撞")

            success_count += 1

        except Exception as e:
            print(f"\n  [ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue

    # 总结
    print(f"\n{'='*70}")
    print(f"处理完成!")
    print(f"  成功: {success_count}/{len(case_dirs)}")
    print(f"  失败: {failed_count}/{len(case_dirs)}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*70}\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 设置路径
    base_dir = Path(r"d:\Data_visualization_code")
    result_dir = base_dir / "result"
    rrt_results_dir = result_dir / "RRT_Results_Tmech"
    output_dir = result_dir / "RRT_Visualizations"

    # 检查RRT结果目录是否存在
    if not rrt_results_dir.exists():
        print(f"错误: RRT结果目录不存在: {rrt_results_dir}")
        return

    # 处理所有RRT结果
    process_rrt_results(rrt_results_dir, result_dir, output_dir)


if __name__ == '__main__':
    main()