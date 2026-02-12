"""
贝塞尔曲线航点可视化工具
基于航点(Waypoints)和中继点(Relay Points)生成平滑贝塞尔曲线

数据结构简洁，代码直截了当 - Linus式设计
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from typing import List, Tuple, Dict, Optional


# ============================================================================
# 核心配置
# ============================================================================

class Config:
    """配置参数"""
    PIXEL_TO_METER = 0.0023
    PATH_RESOLUTION = 0.003  # 路径点间距 (米)
    BEZIER_SMOOTHNESS = 0  # B样条平滑参数 (0=通过所有点)

    # 机器人运动学约束 - 差速驱动（与rrt_bezier_planner.py保持一致）
    WHEEL_V_MAX = 0.035  # m/s 单轮最大线速度
    WHEELBASE = 0.053    # m 轮距
    V_MAX = 0.035        # m/s 最大线速度（直线时）
    OMEGA_MAX = 0.35     # rad/s 最大角速度
    V_MIN = 0.0
    DT = 0.1             # s 默认时间步长

    # 碰撞检测参数（与rrt_bezier_planner.py保持一致）
    ROBOT_RADIUS = 0.0265  # e-puck2 半径 (米)
    SAFETY_MARGIN = 0.01  # 额外安全边距 (米)
    OBSTACLE_EXPANSION = 0.046  # 障碍物膨胀距离 (米)


# ============================================================================
# 数据加载
# ============================================================================

def load_waypoints_data(map_dir: str) -> Dict:
    """
    加载地图目录下的所有相关数据

    Args:
        map_dir: 地图目录路径

    Returns:
        {
            'relay_points': {...},  # 中继点数据
            'waypoint_sequence': [...],  # 航点序列
            'environment': {...},  # 环境数据
            'all_points': {...}  # id->坐标映射
        }
    """
    map_path = Path(map_dir)
    map_name = map_path.name

    # 1. 查找并加载图节点（包含所有节点坐标）
    # 优先使用Graph_new_*.json，否则使用Graph_*.json
    graph_file = None
    for pattern in [f'Graph_new_{map_name.lower()}.json', f'Graph_new_*.json',
                    f'Graph_{map_name.lower()}.json', 'Graph_*.json']:
        matches = list(map_path.glob(pattern))
        if matches:
            graph_file = matches[0]
            break

    if not graph_file:
        raise FileNotFoundError(f"未找到Graph文件在 {map_dir}")

    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    print(f"  使用图文件: {graph_file.name}")

    # 2. 加载中继点（特殊标记点）- 可选
    relay_data = None
    relay_file = map_path / 'relay_points.json'
    if relay_file.exists():
        with open(relay_file, 'r', encoding='utf-8') as f:
            relay_data = json.load(f)
        print(f"  加载中继点: relay_points.json")
    else:
        print(f"  [INFO] 未找到relay_points.json，跳过")

    # 3. 加载航点序列
    waypoint_files = sorted(map_path.glob('WayPointFlag*.json'))
    if not waypoint_files:
        raise FileNotFoundError(f"未找到WayPointFlag*.json文件在 {map_dir}")

    waypoint_sequences = []
    for wf in waypoint_files:
        with open(wf, 'r', encoding='utf-8') as f:
            waypoint_sequences.append({
                'file': wf.name,
                'data': json.load(f)
            })

    # 4. 查找并加载环境文件
    env_file = None
    for pattern in [f'environment_{map_name.lower()}.json', 'environment_*.json', 'environment.json']:
        matches = list(map_path.glob(pattern))
        if matches:
            env_file = matches[0]
            break

    if not env_file:
        raise FileNotFoundError(f"未找到environment文件在 {map_dir}")

    with open(env_file, 'r', encoding='utf-8') as f:
        env_data = json.load(f)

    print(f"  使用环境文件: {env_file.name}")

    # 5. 构建id->坐标映射（从Graph节点 + 特殊起点终点）
    all_points = {}

    # 从Graph加载所有节点
    for node in graph_data['nodes']:
        node_id = node[0]
        pixel_coord = node[1]
        meter_coord = [
            pixel_coord[0] * Config.PIXEL_TO_METER,
            pixel_coord[1] * Config.PIXEL_TO_METER
        ]
        all_points[node_id] = {
            'pixel': pixel_coord,
            'meter': meter_coord,
            'is_graph_node': True,
            'flag': 0
        }

    # 添加特殊起点和终点（如果它们不在Graph节点中）
    # 从environment或Graph获取起点终点
    start_pose = graph_data.get('start_pose', env_data.get('start_pose', [120, 480, 0]))
    goal_pose = graph_data.get('goal_pose', env_data.get('goal_pose', [1000, 150, 0]))

    # 检查relay_points中是否有标记为起点/终点的特殊ID
    if relay_data and 'relay_points' in relay_data:
        for point in relay_data['relay_points']:
            if point.get('is_first', False):
                # 起点
                start_id = point['id']
                all_points[start_id] = {
                    'pixel': start_pose[:2],
                    'meter': [start_pose[0] * Config.PIXEL_TO_METER,
                             start_pose[1] * Config.PIXEL_TO_METER],
                    'is_start': True,
                    'flag': point.get('flag', 0)
                }
            if point.get('is_last', False):
                # 终点
                goal_id = point['id']
                all_points[goal_id] = {
                    'pixel': goal_pose[:2],
                    'meter': [goal_pose[0] * Config.PIXEL_TO_METER,
                             goal_pose[1] * Config.PIXEL_TO_METER],
                    'is_goal': True,
                    'flag': point.get('flag', 0)
                }

    # 获取像素转米比例
    pixel_to_meter = Config.PIXEL_TO_METER
    if relay_data and 'pixel_to_meter_scale' in relay_data:
        pixel_to_meter = relay_data['pixel_to_meter_scale']

    return {
        'graph': graph_data,
        'relay_points': relay_data,
        'waypoint_sequences': waypoint_sequences,
        'environment': env_data,
        'all_points': all_points,
        'pixel_to_meter': pixel_to_meter,
        'map_name': map_name
    }


def extract_waypoint_coordinates(waypoint_ids: List[int],
                                 all_points: Dict) -> Tuple[np.ndarray, List[int]]:
    """
    提取航点序列的坐标（米）

    Args:
        waypoint_ids: 航点ID列表 [11, 1, 2, 3, ...]
        all_points: id->坐标映射

    Returns:
        (坐标数组 shape=(N,2), 有效的ID列表)
    """
    coordinates = []
    valid_ids = []

    for point_id in waypoint_ids:
        if point_id in all_points:
            coordinates.append(all_points[point_id]['meter'])
            valid_ids.append(point_id)
        else:
            print(f"  [WARN] 航点ID {point_id} 未在relay_points中找到，已跳过")

    return np.array(coordinates), valid_ids


# ============================================================================
# 贝塞尔曲线生成
# ============================================================================

def generate_bezier_curve(waypoints: np.ndarray,
                          resolution: float = Config.PATH_RESOLUTION) -> np.ndarray:
    """
    使用B样条生成平滑贝塞尔曲线

    Args:
        waypoints: 航点坐标 shape=(N, 2)
        resolution: 路径点间距 (米)

    Returns:
        平滑路径 shape=(M, 2)
    """
    if len(waypoints) < 2:
        return waypoints

    if len(waypoints) == 2:
        # 两点直接线性插值
        total_length = np.linalg.norm(waypoints[1] - waypoints[0])
        num_points = max(int(total_length / resolution), 2)
        t = np.linspace(0, 1, num_points)
        smooth_path = waypoints[0] + t[:, np.newaxis] * (waypoints[1] - waypoints[0])
        return smooth_path

    # 使用B样条插值
    try:
        # k=3表示3次B样条，s=0表示通过所有点
        k_degree = min(3, len(waypoints) - 1)
        tck, u = splprep([waypoints[:, 0], waypoints[:, 1]],
                        s=Config.BEZIER_SMOOTHNESS, k=k_degree)

        # 计算路径总长度
        total_length = 0
        for i in range(len(waypoints) - 1):
            total_length += np.linalg.norm(waypoints[i+1] - waypoints[i])

        # 根据分辨率确定采样点数
        num_points = max(int(total_length / resolution), len(waypoints) * 10)
        u_new = np.linspace(0, 1, num_points)

        # 插值生成平滑路径
        smooth_x, smooth_y = splev(u_new, tck)
        smooth_path = np.column_stack([smooth_x, smooth_y])

        return smooth_path

    except Exception as e:
        print(f"  [WARN] B样条插值失败: {e}，返回原航点")
        return waypoints


# ============================================================================
# 曲率和速度剖面计算
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
        p0, p1, p2 = path[i-1], path[i], path[i+1]

        v1 = p1 - p0
        v2 = p2 - p1

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 < 1e-6 or len_v2 < 1e-6:
            curvatures[i] = 0
            continue

        # 计算转角
        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        # 曲率
        chord_length = np.linalg.norm(p2 - p0)
        if chord_length > 1e-6:
            curvatures[i] = 2 * np.sin(angle / 2) / chord_length
        else:
            curvatures[i] = 0

    # 端点曲率使用相邻点
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]

    return curvatures


def generate_velocity_profile(path: np.ndarray, verbose: bool = False) -> Dict:
    """
    生成速度剖面，通过缩放确保满足差速驱动运动学约束

    差速驱动运动学：
    - 左轮: v_L = v - ω * L/2
    - 右轮: v_R = v + ω * L/2
    - 约束: |v_L| ≤ WHEEL_V_MAX, |v_R| ≤ WHEEL_V_MAX

    组合约束（转弯时外侧轮速度最高）：
    - v + |ω| * L/2 ≤ WHEEL_V_MAX
    - 由于 ω = v * κ: v * (1 + |κ| * L/2) ≤ WHEEL_V_MAX
    - 所以: v ≤ WHEEL_V_MAX / (1 + |κ| * L/2)

    Args:
        path: 平滑路径 shape=(N, 2)
        verbose: 是否打印详细缩放信息

    Returns:
        {
            'positions': [[x, y, theta], ...],
            'velocities': [[v, omega], ...],
            'wheel_velocities': [[v_L, v_R], ...],  # 左右轮速度
            'timestamps': [t, ...],
            'curvatures': [κ, ...],
            'total_time': float,
            'total_distance': float,
            'max_v': float,
            'max_omega': float,
            'max_wheel_v': float,  # 最大轮子速度
            'max_curvature': float,
            'min_turn_radius': float,
            'scale_info': dict
        }
    """
    n_points = len(path)
    L = Config.WHEELBASE  # 轮距

    # 计算朝向角
    thetas = np.zeros(n_points)
    for i in range(n_points - 1):
        dx = path[i+1, 0] - path[i, 0]
        dy = path[i+1, 1] - path[i, 1]
        thetas[i] = np.arctan2(dy, dx)
    thetas[-1] = thetas[-2]

    # 计算曲率
    curvatures = compute_curvature(path)
    max_curvature = float(np.max(np.abs(curvatures)))

    # ========================================
    # 核心：差速驱动约束下的速度计算
    # ========================================
    velocities = np.zeros((n_points, 2))       # [v, omega]
    wheel_velocities = np.zeros((n_points, 2)) # [v_L, v_R]
    scale_factors = np.zeros(n_points)

    for i in range(n_points):
        kappa = abs(curvatures[i])

        # 计算满足所有约束的最大允许速度
        # 约束1: v ≤ WHEEL_V_MAX (直线时最大速度)
        v_limit_1 = Config.WHEEL_V_MAX

        # 约束2: |ω| ≤ OMEGA_MAX → v ≤ OMEGA_MAX / |κ|
        if kappa > 1e-6:
            v_limit_2 = Config.OMEGA_MAX / kappa
        else:
            v_limit_2 = float('inf')

        # 约束3: 差速约束 v + |ω| * L/2 ≤ WHEEL_V_MAX
        #         → v * (1 + |κ| * L/2) ≤ WHEEL_V_MAX
        #         → v ≤ WHEEL_V_MAX / (1 + |κ| * L/2)
        v_limit_3 = Config.WHEEL_V_MAX / (1 + kappa * L / 2)

        # 取最严格的约束
        v_allowed = min(v_limit_1, v_limit_2, v_limit_3)
        v = max(v_allowed, Config.V_MIN)

        # 计算角速度
        omega = v * curvatures[i]  # 保留符号

        # 计算轮子速度
        v_L = v - omega * L / 2
        v_R = v + omega * L / 2

        velocities[i] = [v, omega]
        wheel_velocities[i] = [v_L, v_R]
        scale_factors[i] = v / Config.WHEEL_V_MAX if Config.WHEEL_V_MAX > 0 else 1.0

    # 计算时间戳
    timestamps = [0.0]
    total_distance = 0.0

    for i in range(1, n_points):
        segment_length = np.linalg.norm(path[i] - path[i-1])
        avg_velocity = (velocities[i-1, 0] + velocities[i, 0]) / 2

        if avg_velocity > 1e-6:
            dt = segment_length / avg_velocity
        else:
            dt = Config.DT

        timestamps.append(timestamps[-1] + dt)
        total_distance += segment_length

    # 构建位置+朝向
    positions = np.column_stack([path, thetas])

    # 统计实际最大值
    max_v = float(np.max(velocities[:, 0]))
    max_omega = float(np.max(np.abs(velocities[:, 1])))
    max_wheel_v = float(np.max(np.abs(wheel_velocities)))
    min_scale = float(np.min(scale_factors))
    avg_scale = float(np.mean(scale_factors))

    # 计算最小转弯半径
    min_turn_radius = 1.0 / max_curvature if max_curvature > 1e-6 else float('inf')

    # 缩放信息
    scale_info = {
        'min_scale_factor': min_scale,
        'avg_scale_factor': avg_scale,
        'scaled_points_ratio': float(np.sum(scale_factors < 0.99) / n_points),
        'max_curvature_point': int(np.argmax(np.abs(curvatures))),
        'min_velocity_point': int(np.argmin(velocities[:, 0]))
    }

    if verbose:
        print(f"    [差速约束] 轮距L={L}m, 单轮最大速度={Config.WHEEL_V_MAX}m/s")
        print(f"    [缩放信息] 最小缩放因子: {min_scale:.3f}, "
              f"平均缩放因子: {avg_scale:.3f}, "
              f"被缩放点比例: {scale_info['scaled_points_ratio']:.1%}")
        print(f"    [轮速范围] v_L: [{np.min(wheel_velocities[:,0]):.4f}, {np.max(wheel_velocities[:,0]):.4f}] m/s")
        print(f"    [轮速范围] v_R: [{np.min(wheel_velocities[:,1]):.4f}, {np.max(wheel_velocities[:,1]):.4f}] m/s")

    return {
        'positions': positions.tolist(),
        'velocities': velocities.tolist(),
        'wheel_velocities': wheel_velocities.tolist(),
        'timestamps': timestamps,
        'curvatures': curvatures.tolist(),
        'total_time': float(timestamps[-1]),
        'total_distance': float(total_distance),
        'max_curvature': max_curvature,
        'min_turn_radius': min_turn_radius,
        'max_v': max_v,
        'max_omega': max_omega,
        'max_wheel_v': max_wheel_v,
        'scale_info': scale_info
    }


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
    bounds = [b * pixel_to_meter for b in env_data['coord_bounds']]  # [x_min, x_max, y_min, y_max]

    return {
        'obstacles_union': obstacles_expanded,
        'obstacles_original': obstacles_original,
        'bounds': bounds
    }


def check_path_collision(path: np.ndarray,
                         obstacles_union,
                         bounds: List[float]) -> Tuple[bool, List[int], Dict]:
    """
    检查贝塞尔路径是否与障碍物或边界碰撞

    Args:
        path: 贝塞尔路径 shape=(N, 2)
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


def check_kinematics_constraints(trajectory: Dict) -> Tuple[bool, str]:
    """
    检查轨迹是否满足差速驱动运动学约束

    约束：
    1. v ≤ WHEEL_V_MAX
    2. |ω| ≤ OMEGA_MAX
    3. |v_L|, |v_R| ≤ WHEEL_V_MAX (差速约束)

    Returns:
        (satisfied: bool, message: str)
    """
    velocities = np.array(trajectory['velocities'])
    wheel_velocities = np.array(trajectory.get('wheel_velocities', []))

    v_vals = velocities[:, 0]
    omega_vals = velocities[:, 1]

    v_max_actual = np.max(v_vals)
    omega_max_actual = np.max(np.abs(omega_vals))

    v_satisfied = v_max_actual <= Config.WHEEL_V_MAX + 1e-6
    omega_satisfied = omega_max_actual <= Config.OMEGA_MAX + 1e-6

    # 检查轮子速度约束
    if len(wheel_velocities) > 0:
        wheel_v_max = np.max(np.abs(wheel_velocities))
        wheel_satisfied = wheel_v_max <= Config.WHEEL_V_MAX + 1e-6
    else:
        wheel_v_max = 0
        wheel_satisfied = True

    satisfied = v_satisfied and omega_satisfied and wheel_satisfied

    message = (f"v_max={v_max_actual:.4f} (限制{Config.WHEEL_V_MAX}), "
               f"|ω|_max={omega_max_actual:.4f} (限制{Config.OMEGA_MAX}), "
               f"|v_wheel|_max={wheel_v_max:.4f} (限制{Config.WHEEL_V_MAX})")

    if satisfied:
        message = "[OK] " + message
    else:
        message = "[FAIL] " + message

    # 转换numpy.bool_为Python原生bool（JSON序列化需要）
    return bool(satisfied), message


# ============================================================================
# 可视化
# ============================================================================

def plot_trajectory_combined(data: Dict, waypoint_seq_data: Dict,
                             smooth_path: np.ndarray, waypoint_ids: List[int],
                             trajectory: Dict, save_path: str,
                             collision_info: Optional[Dict] = None,
                             collision_indices: Optional[List[int]] = None):
    """
    绘制综合轨迹图：轨迹+速度剖面（合并版）

    左侧：轨迹图（航点、贝塞尔曲线、障碍物、速度编码）
    右侧：线速度、角速度、轮子速度剖面

    Args:
        data: 完整数据字典
        waypoint_seq_data: 具体的航点序列数据
        smooth_path: 平滑路径
        waypoint_ids: 有效的航点ID列表
        trajectory: 轨迹数据（包含速度信息）
        save_path: 保存路径
        collision_info: 碰撞信息字典（可选）
        collision_indices: 碰撞点索引列表（可选）
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

    env = data['environment']
    all_points = data['all_points']
    pixel_to_meter = data['pixel_to_meter']

    velocities = np.array(trajectory['velocities'])
    wheel_velocities = np.array(trajectory.get('wheel_velocities', []))
    timestamps = np.array(trajectory['timestamps'])
    v_vals = velocities[:, 0]
    omega_vals = velocities[:, 1]

    # ==================== 左侧：轨迹图 ====================
    # 1. 绘制障碍物
    for poly_data in env['polygons']:
        vertices = np.array(poly_data['vertices']) * pixel_to_meter
        poly = Polygon(vertices)
        x, y = poly.exterior.xy
        ax_traj.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)

    # 1.5 添加最小转弯半径信息到图例（放在第一位）
    min_radius = trajectory.get('min_turn_radius', float('inf'))
    if min_radius != float('inf'):
        ax_traj.plot([], [], ' ', label=f'Min R = {min_radius:.4f} m')

    # 2. 绘制航点序列（橙色标记）
    waypoint_coords = np.array([all_points[pid]['meter'] for pid in waypoint_ids])
    ax_traj.plot(waypoint_coords[:, 0], waypoint_coords[:, 1], 'o--',
                color='orange', linewidth=2, markersize=6,
                label=f'Waypoints ({len(waypoint_ids)})', zorder=4)

    # 4. 绘制贝塞尔曲线（速度编码）
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

        # 标记碰撞点
        if collision_info and collision_info.get('has_collision') and collision_indices:
            collision_points = smooth_path[collision_indices]
            ax_traj.scatter(collision_points[:, 0], collision_points[:, 1],
                           c='red', s=20, alpha=0.6, marker='x',
                           label=f'Collision ({len(collision_indices)})', zorder=15)

    # 5. 起点终点
    start_id = waypoint_ids[0]
    end_id = waypoint_ids[-1]
    start_coord = all_points[start_id]['meter']
    end_coord = all_points[end_id]['meter']

    ax_traj.plot(start_coord[0], start_coord[1], 'go', markersize=15,
                label='Start', zorder=10)
    ax_traj.plot(end_coord[0], end_coord[1], 'r*', markersize=20,
                label='Goal', zorder=10)

    # 6. 边界设置
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

    print(f"    [OK] 综合轨迹图保存: {save_path}")


def plot_waypoints_bezier(data: Dict, waypoint_seq_data: Dict,
                          smooth_path: np.ndarray, waypoint_ids: List[int],
                          trajectory: Dict, save_path: str,
                          collision_info: Optional[Dict] = None,
                          collision_indices: Optional[List[int]] = None):
    """
    绘制航点和贝塞尔曲线（带速度热力图）

    Args:
        data: 完整数据字典
        waypoint_seq_data: 具体的航点序列数据
        smooth_path: 平滑路径
        waypoint_ids: 有效的航点ID列表
        trajectory: 轨迹数据（包含速度信息）
        save_path: 保存路径
        collision_info: 碰撞信息字典（可选）
        collision_indices: 碰撞点索引列表（可选）
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    env = data['environment']
    all_points = data['all_points']
    pixel_to_meter = data['pixel_to_meter']

    # 1. 绘制障碍物
    for poly_data in env['polygons']:
        vertices = np.array(poly_data['vertices']) * pixel_to_meter
        poly = Polygon(vertices)
        x, y = poly.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.4, edgecolor='black', linewidth=1.5)

    # 2. 绘制边界
    bounds = [b * pixel_to_meter for b in env['coord_bounds']]
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.add_patch(plt.Rectangle((bounds[0], bounds[2]),
                               bounds[1]-bounds[0], bounds[3]-bounds[2],
                               fill=False, edgecolor='black', linewidth=2))

    # 3. 绘制所有图节点（淡蓝色小点）
    all_relay_x = [pt['meter'][0] for pt in all_points.values()]
    all_relay_y = [pt['meter'][1] for pt in all_points.values()]
    ax.scatter(all_relay_x, all_relay_y, c='lightblue', s=30,
              alpha=0.4, label='Graph Nodes', zorder=2)

    # 4. 绘制航点序列（橙色标记）
    waypoint_coords = np.array([all_points[pid]['meter'] for pid in waypoint_ids])
    ax.plot(waypoint_coords[:, 0], waypoint_coords[:, 1], 'o--',
           color='orange', linewidth=2, markersize=8,
           label=f'Waypoints ({len(waypoint_ids)} points)', zorder=4)

    # 标注航点ID
    for i, point_id in enumerate(waypoint_ids):
        coord = all_points[point_id]['meter']
        ax.text(coord[0], coord[1] + 0.03, str(point_id),
               fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

    # 5. 绘���贝���尔曲线（速度热力图）
    velocities = np.array(trajectory['velocities'])
    v_vals = velocities[:, 0]  # 线速度

    # 使用速度作为颜色
    points = smooth_path.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='RdYlGn_r', linewidths=3)
    lc.set_array(v_vals[:-1])
    lc.set_clim(0, Config.V_MAX)
    line = ax.add_collection(lc)

    # 添加颜色条
    cbar = fig.colorbar(line, ax=ax, pad=0.08, label='Velocity (m/s)')

    # 5.1 标记碰撞点（如果有）
    if collision_info and collision_info.get('has_collision') and collision_indices:
        collision_points = smooth_path[collision_indices]
        ax.scatter(collision_points[:, 0], collision_points[:, 1],
                  c='red', s=20, alpha=0.6, marker='x',
                  label=f'Collision Points ({len(collision_indices)})', zorder=6)

    # 6. 标记起点和终点
    start_id = waypoint_ids[0]
    end_id = waypoint_ids[-1]
    start_coord = all_points[start_id]['meter']
    end_coord = all_points[end_id]['meter']

    ax.plot(start_coord[0], start_coord[1], 'g*', markersize=20,
           label=f'Start (ID={start_id})', zorder=5)
    ax.plot(end_coord[0], end_coord[1], 'r*', markersize=20,
           label=f'Goal (ID={end_id})', zorder=5)

    # 7. 统计信息
    path_length = trajectory['total_distance']
    straight_line_length = np.linalg.norm(
        np.array(end_coord) - np.array(start_coord)
    )
    min_radius = trajectory['min_turn_radius']
    max_curvature = trajectory['max_curvature']
    total_time = trajectory['total_time']

    # 8. 添加信息框（包含碰撞警告）
    info_text = (
        f"File: {waypoint_seq_data['file']}\n"
        f"Waypoints: {waypoint_ids}\n"
        f"Path Length: {path_length:.3f} m\n"
        f"Straight Dist: {straight_line_length:.3f} m\n"
        f"Path Ratio: {path_length/straight_line_length:.2f}x\n"
        f"\n"
        f"Max Curvature: κ={max_curvature:.4f} (1/m)\n"
        f"Min Turn Radius: R={min_radius:.4f} m\n"
        f"Total Time: {total_time:.2f} s\n"
        f"\n"
        f"v_max = {Config.V_MAX} m/s\n"
        f"ω_max = {Config.OMEGA_MAX} rad/s"
    )

    # 添加碰撞警告（如果有）
    if collision_info and collision_info.get('has_collision'):
        collision_text = (
            f"\n\n⚠ COLLISION WARNING ⚠\n"
            f"Collision Points: {collision_info['total_collision_points']}/{len(smooth_path)} "
            f"({collision_info['collision_ratio']:.1%})\n"
            f"Boundary: {collision_info['boundary_collisions']} pts\n"
            f"Obstacle: {collision_info['obstacle_collisions']} pts"
        )
        info_text += collision_text
        bbox_color = 'lightcoral'  # 红色背景表示碰撞
    else:
        bbox_color = 'wheat'  # 正常背景色

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8))

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Bezier Trajectory with Velocity Profile - {waypoint_seq_data["file"]}',
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 可视化保存: {save_path}")


# ============================================================================
# 主处理流程
# ============================================================================

def process_map_waypoints(map_dir: str, output_dir: str):
    """
    处理MAP5_2目录下的所有航点序列

    Args:
        map_dir: MAP5_2目录路径
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print(f"处理MAP: {map_dir}")
    print(f"{'='*70}\n")

    # 1. 加载数据
    print("[1/3] 加载数据...")
    data = load_waypoints_data(map_dir)
    Config.PIXEL_TO_METER = data['pixel_to_meter']

    print(f"  [OK] 中继点总数: {len(data['all_points'])}")
    print(f"  [OK] 航点序列文件数: {len(data['waypoint_sequences'])}")

    # 1.1. 准备碰撞检测数据
    print(f"\n  准备碰撞检测...")
    collision_data = prepare_collision_detection(data['environment'], data['pixel_to_meter'])
    print(f"  [OK] 障碍物已膨胀 {Config.OBSTACLE_EXPANSION}m, 安全距离: {Config.ROBOT_RADIUS + Config.SAFETY_MARGIN}m")

    # 2. 为每个航点序列生成贝塞尔曲线
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] 生成贝塞尔曲线...")

    all_results = []

    for idx, waypoint_seq in enumerate(data['waypoint_sequences'], 1):
        seq_name = waypoint_seq['file']
        seq_data = waypoint_seq['data']
        waypoint_ids = seq_data['Waypoints']

        print(f"\n  处理序列 {idx}/{len(data['waypoint_sequences'])}: {seq_name}")
        print(f"    航点ID序列: {waypoint_ids}")

        # 提取坐标
        waypoint_coords, valid_ids = extract_waypoint_coordinates(
            waypoint_ids, data['all_points']
        )

        if len(valid_ids) < 2:
            print(f"    [WARN] 有效航点少于2个，跳过")
            continue

        print(f"    有效航点: {len(valid_ids)}/{len(waypoint_ids)}")

        # 生成贝塞尔曲线
        smooth_path = generate_bezier_curve(waypoint_coords)
        print(f"    [OK] 贝塞尔点数: {len(smooth_path)}")

        # 生成速度剖面（带差速驱动约束）
        trajectory = generate_velocity_profile(smooth_path, verbose=True)

        # 检查运动学约束
        satisfied, constraint_msg = check_kinematics_constraints(trajectory)
        print(f"    {constraint_msg}")

        # 输出完成时间（重点突出）
        print(f"\n    {'='*40}")
        print(f"    轨迹完成时间: {trajectory['total_time']:.2f} 秒")
        print(f"    轨迹总距离: {trajectory['total_distance']:.4f} m")
        print(f"    实际最大速度: v_max={trajectory['max_v']:.4f} m/s, |ω|_max={trajectory['max_omega']:.4f} rad/s")
        print(f"    最大轮子速度: |v_wheel|_max={trajectory['max_wheel_v']:.4f} m/s")
        print(f"    {'='*40}")

        # 碰撞检测
        has_collision, collision_indices, collision_info = check_path_collision(
            smooth_path,
            collision_data['obstacles_union'],
            collision_data['bounds']
        )

        if has_collision:
            print(f"    [WARNING] 检测到路径碰撞!")
            print(f"      - 碰撞点数: {collision_info['total_collision_points']}/{len(smooth_path)} "
                  f"({collision_info['collision_ratio']:.1%})")
            print(f"      - 边界碰撞: {collision_info['boundary_collisions']} 点")
            print(f"      - 障碍物碰撞: {collision_info['obstacle_collisions']} 点")
        else:
            print(f"    [OK] 无碰撞")

        # 保存结果
        result = {
            'waypoint_file': seq_name,
            'waypoint_ids': valid_ids,
            'waypoint_coordinates_meter': waypoint_coords.tolist(),
            'bezier_curve_meter': smooth_path.tolist(),
            'trajectory': trajectory,
            'constraints': {
                'wheel_v_max': Config.WHEEL_V_MAX,
                'v_max': Config.V_MAX,
                'omega_max': Config.OMEGA_MAX,
                'v_min': Config.V_MIN,
                'wheelbase': Config.WHEELBASE,
                'satisfied': satisfied
            },
            'summary': {
                'total_time_seconds': trajectory['total_time'],
                'total_distance_meters': trajectory['total_distance'],
                'actual_max_v': trajectory['max_v'],
                'actual_max_omega': trajectory['max_omega'],
                'actual_max_wheel_v': trajectory['max_wheel_v'],
                'scale_info': trajectory['scale_info']
            },
            'num_waypoints': len(valid_ids),
            'num_bezier_points': len(smooth_path),
            'collision_info': collision_info
        }

        all_results.append(result)

        # 保存JSON
        json_filename = f"bezier_trajectory_{Path(seq_name).stem}.json"
        json_path = output_path / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"    [OK] JSON保存: {json_filename}")

        # 生成综合可视化（轨迹+速度剖面合并）
        vis_filename = f"trajectory_{Path(seq_name).stem}.png"
        vis_path = output_path / vis_filename
        plot_trajectory_combined(data, waypoint_seq, smooth_path, valid_ids, trajectory, str(vis_path),
                                collision_info=collision_info, collision_indices=collision_indices)

    # 3. 保存汇总结果
    print(f"\n[3/3] 保存汇总结果...")
    summary = {
        'map_name': data.get('map_name', 'unknown'),
        'total_sequences': len(all_results),
        'pixel_to_meter_scale': data['pixel_to_meter'],
        'sequences': all_results
    }

    summary_path = output_path / 'bezier_curves_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] 汇总保存: bezier_curves_summary.json")

    print(f"\n{'='*70}")
    print(f"处理完成! 输出目录: {output_dir}")
    print(f"  - 生成了 {len(all_results)} 个贝塞尔曲线")
    print(f"  - 保存了 {len(all_results)} 个可视化图像")
    print(f"{'='*70}\n")

    return len(all_results) > 0


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 批量处理MAPS目录下所有地图"""
    base_dir = Path(r"d:\Data_visualization_code\result")

    # 查找所有MAP目录
    map_dirs = []

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        # 情况1: 直接是MAP开头的目录，且包含WayPointFlag文件
        if item.name.startswith('MAP') and list(item.glob('WayPointFlag*.json')):
            map_dirs.append(item)

        # 情况2: MAPS目录，需要检查其子目录
        elif item.name == 'MAPS':
            for subitem in item.iterdir():
                if subitem.is_dir() and list(subitem.glob('WayPointFlag*.json')):
                    map_dirs.append(subitem)

    if not map_dirs:
        print(f"未找到任何MAP目录在 {base_dir}")
        return

    print(f"\n{'='*70}")
    print(f"发现 {len(map_dirs)} 个地图目录:")
    for map_dir in map_dirs:
        print(f"  - {map_dir.name}")
    print(f"{'='*70}\n")

    # 批量处理
    success_count = 0
    failed_count = 0

    for map_dir in map_dirs:
        # 输出到 Waypoint_bezier 目录下，按地图名称分子目录
        output_dir = base_dir / "Waypoint_bezier" / map_dir.name

        try:
            success = process_map_waypoints(str(map_dir), str(output_dir))
            if success:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"\n[ERROR] 处理 {map_dir.name} 时出错: {e}\n")
            failed_count += 1
            continue

    # 总结
    print(f"\n{'='*70}")
    print(f"批量处理完成!")
    print(f"  成功: {success_count}/{len(map_dirs)}")
    print(f"  失败: {failed_count}/{len(map_dirs)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()