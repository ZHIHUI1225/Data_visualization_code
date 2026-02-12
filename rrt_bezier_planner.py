"""
RRT + Bezier 路径规划系统
为每个MAP生成从起点到终点的平滑轨迹

作者: Linus式简洁设计
日期: 2025-01-07
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev
import yaml
from typing import List, Tuple, Dict, Optional
import time


# ============================================================================
# 配置参数
# ============================================================================

class Config:
    """系统配置参数 - 从config.yaml加载或使用默认值"""

    # 坐标转换
    PIXEL_TO_METER = 0.0023

    # 机器人运动学约束（差速驱动）
    WHEEL_V_MAX = 0.035  # m/s 单轮最大线速度
    WHEELBASE = 0.053    # m 轮距
    V_MAX = 0.035        # m/s 最大线速度（直线时）
    OMEGA_MAX = 0.35     # rad/s 最大角速度
    V_MIN = 0.0

    # RRT* 参数
    RRT_MAX_ITER = 8000  # 最大迭代次数（增加以应对狭窄通道）
    RRT_STEP_SIZE = 0.05  # 步长 (m) - 减小以通过warehouse等狭窄环境（通道宽0.089m）
    RRT_GOAL_SAMPLE_RATE = 0.20  # 目标采样率（略微提高以加速收敛）
    RRT_SEARCH_RADIUS = 0.15  # 重布线搜索半径 (m)
    RRT_GOAL_THRESHOLD = 0.05  # 到达目标的距离阈值 (m)

    # Bezier 平滑参数
    BEZIER_SHARP_ANGLE = 40.0  # 度，超过此角度认为是尖锐转角
    BEZIER_SMOOTH_FACTOR = 0.3  # 平滑因子
    PATH_RESOLUTION = 0.003  # 路径点间距 (m)

    # 时间离散化
    DT = 0.1  # 秒，时间步长

    # 安全距离
    ROBOT_RADIUS = 0.0265  # e-puck2 半径 (轮距的一半)
    SAFETY_MARGIN = 0.02  # 额外安全边距
    OBSTACLE_EXPANSION = 0.04  # 障碍物膨胀距离 (m)

    @classmethod
    def load_from_yaml(cls, config_path: str):
        """从config.yaml加载参数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                cls.PIXEL_TO_METER = config.get('pixel_to_meter_scale', cls.PIXEL_TO_METER)
                cls.V_MAX = config.get('linear_velocity_max', cls.V_MAX)
                cls.OMEGA_MAX = config.get('angular_velocity_max', cls.OMEGA_MAX)
                cls.ROBOT_RADIUS = config.get('wheelbase', 0.053) / 2.0
                print(f"[OK] 已加载配置: {config_path}")
        except Exception as e:
            print(f"[WARN] 配置文件加载失败，使用默认值: {e}")


# ============================================================================
# 环境加载模块
# ============================================================================

def load_environment(json_path: str) -> Dict:
    """
    加载环境文件并转换坐标系

    Args:
        json_path: environment_map*.json 文件路径

    Returns:
        {
            'obstacles': [Polygon, ...],  # shapely多边形列表
            'bounds': [x_min, x_max, y_min, y_max],  # 边界 (米)
            'start': [x, y, theta],  # 起点 (米, 弧度)
            'goal': [x, y, theta],  # 终点 (米, 弧度)
            'map_name': str
        }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取地图名称
    map_name = Path(json_path).parent.name

    # 转换障碍物坐标 (像素 → 米)
    obstacles = []
    for poly_data in data.get('polygons', []):
        vertices = np.array(poly_data['vertices']) * Config.PIXEL_TO_METER
        obstacles.append(Polygon(vertices))

    # 合并所有障碍物为单一几何对象（加速碰撞检测）
    if obstacles:
        obstacles_union = unary_union(obstacles)
        # 膨胀障碍物 (用于路径规划时的安全距离)
        obstacles_union_expanded = obstacles_union.buffer(Config.OBSTACLE_EXPANSION)
    else:
        obstacles_union = None
        obstacles_union_expanded = None

    # 转换边界
    bounds_pixel = data.get('coord_bounds', [0, data['width'], 0, data['height']])
    bounds = [b * Config.PIXEL_TO_METER for b in bounds_pixel]

    # 转换起点终点
    start_pose = data.get('start_pose', [100, 100, 0])
    goal_pose = data.get('goal_pose', [500, 500, 0])

    start = [start_pose[0] * Config.PIXEL_TO_METER,
             start_pose[1] * Config.PIXEL_TO_METER,
             start_pose[2] if len(start_pose) > 2 else 0.0]

    goal = [goal_pose[0] * Config.PIXEL_TO_METER,
            goal_pose[1] * Config.PIXEL_TO_METER,
            goal_pose[2] if len(goal_pose) > 2 else 0.0]

    return {
        'obstacles': obstacles,
        'obstacles_union': obstacles_union,  # 原始障碍物（用于可视化）
        'obstacles_union_expanded': obstacles_union_expanded,  # 膨胀障碍物（用于规划）
        'bounds': bounds,
        'start': start,
        'goal': goal,
        'map_name': map_name
    }


def check_collision(point: np.ndarray, obstacles_union, safety_distance: float = None,
                   bounds: List[float] = None) -> bool:
    """
    检查点是否与障碍物或边界碰撞

    注意：传入的obstacles_union应该已经膨胀过(OBSTACLE_EXPANSION=0.05m)

    Args:
        point: [x, y]
        obstacles_union: shapely几何对象（已膨胀）
        safety_distance: 边界安全距离（默认使用OBSTACLE_EXPANSION）
        bounds: [x_min, x_max, y_min, y_max] 边界

    Returns:
        True = 碰撞, False = 安全
    """
    if safety_distance is None:
        safety_distance = Config.OBSTACLE_EXPANSION  # 使用0.05m

    # 检查边界碰撞
    if bounds is not None:
        x, y = point[0], point[1]
        if (x < bounds[0] + safety_distance or x > bounds[1] - safety_distance or
            y < bounds[2] + safety_distance or y > bounds[3] - safety_distance):
            return True

    # 检查障碍物碰撞（直接contains，因为obstacles_union已经膨胀过了）
    if obstacles_union is not None:
        p = Point(point)
        return obstacles_union.contains(p)

    return False


def check_line_collision(p1: np.ndarray, p2: np.ndarray, obstacles_union,
                        safety_distance: float = None, num_checks: int = 10,
                        bounds: List[float] = None) -> bool:
    """
    检查线段是否与障碍物或边界碰撞

    注意：传入的obstacles_union应该已经膨胀过(OBSTACLE_EXPANSION=0.05m)

    Args:
        p1, p2: 线段端点 [x, y]
        obstacles_union: shapely几何对象（已膨胀）
        safety_distance: 边界安全距离
        num_checks: 检查点数量
        bounds: [x_min, x_max, y_min, y_max] 边界

    Returns:
        True = 碰撞, False = 安全
    """
    if safety_distance is None:
        safety_distance = Config.OBSTACLE_EXPANSION  # 使用0.05m

    # 检查端点是否超出边界
    if bounds is not None:
        if check_collision(p1, None, safety_distance, bounds):
            return True
        if check_collision(p2, None, safety_distance, bounds):
            return True

    # 检查障碍物碰撞
    if obstacles_union is not None:
        line = LineString([p1, p2])
        return line.intersects(obstacles_union.buffer(safety_distance))

    return False


# ============================================================================
# RRT* 算法实现
# ============================================================================

class RRTStar:
    """
    RRT* 路径规划算法

    Linus评价: "好的算法应该直截了当，没有花哨的东西"
    """

    class Node:
        """RRT树节点"""
        def __init__(self, pos: np.ndarray):
            self.pos = np.array(pos)  # [x, y]
            self.parent: Optional['RRTStar.Node'] = None
            self.cost: float = 0.0

    def __init__(self, start: np.ndarray, goal: np.ndarray,
                 bounds: List[float], obstacles_union):
        """
        Args:
            start: [x, y]
            goal: [x, y]
            bounds: [x_min, x_max, y_min, y_max]
            obstacles_union: shapely几何对象
        """
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.bounds = bounds
        self.obstacles = obstacles_union

        self.nodes = [self.start]
        self.goal_node = None

    def plan(self, max_iter: int = Config.RRT_MAX_ITER,
             verbose: bool = True) -> Optional[List[np.ndarray]]:
        """
        执行RRT*规划

        Returns:
            路径点列表 [[x, y], ...] 或 None
        """
        start_time = time.time()

        for i in range(max_iter):
            # 1. 随机采样
            rand_point = self._sample()

            # 2. 找最近节点
            nearest_node = self._nearest(rand_point)

            # 3. 扩展
            new_node = self._steer(nearest_node, rand_point)

            # 4. 碰撞检测（包括边界）
            if check_line_collision(nearest_node.pos, new_node.pos, self.obstacles,
                                   bounds=self.bounds):
                continue

            # 5. 找附近节点（用于重布线）
            near_nodes = self._near(new_node)

            # 6. 选择最优父节点
            min_cost_node = nearest_node
            min_cost = nearest_node.cost + np.linalg.norm(new_node.pos - nearest_node.pos)

            for near_node in near_nodes:
                cost = near_node.cost + np.linalg.norm(new_node.pos - near_node.pos)
                if cost < min_cost:
                    if not check_line_collision(near_node.pos, new_node.pos, self.obstacles,
                                               bounds=self.bounds):
                        min_cost_node = near_node
                        min_cost = cost

            # 7. 连接到树
            new_node.parent = min_cost_node
            new_node.cost = min_cost
            self.nodes.append(new_node)

            # 8. 重布线
            for near_node in near_nodes:
                new_cost = new_node.cost + np.linalg.norm(near_node.pos - new_node.pos)
                if new_cost < near_node.cost:
                    if not check_line_collision(new_node.pos, near_node.pos, self.obstacles,
                                               bounds=self.bounds):
                        near_node.parent = new_node
                        near_node.cost = new_cost

            # 9. 检查是否到达目标
            if np.linalg.norm(new_node.pos - self.goal.pos) < Config.RRT_GOAL_THRESHOLD:
                if not check_line_collision(new_node.pos, self.goal.pos, self.obstacles,
                                           bounds=self.bounds):
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost + np.linalg.norm(self.goal.pos - new_node.pos)
                    self.goal_node = self.goal

                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"  [OK] RRT* 找到路径: {i+1} 次迭代, {elapsed:.2f}秒, "
                              f"代价={self.goal.cost:.3f}m")

                    return self._extract_path()

        if verbose:
            print(f"  [FAIL] RRT* 失败: 达到最大迭代次数 {max_iter}")
        return None

    def _sample(self) -> np.ndarray:
        """随机采样点"""
        if np.random.rand() < Config.RRT_GOAL_SAMPLE_RATE:
            return self.goal.pos
        else:
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            y = np.random.uniform(self.bounds[2], self.bounds[3])
            return np.array([x, y])

    def _nearest(self, point: np.ndarray) -> Node:
        """找最近节点"""
        distances = [np.linalg.norm(node.pos - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def _steer(self, from_node: Node, to_point: np.ndarray) -> Node:
        """从from_node向to_point扩展"""
        direction = to_point - from_node.pos
        distance = np.linalg.norm(direction)

        if distance < Config.RRT_STEP_SIZE:
            new_pos = to_point
        else:
            new_pos = from_node.pos + direction / distance * Config.RRT_STEP_SIZE

        return self.Node(new_pos)

    def _near(self, node: Node) -> List[Node]:
        """找附近节点"""
        radius = Config.RRT_SEARCH_RADIUS
        near_nodes = []
        for n in self.nodes:
            if np.linalg.norm(n.pos - node.pos) < radius:
                near_nodes.append(n)
        return near_nodes

    def _extract_path(self) -> List[np.ndarray]:
        """提取路径"""
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node.pos.copy())
            node = node.parent
        path.reverse()
        return path

    def get_tree(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取树的边（用于可视化）"""
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append((node.parent.pos, node.pos))
        return edges


# ============================================================================
# 路径简化
# ============================================================================

def simplify_path(path: List[np.ndarray], obstacles_union,
                  bounds: List[float] = None) -> List[np.ndarray]:
    """
    移除冗余中间点（如果两点间可以直连则移除中间点）

    Linus: "如果能用3个点表达，就不要用10个点"

    Args:
        path: 路径点列表
        obstacles_union: 障碍物
        bounds: [x_min, x_max, y_min, y_max] 边界
    """
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    i = 0

    while i < len(path) - 1:
        # 尝试连接到尽可能远的点
        for j in range(len(path) - 1, i, -1):
            if not check_line_collision(path[i], path[j], obstacles_union, bounds=bounds):
                simplified.append(path[j])
                i = j
                break
        else:
            i += 1

    return simplified


# ============================================================================
# Bezier 平滑模块
# ============================================================================

def check_path_collision(path: np.ndarray, obstacles_union,
                        sample_interval: int = 10,
                        bounds: List[float] = None) -> Tuple[bool, int]:
    """
    检查路径是否与障碍物或边界碰撞

    Args:
        path: 路径点 [[x, y], ...]
        obstacles_union: shapely障碍物对象
        sample_interval: 采样间隔（每N个点检查一次）
        bounds: [x_min, x_max, y_min, y_max] 边界

    Returns:
        (has_collision, first_collision_index)
    """
    if obstacles_union is None and bounds is None:
        return False, -1

    for i in range(0, len(path), sample_interval):
        if check_collision(path[i], obstacles_union, bounds=bounds):
            return True, i

    return False, -1


def check_path_collision_detailed(path: np.ndarray, obstacles_union,
                                  bounds: List[float] = None) -> Tuple[bool, List[int], Dict]:
    """
    检查路径是否与障碍物或边界碰撞（详细版本）

    Args:
        path: 路径点 [[x, y], ...]
        obstacles_union: shapely障碍物对象
        bounds: [x_min, x_max, y_min, y_max] 边界

    Returns:
        (是否碰撞, 碰撞点索引列表, 碰撞统计信息)
    """
    from shapely.geometry import Point

    collision_indices = []
    boundary_collisions = 0
    obstacle_collisions = 0

    # 安全距离：使用0.05m（与障碍物膨胀距离一致）
    safety_distance = Config.OBSTACLE_EXPANSION

    for i, point in enumerate(path):
        x, y = point[0], point[1]

        # 检查边界碰撞
        if bounds:
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


def smooth_path_with_bezier(waypoints: List[np.ndarray],
                            obstacles_union=None,
                            resolution: float = Config.PATH_RESOLUTION,
                            max_iterations: int = 3,
                            bounds: List[float] = None) -> Tuple[np.ndarray, bool]:
    """
    使用B样条曲线平滑路径，并检查碰撞

    Args:
        waypoints: 路径点 [[x, y], ...]
        obstacles_union: 障碍物（用于碰撞检查）
        resolution: 输出路径点间距 (m)
        max_iterations: 最大迭代次数（增加waypoints密度的次数）
        bounds: [x_min, x_max, y_min, y_max] 边界

    Returns:
        (平滑后的路径 shape=(N, 2), 是否有碰撞)
    """
    if len(waypoints) < 3:
        return np.array(waypoints), False

    waypoints = np.array(waypoints)
    current_waypoints = waypoints.copy()
    last_smooth_path = None

    # 使用B样条插值
    # k=3 表示3次样条，s=0表示通过所有点
    for iteration in range(max_iterations):
        try:
            tck, u = splprep([current_waypoints[:, 0], current_waypoints[:, 1]],
                            s=0, k=min(3, len(current_waypoints)-1))

            # 计算路径总长度
            total_length = 0
            for i in range(len(current_waypoints) - 1):
                total_length += np.linalg.norm(current_waypoints[i+1] - current_waypoints[i])

            # 根据分辨率确定采样点数
            num_points = max(int(total_length / resolution), len(current_waypoints) * 2)
            u_new = np.linspace(0, 1, num_points)

            # 插值生成平滑路径
            smooth_x, smooth_y = splev(u_new, tck)
            smooth_path = np.column_stack([smooth_x, smooth_y])
            last_smooth_path = smooth_path  # 保存最后一次平滑结果

            # 检查碰撞（包括边界）
            if obstacles_union is not None or bounds is not None:
                has_collision, collision_idx = check_path_collision(smooth_path, obstacles_union,
                                                                    bounds=bounds)

                if has_collision:
                    print(f"  [WARN] Bezier平滑路径碰撞，尝试增加waypoints密度 (迭代 {iteration+1}/{max_iterations})")

                    # 在每两个waypoint之间插入中点
                    new_waypoints = [current_waypoints[0]]
                    for i in range(len(current_waypoints) - 1):
                        new_waypoints.append(current_waypoints[i])
                        # 插入中点
                        mid_point = (current_waypoints[i] + current_waypoints[i+1]) / 2
                        new_waypoints.append(mid_point)
                    new_waypoints.append(current_waypoints[-1])

                    current_waypoints = np.array(new_waypoints)
                    continue  # 重试
                else:
                    # 无碰撞，返回平滑路径
                    return smooth_path, False
            else:
                # 不检查碰撞，直接返回
                return smooth_path, False

        except Exception as e:
            print(f"  [WARN] B样条平滑失败: {e}")
            break

    # 如果所有迭代都失败或仍有碰撞，返回最后一次平滑结果（带碰撞警告）
    if last_smooth_path is not None:
        print(f"  [WARN] Bezier平滑后仍有碰撞，但仍使用平滑路径（带警告标记）")
        return last_smooth_path, True
    else:
        # 如果连平滑都失败了，返回原路径
        print(f"  [WARN] Bezier平滑失败，返回原路径")
        return waypoints, False


# ============================================================================
# 速度剖面生成
# ============================================================================

def compute_curvature(path: np.ndarray) -> np.ndarray:
    """
    计算路径每点的曲率

    使用三点法: κ = 2*sin(α) / |AB|
    其中α是转角，AB是弦长
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

        # 曲率 = 2*sin(angle/2) / 弦长
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
        'max_v': max_v,
        'max_omega': max_omega,
        'max_wheel_v': max_wheel_v,
        'scale_info': scale_info
    }


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
# 可视化模块
# ============================================================================

def plot_trajectory_combined(env: Dict, rrt_path: List[np.ndarray],
                             smooth_path: np.ndarray, trajectory: Dict,
                             rrt_tree: List[Tuple], save_path: str,
                             has_collision: bool = False,
                             collision_info: Dict = None,
                             collision_indices: List[int] = None):
    """
    绘制综合轨迹图：轨迹+速度剖面（合并版）

    左侧：轨迹图（RRT树、路径、障碍物、速度编码）
    右侧：线速度、角速度、轮子速度剖面

    Args:
        env: 环境数据
        rrt_path: RRT原始路径
        smooth_path: Bezier平滑路径
        trajectory: 轨迹数据
        rrt_tree: RRT树的边
        save_path: 保存路径
        has_collision: 是否有碰撞
        collision_info: 碰撞信息
        collision_indices: 碰撞点索引
    """
    fig = plt.figure(figsize=(20, 12))

    # 创建子图布局: 左侧轨迹图，右侧3个速度剖面
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.25)

    ax_traj = fig.add_subplot(gs[:, 0])    # 左侧：轨迹图
    ax_v = fig.add_subplot(gs[0, 1])       # 右上：线速度
    ax_omega = fig.add_subplot(gs[1, 1])   # 右中：角速度
    ax_wheel = fig.add_subplot(gs[2, 1])   # 右下：轮子速度

    velocities = np.array(trajectory['velocities'])
    wheel_velocities = np.array(trajectory.get('wheel_velocities', []))
    timestamps = np.array(trajectory['timestamps'])
    v_vals = velocities[:, 0]
    omega_vals = velocities[:, 1]

    # ==================== 左侧：轨迹图 ====================
    # 1. 绘制障碍物
    for obs in env['obstacles']:
        if obs.geom_type == 'Polygon':
            x, y = obs.exterior.xy
            ax_traj.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)
        elif obs.geom_type == 'MultiPolygon':
            for poly in obs.geoms:
                x, y = poly.exterior.xy
                ax_traj.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)

    # 2. 绘制RRT树（淡蓝色细线）
    for edge in rrt_tree:
        p1, p2 = edge
        ax_traj.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c-', alpha=0.15, linewidth=0.5)

    # 3. 绘制RRT路径（橙色虚线）
    if rrt_path:
        rrt_array = np.array(rrt_path)
        ax_traj.plot(rrt_array[:, 0], rrt_array[:, 1], 'o--',
                    color='orange', linewidth=2, markersize=5, label='RRT Waypoints')

    # 4. 绘制Bezier平滑路径（速度编码）
    if smooth_path is not None and len(smooth_path) > 0:
        points = smooth_path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='jet', linewidths=3)
        lc.set_array(v_vals[:-1])
        lc.set_clim(0, Config.WHEEL_V_MAX)
        line = ax_traj.add_collection(lc)

        # 颜色条
        cbar = fig.colorbar(line, ax=ax_traj, shrink=0.6, pad=0.02)
        cbar.set_label('Velocity (m/s)', fontsize=11)

        # 标记碰撞点
        if collision_info and collision_info.get('has_collision') and collision_indices:
            collision_points = smooth_path[collision_indices]
            ax_traj.scatter(collision_points[:, 0], collision_points[:, 1],
                           c='red', s=20, alpha=0.6, marker='x',
                           label=f'Collision ({len(collision_indices)})', zorder=15)

    # 5. 起点终点
    start = env['start']
    goal = env['goal']
    ax_traj.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=10)
    ax_traj.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', zorder=10)

    # 6. 边界设置
    bounds = env['bounds']
    ax_traj.set_xlim(bounds[0], bounds[1])
    ax_traj.set_ylim(bounds[2], bounds[3])

    ax_traj.set_xlabel('X (m)', fontsize=12)
    ax_traj.set_ylabel('Y (m)', fontsize=12)
    ax_traj.legend(loc='upper right', fontsize=9)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal')

    # 7. 标题
    if has_collision:
        title_suffix = ' [COLLISION]'
    else:
        title_suffix = ''

    ax_traj.set_title(f'Trajectory - {env["map_name"]}{title_suffix}', fontsize=13, fontweight='bold')

    # ==================== 右上：线速度剖面 ====================
    ax_v.plot(timestamps, v_vals, 'b-', linewidth=1.5, label='v(t)')
    ax_v.axhline(y=Config.WHEEL_V_MAX, color='r', linestyle='--', linewidth=1.5,
                 label=f'V_MAX={Config.WHEEL_V_MAX}')
    ax_v.fill_between(timestamps, 0, v_vals, alpha=0.3)
    ax_v.set_xlabel('Time (s)', fontsize=11)
    ax_v.set_ylabel('v (m/s)', fontsize=11)
    ax_v.set_title('Linear Velocity', fontsize=12, fontweight='bold')
    ax_v.legend(loc='upper right', fontsize=9)
    ax_v.grid(True, alpha=0.3)
    ax_v.set_ylim(0, Config.WHEEL_V_MAX * 1.15)

    # ==================== 右中：角速度剖面 ====================
    ax_omega.plot(timestamps, omega_vals, 'g-', linewidth=1.5, label='ω(t)')
    ax_omega.axhline(y=Config.OMEGA_MAX, color='r', linestyle='--', linewidth=1.5, label=f'±ω_MAX')
    ax_omega.axhline(y=-Config.OMEGA_MAX, color='r', linestyle='--', linewidth=1.5)
    ax_omega.fill_between(timestamps, 0, omega_vals, alpha=0.3, color='green')
    ax_omega.set_xlabel('Time (s)', fontsize=11)
    ax_omega.set_ylabel('ω (rad/s)', fontsize=11)
    ax_omega.set_title('Angular Velocity', fontsize=12, fontweight='bold')
    ax_omega.legend(loc='upper right', fontsize=9)
    ax_omega.grid(True, alpha=0.3)
    ax_omega.set_ylim(-Config.OMEGA_MAX * 1.2, Config.OMEGA_MAX * 1.2)

    # ==================== 右下：轮子速度剖面 ====================
    if len(wheel_velocities) > 0:
        v_L = wheel_velocities[:, 0]
        v_R = wheel_velocities[:, 1]
        ax_wheel.plot(timestamps, v_L, 'b-', linewidth=1.5, label='v_L (left)')
        ax_wheel.plot(timestamps, v_R, 'r-', linewidth=1.5, label='v_R (right)')
        ax_wheel.axhline(y=Config.WHEEL_V_MAX, color='k', linestyle='--', linewidth=1.5,
                        label=f'±V_wheel_MAX')
        ax_wheel.axhline(y=-Config.WHEEL_V_MAX, color='k', linestyle='--', linewidth=1.5)
        ax_wheel.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_wheel.set_xlabel('Time (s)', fontsize=11)
    ax_wheel.set_ylabel('v_wheel (m/s)', fontsize=11)
    ax_wheel.set_title('Wheel Velocities (Differential Drive)', fontsize=12, fontweight='bold')
    ax_wheel.legend(loc='upper right', fontsize=9)
    ax_wheel.grid(True, alpha=0.3)
    ax_wheel.set_ylim(-Config.WHEEL_V_MAX * 1.2, Config.WHEEL_V_MAX * 1.2)

    # 总标题和底部信息
    max_kappa = trajectory.get('max_curvature', 0)
    min_radius = 1.0 / max_kappa if max_kappa > 1e-6 else float('inf')
    max_wheel_v = trajectory.get('max_wheel_v', 0)

    fig.suptitle(f'RRT* + Bezier Trajectory - {env["map_name"]}',
                 fontsize=16, fontweight='bold', y=0.99)

    # 底部信息文本（图外）
    info_line = (f"Finish Time: {trajectory['total_time']:.2f}s  |  "
                 f"Distance: {trajectory['total_distance']:.4f}m  |  "
                 f"Max v: {trajectory['max_v']:.4f} m/s  |  "
                 f"Max |ω|: {trajectory['max_omega']:.4f} rad/s  |  "
                 f"Max |v_wheel|: {max_wheel_v:.4f} m/s  |  "
                 f"Max κ: {max_kappa:.2f} 1/m  |  "
                 f"Min R: {min_radius:.4f}m")

    fig.text(0.5, 0.01, info_line, ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 综合轨迹图保存: {save_path}")


# ============================================================================
# 主处理流程
# ============================================================================

def process_single_map(env_path: str, output_dir: str, verbose: bool = True) -> bool:
    """
    处理单个MAP

    Args:
        env_path: environment_map*.json 路径
        output_dir: 输出目录
        verbose: 是否打印详细信息

    Returns:
        成功返回True
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"处理: {env_path}")
        print(f"{'='*60}")

    try:
        # 1. 加载环境
        env = load_environment(env_path)
        if verbose:
            print(f"[OK] 环境加载完成: {env['map_name']}")
            print(f"  起点: ({env['start'][0]:.3f}, {env['start'][1]:.3f}, {env['start'][2]:.3f})")
            print(f"  终点: ({env['goal'][0]:.3f}, {env['goal'][1]:.3f}, {env['goal'][2]:.3f})")
            print(f"  障碍物: {len(env['obstacles'])} 个")

        # 2. RRT* 规划 (使用膨胀后的障碍物)
        if verbose:
            print(f"\n执行 RRT* 规划 (障碍物膨胀 {Config.OBSTACLE_EXPANSION}m)...")

        rrt = RRTStar(
            start=env['start'][:2],
            goal=env['goal'][:2],
            bounds=env['bounds'],
            obstacles_union=env['obstacles_union_expanded']  # 使用膨胀障碍物
        )

        rrt_path = rrt.plan(verbose=verbose)

        if rrt_path is None:
            print(f"  [FAIL] RRT* 规划失败")
            return False

        # 3. 路径简化 (使用膨胀后的障碍物，检查边界)
        simplified_path = simplify_path(rrt_path, env['obstacles_union_expanded'],
                                       bounds=env['bounds'])
        if verbose:
            print(f"  [OK] 路径简化: {len(rrt_path)} → {len(simplified_path)} 点")

        # 4. Bezier 平滑 (带碰撞和边界检查)
        if verbose:
            print(f"\n执行 Bezier 平滑 (带碰撞和边界检查)...")

        smooth_path, has_collision = smooth_path_with_bezier(simplified_path,
                                                             obstacles_union=env['obstacles_union_expanded'],
                                                             bounds=env['bounds'])
        if verbose:
            if has_collision:
                print(f"  [WARN] 平滑完成但有碰撞: {len(smooth_path)} 点 - 轨迹可能不安全!")
            else:
                print(f"  [OK] 平滑完成: {len(smooth_path)} 点")

        # 5. 生成速度剖面（带缩放以满足约束）
        if verbose:
            print(f"\n生成速度剖面（缩放以满足 v≤{Config.V_MAX} m/s, |ω|≤{Config.OMEGA_MAX} rad/s）...")

        trajectory = generate_velocity_profile(smooth_path, verbose=verbose)

        # 6. 检查运动学约束
        satisfied, constraint_msg = check_kinematics_constraints(trajectory)
        print(f"  {constraint_msg}")

        if verbose:
            max_kappa = trajectory['max_curvature']
            min_radius = 1.0 / max_kappa if max_kappa > 1e-6 else float('inf')
            print(f"  最大曲率: κ_max={max_kappa:.4f} (1/m), 最小转弯半径: R_min={min_radius:.4f} m")

            # 输出完成时间（重点突出）
            print(f"\n  {'='*40}")
            print(f"  轨迹完成时间: {trajectory['total_time']:.2f} 秒")
            print(f"  轨迹总距离: {trajectory['total_distance']:.4f} m")
            print(f"  实际最大速度: v_max={trajectory['max_v']:.4f} m/s, |ω|_max={trajectory['max_omega']:.4f} rad/s")
            print(f"  {'='*40}")

        # 7. 详细碰撞检测（在保存结果前执行）
        _, collision_indices, collision_info = check_path_collision_detailed(
            smooth_path,
            env['obstacles_union_expanded'],
            bounds=env['bounds']
        )

        if verbose and has_collision:
            print(f"\n  [WARNING] 检测到路径碰撞!")
            print(f"    - 碰撞点数: {collision_info['total_collision_points']}/{len(smooth_path)} "
                  f"({collision_info['collision_ratio']:.1%})")
            print(f"    - 边界碰撞: {collision_info['boundary_collisions']} 点")
            print(f"    - 障碍物碰撞: {collision_info['obstacle_collisions']} 点")

        # 8. 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        map_name = env['map_name']

        # 保存RRT路径点
        rrt_output = {
            'map_name': map_name,
            'coordinate_frame': 'world_meter',
            'start_pose': env['start'],
            'goal_pose': env['goal'],
            'waypoints': [p.tolist() for p in simplified_path]
        }

        rrt_json_path = output_path / f'rrt_waypoints_{map_name}.json'
        with open(rrt_json_path, 'w', encoding='utf-8') as f:
            json.dump(rrt_output, f, indent=2)

        if verbose:
            print(f"  [OK] RRT路径保存: {rrt_json_path}")

        # 保存Bezier轨迹
        trajectory_output = {
            'map_name': map_name,
            'coordinate_frame': 'world_meter',
            'start_pose': env['start'],
            'goal_pose': env['goal'],
            'smooth_path': smooth_path.tolist(),
            'trajectory': trajectory,
            'constraints': {
                'v_max_limit': Config.V_MAX,
                'omega_max_limit': Config.OMEGA_MAX,
                'satisfied': satisfied
            },
            'summary': {
                'total_time_seconds': trajectory['total_time'],
                'total_distance_meters': trajectory['total_distance'],
                'actual_max_v': trajectory['max_v'],
                'actual_max_omega': trajectory['max_omega'],
                'scale_info': trajectory['scale_info']
            },
            'collision_info': collision_info
        }

        traj_json_path = output_path / f'bezier_trajectory_{map_name}.json'
        with open(traj_json_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_output, f, indent=2)

        if verbose:
            print(f"  [OK] Bezier轨迹保存: {traj_json_path}")

        # 9. 可视化（合并轨迹图+速度剖面）
        if verbose:
            print(f"\n生成可视化...")

        vis_path = output_path / f'trajectory_{map_name}.png'
        plot_trajectory_combined(env, simplified_path, smooth_path, trajectory, rrt.get_tree(),
                                str(vis_path), has_collision, collision_info, collision_indices)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[OK] {map_name} 处理完成")
            print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n[FAIL] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_process_maps(maps_dir: str, output_base_dir: str):
    """
    批量处理所有MAP

    Args:
        maps_dir: MAPS文件夹路径
        output_base_dir: 输出根目录
    """
    maps_path = Path(maps_dir)

    # 查找所有environment文件
    env_files = sorted(maps_path.glob('*/environment_*.json'))

    if not env_files:
        print(f"错误: 在 {maps_dir} 下未找到 environment_*.json 文件")
        return

    print(f"\n找到 {len(env_files)} 个环境文件")
    print(f"输出目录: {output_base_dir}\n")

    success_count = 0
    failed_maps = []

    for env_file in env_files:
        map_name = env_file.parent.name

        # 跳过 warehouse_N 系列地图（保留纯 warehouse）
        if map_name.startswith('warehouse_'):
            print(f"[SKIP] 跳过 warehouse_N 地图: {map_name}")
            continue

        output_dir = Path(output_base_dir) / map_name

        success = process_single_map(str(env_file), str(output_dir), verbose=True)

        if success:
            success_count += 1
        else:
            failed_maps.append(map_name)

    # 总结
    print(f"\n\n{'='*60}")
    print(f"批量处理完成")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(env_files)}")
    if failed_maps:
        print(f"失败: {failed_maps}")
    print(f"{'='*60}\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""

    # 加载配置
    config_path = r"d:\Data_visualization_code\result\romi\config.yaml"
    if Path(config_path).exists():
        Config.load_from_yaml(config_path)

    # 批量处理所有MAPS
    maps_dir = r"d:\Data_visualization_code\result\MAPS"
    output_dir = r"d:\Data_visualization_code\result\RRT_Results"

    batch_process_maps(maps_dir, output_dir)


if __name__ == '__main__':
    main()
