"""
RRT* + Dubins 路径规划系统
为每个MAP生成从起点到终点的符合Dubins约束的平滑轨迹

Dubins路径: 最优路径由圆弧(C)和直线(S)组成，满足最小转弯半径约束
路径类型: LSL, RSR, LSR, RSL, RLR, LRL (L=左转, R=右转, S=直线)

作者: Dubins式优雅设计
日期: 2025-01-07
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import yaml
from typing import List, Tuple, Dict, Optional
import time
from enum import Enum


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

    # Dubins 参数
    DUBINS_MIN_RADIUS = 0.2  # 最小转弯半径 (m) - 根据运动学约束 V_MAX/OMEGA_MAX
    
    # RRT* 参数
    RRT_MAX_ITER = 30000  # 最大迭代次数（大幅增加以应对Dubins约束和高碰撞率）
    RRT_STEP_SIZE = 0.08  # Dubins连接步长 (m) - 减小以适应狭窄环境
    RRT_GOAL_SAMPLE_RATE = 0.20  # 目标采样率初始值（会自适应增加）
    RRT_SEARCH_RADIUS = 0.15  # 重布线搜索半径 (m)
    RRT_GOAL_THRESHOLD = 0.08  # 到达目标的距离阈值 (m) - 放宽以更容易到达
    RRT_COLLISION_CHECK_RES = 0.02  # 碰撞检测分辨率 (m) - 粗化以提升速度

    # 路径离散化
    PATH_RESOLUTION = 0.003  # 路径点间距 (m)

    # 时间离散化
    DT = 0.1  # 秒，时间步长

    # 安全距离
    ROBOT_RADIUS = 0.0265  # e-puck2 半径
    SAFETY_MARGIN = 0.02  # 额外安全边距
    OBSTACLE_EXPANSION = 0.03  # 障碍物膨胀距离 (m) - 减小以增加可行空间

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
                configured_radius = config.get('dubins_min_radius', None)
                if configured_radius is not None:
                    cls.DUBINS_MIN_RADIUS = configured_radius
                print(f"[OK] 已加载配置: {config_path}")
                print(f"     Dubins最小转弯半径: {cls.DUBINS_MIN_RADIUS:.4f} m")
        except Exception as e:
            print(f"[WARN] 配置文件加载失败，使用默认值: {e}")


# ============================================================================
# Dubins 路径类型
# ============================================================================

class SegmentType(Enum):
    """Dubins路径段类型"""
    L = 1  # 左转 (Left)
    R = 2  # 右转 (Right)
    S = 3  # 直线 (Straight)


class DubinsPathType(Enum):
    """Dubins路径类型"""
    LSL = (SegmentType.L, SegmentType.S, SegmentType.L)
    RSR = (SegmentType.R, SegmentType.S, SegmentType.R)
    LSR = (SegmentType.L, SegmentType.S, SegmentType.R)
    RSL = (SegmentType.R, SegmentType.S, SegmentType.L)
    RLR = (SegmentType.R, SegmentType.L, SegmentType.R)
    LRL = (SegmentType.L, SegmentType.R, SegmentType.L)


# ============================================================================
# 环境加载模块（与原文件相同）
# ============================================================================

def load_environment(json_path: str) -> Dict:
    """加载环境文件并转换坐标系"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    map_name = Path(json_path).parent.name

    obstacles = []
    for poly_data in data.get('polygons', []):
        vertices = np.array(poly_data['vertices']) * Config.PIXEL_TO_METER
        obstacles.append(Polygon(vertices))

    if obstacles:
        obstacles_union = unary_union(obstacles)
        obstacles_union_expanded = obstacles_union.buffer(Config.OBSTACLE_EXPANSION)
    else:
        obstacles_union = None
        obstacles_union_expanded = None

    bounds_pixel = data.get('coord_bounds', [0, data['width'], 0, data['height']])
    bounds = [b * Config.PIXEL_TO_METER for b in bounds_pixel]

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
        'obstacles_union': obstacles_union,
        'obstacles_union_expanded': obstacles_union_expanded,
        'bounds': bounds,
        'start': start,
        'goal': goal,
        'map_name': map_name
    }


def check_collision(point: np.ndarray, obstacles_union, safety_distance: float = None,
                   bounds: List[float] = None) -> bool:
    """检查点是否与障碍物或边界碰撞"""
    if safety_distance is None:
        safety_distance = Config.OBSTACLE_EXPANSION

    if bounds is not None:
        x, y = point[0], point[1]
        if (x < bounds[0] + safety_distance or x > bounds[1] - safety_distance or
            y < bounds[2] + safety_distance or y > bounds[3] - safety_distance):
            return True

    if obstacles_union is not None:
        p = Point(point)
        return obstacles_union.contains(p)

    return False


# ============================================================================
# Dubins 路径生成核心算法
# ============================================================================

def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-π, π]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def mod2pi(angle: float) -> float:
    """将角度归一化到 [0, 2π)"""
    return angle % (2 * np.pi)


class DubinsPath:
    """Dubins路径表示"""
    
    def __init__(self, start: np.ndarray, end: np.ndarray, radius: float,
                 path_type: DubinsPathType, lengths: Tuple[float, float, float]):
        """
        Args:
            start: [x, y, theta] 起点
            end: [x, y, theta] 终点
            radius: 转弯半径
            path_type: 路径类型 (LSL, RSR, etc.)
            lengths: 三段路径长度 (t, p, q)
        """
        self.start = np.array(start)
        self.end = np.array(end)
        self.radius = radius
        self.path_type = path_type
        self.lengths = lengths  # (t, p, q)
        self.total_length = sum(lengths)
        
    def sample(self, resolution: float = 0.01) -> np.ndarray:
        """
        采样Dubins路径点
        
        Args:
            resolution: 采样间距 (m)
            
        Returns:
            路径点数组 shape=(N, 3) [x, y, theta]
        """
        num_points = max(int(self.total_length / resolution), 10)
        params = np.linspace(0, self.total_length, num_points)
        
        path = []
        for s in params:
            point = self._interpolate(s)
            path.append(point)
        
        # 确保最后一个点精确匹配终点姿态（避免数值误差导致方向不连续）
        path[-1] = self.end.copy()
            
        return np.array(path)
    
    def _interpolate(self, s: float) -> np.ndarray:
        """在路径上插值计算位置"""
        t, p, q = self.lengths
        seg1, seg2, seg3 = self.path_type.value
        
        x, y, theta = self.start[0], self.start[1], self.start[2]
        
        # 第一段
        if s <= t:
            return self._segment_point(x, y, theta, s, seg1)
        
        # 第二段
        x1, y1, theta1 = self._segment_point(x, y, theta, t, seg1)
        if s <= t + p:
            return self._segment_point(x1, y1, theta1, s - t, seg2)
        
        # 第三段
        x2, y2, theta2 = self._segment_point(x1, y1, theta1, p, seg2)
        return self._segment_point(x2, y2, theta2, s - t - p, seg3)
    
    def _segment_point(self, x: float, y: float, theta: float, 
                      length: float, seg_type: SegmentType) -> np.ndarray:
        """计算单段路径上的点"""
        if seg_type == SegmentType.L:  # 左转
            cx = x - self.radius * np.sin(theta)
            cy = y + self.radius * np.cos(theta)
            angle = length / self.radius
            new_theta = normalize_angle(theta + angle)
            new_x = cx + self.radius * np.sin(new_theta)
            new_y = cy - self.radius * np.cos(new_theta)
            return np.array([new_x, new_y, new_theta])
            
        elif seg_type == SegmentType.R:  # 右转
            cx = x + self.radius * np.sin(theta)
            cy = y - self.radius * np.cos(theta)
            angle = length / self.radius
            new_theta = normalize_angle(theta - angle)
            new_x = cx - self.radius * np.sin(new_theta)
            new_y = cy + self.radius * np.cos(new_theta)
            return np.array([new_x, new_y, new_theta])
            
        else:  # 直线
            new_x = x + length * np.cos(theta)
            new_y = y + length * np.sin(theta)
            return np.array([new_x, new_y, theta])


def compute_dubins_path(start: np.ndarray, end: np.ndarray,
                       radius: float) -> Optional[DubinsPath]:
    """
    计算两点间的最短Dubins路径（使用RRT_dubins_test中的求解逻辑）

    Args:
        start: [x, y, theta] 起点
        end: [x, y, theta] 终点
        radius: 最小转弯半径

    Returns:
        DubinsPath对象 或 None（无解）
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    d = np.hypot(dx, dy) / radius

    theta = mod2pi(np.arctan2(dy, dx))
    alpha = mod2pi(start[2] - theta)
    beta = mod2pi(end[2] - theta)

    candidates: List[Tuple[DubinsPathType, Tuple[float, float, float]]] = []

    for path_type, solver in [
        (DubinsPathType.LSL, _dubins_LSL),
        (DubinsPathType.RSR, _dubins_RSR),
        (DubinsPathType.LSR, _dubins_LSR),
        (DubinsPathType.RSL, _dubins_RSL),
        (DubinsPathType.RLR, _dubins_RLR),
        (DubinsPathType.LRL, _dubins_LRL),
    ]:
        res = solver(alpha, beta, d)
        if res is None:
            continue
        candidates.append((path_type, res))

    if not candidates:
        return None

    best_type, best_lengths = min(candidates, key=lambda p: sum(p[1]))
    best_lengths = tuple(l * radius for l in best_lengths)
    return DubinsPath(start, end, radius, best_type, best_lengths)


def _dubins_LSL(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """LSL路径（alpha/beta/d归一化参数）"""
    tmp0 = d + np.sin(alpha) - np.sin(beta)
    p_squared = 2 + (d * d) - (2 * np.cos(alpha - beta)) + (2 * d * (np.sin(alpha) - np.sin(beta)))
    if p_squared < 0:
        return None
    tmp1 = np.arctan2((np.cos(beta) - np.cos(alpha)), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = np.sqrt(p_squared)
    q = mod2pi(beta - tmp1)
    return t, p, q


def _dubins_RSR(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """RSR路径（alpha/beta/d归一化参数）"""
    tmp0 = d - np.sin(alpha) + np.sin(beta)
    p_squared = 2 + (d * d) - (2 * np.cos(alpha - beta)) + (2 * d * (np.sin(beta) - np.sin(alpha)))
    if p_squared < 0:
        return None
    tmp1 = np.arctan2((np.cos(alpha) - np.cos(beta)), tmp0)
    t = mod2pi(alpha - tmp1)
    p = np.sqrt(p_squared)
    q = mod2pi(-beta + tmp1)
    return t, p, q


def _dubins_LSR(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """LSR路径（alpha/beta/d归一化参数）"""
    p_squared = -2 + (d * d) + (2 * np.cos(alpha - beta)) + (2 * d * (np.sin(alpha) + np.sin(beta)))
    if p_squared < 0:
        return None
    p = np.sqrt(p_squared)
    tmp2 = np.arctan2((-np.cos(alpha) - np.cos(beta)), (d + np.sin(alpha) + np.sin(beta))) - np.arctan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)
    return t, p, q


def _dubins_RSL(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """RSL路径（alpha/beta/d归一化参数）"""
    p_squared = (d * d) - 2 + (2 * np.cos(alpha - beta)) - (2 * d * (np.sin(alpha) + np.sin(beta)))
    if p_squared < 0:
        return None
    p = np.sqrt(p_squared)
    tmp2 = np.arctan2((np.cos(alpha) + np.cos(beta)), (d - np.sin(alpha) - np.sin(beta))) - np.arctan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)
    return t, p, q


def _dubins_RLR(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """RLR路径（alpha/beta/d归一化参数）"""
    tmp = (6.0 - d * d + 2.0 * np.cos(alpha - beta) + 2.0 * d * (np.sin(alpha) - np.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2 * np.pi - np.arccos(tmp))
    t = mod2pi(alpha - np.arctan2(np.cos(alpha) - np.cos(beta), d - np.sin(alpha) + np.sin(beta)) + p / 2.0)
    q = mod2pi(alpha - beta - t + p)
    return t, p, q


def _dubins_LRL(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    """LRL路径（alpha/beta/d归一化参数）"""
    tmp = (6.0 - d * d + 2.0 * np.cos(alpha - beta) + 2.0 * d * (-np.sin(alpha) + np.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2 * np.pi - np.arccos(tmp))
    t = mod2pi(-alpha - np.arctan2(np.cos(alpha) - np.cos(beta), d + np.sin(alpha) - np.sin(beta)) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + p)
    return t, p, q


def check_dubins_collision(dubins_path: DubinsPath, obstacles_union,
                          bounds: List[float] = None, 
                          resolution: float = None) -> bool:
    """
    检查Dubins路径是否与障碍物碰撞
    
    Args:
        dubins_path: Dubins路径对象
        obstacles_union: 障碍物
        bounds: 边界
        resolution: 检查分辨率 (m) - 默认使用Config中的值
        
    Returns:
        True = 碰撞, False = 安全
    """
    if resolution is None:
        resolution = Config.RRT_COLLISION_CHECK_RES
    path_points = dubins_path.sample(resolution)
    
    for point in path_points:
        if check_collision(point[:2], obstacles_union, bounds=bounds):
            return True
    
    return False


def ensure_path_continuity(path: np.ndarray, max_gap: float = 0.01) -> np.ndarray:
    """
    确保路径连续性，修复任何gap
    
    Args:
        path: 路径点 shape=(N, 3) [x, y, theta]
        max_gap: 允许的最大点间距 (m)
        
    Returns:
        修复后的连续路径
    """
    if len(path) < 2:
        return path
    
    fixed_path = [path[0]]
    
    for i in range(1, len(path)):
        gap = np.linalg.norm(path[i][:2] - path[i-1][:2])
        
        if gap > max_gap:
            # 插入中间点，确保位置和方向都平滑过渡
            num_points = int(np.ceil(gap / max_gap))
            for j in range(1, num_points):
                t = j / num_points
                # 位置线性插值
                interp_x = path[i-1, 0] * (1-t) + path[i, 0] * t
                interp_y = path[i-1, 1] * (1-t) + path[i, 1] * t
                # 角度插值（考虑角度的周期性）
                theta1 = path[i-1, 2]
                theta2 = path[i, 2]
                # 处理角度跨越±π的情况
                dtheta = normalize_angle(theta2 - theta1)
                interp_theta = normalize_angle(theta1 + dtheta * t)
                
                fixed_path.append(np.array([interp_x, interp_y, interp_theta]))
        
        fixed_path.append(path[i])
    
    return np.array(fixed_path)


def ensure_orientation_continuity(path: np.ndarray) -> np.ndarray:
    """
    确保路径方向连续性，平滑段与段之间的方向跳变
    
    Args:
        path: 路径点 shape=(N, 3) [x, y, theta]
        
    Returns:
        方向连续的路径
    """
    if len(path) < 2:
        return path
    
    fixed_path = path.copy()
    
    # 逐点检查并修正方向不连续
    for i in range(1, len(fixed_path)):
        # 计算角度变化
        theta_diff = normalize_angle(fixed_path[i][2] - fixed_path[i-1][2])
        
        # 计算位置变化对应的运动方向
        dx = fixed_path[i][0] - fixed_path[i-1][0]
        dy = fixed_path[i][1] - fixed_path[i-1][1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 1e-6:  # 有足够的位置变化
            motion_angle = np.arctan2(dy, dx)
            
            # 如果当前角度与运动方向差异较大，修正为运动方向
            angle_to_motion_prev = normalize_angle(motion_angle - fixed_path[i-1][2])
            angle_to_motion_curr = normalize_angle(motion_angle - fixed_path[i][2])
            
            # 如果方向与运动不一致（前后都偏差超过45度），则平滑处理
            if abs(angle_to_motion_prev) > np.pi/4 and abs(angle_to_motion_curr) > np.pi/4:
                # 使用运动方向作为参考进行平滑
                fixed_path[i][2] = normalize_angle(
                    fixed_path[i-1][2] + normalize_angle(motion_angle - fixed_path[i-1][2]) * 0.5
                )
    
    return fixed_path


# ============================================================================
# RRT* with Dubins 算法实现
# ============================================================================

class RRTStarDubins:
    """RRT* 路径规划算法（使用Dubins曲线连接）"""

    class Node:
        """RRT树节点（包含位姿）"""
        def __init__(self, pose: np.ndarray):
            self.pose = np.array(pose)  # [x, y, theta]
            self.parent: Optional['RRTStarDubins.Node'] = None
            self.cost: float = 0.0

    def __init__(self, start: np.ndarray, goal: np.ndarray,
                 bounds: List[float], obstacles_union, radius: float):
        """
        Args:
            start: [x, y, theta]
            goal: [x, y, theta]
            bounds: [x_min, x_max, y_min, y_max]
            obstacles_union: shapely几何对象
            radius: Dubins最小转弯半径
        """
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.bounds = bounds
        self.obstacles = obstacles_union
        self.radius = radius

        self.nodes = [self.start]
        self.goal_node = None

    def plan(self, max_iter: int = Config.RRT_MAX_ITER,
             verbose: bool = True) -> Optional[DubinsPath]:
        """
        执行RRT*规划（返回完整Dubins路径）

        Returns:
            DubinsPath对象 或 None
        """
        start_time = time.time()
        
        # 调试统计
        collision_count = 0
        dubins_fail_count = 0
        last_progress_iter = 0
        
        # 自适应目标采样率（随迭代次数增加而提高）
        base_goal_rate = Config.RRT_GOAL_SAMPLE_RATE

        for i in range(max_iter):
            # 自适应提高目标采样率
            progress = i / max_iter
            adaptive_goal_rate = base_goal_rate + (0.4 - base_goal_rate) * (progress ** 2)
            # 1. 随机采样（包含角度）
            rand_pose = self._sample()

            # 2. 找最近节点
            nearest_node = self._nearest(rand_pose)

            # 3. 使用Dubins曲线扩展
            dubins = compute_dubins_path(nearest_node.pose, rand_pose, self.radius)
            if dubins is None:
                dubins_fail_count += 1
                continue

            # 限制扩展步长
            if dubins.total_length > Config.RRT_STEP_SIZE:
                # 在Dubins路径上采样到达RRT_STEP_SIZE距离的点
                new_pose = dubins._interpolate(Config.RRT_STEP_SIZE)
            else:
                new_pose = dubins.end  # 使用Dubins路径的实际终点确保连续性

            new_node = self.Node(new_pose)

            # 4. 碰撞检测
            dubins_to_new = compute_dubins_path(nearest_node.pose, new_node.pose, self.radius)
            if dubins_to_new is None:
                dubins_fail_count += 1
                continue
                
            if check_dubins_collision(dubins_to_new, self.obstacles, bounds=self.bounds):
                collision_count += 1
                continue
            
            # 定期输出进度
            if verbose and i > 0 and i % 1000 == 0 and i != last_progress_iter:
                elapsed = time.time() - start_time
                nodes_explored = len(self.nodes)
                print(f"    迭代 {i}/{max_iter}: {nodes_explored} 节点, "
                      f"碰撞={collision_count}, Dubins失败={dubins_fail_count}, "
                      f"耗时={elapsed:.1f}s")
                last_progress_iter = i

            # 5. 找附近节点
            near_nodes = self._near(new_node)

            # 6. 选择最优父节点
            min_cost_node = nearest_node
            min_cost = nearest_node.cost + dubins_to_new.total_length

            for near_node in near_nodes:
                dubins_near = compute_dubins_path(near_node.pose, new_node.pose, self.radius)
                if dubins_near is None:
                    continue
                    
                cost = near_node.cost + dubins_near.total_length
                if cost < min_cost:
                    if not check_dubins_collision(dubins_near, self.obstacles, bounds=self.bounds):
                        min_cost_node = near_node
                        min_cost = cost

            # 7. 连接到树
            new_node.parent = min_cost_node
            new_node.cost = min_cost
            self.nodes.append(new_node)

            # 8. 重布线
            for near_node in near_nodes:
                dubins_rewire = compute_dubins_path(new_node.pose, near_node.pose, self.radius)
                if dubins_rewire is None:
                    continue
                    
                new_cost = new_node.cost + dubins_rewire.total_length
                if new_cost < near_node.cost:
                    if not check_dubins_collision(dubins_rewire, self.obstacles, bounds=self.bounds):
                        near_node.parent = new_node
                        near_node.cost = new_cost
                        # 更新姿态以匹配新的Dubins路径终点，确保连续性
                        near_node.pose = dubins_rewire.end.copy()

            # 9. 检查是否到达目标（只考虑位置，不考虑方向）
            dist_to_goal = np.linalg.norm(new_node.pose[:2] - self.goal.pose[:2])
            if dist_to_goal < Config.RRT_GOAL_THRESHOLD:
                # 使用new_node的朝向作为终点朝向（不约束终点方向）
                goal_pose_relaxed = np.array([self.goal.pose[0], self.goal.pose[1], new_node.pose[2]])
                dubins_to_goal = compute_dubins_path(new_node.pose, goal_pose_relaxed, self.radius)
                if dubins_to_goal is not None:
                    if not check_dubins_collision(dubins_to_goal, self.obstacles, bounds=self.bounds):
                        # 更新goal节点的实际到达姿态为Dubins路径的精确终点
                        self.goal.pose = dubins_to_goal.end.copy()
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + dubins_to_goal.total_length
                        self.goal_node = self.goal

                        if verbose:
                            elapsed = time.time() - start_time
                            print(f"  [OK] RRT*-Dubins 找到路径: {i+1} 次迭代, {elapsed:.2f}秒, "
                                  f"代价={self.goal.cost:.3f}m")
                            print(f"       起点: ({self.start.pose[0]:.3f}, {self.start.pose[1]:.3f}, {np.rad2deg(self.start.pose[2]):.1f}°)")
                            print(f"       终点: ({self.goal.pose[0]:.3f}, {self.goal.pose[1]:.3f}, {np.rad2deg(self.goal.pose[2]):.1f}°)")

                        return self._extract_full_path()

        if verbose:
            print(f"  [FAIL] RRT*-Dubins 失败: 达到最大迭代次数 {max_iter}")
            print(f"    统计: {len(self.nodes)} 节点, 碰撞={collision_count}, Dubins失败={dubins_fail_count}")
            min_dist = min(np.linalg.norm(node.pose[:2] - self.goal.pose[:2]) for node in self.nodes)
            print(f"    最近距离目标: {min_dist:.4f}m (阈值={Config.RRT_GOAL_THRESHOLD}m)")
        return None

    def _sample(self, goal_bias: float = None) -> np.ndarray:
        """随机采样位姿 [x, y, theta]"""
        if goal_bias is None:
            goal_bias = Config.RRT_GOAL_SAMPLE_RATE
        if np.random.rand() < goal_bias:
            # 目标采样：位置固定，方向随机（因为不约束终点方向）
            goal_x, goal_y = self.goal.pose[0], self.goal.pose[1]
            random_theta = np.random.uniform(-np.pi, np.pi)
            return np.array([goal_x, goal_y, random_theta])
        else:
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            y = np.random.uniform(self.bounds[2], self.bounds[3])
            theta = np.random.uniform(-np.pi, np.pi)
            return np.array([x, y, theta])

    def _nearest(self, pose: np.ndarray) -> Node:
        """找最近节点（欧氏距离，忽略角度）"""
        distances = [np.linalg.norm(node.pose[:2] - pose[:2]) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def _near(self, node: Node) -> List[Node]:
        """找附近节点"""
        radius = Config.RRT_SEARCH_RADIUS
        near_nodes = []
        for n in self.nodes:
            if np.linalg.norm(n.pose[:2] - node.pose[:2]) < radius:
                near_nodes.append(n)
        return near_nodes

    def _extract_full_path(self) -> List[DubinsPath]:
        """提取完整Dubins路径（节点间用Dubins曲线连接）"""
        path_segments = []
        node = self.goal_node
        
        while node.parent is not None:
            dubins = compute_dubins_path(node.parent.pose, node.pose, self.radius)
            if dubins is not None:
                path_segments.append(dubins)
            node = node.parent
        
        path_segments.reverse()
        return path_segments

    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取树的边（用于可视化）"""
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append((node.parent.pose[:2], node.pose[:2]))
        return edges


# ============================================================================
# 速度剖面生成
# ============================================================================

def compute_curvature_from_dubins(path: np.ndarray) -> np.ndarray:
    """
    计算Dubins路径的曲率
    
    由于Dubins路径由圆弧和直线组成，曲率为常数段
    """
    if len(path) < 3:
        return np.zeros(len(path))

    curvatures = np.zeros(len(path))

    for i in range(1, len(path) - 1):
        p0, p1, p2 = path[i-1], path[i], path[i+1]

        v1 = p1[:2] - p0[:2]
        v2 = p2[:2] - p1[:2]

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 < 1e-6 or len_v2 < 1e-6:
            curvatures[i] = 0
            continue

        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        chord_length = np.linalg.norm(p2[:2] - p0[:2])
        if chord_length > 1e-6:
            curvatures[i] = 2 * np.sin(angle / 2) / chord_length
        else:
            curvatures[i] = 0

    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]

    return curvatures


def generate_velocity_profile(path: np.ndarray, verbose: bool = False) -> Dict:
    """
    生成速度剖面（与原文件相同的差速约束逻辑）
    
    Args:
        path: 路径 shape=(N, 3) [x, y, theta]
        verbose: 是否打印详细信息
        
    Returns:
        轨迹字典
    """
    n_points = len(path)
    L = Config.WHEELBASE

    curvatures = compute_curvature_from_dubins(path)
    max_curvature = float(np.max(np.abs(curvatures)))

    velocities = np.zeros((n_points, 2))       # [v, omega]
    wheel_velocities = np.zeros((n_points, 2)) # [v_L, v_R]
    scale_factors = np.zeros(n_points)

    for i in range(n_points):
        kappa = abs(curvatures[i])

        v_limit_1 = Config.WHEEL_V_MAX

        if kappa > 1e-6:
            v_limit_2 = Config.OMEGA_MAX / kappa
        else:
            v_limit_2 = float('inf')

        v_limit_3 = Config.WHEEL_V_MAX / (1 + kappa * L / 2)

        v_allowed = min(v_limit_1, v_limit_2, v_limit_3)
        v = max(v_allowed, Config.V_MIN)

        omega = v * curvatures[i]

        v_L = v - omega * L / 2
        v_R = v + omega * L / 2

        velocities[i] = [v, omega]
        wheel_velocities[i] = [v_L, v_R]
        scale_factors[i] = v / Config.WHEEL_V_MAX if Config.WHEEL_V_MAX > 0 else 1.0

    timestamps = [0.0]
    total_distance = 0.0

    for i in range(1, n_points):
        segment_length = np.linalg.norm(path[i, :2] - path[i-1, :2])
        avg_velocity = (velocities[i-1, 0] + velocities[i, 0]) / 2

        if avg_velocity > 1e-6:
            dt = segment_length / avg_velocity
        else:
            dt = Config.DT

        timestamps.append(timestamps[-1] + dt)
        total_distance += segment_length

    max_v = float(np.max(velocities[:, 0]))
    max_omega = float(np.max(np.abs(velocities[:, 1])))
    max_wheel_v = float(np.max(np.abs(wheel_velocities)))
    min_scale = float(np.min(scale_factors))
    avg_scale = float(np.mean(scale_factors))

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
              f"平均缩放因子: {avg_scale:.3f}")
        print(f"    [轮速范围] v_L: [{np.min(wheel_velocities[:,0]):.4f}, {np.max(wheel_velocities[:,0]):.4f}] m/s")
        print(f"    [轮速范围] v_R: [{np.min(wheel_velocities[:,1]):.4f}, {np.max(wheel_velocities[:,1]):.4f}] m/s")

    return {
        'positions': path.tolist(),
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
    """检查轨迹是否满足差速驱动运动学约束"""
    velocities = np.array(trajectory['velocities'])
    wheel_velocities = np.array(trajectory.get('wheel_velocities', []))

    v_vals = velocities[:, 0]
    omega_vals = velocities[:, 1]

    v_max_actual = np.max(v_vals)
    omega_max_actual = np.max(np.abs(omega_vals))

    v_satisfied = v_max_actual <= Config.WHEEL_V_MAX + 1e-6
    omega_satisfied = omega_max_actual <= Config.OMEGA_MAX + 1e-6

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

    return bool(satisfied), message


# ============================================================================
# 可视化模块
# ============================================================================

def plot_failure_visualization(env: Dict, rrt_tree: List[Tuple], save_path: str):
    """
    绘制规划失败时的可视化（显示已探索的RRT树）
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. 障碍物
    for obs in env['obstacles']:
        if obs.geom_type == 'Polygon':
            x, y = obs.exterior.xy
            ax.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)
        elif obs.geom_type == 'MultiPolygon':
            for poly in obs.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)
    
    # 2. RRT树（显示已探索的空间）
    for edge in rrt_tree:
        p1, p2 = edge
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c-', alpha=0.3, linewidth=1)
    
    # 3. 起点终点
    start = env['start']
    goal = env['goal']
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', zorder=10)
    
    # 起点和终点的朝向箭头
    arrow_length = 0.05
    ax.arrow(start[0], start[1], 
             arrow_length * np.cos(start[2]), 
             arrow_length * np.sin(start[2]),
             head_width=0.02, head_length=0.02, fc='green', ec='green', zorder=10)
    ax.arrow(goal[0], goal[1], 
             arrow_length * np.cos(goal[2]), 
             arrow_length * np.sin(goal[2]),
             head_width=0.02, head_length=0.02, fc='red', ec='red', zorder=10)
    
    bounds = env['bounds']
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(f'RRT*-Dubins Planning FAILED - {env["map_name"]}\n(Explored Tree Shown)', 
                fontsize=14, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] 失败场景图保存: {save_path}")


def plot_trajectory_combined(env: Dict, dubins_segments: List[DubinsPath],
                             full_path: np.ndarray, trajectory: Dict,
                             rrt_tree: List[Tuple], save_path: str):
    """
    绘制Dubins轨迹图
    
    左侧：轨迹图（RRT树、Dubins路径段、速度编码）
    右侧：速度剖面
    """
    fig = plt.figure(figsize=(20, 12))

    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.25)

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_omega = fig.add_subplot(gs[1, 1])
    ax_wheel = fig.add_subplot(gs[2, 1])

    velocities = np.array(trajectory['velocities'])
    wheel_velocities = np.array(trajectory.get('wheel_velocities', []))
    timestamps = np.array(trajectory['timestamps'])
    v_vals = velocities[:, 0]
    omega_vals = velocities[:, 1]

    # ==================== 左侧：轨迹图 ====================
    # 1. 障碍物
    for obs in env['obstacles']:
        if obs.geom_type == 'Polygon':
            x, y = obs.exterior.xy
            ax_traj.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)
        elif obs.geom_type == 'MultiPolygon':
            for poly in obs.geoms:
                x, y = poly.exterior.xy
                ax_traj.fill(x, y, color='gray', alpha=0.5, edgecolor='black', linewidth=1)

    # 2. RRT树
    for edge in rrt_tree:
        p1, p2 = edge
        ax_traj.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c-', alpha=0.15, linewidth=0.5)

    # 3. Dubins路径段（不同颜色表示不同类型）
    colors = {'LSL': 'orange', 'RSR': 'purple', 'LSR': 'green', 
              'RSL': 'blue', 'RLR': 'red', 'LRL': 'brown'}
    
    # 记录所有段的连接点
    connection_points = []
    
    for i, seg in enumerate(dubins_segments):
        seg_path = seg.sample(0.01)
        color = colors.get(seg.path_type.name, 'gray')
        ax_traj.plot(seg_path[:, 0], seg_path[:, 1], '-', 
                    color=color, linewidth=1.5, alpha=0.6,
                    label=seg.path_type.name if i == 0 or seg.path_type.name not in [s.path_type.name for s in dubins_segments[:i]] else "")
        
        # 记录段的起点和终点（用于标记连接点）
        if i > 0:
            connection_points.append(seg_path[0])
    
    # 标记Dubins段之间的连接点（帮助检查连续性）
    if connection_points:
        conn_pts = np.array(connection_points)
        ax_traj.scatter(conn_pts[:, 0], conn_pts[:, 1], 
                       c='yellow', s=50, marker='o', edgecolors='black', linewidth=1,
                       label='Segment Connections', zorder=15, alpha=0.8)

    # 4. 完整路径（速度编码）
    if full_path is not None and len(full_path) > 1:
        points = full_path[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='jet', linewidths=3)
        lc.set_array(v_vals[:-1])
        lc.set_clim(0, Config.WHEEL_V_MAX)
        line = ax_traj.add_collection(lc)

        cbar = fig.colorbar(line, ax=ax_traj, shrink=0.6, pad=0.02)
        cbar.set_label('Velocity (m/s)', fontsize=11)

    # 5. 起点终点
    start = env['start']
    goal = env['goal']
    ax_traj.plot(start[0], start[1], 'go', markersize=15, label='Start', zorder=10)
    ax_traj.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal', zorder=10)

    # 绘制起点和终点的朝向箭头
    arrow_length = 0.05
    ax_traj.arrow(start[0], start[1], 
                 arrow_length * np.cos(start[2]), 
                 arrow_length * np.sin(start[2]),
                 head_width=0.02, head_length=0.02, fc='green', ec='green', zorder=10)
    ax_traj.arrow(goal[0], goal[1], 
                 arrow_length * np.cos(goal[2]), 
                 arrow_length * np.sin(goal[2]),
                 head_width=0.02, head_length=0.02, fc='red', ec='red', zorder=10)

    bounds = env['bounds']
    ax_traj.set_xlim(bounds[0], bounds[1])
    ax_traj.set_ylim(bounds[2], bounds[3])

    ax_traj.set_xlabel('X (m)', fontsize=12)
    ax_traj.set_ylabel('Y (m)', fontsize=12)
    ax_traj.legend(loc='upper right', fontsize=9)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal')
    ax_traj.set_title(f'Dubins Trajectory - {env["map_name"]}', fontsize=13, fontweight='bold')

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

    fig.suptitle(f'RRT* + Dubins Trajectory - {env["map_name"]}',
                 fontsize=16, fontweight='bold', y=0.99)

    info_line = (f"Finish Time: {trajectory['total_time']:.2f}s  |  "
                 f"Distance: {trajectory['total_distance']:.4f}m  |  "
                 f"Max v: {trajectory['max_v']:.4f} m/s  |  "
                 f"Max |ω|: {trajectory['max_omega']:.4f} rad/s  |  "
                 f"Max |v_wheel|: {max_wheel_v:.4f} m/s  |  "
                 f"Dubins R_min: {Config.DUBINS_MIN_RADIUS:.4f}m")

    fig.text(0.5, 0.01, info_line, ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Dubins轨迹图保存: {save_path}")


# ============================================================================
# 主处理流程
# ============================================================================

def process_single_map(env_path: str, output_dir: str, verbose: bool = True) -> bool:
    """处理单个MAP（使用Dubins路径）"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"处理: {env_path}")
        print(f"{'='*60}")

    try:
        # 1. 加载环境
        env = load_environment(env_path)
        if verbose:
            print(f"[OK] 环境加载完成: {env['map_name']}")
            print(f"  起点: ({env['start'][0]:.3f}, {env['start'][1]:.3f}) [方向不约束]")
            print(f"  终点: ({env['goal'][0]:.3f}, {env['goal'][1]:.3f}) [方向不约束]")
            print(f"  障碍物: {len(env['obstacles'])} 个")
            print(f"  Dubins最小转弯半径: {Config.DUBINS_MIN_RADIUS:.4f} m")

        # 2. RRT*-Dubins 规划
        if verbose:
            print(f"\n执行 RRT*-Dubins 规划...")

        # 起点方向不约束：设置初始朝向指向目标点（更自然的探索方向）
        start_pos = env['start'][:2]
        goal_pos = env['goal'][:2]
        initial_heading = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
        start_pose_free = np.array([start_pos[0], start_pos[1], initial_heading])

        attempts = 10
        best_segments = None
        best_cost = float('inf')
        best_rrt = None
        last_rrt = None

        for attempt in range(1, attempts + 1):
            rrt = RRTStarDubins(
                start=start_pose_free,
                goal=env['goal'],
                bounds=env['bounds'],
                obstacles_union=env['obstacles_union_expanded'],
                radius=Config.DUBINS_MIN_RADIUS
            )
            last_rrt = rrt

            candidate_segments = rrt.plan(verbose=verbose)
            if candidate_segments is None or len(candidate_segments) == 0:
                if verbose:
                    print(f"  [FAIL] 候选路径 {attempt}/{attempts} 失败")
                continue

            candidate_cost = sum(seg.total_length for seg in candidate_segments)
            if verbose:
                print(f"  [OK] 候选路径 {attempt}/{attempts}，长度={candidate_cost:.4f}m")

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_segments = candidate_segments
                best_rrt = rrt

        dubins_segments = best_segments

        # 准备输出目录（无论成功失败都要保存）
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        map_name = env['map_name']

        if dubins_segments is None or len(dubins_segments) == 0:
            print(f"  [FAIL] RRT*-Dubins 规划失败，保存失败信息...")
            
            # 保存失败信息
            failure_output = {
                'map_name': map_name,
                'status': 'FAILED',
                'reason': 'RRT*-Dubins planning failed to find path',
                'coordinate_frame': 'world_meter',
                'start_pose': env['start'],
                'goal_pose': env['goal'],
                'dubins_min_radius': Config.DUBINS_MIN_RADIUS,
                'max_iterations': Config.RRT_MAX_ITER,
                'bounds': env['bounds']
            }
            
            failure_json_path = output_path / f'dubins_FAILED_{map_name}.json'
            with open(failure_json_path, 'w', encoding='utf-8') as f:
                json.dump(failure_output, f, indent=2)
            
            print(f"  [OK] 失败信息已保存: {failure_json_path}")
            
            # 保存失败时的可视化（显示已探索的RRT树）
            if verbose:
                print(f"  生成失败场景可视化...")
            
            vis_path = output_path / f'dubins_FAILED_{map_name}.png'
            tree_source = last_rrt if last_rrt is not None else RRTStarDubins(
                start=start_pose_free,
                goal=env['goal'],
                bounds=env['bounds'],
                obstacles_union=env['obstacles_union_expanded'],
                radius=Config.DUBINS_MIN_RADIUS
            )
            plot_failure_visualization(env, tree_source.get_tree_edges(), str(vis_path))
            
            return False

        if verbose:
            print(f"  [OK] Dubins最短路径已选定，长度={best_cost:.4f}m")

        # 3. 采样完整路径（确保位置和方向连续性）
        full_path = []
        for i, seg in enumerate(dubins_segments):
            seg_points = seg.sample(Config.PATH_RESOLUTION)
            
            # 对于非首段，强制当前段起点匹配前一段终点（包括方向）
            if i > 0 and len(full_path) > 0:
                last_point = full_path[-1][-1]  # 前一段的最后一点
                first_point = seg_points[0]     # 当前段的第一点
                
                # 检查距离，如果太远说明有gap
                gap_dist = np.linalg.norm(last_point[:2] - first_point[:2])
                
                # 强制当前段第一个点的方向与前一段最后一点的方向一致
                seg_points[0] = last_point.copy()
                
                if gap_dist > Config.PATH_RESOLUTION * 2:
                    # 有明显gap，插入连接点
                    if verbose:
                        print(f"    [WARN] 段{i}与段{i-1}间有gap {gap_dist*1000:.2f}mm，插入连接")
                    # 在gap中插入几个点来平滑过渡
                    num_bridge = max(2, int(gap_dist / Config.PATH_RESOLUTION))
                    for j in range(1, num_bridge):
                        t = j / num_bridge
                        bridge_x = last_point[0] * (1-t) + first_point[0] * t
                        bridge_y = last_point[1] * (1-t) + first_point[1] * t
                        # 方向也要平滑插值（考虑角度周期性）
                        theta_diff = normalize_angle(first_point[2] - last_point[2])
                        bridge_theta = normalize_angle(last_point[2] + theta_diff * t)
                        # 插入到前一段的末尾
                        full_path[-1] = np.vstack([full_path[-1], 
                                                   np.array([bridge_x, bridge_y, bridge_theta])])
                
                # 跳过当前段的第一个点（因为已经被前一段覆盖）
                seg_points = seg_points[1:]
            
            if len(seg_points) > 0:
                full_path.append(seg_points)
        
        full_path = np.vstack(full_path)
        
        # 检查路径连续性（位置）
        max_gap_before = 0.0
        max_angle_jump = 0.0
        for i in range(1, len(full_path)):
            gap = np.linalg.norm(full_path[i][:2] - full_path[i-1][:2])
            angle_jump = abs(normalize_angle(full_path[i][2] - full_path[i-1][2]))
            max_gap_before = max(max_gap_before, gap)
            max_angle_jump = max(max_angle_jump, angle_jump)
        
        # 如果有大gap，修复连续性
        if max_gap_before > Config.PATH_RESOLUTION * 2:
            if verbose:
                print(f"  [WARN] 检测到不连续点，修复中...")
            full_path = ensure_path_continuity(full_path, max_gap=Config.PATH_RESOLUTION * 1.5)
            
            # 再次检查
            max_gap_after = 0.0
            for i in range(1, len(full_path)):
                gap = np.linalg.norm(full_path[i][:2] - full_path[i-1][:2])
                max_gap_after = max(max_gap_after, gap)
            
            if verbose:
                print(f"       修复前最大间距: {max_gap_before*1000:.3f}mm")
                print(f"       修复后最大间距: {max_gap_after*1000:.3f}mm")
        
        # 额外的方向连续性检查和修复
        if max_angle_jump > 0.3:  # 如果存在大于约17度的跳变
            if verbose:
                print(f"  [WARN] 检测到方向不连续（最大跳变: {np.rad2deg(max_angle_jump):.1f}°），修复中...")
            full_path = ensure_orientation_continuity(full_path)
            
            # 再次检查
            max_angle_jump_after = 0.0
            for i in range(1, len(full_path)):
                angle_jump = abs(normalize_angle(full_path[i][2] - full_path[i-1][2]))
                max_angle_jump_after = max(max_angle_jump_after, angle_jump)
            
            if verbose:
                print(f"       修复前最大角度跳变: {np.rad2deg(max_angle_jump):.1f}°")
                print(f"       修复后最大角度跳变: {np.rad2deg(max_angle_jump_after):.1f}°")
        
        if verbose:
            print(f"  [OK] 完整路径采样: {len(full_path)} 点，位置和方向连续性良好")

        # 4. 生成速度剖面
        if verbose:
            print(f"\n生成速度剖面...")

        trajectory = generate_velocity_profile(full_path, verbose=verbose)

        # 5. 检查运动学约束
        satisfied, constraint_msg = check_kinematics_constraints(trajectory)
        print(f"  {constraint_msg}")

        if verbose:
            print(f"\n  {'='*40}")
            print(f"  轨迹完成时间: {trajectory['total_time']:.2f} 秒")
            print(f"  轨迹总距离: {trajectory['total_distance']:.4f} m")
            print(f"  实际最大速度: v_max={trajectory['max_v']:.4f} m/s, |ω|_max={trajectory['max_omega']:.4f} rad/s")
            print(f"  {'='*40}")

        # 6. 保存结果
        # 保存Dubins路径
        dubins_output = {
            'map_name': map_name,
            'status': 'SUCCESS',
            'coordinate_frame': 'world_meter',
            'start_pose': env['start'],
            'goal_pose': env['goal'],
            'dubins_min_radius': Config.DUBINS_MIN_RADIUS,
            'segments': [
                {
                    'type': seg.path_type.name,
                    'lengths': seg.lengths,
                    'total_length': seg.total_length,
                    'start': seg.start.tolist(),
                    'end': seg.end.tolist()
                }
                for seg in dubins_segments
            ]
        }

        dubins_json_path = output_path / f'dubins_path_{map_name}.json'
        with open(dubins_json_path, 'w', encoding='utf-8') as f:
            json.dump(dubins_output, f, indent=2)

        if verbose:
            print(f"  [OK] Dubins路径保存: {dubins_json_path}")

        # 保存轨迹
        trajectory_output = {
            'map_name': map_name,
            'coordinate_frame': 'world_meter',
            'start_pose': env['start'],
            'goal_pose': env['goal'],
            'full_path': full_path.tolist(),
            'trajectory': trajectory,
            'constraints': {
                'v_max_limit': Config.V_MAX,
                'omega_max_limit': Config.OMEGA_MAX,
                'dubins_min_radius': Config.DUBINS_MIN_RADIUS,
                'satisfied': satisfied
            },
            'summary': {
                'total_time_seconds': trajectory['total_time'],
                'total_distance_meters': trajectory['total_distance'],
                'actual_max_v': trajectory['max_v'],
                'actual_max_omega': trajectory['max_omega'],
                'scale_info': trajectory['scale_info']
            }
        }

        traj_json_path = output_path / f'dubins_trajectory_{map_name}.json'
        with open(traj_json_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_output, f, indent=2)

        if verbose:
            print(f"  [OK] Dubins轨迹保存: {traj_json_path}")

        # 7. 可视化
        if verbose:
            print(f"\n生成可视化...")

        vis_path = output_path / f'dubins_trajectory_{map_name}.png'
        plot_trajectory_combined(env, dubins_segments, full_path, trajectory, 
                               rrt.get_tree_edges(), str(vis_path))

        if verbose:
            print(f"\n{'='*60}")
            print(f"[OK] {map_name} 处理完成")
            print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n[FAIL] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存异常信息
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            map_name = Path(env_path).parent.name
            
            error_output = {
                'map_name': map_name,
                'status': 'ERROR',
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            
            error_json_path = output_path / f'dubins_ERROR_{map_name}.json'
            with open(error_json_path, 'w', encoding='utf-8') as f:
                json.dump(error_output, f, indent=2)
            
            print(f"  [OK] 错误信息已保存: {error_json_path}")
        except Exception as save_error:
            print(f"  [WARN] 无法保存错误信息: {save_error}")
        
        return False


def batch_process_maps(maps_dir: str, output_base_dir: str):
    """批量处理所有MAP（Dubins版本）"""
    maps_path = Path(maps_dir)

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

        if map_name.startswith('warehouse_'):
            print(f"[SKIP] 跳过 warehouse_N 地图: {map_name}")
            continue

        output_dir = Path(output_base_dir) / map_name

        success = process_single_map(str(env_file), str(output_dir), verbose=True)

        if success:
            success_count += 1
        else:
            failed_maps.append(map_name)

    print(f"\n\n{'='*60}")
    print(f"批量处理完成 (Dubins)")
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
    output_dir = r"d:\Data_visualization_code\result\RRT_Dubins_Results"

    batch_process_maps(maps_dir, output_dir)


if __name__ == '__main__':
    main()
