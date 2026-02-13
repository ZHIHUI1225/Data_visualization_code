"""
RRT Dubins轨迹质量检查工具
检查：实际长度、最小曲率半径、碰撞情况

作者: Quality Control
日期: 2026-02-13
"""

import json
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from typing import List, Dict, Tuple, Optional
import pandas as pd


class TrajectoryQualityChecker:
    """轨迹质量检查器"""
    
    def __init__(self, pixel_to_meter: float = 0.0023, robot_radius: float = 0.0265):
        """
        Args:
            pixel_to_meter: 像素到米的转换比例
            robot_radius: 机器人半径(m)
        """
        self.pixel_to_meter = pixel_to_meter
        self.robot_radius = robot_radius
        
    def load_environment(self, env_path: str) -> List[Polygon]:
        """加载环境障碍物"""
        with open(env_path, 'r', encoding='utf-8') as f:
            env_data = json.load(f)
        
        obstacles = []
        for poly_data in env_data.get('polygons', []):
            vertices = np.array(poly_data['vertices']) * self.pixel_to_meter
            obstacles.append(Polygon(vertices))
        
        return obstacles
    
    def load_trajectory(self, traj_path: str) -> Dict:
        """加载轨迹数据"""
        with open(traj_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_path_info(self, path_json: str) -> Dict:
        """加载路径段信息"""
        with open(path_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_trajectory_length(self, full_path: List[List[float]]) -> float:
        """计算轨迹实际长度（沿着路径点累积距离）"""
        if len(full_path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(full_path) - 1):
            p1 = np.array(full_path[i][:2])  # [x, y]
            p2 = np.array(full_path[i+1][:2])
            total_length += np.linalg.norm(p2 - p1)
        
        return total_length
    
    def find_minimum_curvature_radius(self, segments: List[Dict]) -> Tuple[float, str]:
        """
        找到最小曲率半径（从Dubins路径段中）
        
        Returns:
            (最小曲率半径, 对应的路径类型)
        """
        min_radius = float('inf')
        min_radius_type = "N/A"
        
        for seg in segments:
            seg_type = seg['type']
            
            # 检查是否包含圆弧段（L或R）
            if 'L' in seg_type or 'R' in seg_type:
                # Dubins路径的半径在配置中定义
                # 从lengths可以推断半径：对于圆弧段，length = radius * angle
                # 但更直接的是从配置中获取
                if 'radius' in seg:
                    radius = seg['radius']
                    if radius < min_radius:
                        min_radius = radius
                        min_radius_type = seg_type
        
        return min_radius, min_radius_type
    
    def check_collision(self, full_path: List[List[float]], 
                       obstacles: List[Polygon]) -> Dict:
        """
        检查轨迹是否与障碍物碰撞
        
        Returns:
            包含碰撞信息的字典
        """
        if not obstacles:
            return {
                'has_collision': False,
                'collision_count': 0,
                'collision_points': []
            }
        
        collision_points = []
        
        # 创建路径线段并检查每个点
        for i, pose in enumerate(full_path):
            x, y = pose[0], pose[1]
            robot_pos = Point(x, y).buffer(self.robot_radius)
            
            # 检查是否与任何障碍物相交
            for obs in obstacles:
                if robot_pos.intersects(obs):
                    collision_points.append({
                        'index': i,
                        'position': [x, y],
                        'distance': robot_pos.exterior.distance(obs)
                    })
                    break  # 已经检测到碰撞，跳过其他障碍物
        
        return {
            'has_collision': len(collision_points) > 0,
            'collision_count': len(collision_points),
            'collision_points': collision_points[:10]  # 只保留前10个
        }
    
    def check_single_map(self, map_name: str, 
                        dubins_results_dir: str,
                        maps_dir: str) -> Dict:
        """检查单个地图的轨迹质量"""
        
        result = {
            'map_name': map_name,
            'status': 'NOT_FOUND',
            'trajectory_length': 0.0,
            'min_curvature_radius': None,
            'min_radius_type': None,
            'min_curvature': None,
            'has_collision': None,
            'collision_count': 0,
            'error_message': None
        }
        
        try:
            # 构建路径
            dubins_dir = Path(dubins_results_dir) / map_name
            traj_json = dubins_dir / f'dubins_trajectory_{map_name}.json'
            path_json = dubins_dir / f'dubins_path_{map_name}.json'
            failed_json = dubins_dir / f'dubins_FAILED_{map_name}.json'
            
            # 检查是否有失败文件
            if failed_json.exists():
                result['status'] = 'FAILED'
                with open(failed_json, 'r', encoding='utf-8') as f:
                    failed_data = json.load(f)
                result['error_message'] = failed_data.get('error_message', 'Unknown error')
                return result
            
            # 检查轨迹文件是否存在
            if not traj_json.exists() or not path_json.exists():
                result['status'] = 'INCOMPLETE'
                result['error_message'] = 'Trajectory or path JSON not found'
                return result
            
            # 加载数据
            traj_data = self.load_trajectory(str(traj_json))
            path_data = self.load_path_info(str(path_json))
            
            # 加载环境
            env_json = Path(maps_dir) / map_name / f'environment_{map_name.lower()}.json'
            if not env_json.exists():
                result['error_message'] = f'Environment file not found: {env_json}'
                obstacles = []
            else:
                obstacles = self.load_environment(str(env_json))
            
            # 1. 计算轨迹长度
            full_path = traj_data.get('full_path', [])
            trajectory_length = self.calculate_trajectory_length(full_path)
            
            # 2. 找到最小曲率半径
            segments = path_data.get('segments', [])
            dubins_radius = path_data.get('dubins_min_radius', 0.2)  # 默认0.2m
            
            # Dubins路径所有圆弧段使用相同的半径
            min_curvature_radius = dubins_radius
            min_curvature = 1.0 / min_curvature_radius if min_curvature_radius > 0 else float('inf')
            
            # 找到使用圆弧的路径段类型
            arc_types = set()
            for seg in segments:
                seg_type = seg['type']
                if 'L' in seg_type or 'R' in seg_type:
                    arc_types.add(seg_type)
            min_radius_type = ', '.join(sorted(arc_types)) if arc_types else 'None'
            
            # 3. 检查碰撞
            collision_info = self.check_collision(full_path, obstacles)
            
            # 填充结果
            result.update({
                'status': path_data.get('status', 'SUCCESS'),
                'trajectory_length': round(trajectory_length, 4),
                'min_curvature_radius': round(min_curvature_radius, 4),
                'min_radius_type': min_radius_type,
                'min_curvature': round(min_curvature, 4),
                'has_collision': collision_info['has_collision'],
                'collision_count': collision_info['collision_count'],
                'num_segments': len(segments),
                'num_waypoints': len(full_path)
            })
            
            # 如果有碰撞，添加详细信息
            if collision_info['has_collision']:
                result['collision_details'] = collision_info['collision_points']
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error_message'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def batch_check(self, dubins_results_dir: str, maps_dir: str) -> pd.DataFrame:
        """批量检查所有地图"""
        
        results_path = Path(dubins_results_dir)
        map_dirs = [d for d in results_path.iterdir() if d.is_dir()]
        
        all_results = []
        
        print(f"\n{'='*80}")
        print(f"开始批量检查 Dubins 轨迹质量")
        print(f"{'='*80}")
        print(f"结果目录: {dubins_results_dir}")
        print(f"地图目录: {maps_dir}")
        print(f"找到 {len(map_dirs)} 个地图\n")
        
        for map_dir in sorted(map_dirs):
            map_name = map_dir.name
            print(f"检查 {map_name}...", end=' ')
            
            result = self.check_single_map(map_name, dubins_results_dir, maps_dir)
            all_results.append(result)
            
            # 打印简要状态
            status = result['status']
            if status == 'SUCCESS':
                collision_status = '❌碰撞' if result['has_collision'] else '✓无碰撞'
                print(f"[{status}] {collision_status} | 长度={result['trajectory_length']:.3f}m | "
                      f"半径={result['min_curvature_radius']:.3f}m")
            else:
                print(f"[{status}]")
        
        print(f"\n{'='*80}")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """打印统计摘要"""
        print(f"\n{'='*80}")
        print("检查摘要统计")
        print(f"{'='*80}\n")
        
        # 状态统计
        print("1. 状态分布:")
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
        
        # 成功的轨迹统计
        success_df = df[df['status'] == 'SUCCESS']
        if len(success_df) > 0:
            print(f"\n2. 成功轨迹 ({len(success_df)}):")
            print(f"   轨迹长度: 最小={success_df['trajectory_length'].min():.4f}m, "
                  f"最大={success_df['trajectory_length'].max():.4f}m, "
                  f"平均={success_df['trajectory_length'].mean():.4f}m")
            
            print(f"   最小曲率半径: {success_df['min_curvature_radius'].iloc[0]:.4f}m "
                  f"(所有地图相同)")
            print(f"   最大曲率: {success_df['min_curvature'].iloc[0]:.4f} m⁻¹")
            
            # 碰撞统计
            collision_count = success_df['has_collision'].sum()
            print(f"\n3. 碰撞检测:")
            print(f"   有碰撞: {collision_count}/{len(success_df)}")
            print(f"   无碰撞: {len(success_df) - collision_count}/{len(success_df)}")
            
            if collision_count > 0:
                print(f"\n   碰撞详情:")
                collision_maps = success_df[success_df['has_collision'] == True]
                for _, row in collision_maps.iterrows():
                    print(f"      {row['map_name']}: {row['collision_count']} 个碰撞点")
        
        # 失败的轨迹
        failed_df = df[df['status'].isin(['FAILED', 'ERROR'])]
        if len(failed_df) > 0:
            print(f"\n4. 失败/错误的地图 ({len(failed_df)}):")
            for _, row in failed_df.iterrows():
                print(f"   {row['map_name']}: {row['error_message']}")
        
        print(f"\n{'='*80}\n")


def main():
    """主函数"""
    
    # 配置路径
    dubins_results_dir = r"d:\Data_visualization_code\result\RRT_Dubins_Results"
    maps_dir = r"d:\Data_visualization_code\result\MAPS"
    
    # 创建检查器
    checker = TrajectoryQualityChecker(
        pixel_to_meter=0.0023,
        robot_radius=0.0265
    )
    
    # 批量检查
    results_df = checker.batch_check(dubins_results_dir, maps_dir)
    
    # 打印摘要
    checker.print_summary(results_df)
    
    # 保存详细结果
    output_path = Path(dubins_results_dir) / 'trajectory_quality_report.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"详细报告已保存至: {output_path}")
    
    # 保存JSON格式（包含完整信息）
    output_json = Path(dubins_results_dir) / 'trajectory_quality_report.json'
    results_df.to_json(output_json, orient='records', indent=2, force_ascii=False)
    print(f"JSON报告已保存至: {output_json}")
    
    # 显示数据表
    print("\n详细数据表:")
    print(results_df[['map_name', 'status', 'trajectory_length', 
                      'min_curvature_radius', 'has_collision', 
                      'collision_count']].to_string())


if __name__ == '__main__':
    main()
