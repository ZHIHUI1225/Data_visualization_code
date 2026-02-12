"""
测试单个MAP的RRT规划
"""

from rrt_bezier_planner import (
    Config, load_environment, RRTStar, simplify_path,
    smooth_path_with_bezier, generate_velocity_profile,
    check_kinematics_constraints, plot_path
)
from pathlib import Path
import json

# 加载配置
config_path = r"d:\Data_visualization_code\result\romi\config.yaml"
if Path(config_path).exists():
    Config.load_from_yaml(config_path)
    print("✓ 配置已加载")
else:
    print("⚠ 使用默认配置")

# 测试MAP6
env_path = r"d:\Data_visualization_code\result\MAPS\MAP6\environment_map6.json"
output_dir = r"d:\Data_visualization_code\result\RRT_Results\MAP6"

print(f"\n{'='*60}")
print(f"测试: {env_path}")
print(f"{'='*60}\n")

try:
    # 1. 加载环境
    print("1. 加载环境...")
    env = load_environment(env_path)
    print(f"   ✓ 地图: {env['map_name']}")
    print(f"   ✓ 起点: ({env['start'][0]:.3f}, {env['start'][1]:.3f}, {env['start'][2]:.3f})")
    print(f"   ✓ 终点: ({env['goal'][0]:.3f}, {env['goal'][1]:.3f}, {env['goal'][2]:.3f})")
    print(f"   ✓ 障碍物: {len(env['obstacles'])} 个")

    # 2. RRT* 规划
    print(f"\n2. 执行 RRT* 规划 (障碍物膨胀 {Config.OBSTACLE_EXPANSION}m)...")
    rrt = RRTStar(
        start=env['start'][:2],
        goal=env['goal'][:2],
        bounds=env['bounds'],
        obstacles_union=env['obstacles_union_expanded']
    )

    rrt_path = rrt.plan(verbose=True)

    if rrt_path is None:
        print("   ✗ RRT* 规划失败")
        exit(1)

    # 3. 路径简化
    print("\n3. 路径简化...")
    simplified_path = simplify_path(rrt_path, env['obstacles_union_expanded'],
                                   bounds=env['bounds'])
    print(f"   ✓ {len(rrt_path)} → {len(simplified_path)} 点")

    # 4. Bezier 平滑
    print("\n4. Bezier 平滑 (带碰撞和边界检查)...")
    smooth_path, has_collision = smooth_path_with_bezier(simplified_path,
                                                         obstacles_union=env['obstacles_union_expanded'],
                                                         bounds=env['bounds'])
    if has_collision:
        print(f"   [WARN] 生成 {len(smooth_path)} 平滑点 - 但有碰撞!")
    else:
        print(f"   ✓ 生成 {len(smooth_path)} 平滑点")

    # 5. 生成速度剖面
    print("\n5. 生成速度剖面...")
    trajectory = generate_velocity_profile(smooth_path)
    print(f"   ✓ 总距离: {trajectory['total_distance']:.3f} m")
    print(f"   ✓ 总时间: {trajectory['total_time']:.2f} s")

    # 6. 检查约束
    print("\n6. 检查运动学约束...")
    satisfied, msg = check_kinematics_constraints(trajectory)
    print(f"   {msg}")

    # 7. 保存结果
    print("\n7. 保存结果...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存RRT路径
    rrt_output = {
        'map_name': env['map_name'],
        'coordinate_frame': 'world_meter',
        'start_pose': env['start'],
        'goal_pose': env['goal'],
        'waypoints': [p.tolist() for p in simplified_path]
    }

    rrt_json_path = Path(output_dir) / f"rrt_waypoints_{env['map_name']}.json"
    with open(rrt_json_path, 'w', encoding='utf-8') as f:
        json.dump(rrt_output, f, indent=2)
    print(f"   ✓ RRT路径: {rrt_json_path}")

    # 保存Bezier轨迹
    trajectory_output = {
        'map_name': env['map_name'],
        'coordinate_frame': 'world_meter',
        'start_pose': env['start'],
        'goal_pose': env['goal'],
        'smooth_path': smooth_path.tolist(),
        'trajectory': trajectory,
        'constraints': {
            'v_max': Config.V_MAX,
            'omega_max': Config.OMEGA_MAX,
            'satisfied': satisfied
        }
    }

    traj_json_path = Path(output_dir) / f"bezier_trajectory_{env['map_name']}.json"
    with open(traj_json_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_output, f, indent=2)
    print(f"   ✓ Bezier轨迹: {traj_json_path}")

    # 8. 可视化
    print("\n8. 生成可视化...")
    vis_path = Path(output_dir) / f"path_visualization_{env['map_name']}.png"
    plot_path(env, simplified_path, smooth_path, trajectory, rrt.get_tree(), str(vis_path), has_collision)

    print(f"\n{'='*60}")
    print("✓ 测试完成！")
    print(f"{'='*60}")

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
