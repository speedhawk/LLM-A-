import argparse
from llm_pathfinding.train import run

if __name__ == "__main__":

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run PPO training or evaluation.")
    
    # 添加命令行参数
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("--start_x", type=int, default=1, help="Start x coordinate")
    parser.add_argument("--start_y", type=int, default=22, help="Start y coordinate")
    parser.add_argument("--goal_x", type=int, default=19, help="Goal x coordinate")
    parser.add_argument("--goal_y", type=int, default=1, help="Goal y coordinate")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run")
    parser.add_argument("--map_name", type=str, choices=["aisle", "canyon", "double_door"], default="double_door", help="Please choose a map: 'aisle', 'canyon', or 'double_door'")
    parser.add_argument("--map_size", type=int, choices=[16, 24, 32], default=24, help="Map size: 16, 24, or 32")
    parser.add_argument("--train_mode", type=str, choices=["new", "continue"], default="new", help="Mode to train: 'new' or 'continue'")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    
    # 运行 PPO
    run(args.device, args.start_x, args.start_y, args.goal_x, args.goal_y, args.train_mode, args.map_name, args.map_size, args.episodes)

    
    