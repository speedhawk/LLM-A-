import llm_pathfinding.astar_llm as astar_llm
import llm_pathfinding.models.communication as communication
from examples.astar_llm import run

import argparse

def test(key, filename, model_name, env_description, max_margin, start_node, target_node, map_size, action_space):
    
    comm = communication.communication(key, model_name, env_description, max_margin, start_node, target_node)
    astar = astar_llm.AStarLLM(comm, filename, env_description, max_margin, start_node, target_node, map_size, action_space)

    open_queue, close_queue = run(astar)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test A* pathfinding with LLM")
    parser.add_argument("--key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--filename", type=str, required=True, help="Filename for the map")
    parser.add_argument("--model_name", type=str, required=True, help="LLM model name")
    parser.add_argument("--env_description", type=str, required=True, help="Environment description")
    parser.add_argument("--max_margin", type=int, required=True, help="Max margin for the map")
    parser.add_argument("--start_node", type=list, required=True, help="Start node coordinates")
    parser.add_argument("--target_node", type=list, required=True, help="Target node coordinates")
    parser.add_argument("--map_size", type=int, required=True, help="Size of the map")
    parser.add_argument("--action_space", type=list, required=True, help="Action space for the agent")

    argparse_args = parser.parse_args()
    
    # Convert string tuples to actual tuples
    argparse_args.start_node = list(map(int, argparse_args.start_node.strip("()").split(",")))
    argparse_args.target_node = list(map(int, argparse_args.target_node.strip("()").split(",")))
    argparse_args.action_space = list(map(int, argparse_args.action_space.strip("[]").split(",")))
    test(argparse_args.key, argparse_args.filename, argparse_args.model_name, argparse_args.env_description, argparse_args.max_margin, argparse_args.start_node, argparse_args.target_node, argparse_args.map_size, argparse_args.action_space)
