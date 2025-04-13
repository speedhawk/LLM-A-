import numpy as np
import torch
import openpyxl
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

def generate_map(file_path, sheet_name, map_range):
    """
    Load the map into the agent memory
    """
    wb = openpyxl.load_workbook(file_path)
    ws = wb[sheet_name]
    grid_map = []
    for row in ws[map_range]:
        map_row = []
        for cell in row:
            map_row.append(cell.value)
        grid_map.append(map_row)
    return np.array(grid_map)

def calculate_trajectory_metrics(path_nodes, start, goal):
    """
    Path Finding Metrics Calculation
    1. total_steps: The number of steps taken to reach the goal.
    2. avg_steering_angle: The average steering angle of the path.
    3. max_deviation_times: The number of times the path deviated from the straight line between start and goal.
    The deviation is defined as the angle between the current direction and the direction towards the goal.
    4. The angle is calculated using the arccosine of the dot product of the two vectors.
    5. The deviation is considered significant if the angle is greater than 90 degrees.
    6. The average steering angle is calculated as the sum of the angles divided by the number of steps.
    7. The maximum deviation times is the number of times the angle exceeds 90 degrees.
    8. The function returns a dictionary containing the metrics.
    """
    metrics = {
        'total_steps': len(path_nodes)-1,
        'avg_steering_angle': 0.0,
        'max_deviation_times': 0
    }
    
    if len(path_nodes) < 2:
        return metrics
    
    bas_dir_vector = np.array([0.0, 1.0])
    disp_vector = np.array(goal) - np.array(start)
    sum_angle = 0.0
    
    for i in range(1, len(path_nodes)):
        prev_node = np.array(path_nodes[i-1])
        curr_node = np.array(path_nodes[i])
        
        dir_vector = curr_node - prev_node
        dir_norm = np.linalg.norm(dir_vector)
        if dir_norm == 0:
            continue
            
        angle = np.arccos(np.clip(
            np.dot(dir_vector, bas_dir_vector) / 
            (dir_norm * np.linalg.norm(bas_dir_vector)),
            -1.0, 1.0
        ))
        
        dis_angle = np.arccos(np.clip(
            np.dot(dir_vector, disp_vector) /
            (dir_norm * np.linalg.norm(disp_vector)),
            -1.0, 1.0
        ))
        
        sum_angle += abs(angle)
        if abs(dis_angle) > np.pi/2:
            metrics['max_deviation_times'] += 1
            
        bas_dir_vector = dir_vector
    
    metrics['avg_steering_angle'] = round(sum_angle/(len(path_nodes)-1), 4)
    return metrics

def visualize_path(original_map, path_nodes, searched_nodes, start, goal):
    """
    Visualize the path and search area
    1. original_map: original map data
    2. path_nodes: path node list
    3. searched_nodes: search area node list
    4. start: starting point coordinates
    5. goal: end point coordinates
    6. Return the visualized map data
    """
    # 3D array conversion
    vis_map = np.stack([original_map]*3, axis=-1).astype(float)
    
    # Search area coloring (full RGB operation)
    for node in searched_nodes:
        x, y = node
        vis_map[x][y] = [0.0, 1.0, 0.0]  # 绿色
    
    # Path color gradient 
    color_0 = 1.0
    color_2 = 0.0
    delta_color = 1.0 / len(path_nodes)
    
    for i, (x, y) in enumerate(path_nodes):
        vis_map[x][y] = [color_0, 0.0, color_2]
        color_0 -= delta_color
        color_2 += delta_color
    
    # color start and goal
    sx, sy = start
    gx, gy = goal
    vis_map[sx][sy] = [1.0, 0.0, 1.0]  # pink
    vis_map[gx][gy] = [1.0, 1.0, 0.0]  # yellow
    
    return (vis_map * 255).astype(np.uint8)

def save_simulation_result(result_dict, file_path):
    """
    save simulation result to json file
    1. result_dict: dictionary containing simulation results
    2. file_path: path to save the json file
    3. The function will create the directory if it does not exist.
    4. The function will save the result_dict to a json file with indentation for readability.
    5. If the file already exists, it will be overwritten.
    6. If there is an error while saving, it will print the error message and raise an exception.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
    except Exception as e:
        print(f"Error saving result: {str(e)}")
        raise

def parse_node_string(node_str):
    """
    node analysis function
    node_str: string containing node coordinates in the format [x, y], extracing the coordinates using regex.
    """
    pattern = r'\[(\d+),\s*(\d+)\]'
    matches = re.findall(pattern, node_str)
    if not matches:
        raise ValueError("Invalid node format")
    return [int(matches[-1][0]), int(matches[-1][1])]

def validate_map_parameters(map_range):
    """
    Validate the map range format.
    map_range: string containing the map range in the format 'A1:AF32'
    """
    if not re.match(r'^[A-Z]+\d+:[A-Z]+\d+$', map_range):
        raise ValueError("Invalid map range format. Use excel format like 'A1:AF32'")
    
def setup_plot_style():
    """
    keep original plot style
    """
    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'lines.linewidth': 2,
        'axes.titlesize': 16,
        'axes.labelsize': 14
    })

def generate_heatmap(data, title):
    """
    heatmap generating function
    """
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap='viridis')
    fig.colorbar(heatmap, ax=ax)
    ax.set_title(title)
    return fig


def load_episodes(filename):

    """
    Load episodes from file which recording the training process
    """

    file_iter = open('../datafiles/outputs/records/{}.txt'.format(filename), 'r')
    episode = file_iter.read()
    file_iter.close()
    e = int(episode)
    return e


def load_records(filenames, agent):

    """
    Load records from files which recording 
    the training process including avg_scores, 
    avg_steps, total_scores, total_steps. 
    """

    for filename in filenames:
        if filename == 'episodes':
            continue     
        file_iter = open('../datafiles/outputs/records/{}.txt'.format(filename), 'r')
        while True:
            record = file_iter.readline()
            agent.records[filename].append(eval(record))
            if not record:
                file_iter.close()
                break

def save_episodes(filename, e):
    
    """
    Save episodes to file which recording the training process
    """
    file_iter = open('../datafiles/outputs/records/{}.txt'.format(filename), 'w')
    file_iter.write(str(e))
    file_iter.close()

def save_records(agent, filenames):

    """
    Save records to files which recording the training process including avg_scores, avg_steps, total_scores, total_steps
    """
    for filename in filenames:
        if filename == 'episodes':
            continue
        records = agent.records[filename]
        file_iter = open('../datafiles/outputs/records/{}.txt'.format(filename), 'a')
        file_iter.writelines([str(r)+'\n' for r in records[-agent.memory_len:]])
        file_iter.close()

    