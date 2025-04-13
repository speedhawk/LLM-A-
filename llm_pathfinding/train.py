import os
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import numpy as np
from .ppo import PPO
from .utils import generate_map, load_records, load_episodes, save_episodes, save_records

def train_process(agent, e, episodes, filenames):

    """
    Train process for each episode of two cases:
    1. Initialize the agent's start node and reset the agent, meaning this will start a new training
    2. Supporting restrating training process from the last saved episode
    """


    default_is_solved = False

    default_mode = 'Train'

    path_nodes = [agent.start_node.tolist()]
    searched_space = []

    agent.actor.train()
    agent.critic.train()

    Transition = namedtuple('Transition', ['state', 'action', 'pre_prob', 'reward', 'next_state'])
    for e in range(e, episodes):

        agent.start_node = agent.vectorized_start(e)
        agent.reset()
        total_step = 0
        score = 0.0
        done_time = 0

        index = e % agent.memory_len
        agent.memory[index] = []


        while list(agent.cur_node) != list(agent.target_node) and total_step < agent.step_thr:

            obs = np.array(agent.double_hot(agent.cur_node) + agent.situation(agent.cur_node))
            obs = torch.from_numpy(obs).type(torch.FloatTensor).unsqueeze(0).to(agent.device)

            # print(f"state_obs_size: {obs.size()}")

            act, posb = agent.action_select(obs)

            # print(possibilities)

            # v_s_real = 0
            # for i in range(0, 9):
            #   if i == 0:
            #     next_s = obs + torch.tensor([0, 0])
            #   else:
            #     next_s = obs + torch.tensor(agent.action_space[i-1])
            #   if possibilities[i] == 0.0:
            #     v_s_real += 0.0
            #   else:
            #     v_s_real += possibilities[i] * (agent.get_reward(i) + agent.gamma * agent.critic(next_s))

            next_obs, reward, done, _ = agent.step(act)
            n_o = np.array(agent.double_hot(next_obs[0])+agent.situation(next_obs[0]))
            n_o = torch.from_numpy(n_o).type(torch.FloatTensor).unsqueeze(0).to(agent.device)

            # trans = Transition(obs, act, posb, entropy, reward, n_o, agent.x_max, agent.x_min, agent.y_max, agent.y_min)
            trans = Transition(obs, act, posb, reward, n_o)
            agent.save_trans(index, trans)
            score += reward

        # done processing 1

            if done == True:
                done_time += 1
            else:
                agent.pre_node = agent.cur_node
                agent.cur_node = next_obs[0]

            # done processing 2

            # if done == True:
            #   break

            # agent.pre_node = agent.cur_node
            # agent.cur_node = next_obs

            total_step += 1

            if agent.cur_node.tolist() not in searched_space:
                searched_space.append(agent.cur_node.tolist())

        # record the total scores and avg-scores of current episodes


        avg = score / (total_step+1)
        total_avg += avg
        agent.records['scores'].append(total_avg)

        avg_score = total_avg / (e+1)
        agent.records['avg_scores'].append(avg_score)

        # record the total steps and avg-steps of current episodes
        steps += total_step
        agent.records['steps'].append(steps)

        avg_step = steps / (e+1)
        agent.records['avg_steps'].append(avg_step)

        if (e+1) % agent.memory_len == 0:

            # agent.train_process()

            agent.critic_train()
            agent.actor_train()

            save_episodes('episodes', e)
            save_records(agent, filenames)

        # # update learning rate
        if e >= 500 and agent.a_lr > 0.0 and agent.a_lr - ((2e-4) / 1500) >= 0:
            agent.entropy_coe -= 1.5e-5
            agent.a_lr -= ((2e-4) / 1500)
            # agent.c_lr -= (agent.c_lr / 10000)

        # save checkpoints
        agent.save_checkpoints()

        # output records of each episodes
        print(f"episode {e+1}: start node: {agent.start_node.tolist()}, score: {round(score/(total_step+1), 2)}, avg score: {round(agent.avg[e], 2)}, step: {total_step}, avg step: {round(avg_step, 2)}, done_time: {done_time}")


def run(device, start_x, start_y, goal_x, goal_y, train_mode, map_name, map_size, episodes):

    # Step 0: load the maps, records (based on the training mode) and return e;

    # Load the map
    if map_name != 'aisle' or 'canyon' or 'double_door': 
        raise ValueError("Invalid map name. Choose from 'aisle', 'canyon', 'double_door'. ")
    
    if map_size != 16 or 24 or 32:
        raise ValueError("Invalid map size. Choose from 16, 24, or 32.")
    
    file_name = 'datafiles/inputs/{}_{}.xlsx'.format(map_name, map_size)

    if map_size >= 26:
        scope = 'A1:' + chr(ord('A') + map_size - 1) + str(map_size)
    else:
        scope = 'A1:A' + chr(ord('A') + map_size - 26 - 1) + str(map_size)

    env_map = generate_map(file_name, 'Sheet1', scope)
    reward_map = generate_map(file_name, 'Sheet2', scope)

    agent = PPO(device=device, start=[start_x, start_y], goal=[goal_x, goal_y], env_map=env_map, reward_map=reward_map)

    # If the mode is 'new', create the new records files

    file_names = ['avg_scores', 'avg_steps', 'episodes', 'scores', 'steps']
    output_path = '../datafiles/outputs/records/'

    if train_mode == 'new':
        for fn in file_names:
            # check if the directory exists, if not create it
            if os.path.isfile(output_path + '{}.txt'.format(fn)) or os.path.islink(output_path + '{}.txt'.format(fn)):
                os.unlink(output_path + '{}.txt'.format(fn))
            os.makedirs(output_path + '{}.txt'.format(fn))
            if fn == 'episodes':
                with open(output_path + '{}.txt'.format(fn), 'w') as f:
                    f.write('0')

    load_records(file_names, agent)

    if train_mode == 'continue':
        if device == 'cuda':
            agent.load_checkpoints()
        else:
            agent.load_checkpoints_cpu()

    e = load_episodes('episodes')

    # start to train

    train_process(agent, e, episodes, file_names)

    # save the training result graph

    avg_scores = agent.records['avg_scores']
    plt.plot(episodes, avg_scores, 'r--')
    plt.title('Score')
    plt.xlabel('episodes')
    plt.ylabel('average score')
    plt.legend()
    plt.savefig(output_path + 'avg_scores.png')

    avg_steps = agent.records['avg_steps']
    plt.plot(episodes, avg_steps, 'b--')
    plt.title('Steps')
    plt.xlabel('episodes')
    plt.ylabel('average steps')
    plt.legend()
    plt.savefig(output_path + 'avg_steps.png')

