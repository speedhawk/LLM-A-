import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from torch.distributions import Categorical

from collections import namedtuple

import numpy as np
import random
from models.networks import Actor, Critic


class PPO():
  def __init__(self, 
               device, # device to run the model on (cpu or cuda)
               start, 
               goal, 
               border, 
               env_map, 
               reward_map) -> None:

    # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.device = torch.device(device) if device else torch.device('cpu')

    # environment
    self.start_node = np.array(start)
    self.target_node = np.array(goal)
    self.cur_node = self.start_node
    self.pre_node = self.start_node

    # map and reward distribution
    self.map = env_map
    self.reward_distribution = reward_map

    # map scope
    self.max_border = border
    self.x_max = None
    self.x_min = None
    self.y_max = None
    self.y_min = None

    self.action_space = {0: [1, 0],
                           1: [-1, 0],
                           2: [0, 1],
                           3: [0, -1],
                           4: [-1, 1],
                           5: [1, 1],
                           6: [-1, -1],
                           7: [1, -1]}

    # learning param:

    self.memory = {}  # this list consists of transations for the number of n times of batch_size.
    self.memory_len = 50
    self.batch_size = 128
    self.actor_max_grad = 0.05
    # self.critic_max_grad = 1.5
    self.actor = Actor(2 * border + 8, 8).to(self.device)
    self.critic = Critic(2 * border + 8).to(self.device)
    self.gamma = 0.99
    self.lamda = 0.95
    self.a_lr = 2e-4
    self.c_lr = 5e-3
    self.clip = 0.3
    self.entropy_coe = 0.01
    # self.l1_co = 0.001
    # self.l2_co = 0.005

    self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.a_lr)
    self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.c_lr)

    # iteration param:

    self.pre_reward_sign = float(0)
    self.sign_record = 0
    self.pre_unreshaped_reward = 0.0

    self.step_thr = 200
    self.steps = 0

    # statistic param and records:

    self.episodes = [[], []]  # avg, total_avg, avg_step, total_steps
    self.records = {'avg_scores': [], 
                    'avg_steps': [], 
                    'scores': [], 
                    'steps': []}   # avg, total_avg, avg_step, total_steps
    
  def save_checkpoints(self):
    torch.save(self.actor.state_dict(), self.checkpoints['actor'])
    torch.save(self.critic.state_dict(), self.checkpoints['critic'])

  def load_checkpoints_cpu(self):
    self.actor.load_state_dict(torch.load(self.checkpoints['actor_cpu'], map_location=torch.device('cpu')))
    self.critic.load_state_dict(torch.load(self.checkpoints['critic_cpu'], map_location=torch.device('cpu')))

  def load_checkpoints(self):
    self.actor.load_state_dict(torch.load(self.checkpoints['actor']))
    self.critic.load_state_dict(torch.load(self.checkpoints['critic']))

  def save_trans(self, index, trans):
    self.memory[index].insert(0, trans)

  def vectorized_start(self, episode):


    self.x_min = 0
    self.x_max = self.max_border - 1
    self.y_min = 0
    self.y_max = self.max_border - 1

    x = random.randint(self.x_min, self.x_max)
    y = random.randint(self.y_min, self.y_max)

    while self.reward_distribution[x][y] == -1024:
      x = random.randint(self.x_min, self.x_max)
      y = random.randint(self.y_min, self.y_max)
    return np.array([x, y])

  def reset(self):
    self.cur_node = self.start_node

  def get_noise(self, actions):
    sigma = 0.1
    mu = 0.0
    noise_vector = torch.normal(mean=mu, std=sigma, size=actions.size())

  def double_hot(self, obs):
    double_hot_state = [0] * (self.max_border * 2)
    x = obs.tolist()[0]
    y = obs.tolist()[1]
    double_hot_state[x] = 1
    double_hot_state[self.max_border+y] = 1
    return double_hot_state

  def situation(self, obs):
    situation = []
    is_obstacle = 0
    for i in self.action_space:
      sur_obs = (obs + np.array(self.action_space[i])).tolist()
      x = sur_obs[0]
      y = sur_obs[1]
      if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_min or self.reward_distribution[x][y] == -1024:
        is_obstacle = 0
      else:
        is_obstacle = 1
      situation.append(is_obstacle)

    return situation

  def get_next_obs(self, action):

    obs = self.cur_node + np.array(self.action_space[action])
    collide_check = 0

    obs = obs.tolist()
    x = obs[0]
    y = obs[1]

    if x > self.x_max or x < self.x_min or y > self.y_max or y < self.y_min or self.reward_distribution[x][y] == -1024:
      obs = self.cur_node
      collide_check = 1

    return (np.array(obs), collide_check)

  def action_select(self, obs):
    possibilities = self.actor(obs).cpu().detach()
    # print(f"p: {possibilities}")

    distrb = Categorical(possibilities)   # attention: Categorical requires a tensor input!
    act = distrb.sample().item()

    possibilities = possibilities.numpy()
    posb = possibilities.item(act)

    return act, posb

  def get_reward(self, obs):

    if obs[1] == 1:
      # self.pre_unreshaped_reward = 0.0
      return 0.0
    else:
      obs = obs[0]
      x1 = obs[0]
      y1 = obs[1]

      cur_node = self.cur_node.tolist()
      x2 = cur_node[0]
      y2 = cur_node[1]

      r = float(self.reward_distribution[x1][y1]-self.reward_distribution[x2][y2])

      return r

  def reward_reshaping(self, reward):
    """
    Except for reshaping the reward, there are two variables to be sustained in this
    function:
      1. pre_reward_sign: record the last reward sign, used for evaluate sign_record
      2. sign_record: record the frequence of nice actions.
    """

    # reward reshaping
    r_sign = float(np.sign(reward))
    if r_sign == self.pre_reward_sign and r_sign == 1.0:
      self.sign_record += 1
    else:
      self.sign_record = 0

    index = 0.1 * (float(self.sign_record))
    reward += index
    self.pre_reward_sign = r_sign

    return reward

  def is_done(self, reward):

    if reward == -10.0:
      return True

    return False

  def step(self, act):

    obs = self.get_next_obs(act)
    reward = self.get_reward(obs)
    reward = self.reward_reshaping(reward)
    done = self.is_done(reward)
    inf = 'the current step is ' + str(done)
    return obs, reward, done, inf

  def critic_train(self):
    Transition = namedtuple('Transition', ['state', 'action', 'pre_prob', 'reward', 'next_state', 'G'])

    datas = []

    for i in range(len(self.memory)):
      G = 0.0
      for tran in self.memory[i]:
        G = tran.reward + self.gamma * G
        tran_list = list(tran)
        tran = Transition(tran_list[0], tran_list[1], tran_list[2], tran_list[3], tran_list[4], G)
        datas.append(tran)

    random.shuffle(datas)

    states = torch.tensor([trans.state.tolist() for trans in datas], dtype=torch.float).to(self.device)
    actions = torch.tensor([trans.action for trans in datas], dtype=torch.long).view(-1, 1).to(self.device)
    # actions size: (batch_size, 1)
    G = torch.tensor([trans.G for trans in datas], dtype=torch.float).view(-1, 1).to(self.device)
    # G size: (batch_size, 1)

    for _ in range(2 * len(datas) // self.batch_size):

      for index in BatchSampler(SequentialSampler(range(len(datas))), batch_size=self.batch_size, drop_last=False):

        v_s = self.critic(states[index]).squeeze(-1)


        critic_loss = F.mse_loss(G[index], v_s)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

  def actor_train(self):

    Transition = namedtuple('Transition', ['state', 'action', 'pre_prob', 'reward', 'next_state', 'adv'])

    datas = []

    for i in range(len(self.memory)):
      adv = 0.0
      for tran in self.memory[i]:
        Q = tran.reward + self.gamma * self.critic(tran.next_state.to(self.device)).detach().item() # Q = r + V(s')
        v_s = self.critic(tran.state.to(self.device)).detach().item()
        delta = Q - v_s
        adv = delta + self.gamma * self.lamda * adv
        tran_list = list(tran)
        tran = Transition(tran_list[0], tran_list[1], tran_list[2], tran_list[3], tran_list[4], adv)
        datas.append(tran)

    states = torch.tensor([trans.state.tolist() for trans in datas], dtype=torch.float).to(self.device)
    actions = torch.tensor([trans.action for trans in datas], dtype=torch.long).view(-1, 1).to(self.device)
    # actions size: (batch_size, 1)
    advs = torch.tensor([trans.adv for trans in datas], dtype=torch.float).view(-1, 1).to(self.device)
    # advs size: (batch_size, 1)

    pre_probs= torch.tensor([trans.pre_prob for trans in datas], dtype=torch.float).view(-1, 1).to(self.device)


    for _ in range(2 * len(datas) // self.batch_size):

      random.shuffle(datas)

      for index in BatchSampler(SequentialSampler(range(len(datas))), batch_size=self.batch_size, drop_last=False):

        # Obtain pre_action_probability and cur_action_probability, which is used for calculating ratio and clamped ratio

        p = self.actor(states[index].squeeze(1)).squeeze(1).to(device)

        cur_probs = p.gather(1, actions[index])
        entropies = torch.tensor([Categorical(posb.unsqueeze(0)).entropy().item() for posb in p], dtype=torch.float).view(-1, 1).to(self.device)

        ratio = cur_probs / pre_probs[index]
        n_advs = (advs[index]-advs[index].mean()) / advs[index].std()
        sel_a = ratio * n_advs
        sel_b = torch.clamp(ratio, 1-self.clip, 1+self.clip) * n_advs

        # Update Actor by minimum ratio mean loss

        actor_loss = -torch.min(sel_a, sel_b).mean() - self.entropy_coe * entropies.mean() # entropy regularization loss function
        # print(f"a_loss: {actor_loss}")
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_grad)
        self.actor_opt.step()