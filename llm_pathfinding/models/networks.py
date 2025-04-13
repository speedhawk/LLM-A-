import torch
import torch.nn as nn
import torch.nn.functional as F

# AC_PPO_ini Model
class Critic(nn.Module):
  def __init__(self, input):
      super(Critic, self).__init__()
      self.dim = 128
      self.fc1 = nn.Linear(input, self.dim)
      self.fc2 = nn.Linear(self.dim, self.dim)
      self.state_value = nn.Linear(self.dim, 1)

      # layer normalization
      self.bn1 = nn.LayerNorm(self.dim)
      self.bn2 = nn.LayerNorm(self.dim)

      self.initialization()

  def initialization(self):   # weight initialization
    nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
    nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))

  def forward(self, x):
      x = F.tanh(self.fc1(x))
      x = self.bn1(x)
      x = F.tanh(self.fc2(x))
      x = self.bn2(x)
      value = self.state_value(x)
      return value

class Actor(nn.Module):
  """
  INPUT: the position as the state of agent
  OUPPUT: 8-dim Teansor, the probability of each actions
  """
  def __init__(self, input, output):
      super(Actor, self).__init__()
      self.dim = 128
      self.fc1 = nn.Linear(input, self.dim)
      self.fc2 = nn.Linear(self.dim, self.dim)
      self.action_head = nn.Linear(self.dim, output)

      # layer normalization
      self.bn1 = nn.LayerNorm(self.dim)
      self.bn2 = nn.LayerNorm(self.dim)

      # # noise generator
      # self.mu = 0.0
      # self.stdv = 0.1


      self.initialization()

  def initialization(self):
    nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
    nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))

  def noise_generate(self, actions):
    noise = torch.normal(mean=self.mu, std=self.stdv, size=actions.size())
    return noise

  def forward(self, x):
      x = F.tanh(self.fc1(x))
      x = self.bn1(x)
      x = F.tanh(self.fc2(x))
      x = self.bn2(x)
      x = self.action_head(x)
      # x = x + self.noise_generate(x).to('cpu')
      action_prob = F.softmax(x, dim=1)
      return action_prob