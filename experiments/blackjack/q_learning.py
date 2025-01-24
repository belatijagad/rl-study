from collections import defaultdict
import gymnasium as gym
import numpy as np
from utils import visualize_policy

class BlackjackAgent:
  def __init__(
    self,
    env: gym.Env,
    learning_rate: float,
    initial_epsilon: float,
    epsilon_decay: float,
    final_epsilon: float,
    discount_factor: float = 0.95
  ):
    self.env = env
    self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    
    self.lr = learning_rate
    self.discount_factor = discount_factor
    
    self.epsilon = initial_epsilon
    self.epsilon_decay = epsilon_decay
    self.final_epsilon = final_epsilon
    
    self.training_error = []
    
  def get_action(self, obs: tuple[int, int, bool]) -> int:
    if np.random.random() < self.epsilon:
      return self.env.action_space.sample()
    return int(np.argmax(self.q_values[obs]))
  
  def update(
    self,
    obs: tuple[int, int, bool],
    action: int,
    reward: float,
    terminated: bool,
    next_obs: tuple[int, int, bool]
  ):
    future_q_value = (not terminated) * np.max(self.q_values[next_obs])
    temporal_difference = (
      reward + self.discount_factor * future_q_value - self.q_values[obs][action]
    )
    
    self.q_values[obs][action] = (
      self.q_values[obs][action] + self.lr * temporal_difference
    )
    self.training_error.append(temporal_difference)
    
  def decay_epsilon(self):
    self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
if __name__ == "__main__":
  learning_rate = 0.01
  n_episodes = 100_000
  start_epsilon = 1.0
  final_epsilon = 0.1
  epsilon_decay = start_epsilon / (n_episodes / 2)
  
  env = gym.make("Blackjack-v1", sab=False)
  env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
  
  agent = BlackjackAgent(
    env,
    learning_rate,
    start_epsilon,
    epsilon_decay,
    final_epsilon
  )
  
  from tqdm import tqdm
  
  for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    
    while not done:
      action = agent.get_action(obs)
      next_obs, reward, terminated, truncated, info = env.step(action)
      
      agent.update(obs, action, reward, terminated, next_obs)
      
      done = terminated or truncated
      obs = next_obs
    agent.decay_epsilon()
    
  import matplotlib.pyplot as plt
  
  fig1 = visualize_policy(agent, usable_ace=True)
  plt.show()  
  