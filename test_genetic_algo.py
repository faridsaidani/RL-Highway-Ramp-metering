from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import csv
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, n_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        self.model = DQNNetwork(state_dim, n_actions).to(device)
        self.target_model = DQNNetwork(state_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()
        self.best_model = None
        self.best_reward = -float('inf')

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target
            states.append(state)
            targets_f.append(target_f)
        states = torch.cat(states)
        targets_f = torch.cat(targets_f)
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(states)
        loss = self.criterion(output, targets_f)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def mutate(self, mutation_rate=0.01):
        for param in self.model.parameters():
            if np.random.rand() < mutation_rate:
                param.data += torch.randn_like(param) * mutation_rate

    def evaluate(self, env, n_episodes=10):
        total_reward = 0
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
        avg_reward = total_reward / n_episodes
        return avg_reward

    def train(self, env, n_episodes=1000, output_file='genetic_test.csv'):
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Episode', 'Reward', 'Average Highway Speed', 'Average Ramp Wait Time', 'Queue Length']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for e in range(n_episodes):
                state = env.reset()
                done = False
                total_reward = 0
                total_highway_speed = 0
                total_ramp_wait_time = 0
                total_steps = 0
                total_queue_length = 0

                while not done:
                    action = self.act(state)
                    next_state, reward, done = env.step(action)
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    self.replay()

                    # Collect metrics
                    total_reward += reward
                    total_highway_speed += env.get_highway_speed()
                    total_ramp_wait_time += env.get_ramp_wait_time()
                    total_steps += 1

                avg_reward = self.evaluate(env)
                avg_highway_speed = total_highway_speed / total_steps
                avg_ramp_wait_time = total_ramp_wait_time / total_steps

                writer.writerow({
                    'Episode': e + 1,
                    'Reward': avg_reward,
                    'Average Highway Speed': avg_highway_speed,
                    'Average Ramp Wait Time': avg_ramp_wait_time,
                })

                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.best_model = self.model.state_dict()
                    print(f"New best model with reward: {self.best_reward}")
                else:
                    self.model.load_state_dict(self.best_model)
                    self.mutate()
                self.update_target_model()
                print(f"Episode {e+1}/{n_episodes}, Reward: {avg_reward}, Epsilon: {self.epsilon}")

# Example usage
if __name__ == "__main__":
    from highway_env import HighwayEnvironment
    from ramp_meter_env import RampMeterEnv

    parser = argparse.ArgumentParser(description='Train DQN agent with genetic algorithm-like approach.')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes to train the agent')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of steps per episode')
    parser.add_argument('--output_file', type=str, default='genetic_test.csv', help='File to save the rewards and metrics')
    args = parser.parse_args()

    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=args.gui)
    env = RampMeterEnv(sumo_env)
    agent = DQNAgent(state_dim=8, n_actions=2)
    agent.train(env, n_episodes=args.n_episodes, output_file=args.output_file)