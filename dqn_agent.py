from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import csv

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

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, 
                 state_dim=9,
                 n_actions=2,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=64,
                 target_update=10):
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.losses = []
        self.update_count = 0
        
        # Create checkpoints directory
        self.checkpoints_dir = None
        
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None, warmup_steps=1000):
        """Train the DQN agent"""
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            
            # Warmup phase for first episode
            if episode == 0:
                for _ in range(warmup_steps):
                    action = random.randint(0, self.n_actions - 1)
                    next_state, reward, done = env.step(action)
                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        state = env.reset()
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state, training=True)
                next_state, reward, done = env.step(action)
                
                # Store transition
                self.memory.push(state, action, reward, next_state, done)
                
                # Update networks
                self.update()
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.avg_rewards.append(avg_reward)
            
            # Save episode reward to CSV file with lock
            if lock:
                with lock:
                    with open(f'{self.save_dir}/episode_rewards.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([pid, episode + 1, episode_reward])
            
            # Save checkpoint every 50 episodes
            if (episode + 1) % 50 == 0:
                checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_{episode + 1}_{pid}.pth')
                self.save(checkpoint_path)
                print(f"Checkpoint saved at episode {episode + 1}")
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"PID {pid} - Episode {episode + 1}/{n_episodes}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print("-------------------")
    
    def save(self, filepath):
        """Save the trained agent"""
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'avg_rewards': self.avg_rewards,
            'losses': self.losses
        }
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load a trained agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.avg_rewards = checkpoint['avg_rewards']
        self.losses = checkpoint['losses']
    
    def continue_training(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None):
        """Continue training the DQN agent"""
        self.train(env, n_episodes, max_steps, pid, lock)