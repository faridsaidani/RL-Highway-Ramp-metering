import os
import pickle
import numpy as np
from collections import defaultdict
import csv
from datetime import datetime

class QLearningAgent:
    def __init__(self, 
                 state_dims=9,        # Dimensionality of state space
                 n_actions=2,         # Number of possible actions
                 learning_rate=0.1,   
                 gamma=0.95,          # Discount factor
                 epsilon=1.0,         # Initial exploration rate
                 epsilon_min=0.01,    # Minimum exploration rate
                 epsilon_decay=0.995): # Exploration decay rate
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with discretized state space
        self.n_bins = 10  # Number of bins for each state dimension
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        
    def _discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table"""
        discrete_state = []
        for i, value in enumerate(state):
            # Handle one-hot encoded parts of state differently
            if 4 <= i <= 7:  # Traffic light phase one-hot encoding
                discrete_state.append(int(value))
            else:
                # Discretize continuous values into bins
                bin_idx = min(int(value * self.n_bins), self.n_bins - 1)
                discrete_state.append(bin_idx)
        return tuple(discrete_state)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        discrete_state = self._discretize_state(state)
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state):
        """Update Q-value for state-action pair"""
        current_state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)
        
        # Q-learning update rule
        current_q = self.q_table[current_state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        
        self.q_table[current_state][action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None):
        """Train the Q-learning agent"""
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state, training=True)
                next_state, reward, done = env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            avg_reward = np.mean(self.episode_rewards[-100:])  # Moving average of last 100 episodes
            self.avg_rewards.append(avg_reward)
            
            # Save episode reward to CSV file with lock
            if lock:
                with lock:
                    with open(f'{self.save_dir}/episode_rewards.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([pid, episode + 1, episode_reward])
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"PID {pid} - Episode {episode + 1}/{n_episodes}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print("-------------------")
    
    def save(self, filepath):
        """Save the trained agent"""
        save_dict = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'avg_rewards': self.avg_rewards
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, filepath):
        """Load a trained agent"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Convert regular dict back to defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), save_dict['q_table'])
        self.epsilon = save_dict['epsilon']
        self.episode_rewards = save_dict['episode_rewards']
        self.avg_rewards = save_dict['avg_rewards']
    
    def continue_training(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None):
        """Continue training the Q-learning agent"""
        self.train(env, n_episodes, max_steps, pid, lock)