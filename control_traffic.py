import argparse
from collections import defaultdict
import pickle
import numpy as np
from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
import traci

class QLearningAgent:
    def __init__(self, state_dims=9, n_actions=2):
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.q_table = None

    def load(self, filepath):
        """Load the trained agent"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Convert regular dict back to defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), save_dict['q_table'])
        self.epsilon = save_dict['epsilon']
        self.episode_rewards = save_dict['episode_rewards']
        self.avg_rewards = save_dict['avg_rewards']

    def _discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table"""
        discrete_state = []
        for i, value in enumerate(state):
            # Handle one-hot encoded parts of state differently
            if 4 <= i <= 7:  # Traffic light phase one-hot encoding
                discrete_state.append(int(value))
            else:
                # Discretize continuous values into bins
                bin_idx = min(int(value * 10), 9)
                discrete_state.append(bin_idx)
        return tuple(discrete_state)

    def select_action(self, state):
        """Select action using the trained Q-table"""
        discrete_state = self._discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

def run_control(model_path, gui):
    # Initialize environments
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Load the trained agent
    agent = QLearningAgent()
    agent.load(model_path)
    
    # Reset the environment
    state = env.reset()
    
    done = False
    total_reward = 0
    while not done:
        # Select and perform action
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        total_reward += reward
        state = next_state
    
    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control traffic lights using a trained Q-learning agent.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    args = parser.parse_args()
    
    run_control(args.model, args.gui)