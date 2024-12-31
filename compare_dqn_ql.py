import torch
import numpy as np
import argparse
import os
import csv
from dqn_agent import DQNAgent
from q_learning import QLearningAgent
from ramp_meter_env import RampMeterEnv
from highway_env import HighwayEnvironment

def evaluate_ql_agent(env, agent, n_episodes=1, max_steps=3600):
    total_highway_speed = 0
    total_ramp_wait_time = 0
    total_reward = 0
    total_steps = 0

    for episode in range(n_episodes):
        state = env.reset()
        episode_highway_speed = 0
        episode_ramp_wait_time = 0
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action)
            state = next_state

            # Collect metrics
            highway_speed = env.get_highway_speed()
            ramp_wait_time = env.get_ramp_wait_time()
            episode_highway_speed += highway_speed
            episode_ramp_wait_time += ramp_wait_time
            episode_reward += reward

            if done:
                break

        total_highway_speed += episode_highway_speed
        total_ramp_wait_time += episode_ramp_wait_time
        total_reward += episode_reward
        total_steps += step + 1

    avg_highway_speed = total_highway_speed / total_steps
    avg_ramp_wait_time = total_ramp_wait_time / total_steps
    avg_reward = total_reward / n_episodes

    return avg_highway_speed, avg_ramp_wait_time, avg_reward

def evaluate_dqn_agent(gui, model_path, n_episodes=1, max_steps=3600):
    # Initialize environments
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Initialize DQN agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        state_dim=8,  # Updated state dimension to match the new state space
        n_actions=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,  # Set epsilon to 0 to ensure the agent uses the policy network
        epsilon_min=0.0,
        epsilon_decay=0.995
    )
    
    # Load the trained model
    agent.load(model_path)
    
    total_rewards = []
    total_highway_speed = 0
    total_ramp_wait_time = 0
    total_steps = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_highway_speed = 0
        episode_ramp_wait_time = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            # Collect metrics
            highway_speed = env.get_highway_speed()
            ramp_wait_time = env.get_ramp_wait_time()
            episode_highway_speed += highway_speed
            episode_ramp_wait_time += ramp_wait_time
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_highway_speed += episode_highway_speed
        total_ramp_wait_time += episode_ramp_wait_time
        total_steps += step + 1
        print(f"Episode {episode + 1}/{n_episodes} - Total Reward: {episode_reward}")
    
    env.close()
    
    avg_highway_speed = total_highway_speed / total_steps
    avg_ramp_wait_time = total_ramp_wait_time / total_steps
    avg_reward = sum(total_rewards) / n_episodes
    
    return avg_highway_speed, avg_ramp_wait_time, avg_reward

def main(ql_folder, dqn_folder, output_csv, gui=False):
    # Initialize the environment
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=gui)
    env = RampMeterEnv(sumo_env)

    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Type', 'Model File', 'Average Highway Speed', 'Average Ramp Wait Time', 'Average Reward'])

    # Evaluate Q-learning models
    ql_model_files = [f for f in os.listdir(ql_folder) if f.endswith('.pkl')]
    for ql_model_file in ql_model_files:
        ql_model_path = os.path.join(ql_folder, ql_model_file)
        ql_agent = QLearningAgent(state_dims=8, n_actions=2, learning_rate=0.1, gamma=0.95, epsilon=0.0)
        ql_agent.load(ql_model_path)
        ql_avg_highway_speed, ql_avg_ramp_wait_time, ql_avg_reward = evaluate_ql_agent(env, ql_agent)
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Q-learning', ql_model_file, ql_avg_highway_speed, ql_avg_ramp_wait_time, ql_avg_reward])
        print(f"Q-learning Model {ql_model_file} - Average Highway Speed: {ql_avg_highway_speed:.2f}, Average Ramp Wait Time: {ql_avg_ramp_wait_time:.2f}, Average Reward: {ql_avg_reward:.2f}")

    # Evaluate DQN models
    dqn_model_files = [f for f in os.listdir(dqn_folder) if f.endswith('.pth')]
    for dqn_model_file in dqn_model_files:
        dqn_model_path = os.path.join(dqn_folder, dqn_model_file)
        dqn_avg_highway_speed, dqn_avg_ramp_wait_time, dqn_avg_reward = evaluate_dqn_agent(gui, dqn_model_path)
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['DQN', dqn_model_file, dqn_avg_highway_speed, dqn_avg_ramp_wait_time, dqn_avg_reward])
        print(f"DQN Model {dqn_model_file} - Average Highway Speed: {dqn_avg_highway_speed:.2f}, Average Ramp Wait Time: {dqn_avg_ramp_wait_time:.2f}, Average Reward: {dqn_avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Q-learning and DQN models.')
    parser.add_argument('--ql_folder', type=str, required=True, help='Path to the folder containing Q-learning model files')
    parser.add_argument('--dqn_folder', type=str, required=True, help='Path to the folder containing DQN model files')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    args = parser.parse_args()
    
    main(args.ql_folder, args.dqn_folder, args.output_csv, args.gui)