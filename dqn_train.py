from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(agent, save_path=None):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot episode rewards
    plt.subplot(1, 3, 1)
    plt.plot(agent.episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot moving average
    plt.subplot(1, 3, 2)
    plt.plot(agent.avg_rewards)
    plt.title('Average Reward (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    # Plot losses
    plt.subplot(1, 3, 3)
    plt.plot(agent.losses)
    plt.title('Training Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Create directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize environments
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200)
    env = RampMeterEnv(sumo_env)
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_dim=9,
        n_actions=2,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update=10
    )
    
    # Train the agent
    agent.train(env, n_episodes=1000, warmup_steps=1000)
    
    # Save the trained agent
    agent.save('models/dqn_agent.pth')
    
    # Plot and save training results
    plot_training_results(agent, 'plots/dqn_training_results.png')

if __name__ == "__main__":
    main()