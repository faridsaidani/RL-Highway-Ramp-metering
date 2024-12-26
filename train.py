import argparse
from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
from q_learning import QLearningAgent
import matplotlib.pyplot as plt
import os
import multiprocessing

def plot_training_results(agent, save_path=None):
    """Plot training metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(agent.episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    plt.plot(agent.avg_rewards)
    plt.title('Average Reward (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_training(gui, pid, lock):
    # Create directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize environments
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Initialize and train Q-learning agent
    agent = QLearningAgent(
        state_dims=9,
        n_actions=2,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Train the agent
    agent.train(env, n_episodes=150, pid=pid, lock=lock)
    
    # Save the trained agent
    agent.save(f'models/q_learning_agent_{pid}.pkl')
    
    # Plot and save training results
    plot_training_results(agent, f'plots/training_results_{pid}.png')

def main(gui, n_runs):
    lock = multiprocessing.Lock()
    processes = []
    for i in range(n_runs):
        pid = os.getpid() + i
        p = multiprocessing.Process(target=run_training, args=(gui, pid, lock))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Q-learning agent for ramp metering.')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of training runs')
    args = parser.parse_args()
    
    main(gui=args.gui, n_runs=args.n_runs)