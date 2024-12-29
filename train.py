import argparse
import os
import multiprocessing
from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
from q_learning import QLearningAgent
import matplotlib.pyplot as plt

def plot_training_results(agent, save_path=None):
    """Plot training metrics"""
    plt.figure(figsize=(12, 10))
    
    # Plot episode rewards
    plt.subplot(2, 1, 1)
    plt.plot(agent.episode_rewards, label='Episode Reward')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot moving average
    plt.subplot(2, 1, 2)
    plt.plot(agent.avg_rewards, label='Moving Average Reward (last 100 episodes)')
    plt.title('Average Reward (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_training(gui, pid, lock, model_path=None, continue_training=False, n_episodes=500, checkpoint_interval=100):
    # Create directories for saving results
    timestamp = datetime.now().strftime("%d_%m_%H_%M")
    base_dir = f'QL-runs/run_{timestamp}_{n_episodes}'
    model_dir = os.path.join(base_dir, 'models')
    plot_dir = os.path.join(base_dir, 'plots')
    data_dir = os.path.join(base_dir, 'data')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Initialize environments
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Initialize Q-learning agent
    agent = QLearningAgent(
        state_dims=9,  # Updated state dimension to match the new state space
        n_actions=2,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Set directories for saving results
    agent.save_dir = data_dir
    agent.checkpoints_dir = checkpoints_dir
    
    if model_path and continue_training:
        agent.load(model_path)
        agent.continue_training(env, n_episodes=n_episodes, pid=pid, lock=lock, checkpoint_interval=checkpoint_interval)
    else:
        agent.train(env, n_episodes=n_episodes, pid=pid, lock=lock, checkpoint_interval=checkpoint_interval)
    
    # Save the trained agent
    agent.save(f'{model_dir}/q_learning_agent_{pid}.pkl')
    
    # Plot and save training results
    plot_training_results(agent, f'{plot_dir}/training_results_{pid}.png')

def main(gui, n_runs, model_path=None, continue_training=False, n_episodes=500):
    lock = multiprocessing.Lock()
    processes = []
    for i in range(n_runs):
        pid = os.getpid() + i
        p = multiprocessing.Process(target=run_training, args=(gui, pid, lock, model_path, continue_training, n_episodes))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or continue training a Q-learning agent for ramp metering.')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of training runs')
    parser.add_argument('--model', type=str, help='Path to the trained model file to continue training')
    parser.add_argument('--continue_training', action='store_true', help='Continue training an existing model')
    parser.add_argument('--n_episodes', type=int, default=500, help='Number of episodes for training')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Interval for saving checkpoints')
    args = parser.parse_args()
    
    main(gui=args.gui, n_runs=args.n_runs, model_path=args.model, continue_training=args.continue_training, n_episodes=args.n_episodes)