import argparse
import os
import csv
from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
from q_learning import QLearningAgent

def run_q_learning_control(gui, model_path, n_episodes=10, max_steps=3600):
    # Initialize environments
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Initialize Q-learning agent
    agent = QLearningAgent(
        state_dims=9,  # Updated state dimension to match the new state space
        n_actions=2,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=0.0,  # Set epsilon to 0 to ensure the agent uses the policy network
        epsilon_min=0.0,
        epsilon_decay=0.995
    )
    
    # Load the trained model
    agent.load(model_path)
    
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{n_episodes} - Total Reward: {episode_reward}")
    
    env.close()
    return total_rewards

def evaluate_models_in_folder(folder_path, output_csv, gui=False, n_episodes=10, max_steps=3600):
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    results = []

    # Write header to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Average Reward'])

    for model_file in model_files:
        model_path = os.path.join(folder_path, model_file)
        print(f"Evaluating model: {model_path}")
        rewards = run_q_learning_control(gui, model_path, n_episodes, max_steps)
        avg_reward = sum(rewards) / len(rewards)
        results.append([model_file, avg_reward])
        
        # Write result to CSV after each model
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_file, avg_reward])
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Q-learning models for traffic control in SUMO.')
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing the model files')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    parser.add_argument('--n_episodes', type=int, default=10, help='Number of episodes to run for each model')
    parser.add_argument('--max_steps', type=int, default=3600, help='Maximum steps per episode')
    args = parser.parse_args()
    
    # Suppress SUMO warnings
    os.environ['SUMO_HOME'] = 'C:/Program Files (x86)/Eclipse/Sumo'  # Set this to your SUMO installation path
    os.environ['SUMO_OPTS'] = '--no-warnings'

    evaluate_models_in_folder(folder_path=args.folder, output_csv=args.output_csv, gui=args.gui, n_episodes=args.n_episodes, max_steps=args.max_steps)