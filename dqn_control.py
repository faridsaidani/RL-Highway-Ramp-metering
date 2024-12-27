import argparse
from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
from dqn_agent import DQNAgent
import torch

def run_dqn_control(gui, model_path, n_episodes=10, max_steps=3600):
    # Initialize environments
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=200, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Initialize DQN agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        state_dim=7,
        n_actions=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.0,  # Set epsilon to 0 to ensure the agent uses the policy network
        epsilon_min=0.0,
        epsilon_decay=0.995
    )
    
    # Load the trained model
    agent.load(model_path)
    
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
        
        print(f"Episode {episode + 1}/{n_episodes} - Total Reward: {episode_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained DQN agent for traffic control in SUMO.')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--n_episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=3600, help='Maximum steps per episode')
    args = parser.parse_args()
    
    run_dqn_control(gui=args.gui, model_path=args.model, n_episodes=args.n_episodes, max_steps=args.max_steps)