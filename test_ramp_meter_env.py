import traci
from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
import argparse

def run_test(gui, n_steps=3600, output_file='rewards.txt'):
    # Initialize the SUMO environment
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=300, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Reset the environment
    state = env.reset()
    print(f"Initial state: {state}")
    
    total_reward = 0
    total_highway_speed = 0
    total_ramp_wait_time = 0
    total_steps = 0
    
    with open(output_file, 'w') as f:
        for step in range(n_steps):
            # Select an action (0: Red, 1: Green)
            action = 1  # Always green for testing
            
            # Perform the action
            next_state, reward, done = env.step(action)
            
            # Accumulate the reward
            total_reward += reward
            
            # Collect metrics
            highway_speed = env.get_highway_speed()
            ramp_wait_time = env.get_ramp_wait_time()
            total_highway_speed += highway_speed
            total_ramp_wait_time += ramp_wait_time
            total_steps += 1
            
            # Print the results
            print(f"Step: {step + 1}")
            print(f"Action: {'Green' if action == 1 else 'Red'}")
            print(f"Next state: {next_state}")
            print(f"Reward: {reward}")
            print(f"Current Phase: {env.current_phase}")
            print(f"Emergency Stopping Vehicles: {traci.simulation.getEmergencyStoppingVehiclesIDList()}")
            if traci.simulation.getCollisions():
                print(f"Colliding Vehicles: {traci.simulation.getCollisions()}")
            print("-------------------")
            
            # Save the reward to the file
            f.write(f"Step: {step + 1}, Reward: {reward}\n")
            
            if done:
                break
    
    # Calculate averages
    avg_highway_speed = total_highway_speed / total_steps
    avg_ramp_wait_time = total_ramp_wait_time / total_steps
    avg_reward = total_reward / total_steps
    
    # Save the averages to the file
    with open(output_file, 'a') as f:
        f.write(f"Total Reward: {total_reward}\n")
        f.write(f"Average Highway Speed: {avg_highway_speed}\n")
        f.write(f"Average Ramp Wait Time: {avg_ramp_wait_time}\n")
        f.write(f"Average Reward: {avg_reward}\n")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the RampMeterEnv environment with TraCI.')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    parser.add_argument('--n_steps', type=int, default=100, help='Number of steps to run the test')
    parser.add_argument('--output_file', type=str, default='rewards.txt', help='File to save the rewards')
    args = parser.parse_args()
    
    run_test(gui=args.gui, n_steps=args.n_steps, output_file=args.output_file)