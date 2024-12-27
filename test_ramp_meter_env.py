import traci
from highway_env import HighwayEnvironment
from ramp_meter_env import RampMeterEnv
import argparse

def run_test(gui, n_steps=100):
    # Initialize the SUMO environment
    sumo_env = HighwayEnvironment(num_lanes=3, highway_length=1000, ramp_length=300, gui=gui)
    env = RampMeterEnv(sumo_env)
    
    # Reset the environment
    state = env.reset()
    print(f"Initial state: {state}")
    
    for step in range(n_steps):
        # Select an action (0: Red, 1: Green)
        action = 1  # Alternate between red and green for testing
        
        # Perform the action
        next_state, reward, done = env.step(action)
        
        # Print the results
        # print(f"Step: {step + 1}")
        # print(f"Action: {'Green' if action == 1 else 'Red'}")
        # print(f"Next state: {next_state}")
        # print(f"Reward: {reward}")
        print(f"Current Phase: {env.current_phase}")
        print(f"Emergency Stopping Vehicles: {traci.simulation.getEmergencyStoppingVehiclesIDList()}")
        if traci.simulation.getCollisions():
            print(f"Colliding Vehicles: {traci.simulation.getCollisions()}")
        # print(f"Done: {done}")
        # print("-------------------")
        
        if done:
            break
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the RampMeterEnv environment with TraCI.')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for SUMO simulation')
    parser.add_argument('--n_steps', type=int, default=100, help='Number of steps to run the test')
    args = parser.parse_args()
    
    run_test(gui=args.gui, n_steps=args.n_steps)