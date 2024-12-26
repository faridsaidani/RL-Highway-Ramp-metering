import numpy as np
from collections import deque
import traci

class RampMeterEnv:
    def __init__(self, sumo_env, max_steps=3600):
        self.sumo_env = sumo_env
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action space
        # 0: Red light (stop)
        # 1: Green light (go)
        self.action_space = 2
        
        # Store traffic metrics
        self.highway_speeds = deque(maxlen=10)  # Store last 10 speed measurements
        self.ramp_queue = deque(maxlen=10)      # Store last 10 queue lengths
        
    def _get_state(self):
        """
        Get current state of the environment.
        State consists of:
        - Average speed on highway near merge point
        - Number of vehicles in ramp queue
        - Traffic density on highway near merge
        - Ramp waiting time
        """
        # Get vehicle IDs in relevant areas
        highway_vehicles = traci.edge.getLastStepVehicleIDs("highway1")
        ramp_vehicles = traci.edge.getLastStepVehicleIDs("ramp")
        
        # Calculate metrics
        highway_speed = traci.edge.getLastStepMeanSpeed("highway1")
        self.highway_speeds.append(highway_speed)
        
        ramp_queue_length = len(ramp_vehicles)
        self.ramp_queue.append(ramp_queue_length)
        
        highway_density = len(highway_vehicles) / traci.edge.getLength("highway1")
        
        # Calculate average ramp waiting time
        ramp_wait_time = 0
        if ramp_vehicles:
            wait_times = [traci.vehicle.getWaitingTime(v) for v in ramp_vehicles]
            ramp_wait_time = np.mean(wait_times)
        
        # Normalize values
        norm_speed = np.mean(self.highway_speeds) / 27.78  # normalize by max speed
        norm_queue = np.mean(self.ramp_queue) / 20  # normalize assuming max queue of 20
        norm_density = highway_density / 0.1  # normalize by critical density
        norm_wait = min(ramp_wait_time / 120, 1)  # normalize by 120 seconds
        
        return np.array([norm_speed, norm_queue, norm_density, norm_wait])
    
    def _calculate_reward(self, state):
        """
        Calculate reward based on:
        - Highway flow maintenance (higher speeds are better)
        - Ramp waiting times (lower is better)
        - Queue lengths (lower is better)
        """
        highway_speed = state[0]
        ramp_queue = state[1]
        highway_density = state[2]
        ramp_wait = state[3]
        
        # Reward components
        speed_reward = highway_speed  # Higher speeds are better
        queue_penalty = -ramp_queue  # Penalize long queues
        wait_penalty = -ramp_wait    # Penalize long waiting times
        
        # Additional penalty for critical density
        density_penalty = -1.0 if highway_density > 0.8 else 0
        
        # Combine rewards
        reward = (speed_reward + 
                 0.5 * queue_penalty + 
                 0.3 * wait_penalty + 
                 density_penalty)
        
        return reward
    
    def step(self, action):
        """
        Execute action in environment and return next state, reward, done flag
        """
        # Apply action (control traffic light)
        light_state = 'G' if action == 1 else 'r'
        traci.trafficlight.setRedYellowGreenState("ramp_end", light_state)
        
        # Simulate one step
        traci.simulationStep()
        self.current_step += 1
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(next_state)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done
    
    def reset(self):
        """Reset environment to initial state"""
        # End current simulation if any
        try:
            traci.close()
        except:
            pass
        
        # Clear metrics
        self.highway_speeds.clear()
        self.ramp_queue.clear()
        
        # Reset step counter
        self.current_step = 0
        
        # Start new simulation
        self.sumo_env.generate_route_file()  # Generate new traffic
        self.sumo_env.start_simulation()
        
        # Initialize traffic light
        traci.trafficlight.setRedYellowGreenState("ramp_end", "r")
        
        # Return initial state
        return self._get_state()