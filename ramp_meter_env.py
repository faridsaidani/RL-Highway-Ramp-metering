import numpy as np
from collections import deque
import traci
from enum import Enum

class TrafficLightPhase(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2

class RampMeterEnv:
    def __init__(self, sumo_env, max_steps=3600):
        self.sumo_env = sumo_env
        self.max_steps = max_steps
        self.current_step = 0
        
        # Traffic light settings
        self.yellow_duration = 3  # 3 seconds yellow phase
        self.min_green_duration = 5  # minimum green phase duration
        self.current_phase = TrafficLightPhase.RED
        self.phase_duration = 0
        self.yellow_timer = 0
        
        # Define action space
        # 0: Switch to/maintain red
        # 1: Switch to/maintain green
        self.action_space = 2
        
        # Store traffic metrics
        self.highway_speeds = deque(maxlen=10)
        self.ramp_queue = deque(maxlen=10)
        
    def _set_traffic_light(self, phase):
        """Set traffic light state based on phase"""
        phase_to_state = {
            TrafficLightPhase.GREEN: "GGGG",
            TrafficLightPhase.YELLOW: "yGGG",
            TrafficLightPhase.RED: "rGGG"
        }
        
        traci.trafficlight.setRedYellowGreenState("ramp_end", phase_to_state[phase])
        
    def _handle_phase_transition(self, action):
        """Handle traffic light phase transitions including yellow phases"""
        old_phase = self.current_phase
        
        # Update timers
        if self.yellow_timer > 0:
            self.yellow_timer -= 1
            if self.yellow_timer == 0:
                # Yellow phase completed
                if self.current_phase == TrafficLightPhase.YELLOW:
                    self.current_phase = TrafficLightPhase.GREEN if action == 1 else TrafficLightPhase.RED
                    self.phase_duration = 0
        else:
            self.phase_duration += 1
            
            # Handle action
            if action == 1:  # Request Green
                if self.current_phase == TrafficLightPhase.RED:
                    self.current_phase = TrafficLightPhase.YELLOW
                    self.yellow_timer = self.yellow_duration
            else:  # Request Red
                if self.current_phase == TrafficLightPhase.GREEN:
                    if self.phase_duration >= self.min_green_duration:
                        self.current_phase = TrafficLightPhase.YELLOW
                        self.yellow_timer = self.yellow_duration
        
        # Set traffic light state
        self._set_traffic_light(self.current_phase)
        
        return old_phase != self.current_phase  # Return whether phase changed
    
    def _get_state(self):
        """
        Get current state of the environment.
        Additional state components for traffic light phase.
        """
        # Get base metrics
        highway_vehicles = traci.edge.getLastStepVehicleIDs("highway1")
        ramp_vehicles = traci.edge.getLastStepVehicleIDs("ramp")
        
        highway_speed = traci.edge.getLastStepMeanSpeed("highway1")
        self.highway_speeds.append(highway_speed)
        
        ramp_queue_length = len(ramp_vehicles)
        self.ramp_queue.append(ramp_queue_length)
        
        ramp_wait_time = 0
        if ramp_vehicles:
            wait_times = [traci.vehicle.getWaitingTime(v) for v in ramp_vehicles]
            ramp_wait_time = np.mean(wait_times)
        
        # Calculate the distance of the nearest car in the rightmost lane from the traffic light
        right_lane_vehicles = [v for v in highway_vehicles if traci.vehicle.getLaneID(v).endswith("_0")]
        if right_lane_vehicles:
            distances = [self.sumo_env.highway_length - traci.vehicle.getLanePosition(v) for v in right_lane_vehicles]
            min_distance = min(distances)
        else:
            min_distance = self.sumo_env.highway_length  # No vehicles in the rightmost lane
        
        # Normalize values
        norm_speed = np.mean(self.highway_speeds) / 27.78
        norm_queue = np.mean(self.ramp_queue) / 20
        norm_wait = min(ramp_wait_time / 120, 1)
        norm_distance = min_distance / self.sumo_env.highway_length
        
        # Add traffic light phase information
        phase_info = np.zeros(3)  # One-hot encoding of phase
        phase_info[self.current_phase.value] = 1
        
        # Add phase duration
        norm_phase_duration = min(self.phase_duration / 30, 1)  # Normalize to 30 seconds
        
        return np.array([
            norm_speed, norm_queue, norm_wait, norm_distance,
            *phase_info,  # Add phase one-hot encoding
            norm_phase_duration  # Add normalized phase duration
        ])
    
    def _calculate_reward(self, state, phase_changed):
        """
        Calculate reward with additional considerations for phase changes,
        accidents, collisions, emergency braking, and other traffic events.
        """
        highway_speed = state[0]
        ramp_queue = state[1]
        ramp_wait = state[2]
        norm_distance = state[3]
        
        # Base rewards
        speed_reward = highway_speed
        queue_penalty = -ramp_queue
        wait_penalty = -ramp_wait
        
        # Phase change penalty to discourage frequent changes
        change_penalty = -0.2 if phase_changed else 0
        
        # Yellow phase penalty to encourage quick transitions
        yellow_penalty = -0.1 if self.yellow_timer > 0 else 0
        
        # Additional penalties for traffic events
        collision_penalty = -20.0 if traci.simulation.getCollisions() else 0
        emergency_braking_penalty = -0.6 if traci.simulation.getEmergencyStoppingVehiclesNumber() > 0 else 0
        
        # Encourage green light if no cars in the rightmost lane
        green_light_bonus = 1.0 if norm_distance > 0.8 and self.current_phase == TrafficLightPhase.GREEN else 0
        
        # Heavy penalty if the light stays red for more than 10 seconds
        red_light_penalty = -0.2 if self.current_phase == TrafficLightPhase.RED and self.phase_duration > 10 else 0
        
        # Heavy penalty if the light is green and there's a vehicle close to the traffic light in the highway
        close_vehicle_penalty = -5.0 if norm_distance < 0.2 and self.current_phase == TrafficLightPhase.GREEN else 0
        
        return (0.0 * speed_reward + 
                0.5 * queue_penalty + 
                0.3 * wait_penalty + 
                0.5 * change_penalty +
                0.5 * yellow_penalty +
                collision_penalty +
                emergency_braking_penalty +
                green_light_bonus +
                red_light_penalty +
                close_vehicle_penalty)
    
    def step(self, action):
        """Execute action in environment with phase handling"""
        # Handle phase transition
        phase_changed = self._handle_phase_transition(action)
        
        # Simulate one step
        traci.simulationStep()
        self.current_step += 1
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(next_state, phase_changed)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done
    
    def reset(self):
        """Reset environment to initial state"""
        try:
            traci.close()
        except:
            pass
        
        self.highway_speeds.clear()
        self.ramp_queue.clear()
        self.current_step = 0
        self.current_phase = TrafficLightPhase.RED
        self.phase_duration = 0
        self.yellow_timer = 0
        
        self.sumo_env.generate_route_file()
        self.sumo_env.generate_config_file()
        self.sumo_env.start_simulation()
        
        self._set_traffic_light(TrafficLightPhase.RED)
        
        return self._get_state()
    
    def close(self):
        """Close the TraCI connection"""
        traci.close()
    
    def get_highway_speed(self):
        """Calculate and return the average highway speed"""
        if len(self.highway_speeds) == 0:
            return 0
        return np.mean(self.highway_speeds)
    
    def get_ramp_wait_time(self):
        """Calculate and return the average ramp wait time"""
        if len(self.ramp_queue) == 0:
            return 0
        ramp_vehicles = traci.edge.getLastStepVehicleIDs("ramp")
        if not ramp_vehicles:
            return 0
        wait_times = [traci.vehicle.getWaitingTime(v) for v in ramp_vehicles]
        return np.mean(wait_times)