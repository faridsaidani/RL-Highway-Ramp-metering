# File: highway_env.py

This file defines the `HighwayEnvironment` class, which simulates a highway environment with a ramp using the SUMO traffic simulation software. The environment is designed to be used with reinforcement learning algorithms to optimize traffic flow and control.

## Classes:
- `HighwayEnvironment`

### Class: HighwayEnvironment
A class representing a highway environment with a ramp.

#### Methods:
- `__init__(self, num_lanes=3, highway_length=1000, ramp_length=200, gui=False)`
    - Initializes the `HighwayEnvironment` instance.
    - Parameters:
        - `num_lanes`: Number of lanes on the highway (default is 3).
        - `highway_length`: Length of the highway in meters (default is 1000).
        - `ramp_length`: Length of the ramp in meters (default is 200).
        - `gui`: Boolean indicating whether to enable the GUI for the SUMO simulation (default is False).

- `_generate_network(self)`
    - Generates the SUMO network with the highway and ramp.

- `generate_route_file(self, highway_flow=7600, ramp_flow=600)`
    - Generates the route file with traffic flows.
    - Parameters:
        - `highway_flow`: Traffic flow on the highway in vehicles per hour (default is 7600).
        - `ramp_flow`: Traffic flow on the ramp in vehicles per hour (default is 600).

- `generate_config_file(self)`
    - Generates the SUMO configuration file.

- `reset(self)`
    - Resets the environment for a new episode.
    - Returns:
        - `np.array`: The initial state of the environment.

- `get_state(self)`
    - Gets the current state of the environment.
    - Returns:
        - `np.array`: The current state of the environment.

- `step(self, action)`
    - Steps through the environment based on the action taken.
    - Parameters:
        - `action`: The action to perform.
    - Returns:
        - `tuple`: The next state and reward.

- `start_simulation(self)`
    - Starts the SUMO simulation.

- `end_simulation(self)`
    - Ends the SUMO simulation.

### Method Details:

- `__init__(self, num_lanes=3, highway_length=1000, ramp_length=200, gui=False)`
    - Initializes the `HighwayEnvironment` instance with the given parameters. Sets initial values for the number of lanes, highway length, ramp length, and GUI option. Creates necessary paths for SUMO files.

- `_generate_network(self)`
    - Generates the SUMO network with the highway and ramp. Creates the necessary network configuration, nodes, edges, and connections files. Uses the netconvert tool to generate the network.

- `generate_route_file(self, highway_flow=7600, ramp_flow=600)`
    - Generates the route file with traffic flows. Creates a route file with vehicle types and traffic flows for the highway and ramp. Writes the route information to the routes.rou.xml file.

- `generate_config_file(self)`
    - Generates the SUMO configuration file. Creates a configuration file that specifies the network and route files, as well as the simulation start and end times. Writes the configuration information to the highway.sumocfg file.

- `reset(self)`
    - Resets the environment for a new episode. Loads the SUMO configuration file and starts the SUMO simulation. Returns the initial state of the environment.

- `get_state(self)`
    - Gets the current state of the environment. Retrieves the list of vehicle IDs and calculates the average speed and queue length. Returns the state as a numpy array.

- `step(self, action)`
    - Steps through the environment based on the action taken. Sets the traffic light phase, simulates one step in the SUMO environment, calculates the waiting time, and computes the reward. Returns the next state and reward.

- `start_simulation(self)`
    - Starts the SUMO simulation. Checks if the SUMO_HOME environment variable is set, constructs the SUMO command, and starts the simulation using TraCI.

- `end_simulation(self)`
    - Ends the SUMO simulation. Closes the TraCI connection.

# File: ramp_meter_env.py

This file defines the `RampMeterEnv` class, which simulates a ramp metering environment using the SUMO traffic simulation software. The environment is designed to be used with reinforcement learning algorithms to optimize traffic light control at highway on-ramps.

## Classes:
- `TrafficLightPhase`
- `RampMeterEnv`

### Class: TrafficLightPhase
An enumeration representing the different phases of a traffic light.

#### Members:
- `GREEN`: Represents the green light phase.
- `YELLOW`: Represents the yellow light phase.
- `RED`: Represents the red light phase.

### Class: RampMeterEnv
A class representing the ramp metering environment.

#### Methods:
- `__init__(self, sumo_env, max_steps=3600)`
    - Initializes the `RampMeterEnv` instance.
    - Parameters:
        - `sumo_env`: The SUMO environment instance.
        - `max_steps`: The maximum number of steps per episode (default is 3600).

- `_set_traffic_light(self, phase)`
    - Sets the traffic light state based on the given phase.
    - Parameters:
        - `phase`: The traffic light phase to set (`TrafficLightPhase`).

- `_handle_phase_transition(self, action)`
    - Handles traffic light phase transitions, including yellow phases.
    - Parameters:
        - `action`: The action to perform (0 for red, 1 for green).
    - Returns:
        - `bool`: True if the phase changed, False otherwise.

- `_get_state(self)`
    - Gets the current state of the environment, including traffic light phase information.
    - Returns:
        - `np.array`: The current state of the environment.

- `_calculate_reward(self, state, phase_changed)`
    - Calculates the reward with additional considerations for phase changes, accidents, collisions, emergency braking, and other traffic events.
    - Parameters:
        - `state`: The current state of the environment.
        - `phase_changed`: A boolean indicating whether the traffic light phase changed.
    - Returns:
        - `float`: The calculated reward.

- `step(self, action)`
    - Executes the given action in the environment with phase handling.
    - Parameters:
        - `action`: The action to perform (0 for red, 1 for green).
    - Returns:
        - `tuple`: The next state, reward, and done flag.

- `reset(self)`
    - Resets the environment to its initial state.
    - Returns:
        - `np.array`: The initial state of the environment.

- `close(self)`
    - Closes the TraCI connection.

### Method Details:

- `__init__(self, sumo_env, max_steps=3600)`
    - Initializes the `RampMeterEnv` instance with the given SUMO environment and maximum steps per episode. Sets initial values for traffic light settings, action space, and traffic metrics.

- `_set_traffic_light(self, phase)`
    - Sets the traffic light state based on the given phase. Uses the TraCI API to set the traffic light state.

- `_handle_phase_transition(self, action)`
    - Handles traffic light phase transitions, including yellow phases. Updates timers and sets the traffic light state based on the given action. Returns whether the phase changed.

- `_get_state(self)`
    - Gets the current state of the environment, including traffic light phase information. Retrieves traffic metrics such as highway speed, ramp queue length, and ramp wait time. Normalizes the values and returns the state as a numpy array.

- `_calculate_reward(self, state, phase_changed)`
    - Calculates the reward with additional considerations for phase changes, accidents, collisions, emergency braking, and other traffic events. Computes base rewards, penalties for phase changes, yellow phases, collisions, and emergency braking. Returns the calculated reward.

- `step(self, action)`
    - Executes the given action in the environment with phase handling. Handles phase transitions, simulates one step in the SUMO environment, gets the new state, calculates the reward, and checks if the episode is done. Returns the next state, reward, and done flag.

- `reset(self)`
    - Resets the environment to its initial state. Closes any existing TraCI connection, clears traffic metrics, resets traffic light settings, generates route and configuration files, starts the SUMO simulation, sets the traffic light to red, and returns the initial state.

- `close(self)`
    - Closes the TraCI connection.

# File: q_learning.py

This file defines the `QLearningAgent` class, which implements a Q-learning agent for reinforcement learning. The agent can be trained to optimize actions based on the Q-learning algorithm and can save and load its state.

## Classes:
- `QLearningAgent`

### Class: QLearningAgent
A class representing a Q-learning agent.

#### Methods:
- `__init__(self, state_dims=9, n_actions=2, learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)`
    - Initializes the `QLearningAgent` instance.
    - Parameters:
        - `state_dims`: Dimensionality of the state space (default is 9).
        - `n_actions`: Number of possible actions (default is 2).
        - `learning_rate`: Learning rate for Q-learning updates (default is 0.1).
        - `gamma`: Discount factor for future rewards (default is 0.95).
        - `epsilon`: Initial exploration rate for epsilon-greedy policy (default is 1.0).
        - `epsilon_min`: Minimum exploration rate (default is 0.01).
        - `epsilon_decay`: Decay rate for exploration rate (default is 0.995).

- `_discretize_state(self, state)`
    - Converts a continuous state to a discrete state for the Q-table.
    - Parameters:
        - `state`: The continuous state to discretize.
    - Returns:
        - `tuple`: The discretized state.

- `select_action(self, state, training=True)`
    - Selects an action using the epsilon-greedy policy.
    - Parameters:
        - `state`: The current state.
        - `training`: A boolean indicating whether the agent is in training mode (default is True).
    - Returns:
        - `int`: The selected action.

- `update(self, state, action, reward, next_state)`
    - Updates the Q-value for a state-action pair.
    - Parameters:
        - `state`: The current state.
        - `action`: The action taken.
        - `reward`: The reward received.
        - `next_state`: The next state after taking the action.

- `train(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None)`
    - Trains the Q-learning agent.
    - Parameters:
        - `env`: The environment in which to train the agent.
        - `n_episodes`: The number of episodes to train (default is 1000).
        - `max_steps`: The maximum number of steps per episode (default is 3600).
        - `pid`: The process ID for multiprocessing (default is None).
        - `lock`: A multiprocessing lock for synchronizing file writes (default is None).

- `save(self, filepath)`
    - Saves the trained agent to a file.
    - Parameters:
        - `filepath`: The path to the file where the agent will be saved.

- `load(self, filepath)`
    - Loads a trained agent from a file.
    - Parameters:
        - `filepath`: The path to the file from which the agent will be loaded.

- `continue_training(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None)`
    - Continues training the Q-learning agent.
    - Parameters:
        - `env`: The environment in which to continue training the agent.
        - `n_episodes`: The number of episodes to train (default is 1000).
        - `max_steps`: The maximum number of steps per episode (default is 3600).
        - `pid`: The process ID for multiprocessing (default is None).
        - `lock`: A multiprocessing lock for synchronizing file writes (default is None).

### Method Details:

- `__init__(self, state_dims=9, n_actions=2, learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)`
    - Initializes the `QLearningAgent` instance with the given parameters. Sets initial values for the Q-table, exploration rate, and training metrics.

- `_discretize_state(self, state)`
    - Converts a continuous state to a discrete state for the Q-table. Discretizes continuous values into bins and handles one-hot encoded parts of the state differently. Returns the discretized state as a tuple.

- `select_action(self, state, training=True)`
    - Selects an action using the epsilon-greedy policy. If in training mode, selects a random action with probability epsilon, otherwise selects the action with the highest Q-value for the given state. Returns the selected action.

- `update(self, state, action, reward, next_state)`
    - Updates the Q-value for a state-action pair using the Q-learning update rule. Discretizes the current and next states, calculates the new Q-value, and updates the Q-table. Decays the exploration rate.

- `train(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None)`
    - Trains the Q-learning agent in the given environment for the specified number of episodes and steps per episode. Selects and performs actions, updates the Q-table, records metrics, and saves episode rewards to a CSV file if a lock is provided. Prints progress every 10 episodes.

- `save(self, filepath)`
    - Saves the trained agent to a file. Converts the Q-table to a regular dictionary and saves it along with the exploration rate and training metrics using pickle.

- `load(self, filepath)`
    - Loads a trained agent from a file. Loads the Q-table, exploration rate, and training metrics from the file and converts the Q-table back to a defaultdict.

- `continue_training(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None)`
    - Continues training the Q-learning agent in the given environment for the specified number of episodes and steps per episode. Calls the train method to perform the training.

# File: train.py

This file defines the functions and main script for training or continuing the training of a Q-learning agent for ramp metering using the SUMO traffic simulation software. The script supports multiple training runs and can save models, plots, and episode rewards in separate directories.

## Functions:
- `plot_training_results(agent, save_path=None)`
- `run_training(gui, pid, lock, model_path=None, continue_training=False, n_episodes=500)`
- `main(gui, n_runs, model_path=None, continue_training=False, n_episodes=500)`

### Function: plot_training_results(agent, save_path=None)
Plots the training metrics, including episode rewards and moving average rewards.
- Parameters:
    - `agent`: The Q-learning agent whose training results are to be plotted.
    - `save_path`: The path to save the plot (default is None).

### Function: run_training(gui, pid, lock, model_path=None, continue_training=False, n_episodes=500)
Runs the training or continues training of the Q-learning agent.
- Parameters:
    - `gui`: A boolean indicating whether to enable the GUI for the SUMO simulation.
    - `pid`: The process ID for multiprocessing.
    - `lock`: A multiprocessing lock for synchronizing file writes.
    - `model_path`: The path to the trained model file to continue training (default is None).
    - `continue_training`: A boolean indicating whether to continue training an existing model (default is False).
    - `n_episodes`: The number of episodes for training (default is 500).

### Function: main(gui, n_runs, model_path=None, continue_training=False, n_episodes=500)
The main function that sets up and runs the training processes.
- Parameters:
    - `gui`: A boolean indicating whether to enable the GUI for the SUMO simulation.
    - `n_runs`: The number of training runs.
    - `model_path`: The path to the trained model file to continue training (default is None).
    - `continue_training`: A boolean indicating whether to continue training an existing model (default is False).
    - `n_episodes`: The number of episodes for training (default is 500).

### Method Details:

- `plot_training_results(agent, save_path=None)`
    - Plots the training metrics, including episode rewards and moving average rewards. Creates a figure with two subplots: one for episode rewards and one for moving average rewards. Saves the plot to the specified path if provided.

- `run_training(gui, pid, lock, model_path=None, continue_training=False, n_episodes=500)`
    - Runs the training or continues training of the Q-learning agent. Creates directories for saving models, plots, and data based on the current timestamp and number of episodes. Initializes the SUMO environment and the ramp metering environment. Initializes and trains the Q-learning agent, saving the trained model and training results. Plots and saves the training results.

- `main(gui, n_runs, model_path=None, continue_training=False, n_episodes=500)`
    - The main function that sets up and runs the training processes. Creates a multiprocessing lock and starts multiple training processes based on the number of runs specified. Joins the processes after they complete.

### Command-Line Arguments:
- `--gui`: Enable GUI for SUMO simulation.
- `--n_runs`: Number of training runs (default is 1).
- `--model`: Path to the trained model file to continue training.
- `--continue_training`: Continue training an existing model.
- `--n_episodes`: Number of episodes for training (default is 500).