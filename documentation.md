# Project Documentation

## Files Overview

### analyze.py

This script analyzes the rewards from a CSV file. It calculates moving averages, groups rewards by PID and episode, and plots the results.

**Functions:**

- `plot_rewards(csv_file)`: Reads the rewards data from a CSV file, calculates moving averages, and plots the results.

### control_traffic.py

This script runs a trained Q-learning agent to control traffic in the SUMO environment. It initializes the environment, loads the trained agent, and executes the control loop.

**Functions:**

- `run_q_learning_control(gui, model_path, n_episodes=10, max_steps=3600)`: Initializes the environment, loads the trained Q-learning agent, and runs the control loop.

### dqn_agent.py

This script defines the `DQNAgent` class, which implements a Deep Q-Network (DQN) agent for reinforcement learning. It includes methods for training, updating, selecting actions, saving, and loading the agent.

**Functions:**

- `__init__(self, state_dim, n_actions, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, memory_size, batch_size, target_update)`: Initializes the DQN agent.
- `select_action(self, state, training=True)`: Selects an action based on the current state.
- `update(self)`: Updates the DQN agent's knowledge based on new experiences.
- `train(self, env, n_episodes, max_steps, pid, lock, warmup_steps)`: Trains the DQN agent.
- `save(self, filepath)`: Saves the DQN agent's model.
- `load(self, filepath)`: Loads a pre-trained DQN agent's model.
- `continue_training(self, env, n_episodes, max_steps, pid, lock)`: Continues training the DQN agent.

### dqn_control.py

This script runs a trained DQN agent for traffic control in the SUMO environment. It initializes the environment, loads the trained DQN model, and executes the control loop.

**Functions:**

- `run_dqn_control(gui, model_path, n_episodes=10, max_steps=3600)`: Initializes the environment, loads the trained DQN agent, and runs the control loop.

### dqn_train.py

This script trains or continues training a DQN agent for ramp metering. It initializes the environment, trains the agent, saves the trained model, and plots the training results.

**Functions:**

- `plot_training_results(agent, save_path=None)`: Plots the training results, including episode rewards and moving average rewards.
- `run_training(gui, pid, lock, model_path=None, continue_training=False, n_episodes=500, checkpoint_interval=100)`: Initializes the environment, trains the DQN agent, and saves checkpoints.
- `main(gui, n_runs, model_path=None, continue_training=False, n_episodes=500, checkpoint_interval=100)`: Manages multiple training runs using multiprocessing.

### envWrapper.py

This script defines the `RampMeterEnv` class, which simulates a ramp metering environment using the SUMO traffic simulation software. It includes methods for resetting the environment, stepping through the environment, and calculating rewards.

**Functions:**

- `__init__(self, sumo_env)`: Initializes the ramp metering environment.
- `reset(self)`: Resets the environment to its initial state.
- `step(self, action)`: Steps through the environment to simulate time progression.
- `calculate_reward(self)`: Calculates rewards based on the agent's actions.

### evaluate_models.py

This script evaluates multiple DQN models for traffic control in the SUMO environment. It runs each model for a specified number of episodes and saves the average rewards to a CSV file.

**Functions:**

- `run_dqn_control(gui, model_path, n_episodes=10, max_steps=3600)`: Initializes the environment, loads the trained DQN agent, and runs the control loop.
- `evaluate_models_in_folder(folder_path, output_csv, gui=False, n_episodes=10, max_steps=3600)`: Evaluates multiple DQN models and saves the results to a CSV file.

### evaluate_q_learning_models.py

This script evaluates multiple Q-learning models for traffic control in the SUMO environment. It runs each model for a specified number of episodes and saves the average rewards to a CSV file.

**Functions:**

- `run_q_learning_control(gui, model_path, n_episodes=10, max_steps=3600)`: Initializes the environment, loads the trained Q-learning agent, and runs the control loop.
- `evaluate_models_in_folder(folder_path, output_csv, gui=False, n_episodes=10, max_steps=3600)`: Evaluates multiple Q-learning models and saves the results to a CSV file.

### highway_env.py

This file defines the `HighwayEnvironment` class, which simulates a highway environment with a ramp using the SUMO traffic simulation software. The environment is designed to be used with reinforcement learning algorithms to optimize traffic flow and control.

**Classes:**

- `HighwayEnvironment`

**Methods:**

- `__init__(self, num_lanes=3, highway_length=1000, ramp_length=200, gui=False)`: Initializes the `HighwayEnvironment` instance.
- `_generate_network(self)`: Generates the SUMO network with the highway and ramp.
- `generate_route_file(self, highway_flow=7600, ramp_flow=600)`: Generates the route file with traffic flows.
- `generate_config_file(self)`: Generates the SUMO configuration file.
- `reset(self)`: Resets the environment for a new episode.
- `get_state(self)`: Gets the current state of the environment.
- `step(self, action)`: Steps through the environment based on the action taken.
- `start_simulation(self)`: Starts the SUMO simulation.
- `end_simulation(self)`: Ends the SUMO simulation.

### q_learning.py

This script defines the `QLearningAgent` class, which implements a Q-learning agent for reinforcement learning. It includes methods for training, updating, selecting actions, saving, and loading the agent.

**Classes:**

- `QLearningAgent`

**Methods:**

- `__init__(self, state_dims=9, n_actions=2, learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)`: Initializes the Q-learning agent.
- `_discretize_state(self, state)`: Converts a continuous state to a discrete state for the Q-table.
- `select_action(self, state, training=True)`: Selects an action using the epsilon-greedy policy.
- `update(self, state, action, reward, next_state)`: Updates the Q-value for a state-action pair.
- `train(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None)`: Trains the Q-learning agent.
- `save(self, filepath)`: Saves the trained agent to a file.
- `load(self, filepath)`: Loads a trained agent from a file.
- `continue_training(self, env, n_episodes=1000, max_steps=3600, pid=None, lock=None)`: Continues training the Q-learning agent.

### ramp_meter_env.py

This script defines the `RampMeterEnv` class, which simulates a ramp metering environment using the SUMO traffic simulation software. It includes methods for resetting the environment, stepping through the environment, and calculating rewards.

**Classes:**

- `RampMeterEnv`

**Methods:**

- `__init__(self, sumo_env)`: Initializes the ramp metering environment.
- `reset(self)`: Resets the environment to its initial state.
- `step(self, action)`: Steps through the environment to simulate time progression.
- `calculate_reward(self)`: Calculates rewards based on the agent's actions.
- `close(self)`: Closes the TraCI connection.

### test_ramp_meter_env.py

This script tests the `RampMeterEnv` environment with TraCI. It initializes the environment, runs a test loop for a specified number of steps, and prints the results.

**Functions:**

- `test_ramp_meter_env(n_steps, gui)`: Initializes the environment, runs a test loop, and prints the results.

### train.py

This script trains or continues training a Q-learning agent for ramp metering. It initializes the environment, trains the agent, saves the trained model, and plots the training results.

**Functions:**

- `plot_training_results(agent, save_path=None)`: Plots the training results, including episode rewards and moving average rewards.
- `run_training(gui, pid, lock, model_path=None, continue_training=False, n_episodes=500, checkpoint_interval=100)`: Initializes the environment, trains the Q-learning agent, and saves checkpoints.
- `main(gui, n_runs, model_path=None, continue_training=False, n_episodes=500, checkpoint_interval=100)`: Manages multiple training runs using multiprocessing.

## Usage Manual

### Prerequisites

Ensure you have Python installed.
Install the required packages using the following command:

```sh
pip install -r requirements.txt
```

### Running the Scripts

#### Analyzing Rewards

To analyze rewards from a CSV file:

```sh
python analyze.py path/to/rewards.csv
```

#### Controlling Traffic with Q-learning Agent

To run a trained Q-learning agent for traffic control:

```sh
python control_traffic.py --model path/to/trained_model.pkl --gui
```

#### Training a DQN Agent

To train a DQN agent:

```sh
python dqn_train.py --n_runs 3 --n_episodes 1000 --gui
```

#### Evaluating DQN Models

To evaluate multiple DQN models:

```sh
python evaluate_models.py --folder path/to/models --output_csv results.csv --gui
```

#### Evaluating Q-learning Models

To evaluate multiple Q-learning models:

```sh
python evaluate_q_learning_models.py --folder path/to/models --output_csv results.csv --gui
```

#### Testing Ramp Meter Environment

To test the ramp meter environment:

```sh
python test_ramp_meter_env.py --n_steps 100 --gui
```

#### Training a Q-learning Agent

To train a Q-learning agent:

```sh
python train.py --n_runs 3 --n_episodes 1000 --gui
```

#### Continuing Training of a Q-learning Agent

To continue training an existing Q-learning model:

```sh
python train.py --model path/to/trained_model.pkl --continue_training --n_runs 2 --n_episodes 800 --gui
```
