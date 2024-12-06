import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import defaultdict
import torch

from evacuation import WildfireEvacuationEnv
from map_helpers.create_map_info import generate_map_info_new, load_map_info

from algorithms.baseline import RandomAgent, MaxImmediateDistanceAgent
from algorithms.MCTS import MCTS
from algorithms.qlearning_offline import QLearningAgent
from algorithms.deep_qlearning import DQN, DQNAgent


def standard_initialization(n_timesteps=50, grid_size=40, load=False, map_directory_path: Optional[str] = None, save_map=True):
    # Input parameters
    num_rows = grid_size
    num_cols = grid_size
    num_cities = 4
    num_water_bodies = 1
    num_fires = 2
    wind_speed = None
    wind_angle = None
    fuel_mean = 8.5
    fuel_stdev = 3
    fire_propagation_rate = 0.094

    if not load:
        # Generate map info
        initial_position, paths, paths_to_pop, road_cells, city_locations, water_cells, fire_cells = generate_map_info_new(num_rows = num_rows,
                                                                                                    num_cols = num_cols,
                                                                                                    num_cities = num_cities,
                                                                                                    num_water_bodies = num_water_bodies, 
                                                                                                    num_fires = num_fires,
                                                                                                    save_map = save_map,
                                                                                                    steps_lower_bound = 2,
                                                                                                    steps_upper_bound = 4,
                                                                                                    percent_go_straight = 50,
                                                                                                    num_paths_mean = 3,
                                                                                                    num_paths_stdev = 1)
        print("Map generated")
    else:
        if map_directory_path is None:
            raise ValueError("Map path must be provided if load is True")
        initial_position, paths, paths_to_pop, road_cells, city_locations, water_cells, fire_cells = load_map_info(map_directory_path=map_directory_path)
        print("Map loaded")

    # Create environment
    evac_env = WildfireEvacuationEnv(n_timesteps,
                                    num_rows, 
                                    num_cols, 
                                    city_locations, 
                                    water_cells, 
                                    road_cells, 
                                    fire_cells,
                                    initial_position, 
                                    wind_speed, 
                                    wind_angle, 
                                    fuel_mean, 
                                    fuel_stdev, 
                                    fire_propagation_rate)
    print("Environment created")
    return evac_env

def test_first_evacuation(n_timesteps=10):

    # Input parameters
    num_rows = 10
    num_cols = 10
    cities = np.array([[0,0], [0,9], [9,0], [9,9]])
    water_cells = np.array([[5, 5], [6,6]])
    road_cells = road_cells = np.array([[0,i] for i in range(10)] + [[9,i] for i in range(10)] + [[i,0] for i in range(1,9)] + [[i,9] for i in range(1,9)])
    tuple_agent = (0,5)
    initial_position = np.array(tuple_agent)
    num_fire_cells = 2
    custom_fire_locations = None
    wind_speed = None
    wind_angle = None
    fuel_mean = 8.5
    fuel_stdev = 3
    fire_propagation_rate = 0.094

    # Create environment
    evac_env = WildfireEvacuationEnv(num_rows, 
                                num_cols, 
                                cities, 
                                water_cells, 
                                road_cells, 
                                initial_position, 
                                custom_fire_locations, 
                                wind_speed, 
                                wind_angle, 
                                fuel_mean, 
                                fuel_stdev, 
                                fire_propagation_rate)
    evac_env.render()
    for i in range(n_timesteps):
        possible_actions = evac_env.fire_env.actions
        # pick a random action in this list
        action = np.random.choice(possible_actions)
        evac_env.step(action)
        evac_env.render()
    evac_env.close()

def test_MCTS(n_timesteps=10, grid_size=40, load=False, map_directory_path=None):

    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for i in tqdm(range(n_timesteps)):
        mcts = MCTS(evac_env.fire_env, iterations=50, exploration_weight=1.5)
        best_action = mcts.search(evac_env.fire_env.copy())
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()
    print("Final reward using MCTS: ", evac_env.fire_env.reward)
    evac_env.generate_gif()
    evac_env.close()

def test_MaxImmediateDistance(n_timesteps=10, grid_size=40, load=False, map_directory_path=None):

    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for i in tqdm(range(n_timesteps)):
        agent = MaxImmediateDistanceAgent(evac_env.fire_env)
        best_action = agent.get_action()
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()
    print("Final reward using MaxImmediateDistance: ", evac_env.fire_env.reward)
    evac_env.generate_gif()
    evac_env.close()

def test_Random(n_timesteps=10, grid_size=40, load=False, map_directory_path=None):

    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for i in tqdm(range(n_timesteps)):
        agent = RandomAgent(evac_env.fire_env)
        best_action = agent.get_action()
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()
    print("Final reward using Random Agent: ", evac_env.fire_env.reward)
    evac_env.generate_gif()
    evac_env.close()

# Khanh's implementation - modified slightly.
# The issue is that this algorithm only stores the state as a grid, but not the 6 variables in the state -> it can't learn much.
def test_qlearning(n_timesteps=10, grid_size=40, load=False, map_directory_path=None, alpha=0.1, gamma=0.9, epsilon=0.1):
    evac_env = standard_initialization(grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    # Initialize Q-learning input params
    num_states = evac_env.fire_env.state_space.shape[1] * evac_env.fire_env.state_space.shape[2]  # Total number of states in the grid world
    num_actions = len(evac_env.fire_env.actions)

    Q = np.zeros((num_states, num_actions))

    # Map (row, col) to a state index for Q-learning
    def get_state_index(state_space):
        presence = np.where(state_space[5] == 1)
        return presence[0][0] * evac_env.fire_env.num_cols + presence[1][0]

    # Loop through timesteps
    for t in tqdm(range(n_timesteps)):

        # Update possible actions
        evac_env.fire_env.update_possible_actions()

        # Get current state index
        current_state = get_state_index(evac_env.fire_env.get_state())

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(evac_env.fire_env.get_actions())  # Explore
        else:
            action = np.argmax(Q[current_state])  # Exploit

        # Perform action and observe results
        evac_env.fire_env.set_action(action)
        evac_env.fire_env.advance_to_next_timestep()

        # Get the next state index and reward
        next_state = get_state_index(evac_env.fire_env.get_state())
        #reward = evac_env.fire_env.get_state_utility()
        reward = evac_env.fire_env.reward # The function get_state_utility() resets the reward to 0. Instead, we use the reward attribute of the environment.

        # Q-Learning update
        Q[current_state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[current_state, action]
        )

        # Render the environment
        evac_env.render()
    print("Final reward using Q-Learning: ", evac_env.fire_env.reward)
    evac_env.generate_gif()
    evac_env.close()

# Offline implementation of Q-Learning

def encode_state(state_tensor, agent_pos):
    """
    Encodes the state by extracting the full state at the agent's position
    and the 5x5 cells around it, padding with zeros if near the grid border.
    
    Args:
        state_tensor: A tensor of shape (6, grid_size, grid_size) representing the environment state.
        agent_pos: A tuple (row, col) representing the agent's position in the grid.
    
    Returns:
        A 1D numpy array encoding the state (fixed size).
    """
    grid_size = state_tensor.shape[1]  # Assuming state_tensor has shape (6, grid_size, grid_size)
    padded_state = np.zeros((6, grid_size + 4, grid_size + 4))  # Pad with a 2-cell border of zeros
    padded_state[:, 2:-2, 2:-2] = state_tensor  # Copy the original state into the padded grid
    
    # Adjust agent position to match padded grid
    padded_agent_pos = (agent_pos[0] + 2, agent_pos[1] + 2)
    
    # Extract the 5x5 area around the agent
    row_min = padded_agent_pos[0] - 2
    row_max = padded_agent_pos[0] + 3  # +3 to include 5 rows
    col_min = padded_agent_pos[1] - 2
    col_max = padded_agent_pos[1] + 3  # +3 to include 5 columns
    local_window = padded_state[:, row_min:row_max, col_min:col_max]  # Shape: (6, 5, 5)

    # Remove the 2nd variable (index 1: fuel level) from the local window
    local_window = np.delete(local_window, 1, axis=0)

    # Flatten the 5x5 area into a 1D array (5 channels x 25 cells)
    encoded_state = local_window.flatten()  # Shape: (5 * 25)
    
    return encoded_state

def train_qlearning_offline(n_timesteps=10, grid_size=40, n_episodes=100, alpha=0.1, gamma=1.0, epsilon=0.1):
    # Initialize Q-table as a dictionary for sparse representation
    num_actions = 5
    Q = defaultdict(lambda: np.zeros(num_actions))
    
    for episode in tqdm(range(n_episodes)):
        evac_env = standard_initialization(n_timesteps, grid_size, load=False, map_directory_path=None, save_map=False)
        evac_env.fire_env.update_possible_actions()
        
        # Simulate one episode
        done = False
        while not done:
            state_tensor = evac_env.fire_env.get_state()
            current_state = tuple(encode_state(state_tensor, evac_env.fire_env.get_agent_position()))
            feasible_actions = evac_env.fire_env.get_actions()

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(feasible_actions)  # Explore
            else:
                # action = np.argmax(Q[current_state])  # Exploit
                # Exploit: Choose the feasible action with the maximum Q-value
                action = max(feasible_actions, key=lambda a: Q[current_state][a])
            
            # Perform action and observe results
            evac_env.fire_env.set_action(action)
            evac_env.fire_env.advance_to_next_timestep()

            # Get next state and reward
            next_state_tensor = evac_env.fire_env.get_state()
            next_state = tuple(encode_state(next_state_tensor, evac_env.fire_env.get_agent_position()))
            reward = evac_env.fire_env.reward

            # Update Q-value
            Q[current_state][action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[current_state][action]
            )
            
            # Check termination condition
            done = evac_env.fire_env.get_terminated()

        evac_env.close()
        
    # Store the Q-table in CSV format
    for state, actions in Q.items():
        state_str = ','.join(map(str, state))  # This should be 125 values
        actions_str = ','.join(map(str, actions))  # This should be 5 Q-values for actions
        with open('Q_table.csv', 'a') as f:
            f.write(f'{state_str},{actions_str}\n')

    return Q

def test_qlearning_offline(n_timesteps=10, grid_size=40, q_file_path='Q_table.csv', load=False, map_directory_path=None):
    
    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    # Initialize the Q-Learning agent
    agent = QLearningAgent(q_file_path)

    for t in tqdm(range(n_timesteps)):
        # Encode the current state
        state_tensor = evac_env.fire_env.get_state()
        current_state = tuple(encode_state(state_tensor, evac_env.fire_env.get_agent_position()))        
        # Get the best action using the Q-Learning agent
        feasible_actions = evac_env.fire_env.get_actions()
        best_action = agent.get_action(current_state, feasible_actions)
        # Perform the action
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        # Render the environment
        evac_env.render()

    print("Final reward using offline Q-Learning: ", evac_env.fire_env.reward)
    evac_env.generate_gif()
    evac_env.close()

# Deep Q-Networks

def train_dqn_agent(n_timesteps=10, grid_size=40, n_episodes=100, target_update_freq=10):
    state_dim = 5 * 25  # Assuming 5 channels and 5x5 local window
    action_dim = 5
    agent = DQNAgent(state_dim, action_dim)

    for episode in tqdm(range(n_episodes)):
        evac_env = standard_initialization(n_timesteps, grid_size, load=False, map_directory_path=None, save_map=False)
        evac_env.fire_env.update_possible_actions()
        
        done = False
        while not done:
            state_tensor = evac_env.fire_env.get_state()
            agent_pos = evac_env.fire_env.get_agent_position()
            state = encode_state(state_tensor, agent_pos)
            feasible_actions = evac_env.fire_env.get_actions()

            # Select and perform action
            action = agent.act(state, feasible_actions)
            evac_env.fire_env.set_action(action)
            evac_env.fire_env.advance_to_next_timestep()

            # Observe reward and next state
            reward = evac_env.fire_env.reward
            next_state_tensor = evac_env.fire_env.get_state()
            next_state = encode_state(next_state_tensor, evac_env.fire_env.get_agent_position())
            done = evac_env.fire_env.get_terminated()

            # Store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)

            # Train the agent
            agent.train()

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent._update_target_network()

        evac_env.close()
    
    print("DQN Training Complete.")

    torch.save(agent.q_network.state_dict(), "dqn_model.pth")
    print("Model saved successfully.")

    return agent

def load_trained_dqn(state_dim, action_dim, model_path="dqn_model.pth"):
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model

def solve_with_dqn(n_timesteps=10, grid_size=40, model_path="dqn_model.pth", load=False, map_directory_path=None):
    state_dim = 5 * 25  # Assuming 5 channels and 5x5 local window
    action_dim = 5
    model = load_trained_dqn(state_dim, action_dim, model_path)
    
    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path, save_map=False)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for t in tqdm(range(n_timesteps)):
        # Encode the current state
        state_tensor = evac_env.fire_env.get_state()
        agent_pos = evac_env.fire_env.get_agent_position()
        state = encode_state(state_tensor, agent_pos)

        # Select action using the trained model
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        q_values = model(state_tensor).detach().numpy().flatten()
        feasible_actions = evac_env.fire_env.get_actions()
        best_action = max(feasible_actions, key=lambda a: q_values[a])

        # Perform the action
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()
    
    # Evaluate the final reward
    total_reward = evac_env.fire_env.reward
    print("Final reward using the trained DQN: ", total_reward)

    evac_env.generate_gif()
    evac_env.close()

if __name__ == "__main__":

    # Initialize environment information
    grid_size = 100
    n_timesteps = 100
    map_directory_path = "pyrorl_map_info/2024-12-03 19:05:48" # an example
    load = True

    # Testing one of the algorithms
    # test_qlearning(n_timesteps=n_timesteps, grid_size=grid_size, load=load, map_directory_path=map_directory_path)
    # test_MCTS(n_timesteps=n_timesteps, grid_size=grid_size, load=load, map_directory_path=map_directory_path)
    # test_MaxImmediateDistance(n_timesteps=n_timesteps, grid_size=grid_size, load=load, map_directory_path=map_directory_path)
    # test_Random(n_timesteps=n_timesteps, grid_size=grid_size, load=load, map_directory_path=map_directory_path)

    # Testing offline trained QLearning
    # 1. Train with no discount. NB delete the Q_table.csv file before running this.
    n_episodes = 10000
    #train_qlearning_offline(n_timesteps=n_timesteps, grid_size=grid_size, n_episodes=n_episodes, alpha=0.1, gamma=1.0, epsilon=0.2)
    # 2. Apply optimal policy
    #test_qlearning_offline(n_timesteps=n_timesteps, grid_size=grid_size, q_file_path='Q_table.csv', load=load, map_directory_path=map_directory_path)

    # Testing DQN
    n_episodes = 1000
    train_dqn_agent(n_timesteps=n_timesteps, grid_size=grid_size, n_episodes=100, target_update_freq=10)
    #solve_with_dqn(n_timesteps=n_timesteps, grid_size=grid_size, model_path="dqn_model.pth", load=load, map_directory_path=map_directory_path)