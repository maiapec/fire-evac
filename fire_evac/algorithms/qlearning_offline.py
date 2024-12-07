import numpy as np
from tqdm import tqdm
import csv
from collections import defaultdict

from .util import standard_initialization, encode_state

# TODO: change action assigned if not among states stored in the Q table
# TODO: not take the closest state. (fuel level is not on same scale)

class QLearningAgent:
    def __init__(self, q_file_path):
        self.Q = defaultdict(lambda: np.zeros(5))  # Default to 5 actions
        self._load_q_table(q_file_path)
    
    def _load_q_table(self, q_file_path):
        """Loads the Q-table from a CSV file."""
        with open(q_file_path, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                state = tuple(map(float, row[:-5]))  # The first part is the state (125 values)
                actions = list(map(float, row[-5:]))  # The last 5 entries are the Q-values
                self.Q[state] = np.array(actions)
        print("Q-table loaded successfully.")
    
    def _find_closest_state(self, state):
        """Finds the closest state in the Q-table using Euclidean distance."""
        closest_state = None
        min_distance = float('inf')
        
        # Ensure the state is flattened to 1D (125-dimensional)
        state = np.array(state).flatten() 

        for q_state in self.Q.keys():
            # Ensure q_state is flattened to 1D (125-dimensional)
            q_state_flattened = np.array(q_state).flatten()
            
            # Compute the Euclidean distance
            distance = np.linalg.norm(state - q_state_flattened)  # Euclidean distance
            
            if distance < min_distance:
                min_distance = distance
                closest_state = q_state
        
        return closest_state

    
    def get_action(self, state, feasible_actions):
        """Returns the best feasible action based on Q-table."""
        state = tuple(state)  # Ensure state is a tuple for matching with the Q-table
        if state in self.Q:
            # Choose the feasible action with the maximum Q-value
            return max(feasible_actions, key=lambda a: self.Q[state][a])
        else:
            # Find the closest state and use its Q-values
            closest_state = self._find_closest_state(state)
            if closest_state:
                return max(feasible_actions, key=lambda a: self.Q[closest_state][a])
            else:
                # Fallback if no closest state is found
                return np.random.choice(feasible_actions)

def train_qlearning_offline(n_timesteps=10, grid_size=40, n_episodes=100, alpha=0.1, gamma=1.0, epsilon=0.1, q_file_path='Q_table'):
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
        with open('models/'+q_file_path+'.csv', 'a') as f:
            f.write(f'{state_str},{actions_str}\n')

    return Q

def solve_qlearning_offline(n_timesteps=10, grid_size=40, q_file_path='models/Q_table.csv', load=False, map_directory_path=None, gif_name="QLearningOffline"):
    
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
        evac_env.render()

    reward = evac_env.fire_env.reward
    print("Final reward using offline Q-Learning: ", reward)
    evac_env.generate_gif(gif_name=gif_name)
    evac_env.close()
    return reward
