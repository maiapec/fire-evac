import numpy as np
from tqdm import tqdm
import csv
from collections import defaultdict

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

