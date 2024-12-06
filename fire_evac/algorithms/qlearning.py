import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable
from tqdm import tqdm

from .util import standard_initialization

class RLMDP(ABC):
    def __init__(self, A: list[int], gamma: float):
        self.A = A          # action space (assumes 1:nactions)
        self.gamma = gamma  # discount factor

    @abstractmethod
    def lookahead(self, s: int, a: int) -> float:
        pass

    @abstractmethod
    def update(self, s: int, a: int, r: float, s_prime: int):
        pass
class ModelFreeMDP(RLMDP):
    def __init__(self,
                 A: list[int],
                 gamma: float,
                 Q: np.ndarray | Callable[[np.ndarray, float, int], float],
                 alpha: float):
        super().__init__(A, gamma)
        # action value function, either as a numpy array Q[s, a]
        # or a parametrized function Q(theta, s, a) (depending on method)
        self.Q = Q
        self.alpha = alpha  # learning rate

class QLearning(ModelFreeMDP):
    def __init__(self, S: list[int], A: list[int], gamma: float, Q: np.ndarray, alpha: float):
        super().__init__(A, gamma, Q, alpha)
        # The action value function Q[s, a] is a numpy array
        self.S = S  # state space (assumes 1:nstates)
        self.visits = defaultdict(int)  # to track visits for each (state, action) pair
        self.state_values = np.zeros(len(S))  # Track best value seen for each state
        self.novelty_weight = 0.1
        self.progress_weight = 2  # Higher weight for progress
    def lookahead(self, s: int, a: int):
        return self.Q[s, a]

    def update(self, s: int, a: int, r: float, s_prime: int):
        # Update with reward shaping
        shaped_r = self.compute_shaped_reward(s, a, r, s_prime)
        self.Q[s, a] += self.alpha * (shaped_r + self.gamma * np.max(self.Q[s_prime]) - self.Q[s, a])

    def V(self, s: int) -> float:
        return np.max(self.Q[s])

    def compute_shaped_reward(self, s, a, r, s_prime):
        # Bonus for novel exploration
        visit_count = self.visits[(s, a)] + 1
        novelty_bonus = self.novelty_weight / np.sqrt(visit_count)

        # Rewards for improvement
        curr_value = self.state_values[s]
        next_value = self.state_values[s_prime]
        progress_bonus = self.progress_weight * max(0, next_value - curr_value)

        # Update visit count and state value
        self.visits[(s, a)] += 1
        self.state_values[s] = max(self.state_values[s], np.max(self.Q[s]))

        return r + novelty_bonus + progress_bonus
    
def solve_qlearning(n_timesteps=10, grid_size=40, load=False, map_directory_path=None, alpha=0.1, gamma=0.9, epsilon=0.1, gif_name="QLearning"):
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

    reward = evac_env.fire_env.reward
    print("Final reward using Q-Learning: ", reward)
    evac_env.generate_gif(gif_name=gif_name)
    evac_env.close()
    return reward
