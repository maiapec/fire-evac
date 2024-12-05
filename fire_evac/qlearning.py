import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable

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