import numpy as np
import random
import math

# This approach is based on the Maximum Likelihood MDP algorithm, which is a model-based reinforcement learning algorithm.
# The model is trained beforehand by interacting with the environment over multiple episodes.
# During training, the transition probabilities, rewards, and value functions are learned, resulting in an optimal policy.

import numpy as np

class MaximumLikelihoodMDP:
    def __init__(self, num_rows, num_cols, num_actions, gamma=0.9):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_actions = num_actions
        self.gamma = gamma
        
        # State space and action space
        self.S = [(r, c) for r in range(num_rows) for c in range(num_cols)]
        self.A = [0, 1, 2, 3, 4]  # Actions: UP, DOWN, LEFT, RIGHT, NO MOVE
        
        # Transition counts N(s, a, s')
        self.N = np.zeros((num_rows * num_cols, num_actions, num_rows * num_cols))
        
        # Reward sum ρ(s, a)
        self.rho = np.zeros((num_rows * num_cols, num_actions))
        
        # Value function U
        self.U = np.zeros(num_rows * num_cols)
        
    def get_state_index(self, row, col):
        return row * self.num_cols + col

    def get_state(self, state_idx):
        return state_idx // self.num_cols, state_idx % self.num_cols

    def lookahead(self, s, a):
        n = np.sum(self.N[s, a, :])
        if n == 0:
            return 0.0
        r = self.rho[s, a] / n
        T = self.N[s, a, :] / n
        return r + self.gamma * np.sum(T * self.U)

    def backup(self, s):
        return np.max([self.lookahead(s, a) for a in self.A])
    
    def update(self, s, a, r, sp):
        s_idx = self.get_state_index(*s)
        sp_idx = self.get_state_index(*sp)
        self.N[s_idx, a, sp_idx] += 1
        self.rho[s_idx, a] += r
        self.U[s_idx] = self.backup(s_idx)
    
    def mlestimate(self):
        T = np.zeros_like(self.N)
        R = np.zeros_like(self.rho)
        
        # Compute transition probabilities and rewards
        for s in range(len(self.S)):
            for a in range(self.num_actions):
                n = np.sum(self.N[s, a, :])
                if n == 0:
                    T[s, a, :] = 0.0
                else:
                    T[s, a, :] = self.N[s, a, :] / n
                R[s, a] = self.rho[s, a] / n
                
        return T, R
