import numpy as np
import math
import random

"""
The state class must contains methods: get_actions(), set_action(action), get_terminated()
To use the MCTS:
initial_state = YourStateClass()
mcts = MCTS(initial_state)
best_action = mcts.search()
"""

class TreeNode:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.actions)

    def best_child(self, exploration_weight=1.0):
        print("Children: ", self.children)
        weights = [
            (child.total_reward / child.visit_count) +
            exploration_weight * math.sqrt(math.log(self.visit_count) / child.visit_count)
            for child in self.children
        ]
        print("Weights: ", weights)
        return self.children[np.argmax(weights)]

    def add_child(self, child_state, action):
        child_node = TreeNode(state=child_state, action=action, parent=self)
        self.children.append(child_node)
        return child_node

class MCTS:
    def __init__(self, env, iterations=100, exploration_weight=1.0):
        self.env = env
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    def search(self, initial_state):
        root = TreeNode(state=initial_state)

        for _ in range(self.iterations):
            print("Iteration: ", _)
            node = self.select(root)
            if node is not None:
                print("Not none")
                reward = self.simulate(node.state)
                self.backpropagate(node, reward)

        return root.best_child(0).action

    def select(self, node):
        while not self.is_terminal(node.state):
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child(self.exploration_weight)
        return node

    def expand(self, node):
        tried_actions = [child.action for child in node.children]
        for action in node.state.actions:
            if action not in tried_actions:
                next_state = self.simulate_action(node.state, action)
                return node.add_child(next_state, action)

    def simulate(self, state, depth=10):
        total_reward = 0
        for _ in range(depth):
            if self.is_terminal(state):
                break
            action = random.choice(state.actions)
            state = self.simulate_action(state, action)
            total_reward += state.reward
        return total_reward

    def simulate_action(self, state, action):
        new_env = self.env  # Clone environment for simulation
        new_env.set_action(action)
        new_env.advance_to_next_timestep()
        return new_env

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def is_terminal(self, state):
        return state.get_terminated()  # or another condition like max depth
