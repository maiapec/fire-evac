import numpy as np
import random
import math

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
        weights = [
            (child.total_reward / child.visit_count) +
            exploration_weight * math.sqrt(math.log(self.visit_count) / child.visit_count)
            for child in self.children
        ]
        return self.children[np.argmax(weights)]

    def add_child(self, child_state, action):
        child_node = TreeNode(state=child_state, action=action, parent=self)
        self.children.append(child_node)
        return child_node
    
    def describe(self):
        print("State: ", self.state)
        print("Action: ", self.action)
        print("Parent: ", self.parent)
        print("Children: ", self.children)
        print("Visit count: ", self.visit_count)
        print("Total reward: ", self.total_reward)

class MCTS:
    def __init__(self, env, iterations=100, exploration_weight=1.0):
        self.env = env # useless with the current MCTS implementation
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    def search(self, initial_state):
        root = TreeNode(state=initial_state)
        #root.describe()

        for _ in range(self.iterations):
            node = self.select(root)
            if node is not None:
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
        node.state.update_possible_actions()
        for action in node.state.actions:
            if action not in tried_actions:
                next_state = self.simulate_action(node.state, action)
                return node.add_child(next_state, action)

    def simulate(self, state, depth=10):
        total_reward = 0
        for _ in range(depth):
            if self.is_terminal(state):
                break
            state.update_possible_actions()
            action = random.choice(state.actions)
            state = self.simulate_action(state, action)
            total_reward += state.reward
        return total_reward

    def simulate_action(self, state, action):
        new_state = state.copy() # Clone environment for simulation
        new_state.update_possible_actions()
        new_state.set_action(action)
        new_state.advance_to_next_timestep()
        return new_state

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def is_terminal(self, state):
        return state.get_terminated()  # or another condition like max depth
