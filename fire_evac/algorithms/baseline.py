import random

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self):
        self.env.update_possible_actions()
        return random.choice(self.env.actions)
    
# Problem with this agent: it can't go back to take another path if stuck because would require to get closer to the fire for one step
class MaxImmediateDistanceAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self):
        self.env.update_possible_actions()
        # Compute distance to the fire and chooses action that increases distance the most
        distances = [self.env.distance_to_fire(action) for action in self.env.actions]
        index_max_distance = distances.index(max(distances))
        return self.env.actions[index_max_distance]
