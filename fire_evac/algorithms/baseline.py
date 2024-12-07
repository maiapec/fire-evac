import random
from tqdm import tqdm
from .util import standard_initialization

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self):
        self.env.update_possible_actions()
        return random.choice(self.env.actions)
    
class MaxImmediateDistanceAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self):
        self.env.update_possible_actions()
        # Compute distance to the fire and chooses action that increases distance the most
        distances = [self.env.distance_to_fire(action) for action in self.env.actions]
        index_max_distance = distances.index(max(distances))
        return self.env.actions[index_max_distance]

def solve_Random(n_timesteps=10, grid_size=40, load=False, map_directory_path=None, gif_name="Random"):

    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for i in tqdm(range(n_timesteps)):
        agent = RandomAgent(evac_env.fire_env)
        best_action = agent.get_action()
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()
    
    reward = evac_env.fire_env.reward
    print("Final reward using Random Agent: ", reward)
    evac_env.generate_gif(gif_name=gif_name)
    evac_env.close()
    return reward

def solve_MaxImmediateDistance(n_timesteps=10, grid_size=40, load=False, map_directory_path=None, gif_name="MaxImmediateDistance"):

    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for i in tqdm(range(n_timesteps)):
        agent = MaxImmediateDistanceAgent(evac_env.fire_env)
        best_action = agent.get_action()
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()

    reward = evac_env.fire_env.reward
    print("Final reward using MaxImmediateDistance: ", reward)
    evac_env.generate_gif(gif_name=gif_name)
    evac_env.close()
    return reward
