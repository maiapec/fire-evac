import numpy as np
from tqdm import tqdm
from typing import Optional

from evacuation import WildfireEvacuationEnv
from environments.grid_environment import FireWorld
from map_helpers.create_map_info import generate_map_info_new, load_map_info
from algorithms.MCTS import MCTS, TreeNode
from algorithms.baseline import RandomAgent, MaxImmediateDistanceAgent

def standard_initialization(grid_size=40, load=False, map_directory_path: Optional[str] = None):
    # Input parameters
    num_rows = grid_size
    num_cols = grid_size
    num_cities = 4
    num_water_bodies = 1
    #water_cells = np.array([])
    num_fire_cells = 2
    custom_fire_locations = None
    wind_speed = None
    wind_angle = None
    fuel_mean = 8.5
    fuel_stdev = 3
    fire_propagation_rate = 0.094

    if not load:
        # Generate map info
        initial_position, paths, paths_to_pop, road_cells, city_locations, water_cells = generate_map_info_new(num_rows = num_rows,
                                                                                        num_cols = num_cols,
                                                                                        num_cities = num_cities,
                                                                                        num_water_bodies = num_water_bodies, 
                                                                                        save_map = True,
                                                                                        steps_lower_bound = 2,
                                                                                        steps_upper_bound = 4,
                                                                                        percent_go_straight = 50,
                                                                                        num_paths_mean = 3,
                                                                                        num_paths_stdev = 1)
        print("Map generated")
    else:
        if map_directory_path is None:
            raise ValueError("Map path must be provided if load is True")
        initial_position, paths, paths_to_pop, road_cells, city_locations, water_cells = load_map_info(map_directory_path=map_directory_path)
        print("Map loaded")

    # Create environment
    evac_env = WildfireEvacuationEnv(num_rows, 
                                    num_cols, 
                                    city_locations, 
                                    water_cells, 
                                    road_cells, 
                                    initial_position, 
                                    num_fire_cells,
                                    custom_fire_locations, 
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

def test_MCTS(grid_size=40, load=False, map_directory_path=None, n_timesteps=10):

    evac_env = standard_initialization(grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for i in tqdm(range(n_timesteps)):
        mcts = MCTS(evac_env.fire_env, iterations=50, exploration_weight=1.5)
        best_action = mcts.search(evac_env.fire_env.copy())
        # print("Best action: ", best_action)
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()
    print("Final reward using MCTS: ", evac_env.fire_env.reward)
    evac_env.generate_gif()
    evac_env.close()

def test_MaxImmediateDistance(grid_size=40, load=False, map_directory_path=None, n_timesteps=10):

    evac_env = standard_initialization(grid_size, load, map_directory_path)
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

def test_Random(grid_size=40, load=False, map_directory_path=None, n_timesteps=10):

    evac_env = standard_initialization(grid_size, load, map_directory_path)
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

if __name__ == "__main__":
    grid_size = 50
    n_timesteps = 50
    map_directory_path = "pyrorl_map_info/2024-12-03 18:32:45" # an example

    test_MCTS(grid_size=grid_size, load=True, map_directory_path=map_directory_path, n_timesteps=n_timesteps)
    #test_MaxImmediateDistance(grid_size=grid_size, load=True, map_directory_path=map_directory_path, n_timesteps=n_timesteps)
    #test_Random(grid_size=grid_size, load=False, map_directory_path=None, n_timesteps=n_timesteps)