import numpy as np
from tqdm import tqdm
from evacuation import WildfireEvacuationEnv
from environments.grid_environment import FireWorld
from map_helpers.create_map_info import generate_map_info_new
from algorithms.MCTS import MCTS, TreeNode


def test_evacuation(n_timesteps=10):

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

def test_evacuation_with_map(n_timesteps=10):

    # Input parameters
    num_rows = 40
    num_cols = 40
    num_cities = 5
    num_water_cells = 0
    water_cells = np.array([])
    num_populated_areas = 1
    custom_fire_locations = None
    wind_speed = None
    wind_angle = None
    fuel_mean = 8.5
    fuel_stdev = 3
    fire_propagation_rate = 0.094

    initial_position, paths, paths_to_pop, all_path_coords, city_locations = generate_map_info_new(num_rows = num_rows,
                                                                                num_cols = num_cols,
                                                                                num_cities = num_cities,
                                                                                num_water_cells = num_water_cells, 
                                                                                num_populated_areas = num_populated_areas,
                                                                                save_map = True,
                                                                                steps_lower_bound = 2,
                                                                                steps_upper_bound = 4,
                                                                                percent_go_straight = 50,
                                                                                num_paths_mean = 3,
                                                                                num_paths_stdev = 1)
    road_cells = np.array(all_path_coords)
    # Create environment
    evac_env = WildfireEvacuationEnv(num_rows, 
                                    num_cols, 
                                    city_locations, 
                                    water_cells, 
                                    road_cells, 
                                    initial_position, 
                                    custom_fire_locations, 
                                    wind_speed, 
                                    wind_angle, 
                                    fuel_mean, 
                                    fuel_stdev, 
                                    fire_propagation_rate)
    #evac_env.render()
    for i in range(n_timesteps):
        possible_actions = evac_env.fire_env.actions
        # pick a random action in this list
        action = np.random.choice(possible_actions)
        evac_env.step(action)
        #evac_env.render()
    evac_env.generate_gif()
    evac_env.close()

def test_MCTS_with_map(n_timesteps=10):

    # Input parameters
    num_rows = 40
    num_cols = 40
    num_cities = 5
    num_water_cells = 0
    water_cells = np.array([])
    num_populated_areas = 1
    num_fire_cells = 2
    custom_fire_locations = None
    wind_speed = None
    wind_angle = None
    fuel_mean = 8.5
    fuel_stdev = 3
    fire_propagation_rate = 0.094

    initial_position, paths, paths_to_pop, all_path_coords, city_locations = generate_map_info_new(num_rows = num_rows,
                                                                                num_cols = num_cols,
                                                                                num_cities = num_cities,
                                                                                num_water_cells = num_water_cells, 
                                                                                num_populated_areas = num_populated_areas,
                                                                                save_map = True,
                                                                                steps_lower_bound = 2,
                                                                                steps_upper_bound = 4,
                                                                                percent_go_straight = 50,
                                                                                num_paths_mean = 3,
                                                                                num_paths_stdev = 1)
    road_cells = np.array(all_path_coords)
    # Create environment
    evac_env = WildfireEvacuationEnv(num_rows, 
                                    num_cols, 
                                    city_locations, 
                                    water_cells, 
                                    road_cells, 
                                    initial_position, 
                                    custom_fire_locations, 
                                    wind_speed, 
                                    wind_angle, 
                                    fuel_mean, 
                                    fuel_stdev, 
                                    fire_propagation_rate)
    evac_env.fire_env.update_possible_actions()
    evac_env.render()

    for i in tqdm(range(n_timesteps)):
        mcts = MCTS(evac_env.fire_env, iterations=20) # Must be inferior to n_timesteps?
        best_action = mcts.search(evac_env.fire_env.copy())
        # print("Best action: ", best_action)
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        evac_env.render()
    evac_env.generate_gif()
    evac_env.close()

if __name__ == "__main__":
    #test_evacuation()
    #test_evacuation_with_map()
    test_MCTS_with_map(n_timesteps=40)