import numpy as np
from typing import Optional

from environments.environment_evacuation import WildfireEvacuationEnv
from map_helpers.create_map_info import generate_map_info_new, load_map_info

def standard_initialization(n_timesteps=50, grid_size=40, load=False, map_directory_path: Optional[str] = None, save_map=True):
    # Input parameters
    num_rows = grid_size
    num_cols = grid_size
    num_cities = 4
    num_water_bodies = 1
    num_fires = 2
    wind_speed = None
    wind_angle = None
    fuel_mean = 8.5
    fuel_stdev = 3
    fire_propagation_rate = 0.094

    if not load:
        # Generate map info
        initial_position, paths, paths_to_pop, road_cells, city_locations, water_cells, fire_cells = generate_map_info_new(num_rows = num_rows,
                                                                                                    num_cols = num_cols,
                                                                                                    num_cities = num_cities,
                                                                                                    num_water_bodies = num_water_bodies, 
                                                                                                    num_fires = num_fires,
                                                                                                    save_map = save_map,
                                                                                                    steps_lower_bound = 2,
                                                                                                    steps_upper_bound = 4,
                                                                                                    percent_go_straight = 50,
                                                                                                    num_paths_mean = 3,
                                                                                                    num_paths_stdev = 1)
        print("Map generated")
    else:
        if map_directory_path is None:
            raise ValueError("Map path must be provided if load is True")
        initial_position, paths, paths_to_pop, road_cells, city_locations, water_cells, fire_cells = load_map_info(map_directory_path=map_directory_path)
        print("Map loaded")

    # Create environment
    evac_env = WildfireEvacuationEnv(n_timesteps,
                                    num_rows, 
                                    num_cols, 
                                    city_locations, 
                                    water_cells, 
                                    road_cells, 
                                    fire_cells,
                                    initial_position, 
                                    wind_speed, 
                                    wind_angle, 
                                    fuel_mean, 
                                    fuel_stdev, 
                                    fire_propagation_rate)
    print("Environment created")
    return evac_env

def encode_state(state_tensor, agent_pos):
    """
    Encodes the state by extracting the full state at the agent's position
    and the 5x5 cells around it, padding with zeros if near the grid border.
    
    Args:
        state_tensor: A tensor of shape (6, grid_size, grid_size) representing the environment state.
        agent_pos: A tuple (row, col) representing the agent's position in the grid.
    
    Returns:
        A 1D numpy array encoding the state (fixed size).
    """
    grid_size = state_tensor.shape[1]  # Assuming state_tensor has shape (6, grid_size, grid_size)
    padded_state = np.zeros((6, grid_size + 4, grid_size + 4))  # Pad with a 2-cell border of zeros
    padded_state[:, 2:-2, 2:-2] = state_tensor  # Copy the original state into the padded grid
    
    # Adjust agent position to match padded grid
    padded_agent_pos = (agent_pos[0] + 2, agent_pos[1] + 2)
    
    # Extract the 5x5 area around the agent
    row_min = padded_agent_pos[0] - 2
    row_max = padded_agent_pos[0] + 3  # +3 to include 5 rows
    col_min = padded_agent_pos[1] - 2
    col_max = padded_agent_pos[1] + 3  # +3 to include 5 columns
    local_window = padded_state[:, row_min:row_max, col_min:col_max]  # Shape: (6, 5, 5)

    # Remove the 2nd variable (index 1: fuel level) from the local window
    local_window = np.delete(local_window, 1, axis=0)

    # Flatten the 5x5 area into a 1D array (5 channels x 25 cells)
    encoded_state = local_window.flatten()  # Shape: (5 * 25)
    
    return encoded_state
