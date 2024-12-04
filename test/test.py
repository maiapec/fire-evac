import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fire_evac.map_helpers import create_map_info

def main():
    # Set up parameters
    num_rows, num_cols = 20, 20
    num_cities = 5 # For one agent

    agent_loc, paths, paths_to_pops, all_paths_coords, city_locations, water_cells = create_map_info.generate_map_info_new(
        num_rows,
        num_cols,
        num_cities,
        num_water_bodies=3,
        num_populated_areas=1,
        save_map=True,
        steps_lower_bound=2,
        steps_upper_bound=4,
        percent_go_straight=50,
        num_paths_mean=3,
        num_paths_stdev=1,
    )

if __name__ == "__main__":
    main()