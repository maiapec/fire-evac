import random
import numpy as np
import pickle as pkl
import os
from datetime import datetime

DIRECTIONS = {0: "straight", 1: "right", 2: "left"}
ORIENTATONS = {
    "north": {
        "left": [[0, -1], "west"],
        "right": [[0, 1], "east"],
        "straight": [[-1, 0], "north"],
    },
    "south": {
        "left": [[0, 1], "east"],
        "right": [[0, -1], "west"],
        "straight": [[1, 0], "south"],
    },
    "east": {
        "left": [[-1, 0], "north"],
        "right": [[1, 0], "south"],
        "straight": [[0, 1], "east"],
    },
    "west": {
        "left": [[1, 0], "south"],
        "right": [[-1, 0], "north"],
        "straight": [[0, -1], "west"],
    },
}
MAP_DIRECTORY = "pyrorl_map_info"


def generate_pop_locations(num_rows: int, num_cols: int, num_populated_areas: int):
    """
    Randomly generate populated areas.
    """
    populated_areas = set()
    for _ in range(num_populated_areas):
        # We don't generate populated cells on the edge of the
        # map
        pop_row = random.randint(1, num_rows - 2)
        pop_col = random.randint(1, num_cols - 2)
        # Continue generating populated areas until one that
        # has not already been created is made
        while (pop_row, pop_col) in populated_areas:
            pop_row = random.randint(1, num_rows - 2)
            pop_col = random.randint(1, num_cols - 2)
        populated_areas.add((pop_row, pop_col))
    populated_areas = np.array(list(populated_areas))
    return populated_areas


def save_map_info(
    num_rows: int,
    num_cols: int,
    num_populated_areas: int,
    populated_areas: np.ndarray,
    paths: np.ndarray,
    paths_to_pops: dict,
    city_locations: np.ndarray,
):
    """
    This function saves five files:
    - map_info.txt: lets the user easily see the number of rows,
    the number of columns, and the number of populated areas
    - populated_areas_array.pkl: saves the populated areas array
    - city_locs_array.pkl: saves the city locations array
    - paths_array.pkl: saves the paths array
    - paths_to_pops_array.pkl: saves the paths to pops array
    - map_size_and_percent_populated_list.pkl: saves a list that contains
    the number of rows, number of columns, and number of populated areas
    """
    # the map information is saved in the user's current working directory
    user_working_directory = os.getcwd()
    maps_info_directory = os.path.join(user_working_directory, MAP_DIRECTORY)
    if not os.path.exists(maps_info_directory):
        os.makedirs(maps_info_directory)

    # make a new subdirectory for the current map's information
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_map_directory = os.path.join(maps_info_directory, timestamp)
    os.makedirs(current_map_directory)

    # put the number of rows, number of columns, and number of populated areas
    # in text file for user to reference data
    map_info_filename = os.path.join(current_map_directory, "map_info.txt")
    with open(map_info_filename, "w") as f:
        row_info = "num_rows: " + str(num_rows) + "\n"
        f.write(row_info)
        col_info = "num_cols: " + str(num_cols) + "\n"
        f.write(col_info)
        percent_pop_info = "num_populated_areas: " + str(num_populated_areas)
        f.write(percent_pop_info)

    # saved the populated areas array, paths array, and paths_to_pops arrays
    def save_array_to_pickle(current_map_directory, array, name):
        array_filename = os.path.join(current_map_directory, name)
        with open(array_filename, "wb") as f:
            pkl.dump(array, f)

    save_array_to_pickle(
        current_map_directory, populated_areas, "populated_areas_array.pkl"
    )
    save_array_to_pickle(
        current_map_directory, city_locations, "city_locs_array.pkl"
    )
    save_array_to_pickle(current_map_directory, paths, "paths_array.pkl")
    save_array_to_pickle(
        current_map_directory, paths_to_pops, "paths_to_pops_array.pkl"
    )

    # save the number of rows, number of columns, and number of populated areas
    map_size_and_percent_populated_list = [num_rows, num_cols, num_populated_areas]
    map_size_and_percent_populated_list_filename = os.path.join(
        current_map_directory, "map_size_and_percent_populated_list.pkl"
    )
    with open(map_size_and_percent_populated_list_filename, "wb") as f:
        pkl.dump(map_size_and_percent_populated_list, f)


def load_map_info(map_directory_path: str):
    """
    This function loads in six variables to initialize a wildfire environment:
    - number of rows
    - number of columns
    - populated areas array
    - paths array
    - paths to pops array
    - number of populated areas
    - city locations
    """

    def load_pickle_file(name):
        array_filename = os.path.join(map_directory_path, name)
        with open(array_filename, "rb") as f:
            return pkl.load(f)

    # load the populated areas array, paths array, and paths_to_pops arrays
    populated_areas = load_pickle_file("populated_areas_array.pkl")
    city_locations = load_pickle_file("city_locs_array.pkl")
    paths = load_pickle_file("paths_array.pkl")
    paths_to_pops = load_pickle_file("paths_to_pops_array.pkl")

    # load the number of rows, number of columns, and number of populated areas
    map_size_and_percent_populated_list = load_pickle_file(
        "map_size_and_percent_populated_list.pkl"
    )
    num_rows = map_size_and_percent_populated_list[0]
    num_cols = map_size_and_percent_populated_list[1]
    num_populated_areas = map_size_and_percent_populated_list[2]
    return (
        num_rows,
        num_cols,
        populated_areas,
        paths,
        paths_to_pops,
        num_populated_areas,
        city_locations,
    )

def generate_map_info_new(
    num_rows: int,
    num_cols: int,
    num_cities: np.ndarray,  # added
    num_water_cells: np.ndarray,  # TODO after baseline
    num_populated_areas: int = 1,
    save_map: bool = True,
    steps_lower_bound: int = 2,
    steps_upper_bound: int = 4,
    percent_go_straight: int = 50,
    num_paths_mean: int = 3,
    num_paths_stdev: int = 1,
):
    """
    This function generates the populated areas and paths for a map, along
    with added features for cities and water cells.
    """
    if num_populated_areas > (num_rows * num_cols - (2 * num_rows + 2 * num_cols - 4)):
        raise ValueError("Cannot have more than 100 percent of the map be populated!")
    if num_rows <= 0:
        raise ValueError("Number of rows must be a positive value!")
    if num_cols <= 0:
        raise ValueError("Number of columns must be a positive value!")
    if num_cities <= 0:
        raise ValueError("Number of cities must be a positive value!")
    if percent_go_straight > 99:
        raise ValueError(
            "Cannot have the percent chance of going straight be greater than 99!"
        )
    if num_paths_mean < 1:
        raise ValueError("The mean for the number of paths cannot be less than 1!")
    if steps_lower_bound > steps_upper_bound:
        raise ValueError(
            """The lower bound for the number of steps cannot be
            greater than the upper bound!"""
        )
    if steps_lower_bound < 1 or steps_upper_bound < 1:
        raise ValueError("The bounds for the number of steps cannot be less than 1!")

    paths_to_pops = {}
    city_locations = generate_pop_locations(num_rows, num_cols, num_cities)

    # the number of paths for each populated area is chosen from a normal distribution
    num_paths_array = np.random.normal(
        num_paths_mean, num_paths_stdev, num_cities
    ).astype(int)
    # each populated area must have at least one path
    num_paths_array[num_paths_array < 1] = 1

    paths = []
    path_num = 0

    for i in range(len(city_locations)):
        city_row, city_col = city_locations[i]
        # for cases where a path couldn't be made
        num_pop_paths_created = 0
        while num_pop_paths_created < num_paths_array[i]:
            current_path = []

            cur_row, cur_col = city_row, city_col

            # Initialize bounds to not restrict to start
            x_min, x_max = num_rows, -1
            y_min, y_max = num_cols, -1

            # Which orientaion to span out from first
            orientation = random.choice(["north", "south", "east", "west"])

            # We loop until we reach the edge of the map
            done = False
            while not done:
                num_steps = random.randint(steps_lower_bound, steps_upper_bound)

                # We want to make sure that the current
                # path will not intersect with itself
                direction_chosen = False
                while not direction_chosen:
                    # Choose whether to go straight, left, or right based
                    # on percent_go_straight -> if we don't go straight,
                    # we go left or right with equal probability
                    direction_index = 0
                    percent_value = random.randint(0, 100)
                    if percent_value > percent_go_straight:
                        direction_index = random.randint(1, 2)
                    direction = DIRECTIONS[direction_index]

                    if orientation == "north" and direction != "straight":
                        if cur_row == x_min:
                            direction_chosen = True
                    elif orientation == "south" and direction != "straight":
                        if cur_row == x_max:
                            direction_chosen = True
                    elif orientation == "east" and direction != "straight":
                        if cur_col == y_max:
                            direction_chosen = True
                    elif orientation == "west" and direction != "straight":
                        if cur_col == y_min:
                            direction_chosen = True
                    else:
                        direction_chosen = True

                row_update = ORIENTATONS[orientation][direction][0][0]
                col_update = ORIENTATONS[orientation][direction][0][1]

                for _ in range(num_steps):
                    cur_row += row_update
                    cur_col += col_update

                    # Update bounds if necessary
                    # (so that paths do not intersect with themselves)
                    if cur_row > x_max:
                        x_max = cur_row
                    if cur_row < x_min:
                        x_min = cur_row
                    if cur_col > y_max:
                        y_max = cur_col
                    if cur_col < y_min:
                        y_min = cur_col

                    current_path.append([cur_row, cur_col])
                    if (
                            cur_row == 0
                            or cur_row == num_rows - 1
                            or cur_col == 0
                            or cur_col == num_cols - 1
                    ):
                        # we want unique paths
                        done = True
                        if current_path in paths or [city_row, city_col] in current_path:
                            break
                        paths.append(current_path)
                        paths_to_pops[path_num] = [[city_row, city_col]]
                        path_num += 1
                        num_pop_paths_created += 1
                        break

                # update orientation
                orientation = ORIENTATONS[orientation][direction][1]

    # Randomly place agent somewhere along existing path (not in a city!)
    all_path_coords = [coord for path in paths for coord in path]
    city_locations_as_lists = [list(coord) for coord in city_locations] # Converting tuples to lists to match path coords
    non_city_coords = [coord for coord in all_path_coords if coord not in city_locations_as_lists]

    if non_city_coords:
        agent = random.choice(non_city_coords)
    else:
        raise ValueError("No valid path location found for placing the agent.")

    if save_map:
        save_map_info(
            num_rows,
            num_cols,
            num_populated_areas,
            np.array(tuple(agent)),
            paths,
            paths_to_pops,
            city_locations,
        )
    return np.array(tuple(agent)), np.array(paths, dtype=object), paths_to_pops, all_path_coords
