"""
Environment for Wildfire Spread and Evacuation
"""

import numpy as np
import random
import torch
from typing import Optional, Any, Tuple, Dict, List

# For wind bias
from environment_constant import set_fire_mask, linear_wind_transform

"""
State Space
Each cell in the grid world is represented by a 6-tuple:
"""
FIRE_INDEX = 0 # Whether the cell is on fire or not
FUEL_INDEX = 1 # The amount of fuel the cell has (determines how long it can burn)
WATER_INDEX = 2 # Whether the cell represents water or not (water can stop the fire)
CITY_INDEX = 3 # Whether the cell represents a city or not (cities are possible evacuation destinations)
ROAD_INDEX = 4 # Whether the cell represents a road or not (roads are paths for evacuation)
PRESENCE_INDEX = 5 # Whether the individual evacuating is currently in the cell

# PyroRL had:
# FIRE_INDEX = 0
# FUEL_INDEX = 1
# POPULATED_INDEX = 2
# EVACUATING_INDEX = 3
# PATHS_INDEX = 4

"""
Action Space
Option1: At each timestep, the action is a tuple [destination, displacement].
Option2: At each timestep, the action is only a displacement. --> WE CHOOSE THIS ONE FOR NOW.
Displacements are represented by integers:
"""
NOTMOVE_INDEX = 0 # do nothing
UP_INDEX = 1 # go up
DOWN_INDEX = 2 # go down
LEFT_INDEX = 3 # go left
RIGHT_INDEX = 4 # go right

# Question: does the action include the destination? First do without and see how it behaves.
# Do we need to store the list of paths to destinations? I don't think so, as it changes at each timestep.

class FireWorld:
    """
    We represent the world as a 6 by n by m tensor:
    - n by m is the size of the grid world,
    - 5 represents each of the following:
        - [fire, fuel, water, city, road, presence]
    """

    # TODO: Functions to modify/delete from PyroRL (below)
    # - __init__ -> need to adapt to our problem, done modifying
    # - sample_fire_propogation -> no need to change
    # - update_paths_and_evacuations -> need to change completely. made a new function update_possible_actions.
    # - accumulate_reward -> modified
    # - advance_to_next_timestep -> no need to change
    # - set_action -> modified - let's see if it works
    # - get_state_utility -> ok
    # - get_actions -> ok
    # - get_timestep -> ok
    # - get_state -> small change
    # - get_terminated -> ok
    # - get_finished_evacuating -> is it needed?


    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        #populated_areas: np.ndarray,
        cities: np.ndarray, # added
        water_cells: np.ndarray, # added
        road_cells: np.ndarray, # added (a list of grid cells that are roads)
        initial_position: Tuple[int, int], # added
        #paths: np.ndarray,
        #paths_to_pops: dict,
        num_fire_cells: int = 2,
        custom_fire_locations: Optional[np.ndarray] = None,
        wind_speed: Optional[float] = None,
        wind_angle: Optional[float] = None,
        fuel_mean: float = 8.5,
        fuel_stdev: float = 3,
        fire_propagation_rate: float = 0.094,
    ):
        """
        The constructor defines the state and action space, initializes the fires,
        and sets the paths and populated areas.
        - wind angle is in radians
        """
        # Assert that number of rows, columns, and fire cells are both positive
        if num_rows < 1:
            raise ValueError("Number of rows should be positive!")
        if num_cols < 1:
            raise ValueError("Number of rows should be positive!")
        if num_fire_cells < 1:
            raise ValueError("Number of fire cells should be positive!")
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Check that cities are within the grid
        valid_cities = (
            (cities[:, 0] >= 0)
            & (cities[:, 1] >= 0)
            & (cities[:, 0] < num_rows)
            & (cities[:, 1] < num_cols)
        )
        if np.any(~valid_cities):
            raise ValueError("Inputs for cities are not valid with the grid dimensions")

        # Check that each path has squares within the grid
        # valid_road_cells = [
        #     (
        #         (np.array(road_cell)[:, 0] >= 0)
        #         & (np.array(road_cell)[:, 1] >= 0)
        #         & (np.array(road_cell)[:, 0] < num_rows)
        #         & (np.array(road_cell)[:, 1] < num_cols)
        #     )
        #     for road_cell in road_cells
        # ]
        valid_road_cells = (
            (road_cells[:, 0] >= 0)
            & (road_cells[:, 1] >= 0)
            & (road_cells[:, 0] < num_rows)
            & (road_cells[:, 1] < num_cols)
        )
        # if np.any(~np.hstack(valid_road_cells)):
        if np.any(~valid_road_cells):
            raise ValueError("Inputs for road_cells are not valid with the grid dimensions")

        # Define the state and action space
        self.reward = 0
        self.state_space = np.zeros([6, num_rows, num_cols])

        # Set up the city cells
        city_rows, city_cols = cities[:, 0], cities[:, 1]
        self.state_space[CITY_INDEX, city_rows, city_cols] = 1

        # Set up the water cells
        water_rows, water_cols = water_cells[:, 0], water_cells[:, 1]
        self.state_space[WATER_INDEX, water_rows, water_cols] = 1

        # Set up the road cells
        # for road_cell in road_cells:
        #     road_rows, road_cols = np.array(road_cell)[:, 0], np.array(road_cell)[:, 1]
        #     self.state_space[ROAD_INDEX, road_rows, road_cols] = 1
        road_rows, road_cols = road_cells[:, 0], np.array(road_cells)[:, 1]
        self.state_space[ROAD_INDEX, road_rows, road_cols] = 1
        
        # Check that the initial position is within the grid
        if (
            initial_position[0] < 0
            or initial_position[0] >= num_rows
            or initial_position[1] < 0
            or initial_position[1] >= num_cols
        ):
            raise ValueError("Initial position is not within the grid")
        # Check if the initial position is a road cell
        if self.state_space[ROAD_INDEX, initial_position[0], initial_position[1]] == 0:
            raise ValueError("Initial position is not a road cell")
        # Set the initial position
        self.state_space[PRESENCE_INDEX, initial_position[0], initial_position[1]] = 1

        # Set up the actions
        self.actions = list(np.arange(5))

        # # Set up actions -- add extra action for doing nothing
        # num_paths, num_actions = np.arange(len(paths)), 0
        # for key in paths_to_pops:

        #     # First, check that path index actually exists
        #     if not np.isin(key, num_paths):
        #         raise ValueError("Key is not a valid index of a path!")

        #     # Then, check that each populated area exists
        #     areas = np.array(paths_to_pops[key])
        #     if np.any(~np.isin(areas, populated_areas)):
        #         raise ValueError("Corresponding populated area does not exist!")

        #     # Increment total number of actions to be taken
        #     for _ in range(len(paths_to_pops[key])):
        #         num_actions += 1
        # self.actions = list(np.arange(num_actions + 1))

        # # We want to remember which action index corresponds to which population center
        # # and which path (because we just provide an array like [1,2,3,4,5,6,7]) which
        # # would each be mapped to a given population area taking a given path
        # self.action_to_pop_and_path: dict[Any, Optional[Tuple[Any, Any]]] = {
        #     self.actions[-1]: None
        # }

        # # Map each action to a populated area and path
        # index = 0
        # for path in paths_to_pops:
        #     for pop in paths_to_pops[path]:
        #         self.action_to_pop_and_path[index] = (pop, path)
        #         index += 1

        # # State for the evacuation of populated areas
        # self.evacuating_paths: Dict[int, list] = (
        #     {}
        # )  # path_index : list of pop x,y indices that are evacuating [[x,y],[x,y],...]
        # self.evacuating_timestamps = np.full((num_rows, num_cols), np.inf)

        # If the user specifies custom fire locations, set them
        self.num_fire_cells = num_fire_cells
        if custom_fire_locations is not None:

            # Check that populated areas are within the grid
            valid_fire_locations = (
                (custom_fire_locations[:, 0] >= 0)
                & (custom_fire_locations[:, 1] >= 0)
                & (custom_fire_locations[:, 0] < num_rows)
                & (custom_fire_locations[:, 1] < num_cols)
            )
            if np.any(~valid_fire_locations):
                raise ValueError(
                    "Populated areas are not valid with the grid dimensions"
                )

            # Only once valid, set them!
            fire_rows = custom_fire_locations[:, 0]
            fire_cols = custom_fire_locations[:, 1]
            self.state_space[FIRE_INDEX, fire_rows, fire_cols] = 1

        # Otherwise, randomly generate them
        else:
            for _ in range(self.num_fire_cells):
                self.state_space[
                    FIRE_INDEX,
                    random.randint(0, num_rows - 1),
                    random.randint(0, num_cols - 1),
                ] = 1

        # Initialize fuel levels
        # Note: make the fire spread parameters to constants?
        num_values = num_rows * num_cols
        self.state_space[FUEL_INDEX] = np.random.normal(
            fuel_mean, fuel_stdev, num_values
        ).reshape((num_rows, num_cols))
        # TODO: We can modify this modelization to have a more realistic fuel distribution
        # Water cells should have fuel level 0
        self.state_space[FUEL_INDEX, water_rows, water_cols] = 0

        # # Initialize populated areas
        # pop_rows, pop_cols = populated_areas[:, 0], populated_areas[:, 1]
        # self.state_space[POPULATED_INDEX, pop_rows, pop_cols] = 1

        # # Initialize self.paths
        # self.paths: List[List[Any]] = []
        # for path in paths:
        #     path_array = np.array(path)
        #     path_rows, path_cols = path_array[:, 0].astype(int), path_array[
        #         :, 1
        #     ].astype(int)
        #     self.state_space[PATHS_INDEX, path_rows, path_cols] += 1

        #     # Each path in self.paths is a list that records what the path is and
        #     # whether the path still exists (i.e. has not been destroyed by a fire)
        #     self.paths.append([np.zeros((num_rows, num_cols)), True])
        #     self.paths[-1][0][path_rows, path_cols] += 1

        # Set the timestep
        self.time_step = 0

        # set fire mask
        self.fire_mask = set_fire_mask(fire_propagation_rate)

        # Factor in wind speeds
        if wind_speed is not None or wind_angle is not None:
            if wind_speed is None or wind_angle is None:
                raise TypeError(
                    "When setting wind details, "
                    "wind speed and wind angle must both be provided"
                )
            self.fire_mask = linear_wind_transform(wind_speed, wind_angle)
        else:
            self.fire_mask = torch.from_numpy(self.fire_mask)

        # # Record which population cells have finished evacuating
        # self.finished_evacuating_cells = []

    def sample_fire_propogation(self):
        """
        Sample the next state of the wildfire model.
        """
        # Drops fuel level of enflamed cells
        self.state_space[FUEL_INDEX, self.state_space[FIRE_INDEX] == 1] -= 1
        self.state_space[FUEL_INDEX, self.state_space[FUEL_INDEX] < 0] = 0

        # Extinguishes cells that have run out of fuel
        self.state_space[FIRE_INDEX, self.state_space[FUEL_INDEX, :] <= 0] = 0

        # Runs kernel of neighborhing cells where each row
        # corresponds to the neighborhood of a cell
        torch_rep = torch.tensor(self.state_space[FIRE_INDEX]).unsqueeze(0)
        y = torch.nn.Unfold((5, 5), dilation=1, padding=2)
        z = y(torch_rep)

        # The relative importance of each neighboring cell is weighted
        z = z * self.fire_mask

        # Unenflamed cells are set to 1 to eliminate their role to the
        # fire spread equation
        z[z == 0] = 1
        z = z.prod(dim=0)
        z = 1 - z.reshape(self.state_space[FIRE_INDEX].shape)

        # From the probability of an ignition in z, new fire locations are
        # randomly generated
        prob_mask = torch.rand_like(z)
        new_fire = (z > prob_mask).float()

        # These new fire locations are added to the state
        self.state_space[FIRE_INDEX] = np.maximum(
            np.array(new_fire), self.state_space[FIRE_INDEX]
        )

    # def update_paths_and_evactuations(self):
    #     # TODO: check that action space is limited within the grid when we do the update
    #     """
    #     Performs three functions:
    #     1. Remove paths that been burned down by a fire
    #     2. Also stops evacuating any areas that were taking a burned down path
    #     3. Also decrements the evacuation timestamps
    #     """
    #     for i in range(len(self.paths)):
    #         # Decrement path counts and remove path if path is on fire
    #         if (
    #             self.paths[i][1]
    #             and np.sum(
    #                 np.logical_and(self.state_space[FIRE_INDEX], self.paths[i][0])
    #             )
    #             > 0
    #         ):
    #             self.state_space[PATHS_INDEX] -= self.paths[i][0]
    #             self.paths[i][1] = False

    #             # Stop evacuating an area if it was taking the removed path
    #             if i in self.evacuating_paths:
    #                 pop_centers = np.array(self.evacuating_paths[i])
    #                 pop_rows, pop_cols = pop_centers[:, 0], pop_centers[:, 1]

    #                 # Reset timestamp and evacuation index for populated areas
    #                 self.evacuating_timestamps[pop_rows, pop_cols] = np.inf
    #                 self.state_space[EVACUATING_INDEX, pop_rows, pop_cols] = 0
    #                 del self.evacuating_paths[i]

    #         # We need to decrement the evacuating paths timestamp
    #         elif i in self.evacuating_paths:

    #             # For the below, this code works for if multiple population centers
    #             # are taking the same path and finish at the same time, but if we have
    #             # it so that two population centers can't take the same
    #             # path it could probably be simplified
    #             pop_centers = np.array(self.evacuating_paths[i])
    #             pop_rows, pop_cols = pop_centers[:, 0], pop_centers[:, 1]
    #             self.evacuating_timestamps[pop_rows, pop_cols] -= 1
    #             done_evacuating = np.where(self.evacuating_timestamps == 0)

    #             self.state_space[
    #                 EVACUATING_INDEX, done_evacuating[0], done_evacuating[1]
    #             ] = 0
    #             self.state_space[
    #                 POPULATED_INDEX, done_evacuating[0], done_evacuating[1]
    #             ] = 0

    #             # Note that right now it is going to be vastly often the case that two
    #             # population cases don't finish evacuating along the same path at the
    #             # same time right now, so this is an extremely rare edge case, meaning
    #             # that most often this for loop will run for a single iteration
    #             done_evacuating = np.array([done_evacuating[0], done_evacuating[1]])
    #             done_evacuating = np.transpose(done_evacuating)
    #             for j in range(done_evacuating.shape[0]):
    #                 self.evacuating_paths[i].remove(list(done_evacuating[j]))

    #                 # This population center is done evacuating, so we can set its
    #                 # timestamp back to infinity (so we don't try to remove this
    #                 # from self.evacuating paths twice - was causing a bug)
    #                 update_row, update_col = (
    #                     done_evacuating[j, 0],
    #                     done_evacuating[j, 1],
    #                 )
    #                 self.evacuating_timestamps[update_row, update_col] = np.inf
    #                 self.finished_evacuating_cells.append([update_row, update_col])

    #             # No more population centers are using this path, so we delete it
    #             if len(self.evacuating_paths[i]) == 0:
    #                 del self.evacuating_paths[i]

    def update_possible_actions(self):
        # Get the current location of the evacuating individual
        current_location_cell = np.where(self.state_space[PRESENCE_INDEX] == 1)
        current_location_row, current_location_col = current_location_cell[0], current_location_cell[1]

        # Check what actions are possible
        # Possible = not out of bounds and not on fire and keeps you on road
        # TODO: Check consistency with definition of the grid
        possible_actions = [NOTMOVE_INDEX]
        if current_location_row > 0 and self.state_space[FIRE_INDEX, current_location_row - 1, current_location_col] == 0 and self.state_space[ROAD_INDEX, current_location_row - 1, current_location_col] == 1:
            possible_actions.append(UP_INDEX)
        if current_location_row < self.num_rows - 1 and self.state_space[FIRE_INDEX, current_location_row + 1, current_location_col] == 0 and self.state_space[ROAD_INDEX, current_location_row + 1, current_location_col] == 1:
            possible_actions.append(DOWN_INDEX)
        if current_location_col > 0 and self.state_space[FIRE_INDEX, current_location_row, current_location_col - 1] == 0 and self.state_space[ROAD_INDEX, current_location_row, current_location_col - 1] == 1:
            possible_actions.append(LEFT_INDEX)
        if current_location_col < self.num_cols - 1 and self.state_space[FIRE_INDEX, current_location_row, current_location_col + 1] == 0 and self.state_space[ROAD_INDEX, current_location_row, current_location_col + 1] == 1:
            possible_actions.append(RIGHT_INDEX)
        self.actions = possible_actions

    def accumulate_reward(self):
        """
        Mark enflamed areas as no longer populated or evacuating and calculate reward.
        """
        # # Get which populated_areas areas are on fire and evacuating
        # populated_areas = np.where(self.state_space[POPULATED_INDEX] == 1)
        # fire = self.state_space[FIRE_INDEX][populated_areas]
        # evacuating = self.state_space[EVACUATING_INDEX][populated_areas]

        # # Mark enflamed areas as no longer populated or evacuating
        # enflamed_populated_areas = np.where(fire == 1)[0]
        # enflamed_rows = populated_areas[0][enflamed_populated_areas]
        # enflamed_cols = populated_areas[1][enflamed_populated_areas]

        # # Depopulate enflamed areas and remove evacuations
        # self.state_space[POPULATED_INDEX, enflamed_rows, enflamed_cols] = 0
        # self.state_space[EVACUATING_INDEX, enflamed_rows, enflamed_cols] = 0

        # Get the current location of the evacuating population
        current_location_cell = np.column_stack(np.where(self.state_space[PRESENCE_INDEX] == 1))[0]
        enflamed_areas = np.column_stack(np.where(self.state_space[FIRE_INDEX] == 1))
        cities = np.column_stack(np.where(self.state_space[CITY_INDEX] == 1))
        # print("Current location: ", current_location_cell)
        # print("Enflamed areas: ", enflamed_areas)
        # print("Cities: ", cities)

        # Update reward
        if (enflamed_areas == current_location_cell).all(axis=1).any():
            self.reward -= 100
        else:
            self.reward += 1
        if (cities == current_location_cell).all(axis=1).any():
            self.reward += 10

    def advance_to_next_timestep(self):
        """
        Take three steps:
        1. Advance fire forward one timestep
        2. Update paths and evacuation
        3. Accumulate reward and document enflamed areas
        """
        self.sample_fire_propogation()
        #self.update_paths_and_evactuations()
        self.update_possible_actions()
        self.accumulate_reward()
        self.time_step += 1

    def set_action(self, action: int):
        """
        Allow the agent to take an action within the action space.
        """
        # # Check that there is an action to take
        # if (
        #     action in self.action_to_pop_and_path
        #     and self.action_to_pop_and_path[action] is not None
        # ):
        #     action_val = self.action_to_pop_and_path[action]
        #     if action_val is not None and len(action_val) > 0:
        #         pop_cell, path_index = action_val
        #         pop_cell_row, pop_cell_col = pop_cell[0], pop_cell[1]

        #         # Ensure that the path chosen and populated cell haven't
        #         # burned down and it's not already evacuating and it has not
        #         # already evacuated
        #         if (
        #             self.paths[path_index][1]
        #             and self.state_space[POPULATED_INDEX, pop_cell_row, pop_cell_col]
        #             == 1
        #             and self.evacuating_timestamps[pop_cell_row, pop_cell_col] == np.inf
        #         ):

        #             # Add to evacuating paths and update state + timestamp
        #             if path_index in self.evacuating_paths:
        #                 self.evacuating_paths[path_index].append(pop_cell)
        #             else:
        #                 self.evacuating_paths[path_index] = [pop_cell]
        #             self.state_space[EVACUATING_INDEX, pop_cell_row, pop_cell_col] = 1
        #             self.evacuating_timestamps[pop_cell_row, pop_cell_col] = 10
        
        # Check action is in the possible actions list obtained from update_possible_actions
        if action not in self.actions:
            raise ValueError("Invalid action")
        # Move the evacuating individual
        if action == NOTMOVE_INDEX:
            return
        elif action == UP_INDEX:            
            self.state_space[PRESENCE_INDEX] = np.roll(self.state_space[PRESENCE_INDEX], -1, axis=0)
        elif action == DOWN_INDEX:
            self.state_space[PRESENCE_INDEX] = np.roll(self.state_space[PRESENCE_INDEX], 1, axis=0)
        elif action == LEFT_INDEX:
            self.state_space[PRESENCE_INDEX] = np.roll(self.state_space[PRESENCE_INDEX], -1, axis=1)
        elif action == RIGHT_INDEX:
            self.state_space[PRESENCE_INDEX] = np.roll(self.state_space[PRESENCE_INDEX], 1, axis=1)
        else:
            raise ValueError("Invalid action")

    def get_state_utility(self) -> int:
        """
        Get the total amount of utility given a current state.
        """
        present_reward = self.reward
        self.reward = 0 # I think this step is used to reset the reward to 0 after it has been accumulated
        return present_reward

    def get_actions(self) -> list:
        """
        Get the set of actions available to the agent.
        """
        return self.actions

    def get_timestep(self) -> int:
        """
        Get current timestep of simulation
        """
        return self.time_step

    def get_state(self) -> np.ndarray:
        """
        Get the state space of the current configuration of the gridworld.
        """
        returned_state = np.copy(self.state_space)
        #returned_state[PATHS_INDEX] = np.clip(returned_state[PATHS_INDEX], 0, 1)
        returned_state[ROAD_INDEX] = np.clip(returned_state[ROAD_INDEX], 0, 1) # added, not sure if needed
        return returned_state

    def get_terminated(self) -> bool:
        """
        Get the status of the simulation.
        """
        return self.time_step >= 100

    # def get_finished_evacuating(self) -> list:
    #     """
    #     Get the populated areas that are finished evacuating.
    #     """
    #     return self.finished_evacuating_cells