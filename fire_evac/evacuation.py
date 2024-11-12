"""
OpenAI Gym Environment Wrapper Class
"""
# The code was adapted from the PyroRL repository.
# Sarted modifying the functions to adapt to the new environment

from environments.grid_environment import FireWorld
import gymnasium as gym
from gymnasium import spaces
import imageio.v2 as imageio
import numpy as np
import os
import pygame
import shutil
import sys
from typing import Optional, Any, Tuple

# Constants for visualization
IMG_DIRECTORY = "grid_screenshots/"
FIRE_COLOR = pygame.Color("#ef476f")
# POPULATED_COLOR = pygame.Color("#073b4c")
# EVACUATING_COLOR = pygame.Color("#118ab2")
# PATH_COLOR = pygame.Color("#ffd166")
CITY_COLOR = pygame.Color("#073b4c")
WATER_COLOR = pygame.Color("#118ab2")
ROAD_COLOR = pygame.Color("#ffd166")
GRASS_COLOR = pygame.Color("#06d6a0")
PRESENCE_COLOR = pygame.Color("#f4a261")
# FINISHED_COLOR = pygame.Color("#BF9ACA")

FIRE_INDEX = 0 
FUEL_INDEX = 1 
WATER_INDEX = 2 
CITY_INDEX = 3 
ROAD_INDEX = 4
PRESENCE_INDEX = 5 


class WildfireEvacuationEnv(gym.Env):
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
        custom_fire_locations: Optional[np.ndarray] = None,
        wind_speed: Optional[float] = None,
        wind_angle: Optional[float] = None,
        fuel_mean: float = 8.5,
        fuel_stdev: float = 3,
        fire_propagation_rate: float = 0.094,
        skip: bool = False,
    ):
        """
        Set up the basic environment and its parameters.
        """
        # Save parameters and set up environment
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cities = cities # added
        self.water_cells = water_cells # added
        self.road_cells = road_cells # added
        self.initial_position = initial_position # added
        #self.populated_areas = populated_areas
        #self.paths = paths
        #self.paths_to_pops = paths_to_pops
        self.custom_fire_locations = custom_fire_locations
        self.wind_speed = wind_speed
        self.wind_angle = wind_angle
        self.fuel_mean = fuel_mean
        self.fuel_stdev = fuel_stdev
        self.fire_propagation_rate = fire_propagation_rate
        self.skip = skip
        self.fire_env = FireWorld(
            num_rows,
            num_cols,
            cities, # added
            water_cells, # added
            road_cells, # added
            initial_position, # added
            #populated_areas,
            #paths,
            #aths_to_pops,
            custom_fire_locations=custom_fire_locations,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            fuel_mean=fuel_mean,
            fuel_stdev=fuel_stdev,
            fire_propagation_rate=fire_propagation_rate,
        )

        # Set up action space
        actions = self.fire_env.get_actions()
        self.action_space = spaces.Discrete(len(actions))

        # Set up observation space
        observations = self.fire_env.get_state()
        self.observation_space = spaces.Box(
            low=0, high=200, shape=observations.shape, dtype=np.float64
        )

        # Create directory to store screenshots
        if os.path.exists(IMG_DIRECTORY) is False:
            os.mkdir(IMG_DIRECTORY)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to its initial state.
        """
        self.fire_env = FireWorld(
            self.num_rows,
            self.num_cols,
            self.cities, # added
            self.water_cells, # added
            self.road_cells, # added
            self.initial_position, # added
            #self.populated_areas,
            #self.paths,
            #self.paths_to_pops,
            wind_speed=self.wind_speed,
            wind_angle=self.wind_angle,
            fuel_mean=self.fuel_mean,
            fuel_stdev=self.fuel_stdev,
            fire_propagation_rate=self.fire_propagation_rate,
        )

        state_space = self.fire_env.get_state()
        return state_space, {"": ""}

    def step(self, action: int) -> tuple:
        """
        Take a step and advance the environment after taking an action.
        """
        # Take the action and advance to the next timestep
        self.fire_env.set_action(action)
        self.fire_env.advance_to_next_timestep()

        # Gather observations and rewards
        observations = self.fire_env.get_state()
        rewards = self.fire_env.get_state_utility()
        terminated = self.fire_env.get_terminated()
        return observations, rewards, terminated, False, {"": ""}

    def render_hf(
        self, screen: pygame.Surface, font: pygame.font.Font
    ) -> pygame.Surface:
        """
        Set up header and footer
        """
        # Get width and height of the screen
        surface_width = screen.get_width()
        surface_height = screen.get_height()

        # Starting locations and timestep
        x_offset, y_offset = 0.05, 0.05
        timestep = self.fire_env.get_timestep()

        # Set title of the screen
        text = font.render("Timestep #: " + str(timestep), True, (0, 0, 0))
        screen.blit(text, (surface_width * x_offset, surface_height * y_offset))

        # Set initial grid squares and offsets
        grid_squares = [
            (GRASS_COLOR, "Grass"),
            (FIRE_COLOR, "Fire"),
            # (POPULATED_COLOR, "Populated"),
            # (EVACUATING_COLOR, "Evacuating"),
            # (PATH_COLOR, "Path"),
            # (FINISHED_COLOR, "Finished"),
            (CITY_COLOR, "City"), # added
            (WATER_COLOR, "Water"), # added
            (ROAD_COLOR, "Road"), # added
            (PRESENCE_COLOR, "Presence"), # added
        ]
        x_offset, y_offset = 0.2, 0.045

        # Iterate through, create the grid squares
        for i in range(len(grid_squares)):

            # Get the color and name, set in the screen
            (color, name) = grid_squares[i]
            pygame.draw.rect(
                screen,
                color,
                (surface_width * x_offset, surface_height * y_offset, 25, 25),
            )
            text = font.render(name, True, (0, 0, 0))
            screen.blit(
                text, (surface_width * x_offset + 35, surface_height * y_offset + 5)
            )

            # Adjust appropriate offset
            x_offset += 0.125

        return screen

    def render(self):
        """
        Render the environment
        """
        # Set up the state space
        state_space = self.fire_env.get_state()
        # finished_evacuating = self.fire_env.get_finished_evacuating()
        (_, rows, cols) = state_space.shape

        # Get dimensions of the screen
        pygame.init()
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h

        # Set up screen and font
        surface_width = screen_width * 0.8
        surface_height = screen_height * 0.8
        screen = pygame.display.set_mode([surface_width, surface_height])
        font = pygame.font.Font(None, 25)

        # Set screen details
        screen.fill((255, 255, 255))
        pygame.display.set_caption("PyroRL")
        screen = self.render_hf(screen, font)

        # Calculation for square
        total_width = 0.85 * surface_width - 2 * (cols - 1)
        total_height = 0.85 * surface_height - 2 * (rows - 1)
        square_dim = min(int(total_width / cols), int(total_height / rows))

        # Calculate start x, start y
        start_x = surface_width - 2 * (cols - 1) - square_dim * cols
        start_y = (
            surface_height - 2 * (rows - 1) - square_dim * rows + 0.05 * surface_height
        )
        start_x /= 2
        start_y /= 2

        # Running the loop!
        running = True
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    timestep = self.fire_env.get_timestep()
                    pygame.image.save(screen, IMG_DIRECTORY + str(timestep) + ".png")
                    running = False

            # Iterate through all of the squares
            # Note: try to vectorize?
            for x in range(cols):
                for y in range(rows):

                    # # Set color of the square
                    # color = GRASS_COLOR
                    # if state_space[4][y][x] > 0:
                    #     color = PATH_COLOR
                    # if state_space[0][y][x] == 1:
                    #     color = FIRE_COLOR
                    # if state_space[2][y][x] == 1:
                    #     color = POPULATED_COLOR
                    # if state_space[3][y][x] > 0:
                    #     color = EVACUATING_COLOR
                    # if [y, x] in finished_evacuating:
                    #     color = FINISHED_COLOR

                    # Set color of the square
                    color = GRASS_COLOR
                    if state_space[ROAD_INDEX][y][x] == 1:
                        color = ROAD_COLOR
                    if state_space[CITY_INDEX][y][x] == 1:
                        color = CITY_COLOR
                    if state_space[FIRE_INDEX][y][x] == 1:
                        color = FIRE_COLOR
                    if state_space[WATER_INDEX][y][x] == 1:
                        color = WATER_COLOR
                    if state_space[PRESENCE_INDEX][y][x] == 1:
                        color = PRESENCE_COLOR

                    # Draw the square
                    # self.grid_dim = min(self.grid_width, self.grid_height)
                    square_rect = pygame.Rect(
                        start_x + x * (square_dim + 2),
                        start_y + y * (square_dim + 2),
                        square_dim,
                        square_dim,
                    )
                    pygame.draw.rect(screen, color, square_rect)

            # Render and then quit outside
            pygame.display.flip()

            # If we skip, then we basically just render the canvas and then quit outside
            if self.skip:
                timestep = self.fire_env.get_timestep()
                pygame.image.save(screen, IMG_DIRECTORY + str(timestep) + ".png")
                running = False
        pygame.quit()

    def generate_gif(self):
        """
        Save run as a GIF.
        """
        files = [str(i) for i in range(1, self.fire_env.get_timestep() + 1)]
        images = [imageio.imread(IMG_DIRECTORY + f + ".png") for f in files]
        imageio.mimsave("training.gif", images, loop=0)
        shutil.rmtree(IMG_DIRECTORY)