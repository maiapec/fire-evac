import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

'''
Possible states
'''

VEGETATION_INDEX = 0
WATER_INDEX = 1
CITY_INDEX = 2
ROAD_INDEX = 3
FIRE_INDEX = 4

# Simulate a grid world with fire propagation parameters

class GridWorld:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))

    def set_fire(self, x, y):
        self.grid[x, y] = FIRE_INDEX

    def set_vegetation(self, x, y):
        self.grid[x, y] = VEGETATION_INDEX

    def set_water(self, x, y):
        self.grid[x, y] = WATER_INDEX

    def set_city(self, x, y):
        self.grid[x, y] = CITY_INDEX

    def set_road(self, x, y):
        self.grid[x, y] = ROAD_INDEX

    # Generate a map with n_cities, n_waters, and a connected road network
    def generate_map(self, n_cities, n_waters, n_roads):
        # Generate city locations and store their coordinates
        city_coords = []
        for i in range(n_cities):
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            while self.grid[x, y] != VEGETATION_INDEX:  # Avoid overlaps
                x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            self.set_city(x, y)
            city_coords.append((x, y))

        # Generate water bodies
        for i in range(n_waters):
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            while self.grid[x, y] != VEGETATION_INDEX:  # Avoid overlaps
                x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            self.set_water(x, y)

        # Connect cities with roads
        self.connect_cities(city_coords, n_roads)

    # Helper function to create a connected road network between cities
    def connect_cities(self, city_coords, n_roads):
        # Use a greedy approach to connect each city to its nearest neighbor
        for i, start_city in enumerate(city_coords):
            if i < len(city_coords) - 1:
                distances = [distance.cityblock(start_city, city) for city in city_coords[i+1:]]
                nearest_city = city_coords[i + 1 + np.argmin(distances)]
                self.create_road_path(start_city, nearest_city)
        
        # Optionally add extra roads for a denser network
        extra_roads = max(0, n_roads - len(city_coords) + 1)
        for _ in range(extra_roads):
            idx1, idx2 = np.random.choice(len(city_coords), 2, replace=False)
            city1, city2 = city_coords[idx1], city_coords[idx2]
            self.create_road_path(city1, city2)

    # Create a simple straight road between two points (Manhattan path)
    def create_road_path(self, start, end):
        x1, y1 = start
        x2, y2 = end
        x, y = x1, y1

        # Move horizontally and then vertically
        while (x, y) != (x2, y2):
            if x != x2:
                x += 1 if x < x2 else -1
            elif y != y2:
                y += 1 if y < y2 else -1

            # Ensure roads avoid cities and water
            if self.grid[x, y] in [VEGETATION_INDEX, FIRE_INDEX]:
                self.set_road(x, y)

    # Create a visualization of the grid world

    def visualize(self):
        plt.imshow(self.grid, cmap='viridis')
        plt.show()

if __name__ == "__main__":
    world = GridWorld(40)
    world.generate_map(n_cities=3, n_waters=10, n_roads=20)
    world.visualize()