""" Testing the grid_environment.py module. """

import numpy as np
from grid_environment import FireWorld

PRESENCE_INDEX = 5

num_rows = 10
num_cols = 10
cities = np.array([[0,0], [0,9], [9,0], [9,9]])
water_cells = np.array([[5, 5], [6,6]])
road_cells = road_cells = np.array([[0,i] for i in range(10)] + [[9,i] for i in range(10)] + [[i,0] for i in range(1,9)] + [[i,9] for i in range(1,9)])
initial_position = np.array([0,5])
num_fire_cells = 2
custom_fire_locations = None
wind_speed = None
wind_angle = None
fuel_mean = 8.5
fuel_stdev = 3
fire_propagation_rate = 0.094

def test_fireworld():   
    # Create a FireWorld environment
    env = FireWorld(num_rows, 
                    num_cols, 
                    cities, 
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
    
    # Show initial state
    print('Initial Position:')
    initial_state = env.get_state()
    initial_agent_position = np.column_stack(np.where(initial_state[PRESENCE_INDEX] == 1))[0]
    print(initial_agent_position)

    # Check possible actions
    env.update_possible_actions()
    print('Possible actions:')
    print(env.get_actions())

    # Move agent to the right
    next_action = 4
    env.set_action(next_action)
    env.advance_to_next_timestep()

    # Show state after moving right
    print('Final position after moving right:')
    current_state = env.get_state()
    agent_position = np.column_stack(np.where(current_state[PRESENCE_INDEX] == 1))[0]
    print(agent_position)

    rewards = env.get_state_utility()
    print('Rewards:')
    print(rewards)

if __name__ == '__main__':
    test_fireworld()