import numpy as np
from evacuation import WildfireEvacuationEnv

def test_evacuation():

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
    for i in range(10):
        possible_actions = evac_env.fire_env.actions
        # pick a random action in this list
        action = np.random.choice(possible_actions)
        evac_env.step(action)
        evac_env.render()
    evac_env.close()

if __name__ == "__main__":
    test_evacuation()