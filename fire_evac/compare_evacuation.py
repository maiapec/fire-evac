""" 
Ideas for validating and comaaring evacuation models. 

1. Basic models to compare to:
        - random evacuation
        - greedy evacuation
        - parametrized rule-based evacuation (to define)

2. Experiments to validate behaviors of models:
        - In a very simple grid, with a single fire cell, the agent should move away from the fire cell.

3. Metrics to compare models:
        - Average final reward over multiple runs (for many maps)
        - Time to choose next action (-> to make it realistic)
        - ?

Note: the immediate max distance strategy can get stuck!


Other things:
TODO: Modify map generation to make sure evacuation is possible
TODO: When comparing strategies: should be compared on the same maps --> e.g. generate 1000 maps and calculate average reward for each strategy
TODO: Store the initial fire cells in the map and load it to make sure we compare same scenarios
"""