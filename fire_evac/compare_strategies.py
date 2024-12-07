import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import defaultdict
import os
import pandas as pd
import time

from algorithms.baseline import solve_MaxImmediateDistance, solve_Random
from algorithms.MCTS import solve_MCTS
from algorithms.qlearning_offline import solve_qlearning_offline, train_qlearning_offline
from algorithms.DQN import solve_dqn, train_dqn_agent
from algorithms.util import standard_initialization

def train_all(grid_size):
    n_timesteps = int(grid_size / 1.25)
    
    # Train DQN
    n_episodes = 1000
    target_update_freq = 10
    train_dqn_agent(n_timesteps, grid_size, n_episodes, target_update_freq, dqn_model_name="DQN_model_grid_"+str(grid_size))

    # Train QLearning
    n_episodes = 1000
    train_qlearning_offline(n_timesteps, grid_size, n_episodes, alpha=0.1, gamma=1.0, epsilon=0.2, q_file_path="Q_table_grid_"+str(grid_size))

def generate_all_maps(grid_size, n_maps):
    n_timesteps = int(grid_size / 1.25)
    for _ in range(n_maps):
        standard_initialization(n_timesteps=n_timesteps, grid_size=grid_size, load=False, map_directory_path=None, save_map=True)
        # Wait 1 second to avoid overwriting the map
        time.sleep(1)

def implement_strategy_all_maps(grid_size, strategy):
    # Define the CSV file path
    csv_file = f"results_{grid_size}.csv"
    
    # Load or create the CSV file
    if os.path.exists(csv_file):
        results_df = pd.read_csv(csv_file, index_col=0)  # Use 'Map' as the index
    else:
        # Predefine columns for strategies
        strategies = ["Random", "MaxImmediateDistance", "MCTS", "QLearning", "DQN"]
        results_df = pd.DataFrame(columns=strategies)
        results_df.index.name = "Map"  # Set the index name
    
    n_timesteps = int(grid_size / 1.25)
    all_maps = os.listdir("pyrorl_map_info")

    # Only keep first 500 maps for now
    all_maps = all_maps[:500]

    for map_name in all_maps:
        # Extract map
        map_directory_path = os.path.join("pyrorl_map_info", map_name)
        load = True

        # Implement strategy
        if strategy == 'Random':
            reward = solve_Random(n_timesteps=n_timesteps, grid_size=grid_size, load=load, map_directory_path=map_directory_path)
        elif strategy == "MaxImmediateDistance":
            reward = solve_MaxImmediateDistance(n_timesteps=n_timesteps, grid_size=grid_size, load=load, map_directory_path=map_directory_path)
        elif strategy == "MCTS":
            reward = solve_MCTS(n_timesteps=n_timesteps, grid_size=grid_size, load=load, map_directory_path=map_directory_path)
        elif strategy == "QLearning":
            q_file_path = f"models/Q_table_grid_{grid_size}.csv"
            reward = solve_qlearning_offline(n_timesteps=n_timesteps, grid_size=grid_size, q_file_path=q_file_path, load=load, map_directory_path=map_directory_path)
        elif strategy == "DQN":
            model_path = f"models/DQN_model_grid_{grid_size}.pth"
            reward = solve_dqn(n_timesteps=n_timesteps, grid_size=grid_size, model_path=model_path, load=load, map_directory_path=map_directory_path)

        # Ensure the map exists as a row
        if map_name not in results_df.index:
            results_df.loc[map_name] = [None] * len(results_df.columns)  # Initialize all strategies with None
        
        # Update the DataFrame with the reward
        results_df.loc[map_name, strategy] = reward
        
    # Save the updated DataFrame back to CSV
    results_df.to_csv(csv_file)

if __name__ == "__main__":

    # We will use grid sizes 20, 40, 60, 80, 100

    # Train models 
    # train_all(grid_size=40) # trained with 1000 episodes

    # Generate maps
    generate_all_maps(grid_size=80, n_maps=500) # only use 500 maps 

    # Implement strategies
    implement_strategy_all_maps(grid_size=80, strategy="Random")
    implement_strategy_all_maps(grid_size=80, strategy="MaxImmediateDistance")
    implement_strategy_all_maps(grid_size=80, strategy="MCTS")
    implement_strategy_all_maps(grid_size=60, strategy="QLearning")
    implement_strategy_all_maps(grid_size=60, strategy="DQN")
