import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

from .util import standard_initialization, encode_state

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  # Flatten the input tensor if needed
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

        # Update target network initially
        self._update_target_network()

    def _update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, feasible_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(feasible_actions)  # Explore
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor).detach().numpy().flatten()
            return max(feasible_actions, key=lambda a: q_values[a])  # Exploit

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Skip training if not enough samples

        # Sample minibatch from replay buffer
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_agent(n_timesteps=10, grid_size=40, n_episodes=100, target_update_freq=10, dqn_model_name="DQN_model"):
    state_dim = 5 * 25  # Assuming 5 channels and 5x5 local window
    action_dim = 5
    agent = DQNAgent(state_dim, action_dim)

    for episode in tqdm(range(n_episodes)):
        evac_env = standard_initialization(n_timesteps, grid_size, load=False, map_directory_path=None, save_map=False)
        evac_env.fire_env.update_possible_actions()
        
        done = False
        while not done:
            state_tensor = evac_env.fire_env.get_state()
            agent_pos = evac_env.fire_env.get_agent_position()
            state = encode_state(state_tensor, agent_pos)
            feasible_actions = evac_env.fire_env.get_actions()

            # Select and perform action
            action = agent.act(state, feasible_actions)
            evac_env.fire_env.set_action(action)
            evac_env.fire_env.advance_to_next_timestep()

            # Observe reward and next state
            reward = evac_env.fire_env.reward
            next_state_tensor = evac_env.fire_env.get_state()
            next_state = encode_state(next_state_tensor, evac_env.fire_env.get_agent_position())
            done = evac_env.fire_env.get_terminated()

            # Store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)

            # Train the agent
            agent.train()

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent._update_target_network()

        evac_env.close()
    
    print("DQN Training Complete.")

    torch.save(agent.q_network.state_dict(), "models/"+dqn_model_name+".pth")
    print("Model saved successfully.")

    return agent

def load_trained_dqn(state_dim, action_dim, model_path="models/DQN_model.pth"):
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model

def solve_dqn(n_timesteps=10, grid_size=40, model_path="models/DQN_model.pth", load=False, map_directory_path=None, gif_name="DQN"):
    state_dim = 5 * 25  # Assuming 5 channels and 5x5 local window
    action_dim = 5
    model = load_trained_dqn(state_dim, action_dim, model_path)
    
    evac_env = standard_initialization(n_timesteps, grid_size, load, map_directory_path)
    evac_env.fire_env.update_possible_actions()
    #evac_env.render()

    for t in tqdm(range(n_timesteps)):
        # Encode the current state
        state_tensor = evac_env.fire_env.get_state()
        agent_pos = evac_env.fire_env.get_agent_position()
        state = encode_state(state_tensor, agent_pos)

        # Select action using the trained model
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        q_values = model(state_tensor).detach().numpy().flatten()
        feasible_actions = evac_env.fire_env.get_actions()
        best_action = max(feasible_actions, key=lambda a: q_values[a])

        # Perform the action
        evac_env.fire_env.set_action(best_action)
        evac_env.fire_env.advance_to_next_timestep()
        #evac_env.render()
    
    # Evaluate the final reward
    total_reward = evac_env.fire_env.reward
    print("Final reward using the trained DQN: ", total_reward)

    #evac_env.generate_gif(gif_name=gif_name)
    evac_env.close()
    return total_reward
