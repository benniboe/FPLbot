import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class SimplifiedFPLAgent:
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.0001)
        
        # Experience replay memory
        self.memory = deque(maxlen=10000)
        
        # Exploration parameter
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
    
    def get_state_vector(self, state):
        squad_ids = state['squad']
        player_data = state['player_features']
        squad_data = player_data[player_data['id'].isin(squad_ids)]
        
        if len(squad_data) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        features = [
            state['gameweek'] / 38.0,
            state['budget'] / 100.0,
            state['transfers_available'] / 2.0,
            
            squad_data['form'].astype(float).mean() / 10.0,
            squad_data['now_cost'].astype(float).mean() / 10.0,
            squad_data['GW_points'].astype(float).mean() / 10.0,
            squad_data['minutes'].astype(float).mean() / 90.0,
            
            len(squad_data[squad_data['position'] == 1]) / 15.0,
            len(squad_data[squad_data['position'] == 2]) / 15.0,
            len(squad_data[squad_data['position'] == 3]) / 15.0,
            len(squad_data[squad_data['position'] == 4]) / 15.0,
            
            squad_data['form'].astype(float).max() / 10.0,
            squad_data['form'].astype(float).min() / 10.0,
            squad_data['now_cost'].astype(float).max() / 15.0,
            squad_data['now_cost'].astype(float).min() / 15.0,
            
            squad_data['form'].astype(float).std() / 10.0,
            squad_data['now_cost'].astype(float).std() / 10.0,
            
            state['total_points'] / 3000.0,
            
            0.0,
            0.0,
        ]
        
        features = features[:self.state_dim]
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def select_action(self, state, valid_actions):
        # Epsilon-greedy action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        state_vector = self.get_state_vector(state)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        valid_mask = torch.zeros(self.action_dim)
        for action_idx in valid_actions:
            valid_mask[action_idx] = 1
        
        q_values = q_values * valid_mask - 1e9 * (1 - valid_mask)
        
        action_idx = q_values.argmax().item()
        return action_idx
    
    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        
        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.int64)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        # Q-network kopieren nach target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        #Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)