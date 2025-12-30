#optimiertes DQN_Training für FPL


import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loader import load_real_season_data
from fpl_environment import FPLEnvironment
from baseline_algorithm import GreedyBaseline
from rl_agent_dql import SimplifiedFPLAgent

def train_dqn_optimized(season_data, num_episodes=250):
    
    print(f"Training DQN Agent for {num_episodes} episodes\n")
    
    env = FPLEnvironment(season_data)
    agent = SimplifiedFPLAgent(state_dim=20, action_dim=100)
    
    # aggressives learning
    agent.optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=0.0001) # Höhere Lernrate
    agent.epsilon_decay = 0.995  # Schnellere Abnahme der Exploration
    
    episode_rewards = []
    episode_losses = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        
        for gw in range(1, 39):
            state_vector = agent.get_state_vector(state)
            
            # exploration boost
            if episode < 20:
                action_idx = 0 if np.random.rand() < 0.3 else 1 
            else:
                action_idx = 0 if np.random.rand() < agent.epsilon else 1
            
            if action_idx == 0:
                fpl_action = {
                    'transfer_in': None,
                    'transfer_out': None,
                    'captain': _pick_captain(state),
                    'vice_captain': _pick_vice_captain(state),
                    'starting_11': _pick_starting_11(state),
                    'chip': None
                }
            else:
                baseline = GreedyBaseline()
                fpl_action = baseline.select_action(state)
            
            next_state, reward, done, _ = env.step(fpl_action)
            
            # Normalisierung
            next_state_vector = agent.get_state_vector(next_state)
            normalized_reward = reward / 50.0
            agent.memory.append((state_vector, action_idx, normalized_reward, next_state_vector, done))
            
            if len(agent.memory) >= 32:
                loss = agent.train_step(batch_size=32)
                if loss is not None:
                    episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        if episode % 5 == 0:
            print(f"Episode {episode:3d}: {total_reward:4.0f} pts | Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}")
    
    return agent, episode_rewards, episode_losses


def _pick_captain(state):
    squad = state['squad']
    players = state['player_features']
    squad_df = players[players['id'].isin(squad)]
    if len(squad_df) > 0:
        return int(squad_df.sort_values('form', ascending=False).iloc[0]['id'])
    return squad[0] if squad else 1


def _pick_vice_captain(state):
    squad = state['squad']
    players = state['player_features']
    squad_df = players[players['id'].isin(squad)]
    if len(squad_df) > 1:
        return int(squad_df.sort_values('form', ascending=False).iloc[1]['id'])
    return squad[1] if len(squad) > 1 else squad[0]


def _pick_starting_11(state):
    squad = state['squad']
    players = state['player_features']
    squad_df = players[players['id'].isin(squad)]
    
    starting = []
    formation = {1: 1, 2: 3, 3: 4, 4: 3}
    
    for position, count in formation.items():
        pos_players = squad_df[squad_df['position'] == position]
        pos_players = pos_players.sort_values('form', ascending=False)
        for i in range(min(count, len(pos_players))):
            starting.append(int(pos_players.iloc[i]['id']))
    
    return starting


def plot_comparison(rewards_stable, rewards_optimized):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    episodes = range(len(rewards_optimized))
    
    # Plot both
    ax.plot(episodes, rewards_optimized, alpha=0.4, color='blue', label='Optimized (Raw)')
    
    # Moving averages
    window = 10
    if len(rewards_optimized) >= window:
        ma_opt = np.convolve(rewards_optimized, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards_optimized)), ma_opt, 
               linewidth=3, color='darkblue', label='Optimized (10-EP MA)')
    
    # Benchmarks
    ax.axhline(y=1769, color='red', linestyle='--', linewidth=2, label='Baseline (1769)', alpha=0.7)
    ax.axhline(y=2000, color='orange', linestyle='--', linewidth=2, label='Average FPL (2000)', alpha=0.7)
    ax.axhline(y=2200, color='green', linestyle='--', linewidth=2, label='Top 100k (2200)', alpha=0.7)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Points', fontsize=12)
    ax.set_title('Optimized DQN Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_optimized.png', dpi=200, bbox_inches='tight')
    print(f"\Saved: training_optimized.png")
    plt.show()


def print_summary(episode_rewards):
    best = max(episode_rewards)
    avg = np.mean(episode_rewards)
    recent_avg = np.mean(episode_rewards[-10:])
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Best Score:          {best:4.0f} points")
    print(f"Average Score:       {avg:4.0f} points")
    print(f"Last 10 Avg:         {recent_avg:4.0f} points")
    print(f"\nBaseline:            1769 points")
    print(f"Improvement:         {recent_avg - 1769:+4.0f} points")


def main():
    print("DQN Training for FPL\n")
    
    season_data = load_real_season_data('2023-24')
    
    if not season_data:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(season_data)} gameweeks\n")
    
    # Train
    agent, rewards, losses = train_dqn_optimized(season_data, num_episodes=250)
    
    # Results
    print_summary(rewards)
    
    # Plot
    plot_comparison([], rewards)
    
    # Save
    torch.save(agent.q_network.state_dict(), 'dqn_model_optimized.pth')
    print(f"\nModel saved: dqn_model_optimized.pth")
    
    print("\nTraining Complete!")


if __name__ == "__main__":
    main()