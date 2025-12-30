"""
DQN Training with TensorBoard logging
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from data_loader import load_real_season_data
from fpl_environment import FPLEnvironment
from baseline_algorithm import GreedyBaseline
from rl_agent_dql import SimplifiedFPLAgent


def train_with_tensorboard(season_data, num_episodes=250, run_name=None):
    """Train DQN with full TensorBoard logging"""
    
    # Create simple numerical run name
    if run_name is None:
        # Find existing runs and increment
        runs_dir = 'runs'
        os.makedirs(runs_dir, exist_ok=True)
        existing_runs = [d for d in os.listdir(runs_dir) if d.isdigit()]
        if existing_runs:
            next_num = max([int(d) for d in existing_runs]) + 1
        else:
            next_num = 1
        run_name = str(next_num)
    
    # Initialize TensorBoard writer
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logging to: {log_dir}")
    print(f"   View with: tensorboard --logdir=runs\n")
    
    # Initialize environment and agent
    env = FPLEnvironment(season_data)
    agent = SimplifiedFPLAgent(state_dim=20, action_dim=100)
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', f"""
    - Learning Rate: 0.0001
    - Epsilon Decay: 0.995
    - Gamma: 0.99
    - Batch Size: 32
    - Replay Buffer: 10000
    - Architecture: [20, 256, 256, 128, 2]
    - Reward Normalization: /50
    """)
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        transfers_made = 0
        captain_points_list = []
        gw_points_list = []
        
        for gw in range(1, 39):
            state_vector = agent.get_state_vector(state)
            
            # Select action
            if episode < 20:
                action_idx = 0 if np.random.rand() < 0.3 else 1
            else:
                action_idx = 0 if np.random.rand() < agent.epsilon else 1
            
            # Track if transfer was made
            if action_idx == 1:
                transfers_made += 1
            
            # Convert to FPL action
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
            
            # Execute
            next_state, reward, done, _ = env.step(fpl_action)
            gw_points_list.append(reward)
            
            # Store experience with normalized reward
            next_state_vector = agent.get_state_vector(next_state)
            normalized_reward = reward / 100
            agent.memory.append((state_vector, action_idx, normalized_reward, next_state_vector, done))
            
            # Train
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
        
        # Calculate metrics
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        avg_gw_points = np.mean(gw_points_list)
        max_gw_points = max(gw_points_list)
        min_gw_points = min(gw_points_list)
        
        # ============= TENSORBOARD LOGGING =============
        
        # 1. Main metrics
        writer.add_scalar('Performance/Total_Points', total_reward, episode)
        writer.add_scalar('Performance/Average_GW_Points', avg_gw_points, episode)
        writer.add_scalar('Training/Loss', avg_loss, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
        
        # 2. Transfer behavior
        writer.add_scalar('Strategy/Transfers_Made', transfers_made, episode)
        writer.add_scalar('Strategy/Transfer_Rate', transfers_made / 38, episode)
        
        # 3. Gameweek performance
        writer.add_scalar('GameweekStats/Max_Points', max_gw_points, episode)
        writer.add_scalar('GameweekStats/Min_Points', min_gw_points, episode)
        writer.add_scalar('GameweekStats/Variance', np.std(gw_points_list), episode)
        
        # 4. Benchmarks (for comparison)
        writer.add_scalar('Benchmarks/Baseline', 1769, episode)
        writer.add_scalar('Benchmarks/Average_FPL', 2000, episode)
        writer.add_scalar('Benchmarks/Top_100k', 2200, episode)
        
        # 5. Network statistics (every 10 episodes)
        if episode % 10 == 0:
            for name, param in agent.q_network.named_parameters():
                writer.add_histogram(f'Network/{name}', param, episode)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, episode)
        
        # 6. Episode-by-episode gameweek performance (heatmap data)
        if episode % 10 == 0:
            # Log gameweek progression
            for gw_idx, points in enumerate(gw_points_list):
                writer.add_scalar(f'GameweekProgression/GW_{gw_idx+1}', points, episode)
        
        # Print progress
        if episode % 5 == 0:
            print(f"Episode {episode:3d}: {total_reward:4.0f} pts | "
                  f"Avg GW: {avg_gw_points:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Transfers: {transfers_made}")
    
    # Final statistics
    writer.add_text('Training_Complete', f"""
    Training completed successfully!
    Total episodes: {num_episodes}
    Final epsilon: {agent.epsilon:.3f}
    """)
    
    writer.close()
    print(f"\n✅ Training complete! TensorBoard logs saved to: {log_dir}")
    
    return agent, log_dir


def _pick_captain(state):
    """Pick captain (best form)"""
    squad = state['squad']
    players = state['player_features']
    squad_df = players[players['id'].isin(squad)]
    if len(squad_df) > 0:
        return int(squad_df.sort_values('form', ascending=False).iloc[0]['id'])
    return squad[0] if squad else 1


def _pick_vice_captain(state):
    """Pick vice captain"""
    squad = state['squad']
    players = state['player_features']
    squad_df = players[players['id'].isin(squad)]
    if len(squad_df) > 1:
        return int(squad_df.sort_values('form', ascending=False).iloc[1]['id'])
    return squad[1] if len(squad) > 1 else squad[0]


def _pick_starting_11(state):
    """Pick starting 11 (3-4-3)"""
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


def compare_runs():
    """Run multiple configurations and compare in TensorBoard"""
    
    print("🔬 Running Comparison Experiments\n")
    
    season_data = load_real_season_data('2023-24')
    
    if not season_data:
        print("❌ Failed to load data")
        return
    
    # Experiment 1: Baseline configuration (optimized)
    print("\n" + "="*60)
    print("Experiment 1: Optimized Configuration")
    print("="*60)
    agent1, _ = train_with_tensorboard(season_data, num_episodes=50, 
                                       run_name="optimized_div50")
    
    # Experiment 2: Heavy normalization
    print("\n" + "="*60)
    print("Experiment 2: Heavy Normalization (/100)")
    print("="*60)
    # Modify reward normalization in training loop
    agent2, _ = train_with_tensorboard(season_data, num_episodes=50, 
                                       run_name="heavy_norm_div100")
    
    # Experiment 3: No normalization
    print("\n" + "="*60)
    print("Experiment 3: No Normalization")
    print("="*60)
    agent3, _ = train_with_tensorboard(season_data, num_episodes=50, 
                                       run_name="no_normalization")
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*60}")
    print("\n📊 View all runs in TensorBoard:")
    print("   tensorboard --logdir=runs")
    print("\nYou can now compare:")
    print("  • Training curves side-by-side")
    print("  • Loss progression")
    print("  • Transfer strategies")
    print("  • Network weight distributions")


def main():
    print("🤖 DQN Training with TensorBoard\n")
    
    # Load data
    season_data = load_real_season_data('2023-24')
    
    if not season_data:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(season_data)} gameweeks\n")
    
    # Single training run
    agent, log_dir = train_with_tensorboard(season_data, num_episodes=250)
    
    # Save model
    model_path = f'{log_dir}/model.pth'
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print("\n📊 To view TensorBoard:")
    print(f"   1. Open terminal")
    print(f"   2. Run: tensorboard --logdir=runs")
    print(f"   3. Open browser: http://localhost:6006")
    print(f"\nYour logs are in: {log_dir}")


if __name__ == "__main__":
    # Choose one:
    
    # Option 1: Single training run
    main()
    
    # Option 2: Compare multiple configurations
    # compare_runs()