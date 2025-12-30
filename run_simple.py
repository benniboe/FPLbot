
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_real_season_data
from fpl_environment import FPLEnvironment
from baseline_algorithm import GreedyBaseline

def run_agent(agent, season_data, agent_name="Agent"):
    env = FPLEnvironment(season_data)
    state = env.reset()
    
    gw_points = []
    total_points = 0
    
    print(f"\n{'='*50}")
    print(f"Running {agent_name}")
    print(f"{'='*50}")
    
    for gw in range(1, len(season_data) + 1):
        try:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            gw_points.append(reward)
            total_points += reward
            
            if gw % 5 == 0 or gw == 1:
                print(f"GW {gw:2d}: {reward:3.0f} pts | Total: {total_points:4.0f} pts")
            
            if done:
                break
                
        except Exception as e:
            print(f"Error at GW{gw}: {e}")
            break
    
    print(f"\n{agent_name} Final Score: {total_points:.0f} points")
    print(f"Average per GW: {np.mean(gw_points):.1f} points")
    
    return gw_points, total_points


def visualize_results(results_dict, season_name=""):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gameweek punkte
    for agent_name, (gw_points, _) in results_dict.items():
        gameweeks = range(1, len(gw_points) + 1)
        axes[0].plot(gameweeks, gw_points, marker='o', label=agent_name, linewidth=2)
    
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Gameweek')
    axes[0].set_ylabel('Points')
    axes[0].set_title(f'Gameweek Points - {season_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Punkte kumuliert
    for agent_name, (gw_points, _) in results_dict.items():
        cumulative = np.cumsum(gw_points)
        gameweeks = range(1, len(gw_points) + 1)
        axes[1].plot(gameweeks, cumulative, marker='o', label=agent_name, linewidth=2.5)
    
    axes[1].axhline(y=2000, color='orange', linestyle='--', label='Average FPL', alpha=0.6)
    axes[1].axhline(y=2200, color='red', linestyle='--', label='Top 100k', alpha=0.6)
    
    axes[1].set_xlabel('Gameweek')
    axes[1].set_ylabel('Cumulative Points')
    axes[1].set_title('Season Progress')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'fpl_results_{season_name}.png'
    plt.savefig(filename, dpi=200)
    print(f"\nSaved: {filename}")
    plt.show()


def main():
    print("FPL RL Agent\n")
    
    # Load data
    season = '2023-24'
    print(f"Loading season {season}...")
    
    season_data = load_real_season_data(season)
    
    if not season_data:
        print("Failed to load data. Run 'python download_fpl_data.py' first!")
        return
    
    print(f"Loaded {len(season_data)} gameweeks\n")
    
    # Run baseline
    baseline = GreedyBaseline()
    results = {}
    
    gw_points, total_points = run_agent(baseline, season_data, "Greedy Baseline")
    results["Greedy Baseline"] = (gw_points, total_points)
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Greedy Baseline: {total_points:.0f} points")
    print(f"\nFPL Benchmarks:")
    print(f"  Average: ~2000 points")
    print(f"  Top 100k: ~2200 points")
    
    # Visualize
    visualize_results(results, season)
    
    print("\nDone!")


if __name__ == "__main__":
    main()