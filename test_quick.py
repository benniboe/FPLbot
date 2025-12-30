# Schnelltest
from data_loader import load_simple_season_data
from fpl_environment import FPLEnvironment
from baseline_algorithm import GreedyBaseline

# Load data
print("Loading data...")
season_data = load_simple_season_data()
print(f"Loaded {len(season_data)} gameweeks")

# Test environment
print("\nTesting environment...")
env = FPLEnvironment(season_data)
state = env.reset()
print(f"Environment initialized")
print(f"   Squad size: {len(state['squad'])}")
print(f"   Budget remaining: £{state['budget']:.1f}m")

# Test baseline
print("\nTesting baseline agent...")
baseline = GreedyBaseline()
action = baseline.select_action(state)
print(f"Baseline working")
print(f"   Captain: {action['captain']}")
print(f"   Starting 11: {len(action['starting_11'])} players")

# Test one step
print("\nTesting one gameweek...")
state, reward, done, _ = env.step(action)
print(f"Step executed")
print(f"   Points scored: {reward}")
print(f"   Total points: {state['total_points']}")

print("\nAll tests passed!")