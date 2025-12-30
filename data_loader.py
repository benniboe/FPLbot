import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def load_real_season_data(season='2023-24', data_dir='data'):
    season_path = Path(data_dir) / season
    
    if not season_path.exists():
        print(f"Season {season} not found at {season_path}")
        return None
    
    print(f"Loading season {season}...")
    
    season_data = {}
    gws_path = season_path / 'gws'
    
    if not gws_path.exists():
        print(f"Gameweeks folder not found")
        return None
    
    # Gameweeks laden
    loaded_gws = 0
    for gw in range(1, 39):
        gw_file = gws_path / f'gw{gw}.csv'
        
        if gw_file.exists():
            try:
                df = pd.read_csv(gw_file)
                df = process_gameweek_data(df, gw)
                season_data[gw] = df
                loaded_gws += 1
            except Exception as e:
                print(f"Error loading GW{gw}: {e}")
    
    print(f"Loaded {loaded_gws} gameweeks")
    
    return season_data


def process_gameweek_data(df, gw_num):    
    column_mapping = {
        'element': 'id',
        'name': 'player_name',
        'position': 'position',
        'team': 'team',
        'value': 'now_cost',
        'total_points': 'GW_points',
        'minutes': 'minutes',
    }
    
    for old, new in column_mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    # KOsten konvertieren
    if 'now_cost' in df.columns and df['now_cost'].max() > 100:
        df['now_cost'] = df['now_cost'] / 10.0
    
    # form hinzufpgen
    if 'GW_points' in df.columns:
        df['form'] = df['GW_points'].astype(float).clip(0, 20)
    else:
        df['form'] = 5.0
    
    # position numerisch?
    if 'position' in df.columns and df['position'].dtype == 'object':
        position_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['position'] = df['position'].map(position_map)
    
    # Fehlende Werte auffüllen
    df = df.fillna(0)
    
    # columns exist?
    required = ['id', 'position', 'now_cost', 'GW_points', 'form']
    for col in required:
        if col not in df.columns:
            df[col] = 0
    
    return df


if __name__ == "__main__":
    # Test
    season_data = load_real_season_data('2023-24')
    
    if season_data:
        print(f"\nSuccessfully loaded {len(season_data)} gameweeks")
        print(f"\nSample from GW1:")
        print(season_data[1][['id', 'position', 'now_cost', 'GW_points']].head())
    else:
        print("\nFailed to load data")