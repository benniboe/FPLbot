import numpy as np
import pandas as pd

class GreedyBaseline:
    """
    Simple baseline: pick best players by form, make sensible transfers
    """
    
    def __init__(self):
        self.name = "Greedy Baseline"
    
    def draft_squad(self, players_df, budget=100.0):
        """Draft initial squad - best value players"""
        players = players_df.copy()
        players['value'] = players['form'] / players['now_cost']
        
        squad = []
        remaining_budget = budget
        
        # Positionen: 2 GK, 5 DEF, 5 MID, 3 FWD
        requirements = {1: 2, 2: 5, 3: 5, 4: 3}
        
        for position, count in requirements.items():
            pos_players = players[players['position'] == position].copy()
            pos_players = pos_players.sort_values('value', ascending=False)
            
            picked = 0
            for _, player in pos_players.iterrows():
                if picked >= count:
                    break
                    
                if player['now_cost'] <= remaining_budget:
                    squad.append(player['id'])
                    remaining_budget -= player['now_cost']
                    picked += 1
        
        return squad
    
    def select_action(self, state):
        """Select action for this gameweek"""
        squad = state['squad']
        players_df = state['player_features']
        budget = state['budget']
        
        squad_df = players_df[players_df['id'].isin(squad)]
        
        # Wählt den Spieler mit der höchsten Form als Captain aus
        captain_id = squad_df.sort_values('form', ascending=False).iloc[0]['id']
        
        # Wähle Start 11
        starting_11 = self._pick_starting_11(squad_df)
        
        # transfer ja/nein
        transfer_in, transfer_out = self._find_best_transfer(state)
        
        return {
            'transfer_in': transfer_in,
            'transfer_out': transfer_out,
            'captain': captain_id,
            'vice_captain': squad_df.sort_values('form', ascending=False).iloc[1]['id'],
            'starting_11': starting_11,
            'chip': None
        }
    
    # Wählt die besten 11 Spieler in einer gültigen Formation (z.B. 3-4-3) aus
    def _pick_starting_11(self, squad_df):
        """Pick best 11 players in valid formation (e.g., 3-4-3)"""
        starting = []
        
        # Formation: 1 GK, 3 DEF, 4 MID, 3 FWD
        formation = {1: 1, 2: 3, 3: 4, 4: 3}
        
        for position, count in formation.items():
            pos_players = squad_df[squad_df['position'] == position]
            pos_players = pos_players.sort_values('form', ascending=False)
            
            for i in range(min(count, len(pos_players))):
                starting.append(pos_players.iloc[i]['id'])
        
        return starting
    
    # Findet den besten Transfer (falls vorhanden)
    def _find_best_transfer(self, state):
        """Find best transfer (if any)"""
        squad = state['squad']
        players_df = state['player_features']
        budget = state['budget']
        
        squad_df = players_df[players_df['id'].isin(squad)]
        
        # Schlechtesten Spieler im Team finden
        worst_player = squad_df.sort_values('form').iloc[0]
        
        # Beste finanzierbare Spieler finden
        available = players_df[~players_df['id'].isin(squad)]
        available = available[available['position'] == worst_player['position']]
        available = available[available['now_cost'] <= budget + worst_player['now_cost']]
        
        if len(available) > 0:
            best_available = available.sort_values('form', ascending=False).iloc[0]
            
            # Transfer nur durchführen, wenn der neue Spieler deutlich besser ist
            if best_available['form'] > worst_player['form'] + 2.0:
                return best_available['id'], worst_player['id']
        
        return None, None