import pandas as pd
import numpy as np

class FPLEnvironment:
    def __init__(self, season_data, starting_budget=100.0):
        self.season_data = season_data
        self.current_gw = 1
        self.budget = starting_budget
        self.squad = []
        self.transfers_available = 1
        self.total_points = 0
        
    def reset(self):
        """Start new season"""
        self.current_gw = 1
        self.budget = 100.0
        self.squad = self._auto_draft_squad()
        self.transfers_available = 1
        self.total_points = 0
        return self._get_state()
    
    def _auto_draft_squad(self):
        """Draft initial squad"""
        players = self.season_data[1].copy()
        players['value'] = players['form'] / players['now_cost'].clip(lower=0.1)
        
        squad = []
        remaining_budget = self.budget
        
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
                    squad.append(int(player['id']))
                    remaining_budget -= player['now_cost']
                    picked += 1
        
        # Update budget
        squad_df = players[players['id'].isin(squad)]
        self.budget = remaining_budget
        
        return squad
    
    def step(self, action):
        """Execute one gameweek"""
        transfer_cost = 0
        
        # transferieren
        if action['transfer_in'] is not None:
            if self.transfers_available == 0:
                transfer_cost = -4
            self._make_transfer(action['transfer_out'], action['transfer_in'])
            self.transfers_available = max(0, self.transfers_available - 1)
        else:
            self.transfers_available = min(2, self.transfers_available + 1)
        
        # punkte berechnen
        gw_points = self._calculate_gameweek_points(
            action['starting_11'], 
            action['captain']
        )
        
        gw_points += transfer_cost
        self.total_points += gw_points
        
        # nächste gameweek
        self.current_gw += 1
        
        # Check
        done = self.current_gw > 38 or self.current_gw > max(self.season_data.keys())
        
        # stoppen wenn fertig
        if done:
            self.current_gw = min(self.current_gw, 38)
        
        return self._get_state(), gw_points, done, {}
    
    def _calculate_gameweek_points(self, starting_11, captain_id):
        if self.current_gw not in self.season_data:
            return 0
            
        gw_data = self.season_data[self.current_gw]
        points = 0
        
        for player_id in starting_11:
            player = gw_data[gw_data['id'] == player_id]
            if len(player) > 0:
                player_points = float(player['GW_points'].values[0])
                
                if player_id == captain_id:
                    points += player_points * 2
                else:
                    points += player_points
        
        return points
    
    def _get_state(self):
        if self.current_gw > 38:
            self.current_gw = 38
        
        if self.current_gw not in self.season_data:
            available_gws = sorted(self.season_data.keys())
            self.current_gw = available_gws[-1] if available_gws else 1
        
        gw_data = self.season_data[self.current_gw]
        
        return {
            'gameweek': self.current_gw,
            'squad': self.squad,
            'budget': self.budget,
            'transfers_available': self.transfers_available,
            'player_features': gw_data,
            'total_points': self.total_points
        }
    
    def _make_transfer(self, out_id, in_id):
        # Transfers machen
        if self.current_gw not in self.season_data:
            return
            
        gw_data = self.season_data[self.current_gw]
        
        out_player = gw_data[gw_data['id'] == out_id]
        in_player = gw_data[gw_data['id'] == in_id]
        
        if len(out_player) > 0 and len(in_player) > 0:
            self.budget += float(out_player['now_cost'].values[0])
            self.budget -= float(in_player['now_cost'].values[0])
            
            if out_id in self.squad:
                self.squad.remove(out_id)
            self.squad.append(int(in_id))