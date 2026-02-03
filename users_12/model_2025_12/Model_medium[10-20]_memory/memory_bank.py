"""
时空记忆库模块 - 3-10min 补全版本
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict


class SpatioTemporalMemoryBank:
    def __init__(self, time_slots: int = 6):
        self.time_slots = time_slots
        self.memory = defaultdict(lambda: defaultdict(list))
        self.user_trajectories = defaultdict(list)
        
    def get_time_slot(self, time_feat: float) -> int:
        hour = int(time_feat * 24) % 24
        return hour // (24 // self.time_slots)
    
    def add_trajectory(self, user_id: int, tokens: np.ndarray, 
                       time_feats: np.ndarray, weekday: int = 0):
        if len(tokens) < 3:
            return
        start_station = int(tokens[0])
        end_station = int(tokens[-1])
        mid_idx = len(time_feats) // 2
        time_slot = self.get_time_slot(time_feats[mid_idx])
        
        key = (weekday, time_slot)
        entry = {
            'start': start_station,
            'end': end_station,
            'tokens': tokens.copy(),
            'time_feats': time_feats.copy(),
            'length': len(tokens)
        }
        self.memory[user_id][key].append(entry)
        self.user_trajectories[user_id].append(entry)
    
    def retrieve(self, user_id: int, weekday: int, time_slot: int,
                 start_station: int, end_station: int, 
                 top_k: int = 3, min_length: int = 5) -> List[Dict]:
        candidates = []
        
        key = (weekday, time_slot)
        if user_id in self.memory and key in self.memory[user_id]:
            for entry in self.memory[user_id][key]:
                if entry['start'] == start_station and entry['end'] == end_station:
                    if entry['length'] >= min_length:
                        candidates.append((entry, 1.0))
        
        if len(candidates) < top_k:
            if user_id in self.memory and key in self.memory[user_id]:
                for entry in self.memory[user_id][key]:
                    if entry['start'] == start_station and entry['length'] >= min_length:
                        if not any(c[0] is entry for c in candidates):
                            candidates.append((entry, 0.7))
        
        if len(candidates) < top_k:
            for ts_offset in [-1, 1]:
                adj_slot = (time_slot + ts_offset) % self.time_slots
                adj_key = (weekday, adj_slot)
                if user_id in self.memory and adj_key in self.memory[user_id]:
                    for entry in self.memory[user_id][adj_key]:
                        if entry['start'] == start_station and entry['length'] >= min_length:
                            if not any(c[0] is entry for c in candidates):
                                candidates.append((entry, 0.5))
        
        if len(candidates) < top_k:
            for entry in self.user_trajectories[user_id]:
                if entry['length'] >= min_length:
                    if not any(c[0] is entry for c in candidates):
                        candidates.append((entry, 0.3))
                if len(candidates) >= top_k * 2:
                    break
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:top_k]]
    
    def build_from_dataset(self, dataset, weekday_fn=None):
        print("Building Memory Bank...")
        for idx in range(len(dataset.rows)):
            sample = dataset.rows[idx]
            user_id = int(sample['user_id'])
            tokens = np.array(sample['tokens'])
            time_feats = np.array(sample['time_feat'])
            weekday = idx % 7 if weekday_fn is None else weekday_fn(idx)
            self.add_trajectory(user_id, tokens, time_feats, weekday)
        
        total_entries = sum(
            sum(len(v) for v in user_mem.values()) 
            for user_mem in self.memory.values()
        )
        print(f"Memory Bank built: {len(self.memory)} users, {total_entries} entries")
