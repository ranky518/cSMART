"""
时空记忆库模块
- 构建用户历史轨迹记忆
- 基于时空特征检索相似轨迹
"""
import numpy as np
import torch
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class SpatioTemporalMemoryBank:
    """
    时空感知的记忆库
    存储结构: {user_id: {(weekday, time_slot): [(start, end, trajectory), ...]}}
    """
    
    def __init__(self, time_slots: int = 6):
        """
        Args:
            time_slots: 时段划分数量 (默认6: 每4小时一个时段)
                - 0: 00:00-04:00 (深夜)
                - 1: 04:00-08:00 (早晨)
                - 2: 08:00-12:00 (上午/早高峰)
                - 3: 12:00-16:00 (下午)
                - 4: 16:00-20:00 (傍晚/晚高峰)
                - 5: 20:00-24:00 (夜间)
        """
        self.time_slots = time_slots
        self.memory = defaultdict(lambda: defaultdict(list))
        self.user_trajectories = defaultdict(list)  # 用于快速访问用户所有轨迹
        
    def get_time_slot(self, time_feat: float) -> int:
        """将归一化时间 [0,1] 转换为时段索引"""
        hour = int(time_feat * 24) % 24
        return hour // (24 // self.time_slots)
    
    def get_weekday(self, time_feat: float, day_offset: int = 0) -> int:
        """从时间特征推断星期几 (简化：使用随机或外部提供)"""
        # 实际应用中需要从原始数据获取
        # 这里简化处理，假设 time_feat 只包含一天内的时间
        return day_offset % 7
    
    def add_trajectory(self, user_id: int, tokens: np.ndarray, 
                       time_feats: np.ndarray, weekday: int = 0):
        """
        添加一条完整轨迹到记忆库
        
        Args:
            user_id: 用户ID
            tokens: 基站token序列 [L]
            time_feats: 时间特征序列 [L] (归一化到 [0,1])
            weekday: 星期几 (0-6)
        """
        if len(tokens) < 3:
            return
            
        # 提取关键信息
        start_station = int(tokens[0])
        end_station = int(tokens[-1])
        
        # 计算主要时段 (取序列中间点的时段)
        mid_idx = len(time_feats) // 2
        time_slot = self.get_time_slot(time_feats[mid_idx])
        
        # 存储
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
        """
        检索相似的历史轨迹
        
        检索策略 (逐步放松):
        1. 精确匹配: 同用户 + 同时段 + 同起点 + 同终点
        2. 放松终点: 同用户 + 同时段 + 同起点
        3. 放松时段: 同用户 + 同起点
        4. 全局回退: 同用户的任意轨迹
        
        Returns:
            List of trajectory dicts, sorted by relevance
        """
        candidates = []
        
        # Level 1: 精确匹配
        key = (weekday, time_slot)
        if user_id in self.memory and key in self.memory[user_id]:
            for entry in self.memory[user_id][key]:
                if entry['start'] == start_station and entry['end'] == end_station:
                    if entry['length'] >= min_length:
                        candidates.append((entry, 1.0))  # 最高分
        
        # Level 2: 放松终点
        if len(candidates) < top_k:
            if user_id in self.memory and key in self.memory[user_id]:
                for entry in self.memory[user_id][key]:
                    if entry['start'] == start_station and entry['length'] >= min_length:
                        if not any(c[0] is entry for c in candidates):
                            candidates.append((entry, 0.7))
        
        # Level 3: 放松时段 (相邻时段)
        if len(candidates) < top_k:
            for ts_offset in [-1, 1]:
                adj_slot = (time_slot + ts_offset) % self.time_slots
                adj_key = (weekday, adj_slot)
                if user_id in self.memory and adj_key in self.memory[user_id]:
                    for entry in self.memory[user_id][adj_key]:
                        if entry['start'] == start_station and entry['length'] >= min_length:
                            if not any(c[0] is entry for c in candidates):
                                candidates.append((entry, 0.5))
        
        # Level 4: 同用户任意轨迹
        if len(candidates) < top_k:
            for entry in self.user_trajectories[user_id]:
                if entry['length'] >= min_length:
                    if not any(c[0] is entry for c in candidates):
                        candidates.append((entry, 0.3))
                if len(candidates) >= top_k * 2:
                    break
        
        # 排序并返回 Top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:top_k]]
    
    def build_from_dataset(self, dataset, weekday_fn=None):
        """
        从数据集构建记忆库
        
        Args:
            dataset: MinuteTrajDataset 实例
            weekday_fn: 可选的函数，用于从索引推断星期几
        """
        print("Building Memory Bank...")
        for idx in range(len(dataset)):
            sample = dataset.rows[idx]
            user_id = int(sample['user_id'])
            tokens = np.array(sample['tokens'])
            time_feats = np.array(sample['time_feat'])
            
            # 简化: 使用索引模7作为星期几 (实际应从原始数据获取)
            weekday = idx % 7 if weekday_fn is None else weekday_fn(idx)
            
            self.add_trajectory(user_id, tokens, time_feats, weekday)
        
        # 统计
        total_entries = sum(
            sum(len(v) for v in user_mem.values()) 
            for user_mem in self.memory.values()
        )
        print(f"Memory Bank built: {len(self.memory)} users, {total_entries} entries")
    
    def __len__(self):
        return sum(len(trajs) for trajs in self.user_trajectories.values())
