import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class MinuteTrajDataset(Dataset):
    def __init__(self, pkl_path, mask_ratio=0.15, L=96, vocab_size=0, 
                 span_mask=False, span_len_min=2, span_len_max=4,
                 sample_step=5, preprocess_mode="ZOH"): 
        """
        vocab_size: 基础词表大小 (0 ~ vocab_size-1 为有效Token)
        """
        self.data = pd.read_pickle(pkl_path)
        self.rows = self.data['sequences']
        self.L = L
        self.mask_ratio = mask_ratio
        
        # [Fix] 分离 Mask 和 Pad 
        # Token IDs: [0...vocab_size-1] -> Cluster Tokens
        #            [vocab_size]       -> MASK Token (Attended)
        #            [vocab_size + 1]   -> PAD Token  (Ignored)
        self.vocab_size = vocab_size 
        self.mask_token = vocab_size 
        self.pad_token  = vocab_size + 1 
        self.empty_token = vocab_size  # For NO_FILL mode

        self.preprocess_mode = preprocess_mode
        self.span_mask = span_mask
        self.span_len_min = span_len_min
        self.span_len_max = span_len_max
        self.sample_step = sample_step
        
        # [Variant B] Linear: 需要聚类中心坐标
        if 'cluster_centers' in self.data:
            self.cluster_centers = np.array(self.data['cluster_centers']) 
        else:
            self.cluster_centers = None
            if preprocess_mode == 'LINEAR':
                print("Warning: 'cluster_centers' not found. LINEAR mode implies ZOH fallback.")

    def __len__(self):
        return len(self.rows)

    def _apply_linear_interpolation(self, tokens_dense):
        """物理坐标线性插值逻辑"""
        if self.cluster_centers is None:
            return tokens_dense

        # 1. 找到值变化的时刻 (Transition Points)
        is_transition = np.concatenate(([True], tokens_dense[1:] != tokens_dense[:-1]))
        change_indices = np.where(is_transition)[0]
        
        if len(change_indices) < 2:
            return tokens_dense

        tokens_interp = tokens_dense.copy()
        
        # 2. 逐段插值
        for k in range(len(change_indices) - 1):
            t_start = change_indices[k]
            t_end = change_indices[k+1]
            if t_end - t_start <= 1: continue
                
            token_start = tokens_dense[t_start]
            token_end = tokens_dense[t_end]
            coord_start = self.cluster_centers[token_start]
            coord_end = self.cluster_centers[token_end]
            
            steps = t_end - t_start
            for step in range(1, steps):
                t_curr = t_start + step
                # 线性混合
                alpha = step / float(steps)
                coord_curr = (1 - alpha) * coord_start + alpha * coord_end
                # 找最近的 Cluster ID
                dists = np.linalg.norm(self.cluster_centers - coord_curr, axis=1)
                tokens_interp[t_curr] = np.argmin(dists)
                
        return tokens_interp

    def __getitem__(self, idx):
        row = self.rows[idx]
        # 1. 获取原始 Dense Token (ZOH形式)
        tokens_dense = np.array(row['tokens'], dtype=np.int64)
        
        # 应用下采样 (如果需要)
        if self.sample_step > 1:
            tokens_dense = tokens_dense[::self.sample_step]
        
        # 2. 根据模式进行处理
        if self.preprocess_mode == 'NO_FILL':
            # 只保留跳变点，其他位置设为 mask_token (empty)
            tokens_processed = np.full_like(tokens_dense, self.empty_token)
            transitions = np.concatenate(([True], tokens_dense[1:] != tokens_dense[:-1]))
            tokens_processed[transitions] = tokens_dense[transitions]
            tokens = tokens_processed[:self.L]
            
        elif self.preprocess_mode == 'LINEAR':
            # 物理插值
            tokens_interp = self._apply_linear_interpolation(tokens_dense)
            tokens = tokens_interp[:self.L]
            
        else: 
            # ZOH (默认)
            tokens = tokens_dense[:self.L]
        
        # 3. 处理时间特征 (需同步截断)
        time_feat = np.array(row['time_feat'], dtype=np.float32)
        if self.sample_step > 1:
            time_feat = time_feat[::self.sample_step]
        time_feat = time_feat[:self.L]

        # [Fix] 使用 pad_token 填充
        length = len(tokens)
        if length < self.L:
            pad = self.L - length
            tokens = np.pad(tokens, (0, pad), constant_values=self.pad_token)
            # time_feat shape is [L, 4], pad first dim
            time_feat = np.pad(time_feat, ((0, pad), (0, 0)), constant_values=0.0)
        
        # 5. Ground Truth 准备 (始终使用 Dense ZOH 供评估)
        gt_tokens = tokens_dense[:self.L]
        if len(gt_tokens) < self.L:
            gt_tokens = np.pad(gt_tokens, (0, self.L - len(gt_tokens)), constant_values=self.pad_token)

        # 6. 生成 Mask (用于训练/评估的遮盖任务)
        mask = np.zeros(self.L, dtype=bool)
        # 仅对非 Pad 区域进行 Mask
        valid_indices = np.where((gt_tokens != self.pad_token) & (gt_tokens != self.mask_token))[0]
        
        if len(valid_indices) > 0:
            if self.span_mask:
                 span_len = np.random.randint(self.span_len_min, self.span_len_max + 1)
                 valid_len = len(valid_indices)
                 if valid_len > span_len:
                      start_idx_rel = np.random.randint(0, valid_len - span_len)
                      chosen_indices = valid_indices[start_idx_rel : start_idx_rel + span_len]
                      mask[chosen_indices] = True
                 else:
                      mask[valid_indices] = True
            else:
                 mask_indices = np.random.choice(valid_indices, 
                                               size=int(len(valid_indices) * self.mask_ratio), 
                                               replace=False)
                 mask[mask_indices] = True

        # 7. 构建输入
        tokens_input = tokens.copy()
        tokens_input[mask] = self.mask_token # 强制遮盖 Mask 区域

        return {
            'token_id': torch.from_numpy(tokens_input),
            'token_true': torch.from_numpy(gt_tokens),  # 评估对照
            'mask': torch.from_numpy(mask),
            'time_feat': torch.from_numpy(time_feat), 
            'user_id': int(row['user_id'])
        }
