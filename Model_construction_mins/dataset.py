import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class MinuteTrajDataset(Dataset):
    def __init__(self, pkl_path, mask_ratio=0.15, L=96, vocab_size=2000,
                 span_mask=False, span_len_min=1, span_len_max=1, span_count=1):  # 1. 新增 sample_step 参数
        data = pickle.load(open(pkl_path, 'rb'))          # dict
        self.rows = data['sequences']                     # list of dict
        self.mask_ratio = mask_ratio
        self.L = L
        self.base_vocab = vocab_size
        self.mask_id = vocab_size                         # mask 索引 = 原簇数
        # 新增：连续片段掩码配置
        self.span_mask = span_mask
        self.span_len_min = span_len_min
        self.span_len_max = span_len_max
        self.span_count = span_count

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tokens = np.array(row['tokens'], dtype=np.int64)
        time_feat = np.array(row['time_feat'], dtype=np.float32)
        user_id = int(row['user_id'])
        # 截断
        tokens = tokens[:self.L]
        time_feat = time_feat[:self.L]
        length = len(tokens)

        # 构造 mask
        mask = np.zeros(self.L, dtype=bool)
        if length > 0:
            if self.span_mask:
                # 连续片段掩码：生成 span_count 个片段，长度在 [span_len_min, span_len_max]
                for _ in range(self.span_count):
                    span_len = np.random.randint(self.span_len_min, self.span_len_max + 1)
                    if span_len <= 0:
                        continue
                    if span_len > length:
                        span_len = length
                    # 起点随机，但保证片段不越过有效长度
                    start = np.random.randint(0, max(1, length - span_len + 1))
                    mask[start:start + span_len] = True
            else:
                # 原随机点掩码（独立伯努利）
                rand_mask = (np.random.rand(length) < self.mask_ratio)
                mask[:length] = rand_mask

        # padding
        if length < self.L:
            pad_len = self.L - length
            tokens = np.pad(tokens, (0, pad_len), constant_values=self.mask_id)
            time_feat = np.pad(time_feat, ((0, pad_len), (0, 0)), constant_values=0.0)

        tokens_masked = tokens.copy()
        tokens_masked[mask] = self.mask_id

        return {
            'token_id': torch.from_numpy(tokens_masked),   # [L]
            'token_true': torch.from_numpy(tokens),        # [L]
            'time_feat': torch.from_numpy(time_feat),      # [L,4]
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'mask': torch.from_numpy(mask)                 # [L] bool
        }