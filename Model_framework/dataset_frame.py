import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class MinuteTrajDataset(Dataset):
    def __init__(self, pkl_path, mask_ratio=0.15, L=96, vocab_size=2000,
                 span_mask=True, span_len_min=2, span_len_max=4, span_count=1,
                 sample_step=5):  # <--- 确保这里有 sample_step
        # 下限 (10分钟)：$10 \div 5 = 2 \text{point}$。
        # 上限 (20分钟)：$20 \div 5 =  4 \text{point}$ 。
        data = pickle.load(open(pkl_path, 'rb'))          # dict
        self.rows = data['sequences']                     # list of dict
        self.mask_ratio = mask_ratio
        self.L = L
        self.base_vocab = vocab_size
        self.mask_id = vocab_size                         # mask 索引 = 原簇数
        self.span_mask = span_mask
        self.span_len_min = span_len_min
        self.span_len_max = span_len_max
        self.span_count = span_count
        self.sample_step = sample_step                    # 保存采样步长

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tokens = np.array(row['tokens'], dtype=np.int64)
        time_feat = np.array(row['time_feat'], dtype=np.float32)
        user_id = int(row['user_id'])

        # --- 核心修改：下采样 (Downsampling) ---
        # 每隔 sample_step 取一个点 (例如3分钟取一次)
        if self.sample_step > 1:
            tokens = tokens[::self.sample_step]
            time_feat = time_feat[::self.sample_step]

        # 截断
        tokens = tokens[:self.L]
        time_feat = time_feat[:self.L]
        length = len(tokens)

        # 构造 mask
        mask = np.zeros(self.L, dtype=bool)
        if length > 0:
            if self.span_mask:
                # 连续片段掩码
                # 在3分钟采样下，mask 1个点=3分钟，mask 2个点=6分钟
                # 对应 "3-5分钟补全" 任务，span_len 应设为 1~2
                for _ in range(self.span_count):
                    span_len = np.random.randint(self.span_len_min, self.span_len_max + 1)
                    if span_len <= 0: continue
                    if span_len > length: span_len = length
                    # 起点随机
                    start = np.random.randint(0, max(1, length - span_len + 1))
                    mask[start:start + span_len] = True
            else:
                # 原随机点掩码
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