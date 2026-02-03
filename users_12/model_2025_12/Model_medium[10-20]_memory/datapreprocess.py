import pickle
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class MinuteTrajDataset(Dataset):
    """3-10分钟缺失补全数据集 (sample_step=5, span_len=1-2)"""
    
    def __init__(self, pkl_path, mask_ratio=0.15, L=96, vocab_size=None,
                 span_mask=True, span_len_min=2, span_len_max=4, span_count=1,
                 sample_step=5):
        
        print(f"Loading dataset from {pkl_path} ...")
        self.sample_step = sample_step
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        self.base_vocab = data.get('vocab_size', vocab_size) if vocab_size is None else vocab_size
        
        if 'sequences' in data:
            self.rows = self._process_raw_data(data, L)
        else:
            raise ValueError("Invalid pickle format")

        self.mask_ratio = mask_ratio
        self.L = L
        self.mask_id = self.base_vocab
        self.span_mask = span_mask
        self.span_len_min = span_len_min
        self.span_len_max = span_len_max
        self.span_count = span_count

        print(f"Dataset loaded. Total samples: {len(self.rows)}")

    def _process_raw_data(self, data, seq_len):
        processed_samples = []
        raw_seqs = data['sequences']
        required_len = seq_len * self.sample_step
        stride = required_len // 2

        for seq in tqdm(raw_seqs, desc="Pre-processing (5min Res)"):
            tokens = seq['tokens']
            time_feat = seq['time_feat']
            user_id = seq['user_id']
            
            if len(tokens) < 2:
                continue
            
            tf = np.array(time_feat)
            minutes = tf * 1440.0 if tf.max() <= 1.0 + 1e-6 else tf
            td_index = pd.to_timedelta(minutes, unit='m')
            df = pd.DataFrame({'token': tokens}, index=td_index)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            df_res = df.resample('1T').last()
            df_res['token'] = df_res['token'].ffill()
            df_res = df_res.dropna()
            
            if len(df_res) < required_len:
                continue
            
            new_tokens = df_res['token'].astype(int).values
            mins_processed = df_res.index.total_seconds() / 60.0
            new_time_feat = mins_processed / 1440.0
            total_len = len(new_tokens)
            
            for i in range(0, total_len - required_len + 1, stride):
                processed_samples.append({
                    'user_id': user_id,
                    'tokens': new_tokens[i:i + required_len],
                    'time_feat': new_time_feat[i:i + required_len]
                })
        return processed_samples

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tokens = np.array(row['tokens'], dtype=np.int64)
        time_feat = np.array(row['time_feat'], dtype=np.float32)
        user_id = int(row['user_id'])

        if self.sample_step > 1:
            tokens = tokens[::self.sample_step]
            time_feat = time_feat[::self.sample_step]

        tokens = tokens[:self.L]
        time_feat = time_feat[:self.L]
        length = len(tokens)

        if length < self.L:
            pad_len = self.L - length
            tokens = np.pad(tokens, (0, pad_len), constant_values=self.mask_id)
            time_feat = np.pad(time_feat, (0, pad_len), constant_values=0.0)

        mask = np.zeros(self.L, dtype=bool)
        mask_start, mask_end = 0, 0
        
        if length > 0 and self.span_mask:
            for _ in range(self.span_count):
                span_len = np.random.randint(self.span_len_min, self.span_len_max + 1)
                span_len = min(span_len, length)
                if span_len <= 0:
                    continue
                start = np.random.randint(0, max(1, length - span_len + 1))
                mask[start:start + span_len] = True
                mask_start, mask_end = start, start + span_len

        tokens_masked = tokens.copy()
        tokens_masked[mask] = self.mask_id
        
        start_station = int(tokens[max(0, mask_start - 1)]) if mask_start > 0 else int(tokens[0])
        end_station = int(tokens[min(length - 1, mask_end)]) if mask_end < length else int(tokens[-1])

        return {
            'token_id': torch.from_numpy(tokens_masked),
            'token_true': torch.from_numpy(tokens),
            'time_feat': torch.from_numpy(time_feat),
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'mask': torch.from_numpy(mask),
            'start_station': torch.tensor(start_station, dtype=torch.long),
            'end_station': torch.tensor(end_station, dtype=torch.long),
            'idx': idx
        }
