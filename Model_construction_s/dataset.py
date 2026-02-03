import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class SecondTrajDataset(Dataset):
    def __init__(self, data_pkl, L=96, mask_ratio=0.15):
        with open(data_pkl, 'rb') as f:
            data = pickle.load(f)
        
        self.sequences = data['sequences']
        self.norm_params = data['norm_params'] # Useful if needed externally
        self.L = L
        self.span_mask = True
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        
        # [L, 2] float32 normalized
        traj_norm = item['traj_norm']
        user_id = int(item['user_id'])
        
        # Convert to torch
        traj_true = torch.from_numpy(traj_norm).float()
        traj_in = traj_true.clone()
        mask = torch.zeros(self.L, dtype=torch.bool)
        
        # Masking logic (Span)
        if self.span_mask:
            # Random span 6 (30s) to 12 (60s) points
            span_len = np.random.randint(6, 13) 
            start_idx = np.random.randint(0, self.L - span_len + 1)
            
            traj_in[start_idx : start_idx+span_len] = 0.0 # Zero out
            mask[start_idx : start_idx+span_len] = True
            
        # Time feat: normalized position in sequence for embedding
        time_feat = torch.linspace(0, 1, self.L).unsqueeze(-1) # Simplified [L, 1]

        return {
            'traj_in': traj_in,     # Input (masked)
            'traj_true': traj_true, # GT
            'mask': mask,           # Bool mask
            'time_feat': time_feat,
            'user_id': torch.tensor(user_id, dtype=torch.long)
        }