import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

class SecondTrajDataset(Dataset):
    def __init__(self, data_pkl, L=96, mask_ratio=0.5):
        """
        Args:
            data_pkl: Path to pickle file containing trajectories and norm_params
            L: Max sequence length
            mask_ratio: Masking ratio for training
        """
        super().__init__()
        with open(data_pkl, 'rb') as f:
            meta = pickle.load(f)
        
        # 1. Load Normalization Params
        self.norm_params = meta.get('norm_params')
        if not self.norm_params:
            raise ValueError(f"Key 'norm_params' not found in {data_pkl}. Keys: {list(meta.keys())}")

        self.lat_min = self.norm_params['lat_min']
        self.lat_max = self.norm_params['lat_max']
        self.lon_min = self.norm_params['lon_min']
        self.lon_max = self.norm_params['lon_max']
        
        self.num_users = meta.get('num_users', 1)
        self.L = L
        self.mask_ratio = mask_ratio

        # 2. Smartly detect the data list
        possible_keys = ['data', 'trajectories', 'trajs', 'sequences', 'traj_list']
        self.data = None
        
        # Try known keys
        for key in possible_keys:
            if key in meta:
                self.data = meta[key]
                print(f"Dataset: Loaded data from key '{key}'.")
                break
        
        # Fallback: look for the largest list in the dict
        if self.data is None:
            print(f"Dataset: Key 'data' not found. Searching in keys: {list(meta.keys())}...")
            candidates = []
            for k, v in meta.items():
                if isinstance(v, list) and len(v) > 0:
                    candidates.append((k, len(v)))
            
            if candidates:
                # Pick the longest list, assuming it's the dataset
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_key = candidates[0][0]
                self.data = meta[best_key]
                print(f"Dataset: Auto-detected data under key '{best_key}' (len={candidates[0][1]}).")
            else:
                raise KeyError(f"Could not find trajectory data in pickle. Keys found: {list(meta.keys())}")

    def normalize(self, traj):
        """将经纬度归一化到 [-1, 1]"""
        # copy to avoid modifying original data
        traj_norm = traj.copy()
        traj_norm[:, 0] = 2 * (traj[:, 0] - self.lat_min) / (self.lat_max - self.lat_min) - 1
        traj_norm[:, 1] = 2 * (traj[:, 1] - self.lon_min) / (self.lon_max - self.lon_min) - 1
        return traj_norm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 获取原始轨迹
        item = self.data[idx]
        
        # Determine if item is dict or array
        if isinstance(item, dict):
            # Try common keys for coordinates
            if 'traj' in item:
                traj_raw = item['traj']
            elif 'coordinates' in item:
                traj_raw = item['coordinates']
            elif 'data' in item:
                traj_raw = item['data']
            elif 'traj_norm' in item:
                traj_raw = item['traj_norm']
            else:
                # Fallback: assume values() might contain the array or just fail
                raise KeyError(f"Could not find trajectory array in dict item at index {idx}. Keys: {list(item.keys())}")
        else:
            traj_raw = item # Assume it is already numpy array or tensor

        # Ensure it's numpy array
        if isinstance(traj_raw, list):
            traj_raw = np.array(traj_raw)
        elif torch.is_tensor(traj_raw):
            traj_raw = traj_raw.numpy()

        # 2. 截断或填充
        seq_len = traj_raw.shape[0]
        if seq_len > self.L:
            # 随机截取一段或取前段
            start = np.random.randint(0, seq_len - self.L + 1)
            traj_raw = traj_raw[start:start+self.L]
            valid_len = self.L
        else:
            valid_len = seq_len
            
        # 3. 归一化 [-1, 1]
        traj_norm = self.normalize(traj_raw)
        
        # 4. 构建 Tensor 和 Padding
        # traj_true: 完整的归一化轨迹 (Padding部分填0)
        traj_true = torch.zeros((self.L, 2), dtype=torch.float32)
        traj_true[:valid_len] = torch.from_numpy(traj_norm)
        
        # mask_padding: 标识哪些位置是真实数据 (1为真实，0为填充)
        mask_padding = torch.zeros((self.L,), dtype=torch.float32)
        mask_padding[:valid_len] = 1.0

        # 5. 生成 traj_in (Condition)
        # 随机丢弃一部分点作为输入，模拟补全任务
        # mask_observed: 1表示该点可见(作为条件输入)，0表示该点缺失(需要预测)
        # 注意：Padding部分也视为不可见(0)
        prob = torch.rand(valid_len)
        mask_obs_bool = prob > self.mask_ratio # 也就是保留 (1-mask_ratio) 的点
        
        mask_observed = torch.zeros((self.L,), dtype=torch.float32)
        mask_observed[:valid_len] = mask_obs_bool.float()
        
        # traj_in: 仅保留可见点，其余置0
        traj_in = traj_true.clone()
        traj_in[mask_observed == 0] = 0 # 将不可见部分抹去
        
        sample = {
            'traj_in': traj_in,          # 稀疏/带噪声的输入
            'traj_true': traj_true,      # 完整 Ground Truth
            'mask': mask_padding,        # 有效长度 Mask
            'mask_obs': mask_observed,   # 观测点 Mask
            'time_feat': torch.zeros((self.L,)), 
            'user_id': torch.tensor(0)   
        }
        
        return sample
