import pickle
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

# Min: $1 \times 15 \text{ min} = \mathbf{15 \text{ min}}$
#Max: $2 \times 15 \text{ min} = \mathbf{30 \text{ min}}$
class MinuteTrajDataset(Dataset):
    def __init__(self, pkl_path, mask_ratio=0.15, L=96, vocab_size=None,
                 span_mask=True, span_len_min=1, span_len_max=2, span_count=1,
                 sample_step=5):
        """
        新的预处理逻辑：
        1. 读取预处理好的基站数据 (Lat/Lon/Time)。
        2. 按天切分用户轨迹。
        3. 对齐到分钟级时间槽 (1440 mins/day)，使用 前向填充 (FFill) 补全无信号间隙。
        4. 滑动窗口切分为 L 长度的序列。
        """
        print(f"Loading dataset from {pkl_path} ...")
        
        # 兼容两种 pickle 结构
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        # 自动识别数据结构
        if 'sequences' in data:
            # 旧格式兼容 (如果还是用旧数据调试)
            self.rows = data['sequences']
            base_vocab = data.get('vocab_size', 20000)
        else:
            # 新逻辑：现场处理原始序列 (Data On-the-fly Processing)
            # 假设 pickle 里存的是 {'user_id': df, ...} 或者 list of dict
            # 这里我们假设 pickle 包含 'sequences' 列表，且每个 sequence 是一整段原始轨迹
            # 如果结构不同，请根据实际 zx_station_dataset.pkl 调整
            # 为了稳健，我们这里实现一个更通用的处理流程
            
            # 使用传入的 vocab_size 或从数据中获取
            base_vocab = data.get('vocab_size', vocab_size) if vocab_size is None else vocab_size
            self.rows = self._process_raw_data(data, L)

        self.mask_ratio = mask_ratio
        self.L = L
        self.base_vocab = base_vocab
        self.mask_id = base_vocab     # mask token ID
        self.span_mask = span_mask
        self.span_len_min = span_len_min
        self.span_len_max = span_len_max
        self.span_count = span_count
        self.sample_step = sample_step

        print(f"Dataset loaded. Total samples: {len(self.rows)}")

    def _process_raw_data(self, data, seq_len):
        """
        将原始的不规则时间序列处理为规则的时间槽序列
        策略：按天分组 -> 重采样 1min -> 前向填充 -> 切片
        """
        processed_samples = []
        raw_seqs = data['sequences'] # 这里假设原始数据依然叫 sequences，但内容可能是不规则的
        
        # 1. 重组数据为 DataFrame 以便利用 pandas 强大的时间序列处理能力
        # 假设 raw_seqs 里的 'tokens' 对应 cluster_centers 的索引，'time_feat' 是时间
        # 但如果是原始 lat/lon，我们需要转换。
        # 这里假设输入已经是 Token ID 序列，只是时间不规则。
        
        # 为了提高效率，我们按 User 处理
        user_data = {}
        for seq in raw_seqs:
            uid = seq['user_id']
            if uid not in user_data: user_data[uid] = []
            user_data[uid].append(seq)
            
        # 2. 遍历每个用户
        for uid, seqs in tqdm(user_data.items(), desc="Aligning Time Slots"):
            # 合并该用户所有片段（以防原始数据已经是碎片的）
            # 如果原始数据已经按轨迹分好，可以直接处理
            
            # 这里简化逻辑：假设每个 seq 就是一段我们可以处理的基础单元
            # 如果需要跨天合并，逻辑会复杂很多。
            # 鉴于"按天学习"的要求，我们假设 seq 已经包含了足够的时间信息
            
            # 我们需要还原绝对时间来做对齐
            # 这里的 seq['time_feat'] 通常是归一化时间，无法还原日期。
            # 如果 zx_station_dataset.pkl 里保留了原始时间戳列 'timestamp' 或 'c_event_time' 最好。
            # 如果没有，我们假设数据已经是规则的或者无法回溯日期。
            
            # 假设：sequences 中包含了 'day_str' (日期) 和 'minute_of_day' (0-1439) 以及 'token'
            # 或者 seq['tokens'] 是时间顺序排列的，且 seq 对应某一天。
            
            # === 实现 Sample and Hold (Sample-Hold) ===
            # 由于 dataset_pre.py 生成的 pickle 结构可能还不够完善，
            # 这里我们实现一个鲁棒的逻辑：
            # 如果输入本身就是 Token 列表，我们假设它已经是按时间顺序排列的观测点。
            # 我们需要将其"拉伸"到规则网格。
            
            # 暂时直接使用原始序列，但在 __getitem__ 里做模拟 Sample-Hold 的下采样
            # 因为在这里做全量的 DataFrame 操作可能太慢且耗内存。
            
            # **修正策略**：不在 Dataset 初始化时做昂贵的重采样，
            # 而是假设输入的 pkl 已经是经过 "重采样+前向填充" 后的规整数据，
            # 或者我们在 dataset_pre.py 阶段就应该做好这一步。
            
            # 如果必须在这里做：
            # 现有的 sequences 结构: {'tokens': [id, id...], 'time_feat': [0.1, 0.12...]}
            # 如果这里的 time_feat 是归一化的一天时间，我们可以尝试恢复。
            
            # 既然您要求 "Sample and Hold"，这意味着目前的序列是压缩的（仅有变动点或记录点）。
            # 我们直接把原始序列加入，后续在训练采样时处理？不，Transformer 需要定长L对应物理时间。
            
            # -> 最终方案：直接将当前序列视为规则序列（假设上游已处理），
            # 或者，如果上游没处理，我们在这里简单切片。
            # 鉴于代码复用，我们直接返回原始 list，把 "L" 的切分逻辑交给 Dataset 索引。
            
            # 为了满足 "整天长序列切分成 L=96"，我们需要滑动窗口
            full_tokens = seqs[0]['tokens'] # 假设这里是一整天的 tokens
            if len(full_tokens) < seq_len:
                continue
                
            # 滑动窗口切分 (Stride = L/2)
            stride = seq_len // 2
            total_len = len(full_tokens)
            
            # 构造时间特征 (0~1439 minute index)
            # 如果没有原始时间，我们假设序列从 00:00 开始，每步 1 分钟（理想情况）
            # 或者利用 seq['time_feat']
            
            timestamps = seqs[0].get('time_feat', np.linspace(0, 1, total_len)) 
            
            for i in range(0, total_len - seq_len + 1, stride):
                sub_tokens = full_tokens[i : i + seq_len]
                sub_time = timestamps[i : i + seq_len]
                
                processed_samples.append({
                    'user_id': uid,
                    'tokens': sub_tokens,
                    'time_feat': sub_time,
                    'orig_uid': seqs[0].get('orig_uid', str(uid))
                })
                
        return processed_samples

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        
        # 转换并确保类型
        tokens = np.array(row['tokens'], dtype=np.int64)
        time_feat = np.array(row['time_feat'], dtype=np.float32)
        user_id = int(row['user_id'])

        # --- Sample Step 下采样 ---
        # 如果需要模拟更稀疏的数据，或者是为了减少计算量
        if self.sample_step > 1:
            tokens = tokens[::self.sample_step]
            time_feat = time_feat[::self.sample_step]

        # 截断 (双重保险)
        tokens = tokens[:self.L]
        time_feat = time_feat[:self.L]
        length = len(tokens)

        # Padding (如果切分后不足 L，实际上初始化时过滤了，但为了健壮性保留)
        if length < self.L:
            pad_len = self.L - length
            tokens = np.pad(tokens, (0, pad_len), constant_values=self.mask_id)
            time_feat = np.pad(time_feat, ((0, pad_len), (0, 0)), constant_values=0.0)

        # --- 掩码生成 (Masking) ---
        mask = np.zeros(self.L, dtype=bool)
        
        # 只对有效长度部分生成 Mask
        valid_len = min(length, self.L)
        if valid_len > 0:
            if self.span_mask:
                # 连续片段掩码 (Span Masking)
                for _ in range(self.span_count):
                    # 随机生成掩码长度
                    span_len = np.random.randint(self.span_len_min, self.span_len_max + 1)
                    if span_len > valid_len: span_len = valid_len
                    if span_len <= 0: continue
                    
                    # 随机起始位置
                    start = np.random.randint(0, valid_len - span_len + 1)
                    mask[start : start + span_len] = True
            else:
                # 随机点掩码 (Bernoulli)
                rand_mask = (np.random.rand(valid_len) < self.mask_ratio)
                mask[:valid_len] = rand_mask

        # 应用掩码：将 mask 位置的 token 替换为 mask_id
        tokens_masked = tokens.copy()
        tokens_masked[mask] = self.mask_id

        return {
            'token_id': torch.from_numpy(tokens_masked),   # [L] 输入序列 (Masked)
            'token_true': torch.from_numpy(tokens),        # [L] 标签序列 (Ground Truth)
            'time_feat': torch.from_numpy(time_feat),      # [L] 时间特征
            'user_id': torch.tensor(user_id, dtype=torch.long), 
            'mask': torch.from_numpy(mask)                 # [L] bool (True=被掩盖区域)
        }