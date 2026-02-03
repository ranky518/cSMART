import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, mask_id=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=mask_id)
    def forward(self, x):
        return self.embedding(x)

class UserEmbedding(nn.Module):
    def __init__(self, num_users, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_users, embed_dim)
    def forward(self, user_id):
        return self.embedding(user_id)          # [B] -> [B,d]

class PositionalTimeEmbedding(nn.Module):
    """sin/cos 位置编码，可接分钟级"""
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]          # x:[B,L,d]

class StepEmbedding(nn.Module):
    """扩散步 k 的 4d sin/cos"""
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, k):                       # k:[B,L] or [B]
        if k.dim() == 1:
            k = k.unsqueeze(1)
        half = 2
        inv = torch.exp(torch.arange(half, device=k.device, dtype=torch.float32) *
                        (-math.log(10000.0) / (half - 1)))
        angles = k.float().unsqueeze(-1) * inv      # [B,L,2]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B,L,4]
        return self.mlp(emb)                        # [B,L,d]

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.wk_emb = nn.Embedding(7, embed_dim)
        self.min_emb = nn.Embedding(1440, embed_dim) # Minutes in day

    def forward(self, time_feat):
        # Placeholder if time_feat is continuous
        # If time_feat is [B, L, 1], project it
        # Adapted for the new continuous pipeline
        return torch.zeros(time_feat.size(0), time_feat.size(1), self.embed_dim, device=time_feat.device)