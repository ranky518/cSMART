"""
记忆编码器模块 - 3-10min 版本
"""
import torch
import torch.nn as nn


class MemoryEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 96):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        self.token_emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.aggregate = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
    def encode_single(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L = tokens.shape
        device = tokens.device
        h = self.token_emb(tokens)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_emb(pos)
        key_padding_mask = ~mask if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return h
    
    def forward(self, memory_tokens: torch.Tensor, memory_masks: torch.Tensor,
                query: torch.Tensor) -> torch.Tensor:
        B, K, L = memory_tokens.shape
        device = memory_tokens.device
        
        if K == 0:
            return torch.zeros_like(query)
        
        memory_tokens_flat = memory_tokens.view(B * K, L)
        memory_masks_flat = memory_masks.view(B * K, L)
        encoded = self.encode_single(memory_tokens_flat, memory_masks_flat)
        encoded = encoded.view(B, K, L, -1)
        
        memory_masks_expanded = memory_masks.unsqueeze(-1).float()
        trajectory_repr = (encoded * memory_masks_expanded).sum(dim=2) / (memory_masks_expanded.sum(dim=2) + 1e-6)
        
        attn_output, _ = self.aggregate(query, trajectory_repr, trajectory_repr)
        return attn_output


class MemoryGating(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([context, memory], dim=-1)
        return self.gate(combined)
