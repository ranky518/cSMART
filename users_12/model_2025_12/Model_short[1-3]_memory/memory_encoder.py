"""
记忆编码器模块
将检索到的历史轨迹编码为条件向量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEncoder(nn.Module):
    """
    将检索到的历史轨迹编码为记忆向量
    支持多条历史轨迹的聚合
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 96):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # 轨迹编码器 (轻量级 Transformer)
        self.token_emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 多轨迹聚合
        self.aggregate = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
    def encode_single(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        编码单条轨迹
        
        Args:
            tokens: [B, L] 基站token序列
            mask: [B, L] padding mask (True=有效)
            
        Returns:
            [B, L, D] 编码后的轨迹表示
        """
        B, L = tokens.shape
        device = tokens.device
        
        # Embedding
        h = self.token_emb(tokens)
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_emb(pos)
        
        # Transformer 编码
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
            
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        
        return h
    
    def forward(self, memory_tokens: torch.Tensor, memory_masks: torch.Tensor,
                query: torch.Tensor) -> torch.Tensor:
        """
        编码多条历史轨迹并与当前查询融合
        
        Args:
            memory_tokens: [B, K, L] K条历史轨迹
            memory_masks: [B, K, L] 有效位置mask
            query: [B, L_q, D] 当前序列的查询向量
            
        Returns:
            [B, L_q, D] 融合记忆后的向量
        """
        B, K, L = memory_tokens.shape
        device = memory_tokens.device
        
        if K == 0:
            # 无记忆时返回零向量
            return torch.zeros_like(query)
        
        # 编码每条历史轨迹
        memory_tokens_flat = memory_tokens.view(B * K, L)
        memory_masks_flat = memory_masks.view(B * K, L)
        
        encoded = self.encode_single(memory_tokens_flat, memory_masks_flat)  # [B*K, L, D]
        encoded = encoded.view(B, K, L, -1)  # [B, K, L, D]
        
        # 取每条轨迹的平均表示 (或CLS位置)
        # 这里使用 masked mean pooling
        memory_masks_expanded = memory_masks.unsqueeze(-1).float()  # [B, K, L, 1]
        trajectory_repr = (encoded * memory_masks_expanded).sum(dim=2) / (memory_masks_expanded.sum(dim=2) + 1e-6)
        # trajectory_repr: [B, K, D]
        
        # 使用 Cross-Attention 让 query attend 到历史轨迹
        # query: [B, L_q, D], key/value: [B, K, D]
        attn_output, _ = self.aggregate(query, trajectory_repr, trajectory_repr)
        
        return attn_output  # [B, L_q, D]


class MemoryGating(nn.Module):
    """
    记忆门控模块
    动态决定记忆的可信度
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, context: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        计算记忆的可信度门控
        
        Args:
            context: [B, L, D] 当前上下文
            memory: [B, L, D] 记忆表示
            
        Returns:
            [B, L, 1] 门控权重
        """
        combined = torch.cat([context, memory], dim=-1)
        gate = self.gate(combined)
        return gate
