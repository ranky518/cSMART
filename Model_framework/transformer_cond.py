import torch
import torch.nn as nn
from embeddings import (TokenEmbedding, UserEmbedding, PositionalTimeEmbedding)
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, time_emb=None):
        L, B, _ = x.shape
        pos_emb = self.pe[:L, :].unsqueeze(1).repeat(1, B, 1)
        x = x + pos_emb
        if time_emb is not None:
            x = x + time_emb.permute(1, 0, 2)
        return self.dropout(x)


class TransformerCond(nn.Module):
    """
    条件 Transformer 编码器
    
    Args:
        use_direct_classifier: 
            - False (default): 返回上下文向量供 Diffusion 使用 (TC-DPM)
            - True: 直接输出分类 logits (Transformer-Direct Baseline)
    """
    def __init__(self, vocab_size, num_users, embed_dim=512,
                 nhead=8, num_layers=6, mask_id=None, max_len=288, 
                 use_direct_classifier=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_id = mask_id
        self.use_direct_classifier = use_direct_classifier
        
        # Embeddings
        self.token_emb = TokenEmbedding(vocab_size, embed_dim, mask_id)
        self.user_emb = UserEmbedding(num_users, embed_dim)
        self.time_proj = nn.Linear(4, embed_dim)
        self.pos_enc = PositionalTimeEmbedding(embed_dim, max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=2048,
            activation='gelu', batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # [ABLATION] 直接分类头 (Baseline: Transformer-Direct)
        if self.use_direct_classifier:
            # 注意：输出维度是 base_vocab（不含 mask/pad），与 Diffusion classifier 对齐
            # 但这里 vocab_size 可能已经包含 mask，需要外部传入正确的 base_vocab
            self.classifier_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, vocab_size)  # 输出到完整词表
            )

    def forward(self, token_id, time_feat, user_id):
        """
        Args:
            token_id: [B, L] 输入 token（mask 位置为 mask_id）
            time_feat: [B, L, 4] 时间特征
            user_id: [B] 用户 ID
            
        Returns:
            - use_direct_classifier=False: context [B, L, D] 供 Diffusion 使用
            - use_direct_classifier=True: logits [B, L, vocab_size] 直接预测
        """
        B, L = token_id.shape
        
        # 1. 嵌入融合
        h_tok = self.token_emb(token_id)                                      # [B, L, D]
        h_usr = self.user_emb(user_id).unsqueeze(1).expand(-1, L, -1)        # [B, L, D]
        h_tim = self.time_proj(time_feat)                                     # [B, L, D]
        h = h_tok + h_usr + h_tim
        h = self.pos_enc(h)
        
        # 2. Padding Mask（mask_id 位置不参与 attention）
        # 注意：这里是 key_padding_mask，True 表示忽略
        if self.mask_id is not None:
            key_padding_mask = (token_id == self.mask_id)
        else:
            key_padding_mask = None
        
        # 3. Transformer 编码
        context = self.transformer(h, src_key_padding_mask=key_padding_mask)  # [B, L, D]
        
        # 4. 分支输出
        if self.use_direct_classifier:
            # Baseline: 直接预测 token logits
            logits = self.classifier_head(context)  # [B, L, vocab_size]
            return logits
        else:
            # Ours: 返回上下文供 Diffusion 使用
            return context