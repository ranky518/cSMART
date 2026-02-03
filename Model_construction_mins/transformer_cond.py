import torch.nn as nn
from embeddings import (TokenEmbedding, UserEmbedding,
                        PositionalTimeEmbedding)

class TransformerCond(nn.Module):
    def __init__(self, vocab_size, num_users, embed_dim=512,
                 nhead=8, num_layers=6, mask_id=None, max_len=288):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_dim, mask_id)
        self.user_emb = UserEmbedding(num_users, embed_dim)
        self.time_proj = nn.Linear(4, embed_dim)               # 4->d
        self.pos_enc = PositionalTimeEmbedding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=2048,
            activation='gelu', batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, token_id, time_feat, user_id):
        # 1. 各种嵌入相加
        h_tok = self.token_emb(token_id)                      # [B,L,d]
        h_usr = self.user_emb(user_id).unsqueeze(1).expand(-1, token_id.size(1), -1)
        h_tim = self.time_proj(time_feat)                     # [B,L,d]
        h = h_tok + h_usr + h_tim
        h = self.pos_enc(h)
        # 2. key-padding-mask
        key_padding_mask = (token_id == self.token_emb.embedding.padding_idx)
        cond = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return cond                                           # [B,L,d]