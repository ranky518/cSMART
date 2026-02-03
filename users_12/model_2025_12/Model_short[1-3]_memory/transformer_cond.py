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
        return self.embedding(user_id)


class PositionalTimeEmbedding(nn.Module):
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
        return x + self.pe[:x.size(1)]


class TransformerCond(nn.Module):
    def __init__(self, vocab_size, num_users, embed_dim=512,
                 nhead=8, num_layers=6, mask_id=None, max_len=288):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_dim, mask_id)
        self.user_emb = UserEmbedding(num_users, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        self.pos_enc = PositionalTimeEmbedding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=2048,
            activation='gelu', batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, token_id, time_feat, user_id):
        h_tok = self.token_emb(token_id)
        h_usr = self.user_emb(user_id).unsqueeze(1).expand(-1, token_id.size(1), -1)
        
        if time_feat.dim() == 2:
            time_feat = time_feat.unsqueeze(-1)
            
        h_tim = self.time_proj(time_feat)
        h = h_tok + h_usr + h_tim
        h = self.pos_enc(h)
        
        key_padding_mask = (token_id == self.token_emb.embedding.padding_idx)
        cond = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return cond
