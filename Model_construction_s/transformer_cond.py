import torch
import torch.nn as nn
from embeddings import TimeEmbedding  # Assuming this exists or simple replacement

class TransformerCond(nn.Module):
    def __init__(self, input_dim=2, num_users=1, embed_dim=512, nhead=8, num_layers=4, max_len=96):
        super().__init__()
        
        # Continuous input projection: [B, L, 2] -> [B, L, d_model]
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # User Embedding
        self.user_emb = nn.Embedding(num_users, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, traj, time_feat, user_id):
        # traj: [B, L, 2]
        x = self.input_proj(traj)
        
        # Add position
        x = x + self.pos_emb[:, :x.size(1), :]
        
        # Add User
        u = self.user_emb(user_id).unsqueeze(1) # [B, 1, D]
        x = x + u
        
        x = self.dropout(x)
        
        # Encode
        z = self.transformer_encoder(x)
        return self.out_proj(z)