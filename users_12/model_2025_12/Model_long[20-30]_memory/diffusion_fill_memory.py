"""
记忆增强扩散模型 - 3-10min 版本
"""
import torch
import torch.nn as nn
import math
from memory_encoder import MemoryEncoder, MemoryGating


def cosine_beta_schedule(steps, s=0.008):
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class StepEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, k):
        if k.dim() == 1:
            k = k.unsqueeze(1)
        half = 2
        inv = torch.exp(torch.arange(half, device=k.device, dtype=torch.float32) *
                        (-math.log(10000.0) / (half - 1)))
        angles = k.float().unsqueeze(-1) * inv
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(emb)


class MemoryAugmentedDenoiser(nn.Module):
    def __init__(self, d_cond: int, d_embed: int, d_memory: int = None):
        super().__init__()
        d_memory = d_memory or d_cond
        self.step_emb = StepEmbedding(d_embed)
        self.input_proj = nn.Linear(d_embed + d_cond + d_memory, d_embed)
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(d_embed * 2, d_embed), nn.SiLU(), nn.Linear(d_embed, d_embed)
        )
        
    def forward(self, x_noisy, z_cond, z_memory, t, mask):
        t_emb = self.step_emb(t)
        h = torch.cat([x_noisy + t_emb, z_cond, z_memory], dim=-1)
        h = self.input_proj(h)
        return self.net(h)


class DiffusionFillWithMemory(nn.Module):
    def __init__(self, d_cond: int, d_embed: int, vocab_size: int, 
                 num_steps: int = 1000, use_memory_gating: bool = True):
        super().__init__()
        self.d_cond = d_cond
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.use_memory_gating = use_memory_gating
        
        betas = cosine_beta_schedule(num_steps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        self.emb = None
        self.memory_encoder = MemoryEncoder(vocab_size, d_embed)
        if use_memory_gating:
            self.memory_gate = MemoryGating(d_embed)
        self.denoiser = MemoryAugmentedDenoiser(d_cond, d_embed, d_embed)
        self.classifier = nn.Linear(d_embed, vocab_size)
        
    def add_noise(self, x0, t):
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        am1 = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noise = torch.randn_like(x0)
        return a * x0 + am1 * noise, noise
    
    def encode_memory(self, memory_tokens, memory_masks, query):
        return self.memory_encoder(memory_tokens, memory_masks, query)
    
    def forward(self, x_noisy, z_cond, z_memory, t, mask):
        if self.use_memory_gating and z_memory is not None:
            gate = self.memory_gate(z_cond, z_memory)
            z_memory_gated = gate * z_memory
        else:
            z_memory_gated = z_memory if z_memory is not None else torch.zeros_like(z_cond)
        return self.denoiser(x_noisy, z_cond, z_memory_gated, t, mask)
