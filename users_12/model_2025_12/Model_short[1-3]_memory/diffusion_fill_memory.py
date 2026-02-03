"""
记忆增强的扩散模型
将历史轨迹记忆作为扩散过程的条件先验
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
    """扩散步 t 的嵌入"""
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
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
    """
    记忆增强的去噪网络
    eps = Net(x_t, z_cond, z_memory, t)
    """
    
    def __init__(self, d_cond: int, d_embed: int, d_memory: int = None):
        super().__init__()
        d_memory = d_memory or d_cond
        
        self.step_emb = StepEmbedding(d_embed)
        
        # 主干网络: 融合 x_t, z_cond, z_memory
        self.input_proj = nn.Linear(d_embed + d_cond + d_memory, d_embed)
        
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_embed * 2, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed)
        )
        
    def forward(self, x_noisy: torch.Tensor, z_cond: torch.Tensor, 
                z_memory: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy: [B, L, D] 加噪后的嵌入
            z_cond: [B, L, D] Transformer 编码的上下文
            z_memory: [B, L, D] 记忆编码
            t: [B, L] 扩散时间步
            mask: [B, L] 缺失位置mask
            
        Returns:
            [B, L, D] 预测的噪声
        """
        t_emb = self.step_emb(t)  # [B, L, D]
        
        # 融合所有条件
        h = torch.cat([x_noisy + t_emb, z_cond, z_memory], dim=-1)
        h = self.input_proj(h)
        
        eps = self.net(h)
        return eps


class DiffusionFillWithMemory(nn.Module):
    """
    记忆增强的扩散补全模型
    """
    
    def __init__(self, d_cond: int, d_embed: int, vocab_size: int, 
                 num_steps: int = 1000, use_memory_gating: bool = True):
        super().__init__()
        self.d_cond = d_cond
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.use_memory_gating = use_memory_gating
        
        # 扩散参数
        betas = cosine_beta_schedule(num_steps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # Token 嵌入 (将在训练时与 Transformer 共享)
        self.emb = None  # 稍后设置
        
        # 记忆编码器
        self.memory_encoder = MemoryEncoder(vocab_size, d_embed)
        
        # 记忆门控
        if use_memory_gating:
            self.memory_gate = MemoryGating(d_embed)
        
        # 去噪网络
        self.denoiser = MemoryAugmentedDenoiser(d_cond, d_embed, d_embed)
        
        # 分类头
        self.classifier = nn.Linear(d_embed, vocab_size)
        
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """前向扩散：添加噪声"""
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        am1 = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noise = torch.randn_like(x0)
        return a * x0 + am1 * noise, noise
    
    def encode_memory(self, memory_tokens: torch.Tensor, memory_masks: torch.Tensor,
                      query: torch.Tensor) -> torch.Tensor:
        """
        编码历史轨迹记忆
        
        Args:
            memory_tokens: [B, K, L] K条历史轨迹
            memory_masks: [B, K, L] 有效位置mask
            query: [B, L_q, D] 当前序列的查询向量
            
        Returns:
            [B, L_q, D] 记忆表示
        """
        return self.memory_encoder(memory_tokens, memory_masks, query)
    
    def forward(self, x_noisy: torch.Tensor, z_cond: torch.Tensor, 
                z_memory: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        去噪前向传播
        
        Args:
            x_noisy: [B, L, D] 加噪嵌入
            z_cond: [B, L, D] Transformer 上下文
            z_memory: [B, L, D] 记忆编码
            t: [B, L] 时间步
            mask: [B, L] 缺失mask
            
        Returns:
            [B, L, D] 预测噪声
        """
        # 可选：门控融合记忆
        if self.use_memory_gating and z_memory is not None:
            gate = self.memory_gate(z_cond, z_memory)
            z_memory_gated = gate * z_memory
        else:
            z_memory_gated = z_memory if z_memory is not None else torch.zeros_like(z_cond)
        
        eps = self.denoiser(x_noisy, z_cond, z_memory_gated, t, mask)
        return eps
    
    @torch.no_grad()
    def sample_tokens(self, z_cond: torch.Tensor, z_memory: torch.Tensor,
                      mask: torch.Tensor, token_obs: torch.Tensor = None,
                      steps: int = 200) -> torch.Tensor:
        """
        条件采样
        
        Args:
            z_cond: [B, L, D] 上下文条件
            z_memory: [B, L, D] 记忆条件
            mask: [B, L] True=需要生成
            token_obs: [B, L] 观测token (可选)
            steps: 采样步数
            
        Returns:
            [B, L] 预测的token序列
        """
        device = z_cond.device
        B, L, d = z_cond.shape
        
        # 初始噪声
        x = torch.randn(B, L, d, device=device)
        
        # 观测位置约束
        have_obs = token_obs is not None
        if have_obs:
            obs_mask = (~mask.bool()).unsqueeze(-1)
            x0_obs = self.emb(token_obs)
            z0_obs = torch.randn_like(x0_obs)
        
        # 时间步序列
        time_seq = torch.linspace(self.num_steps - 1, 0, steps, device=device).long()
        a_s = self.alphas_cumprod[time_seq]
        a_0 = torch.tensor(1.0, device=device, dtype=a_s.dtype)
        a_s_prev = torch.cat([a_s[1:], a_0.unsqueeze(0)])
        
        for i in range(steps):
            t = time_seq[i]
            t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)
            a_t = a_s[i]
            
            # 观测位约束
            if have_obs:
                x_obs_t = torch.sqrt(a_t) * x0_obs + torch.sqrt(1.0 - a_t) * z0_obs
                x = torch.where(obs_mask, x_obs_t, x)
            
            # 去噪
            eps = self.forward(x, z_cond, z_memory, t_tensor, mask.float())
            
            # DDIM 更新
            a_t_prev = a_s_prev[i]
            x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
            x = torch.sqrt(a_t_prev) * x0_pred + torch.sqrt(1.0 - a_t_prev) * eps
        
        # 分类
        logits = self.classifier(x)
        return logits.argmax(dim=-1)
