import torch
import torch.nn as nn
import math
from embeddings import StepEmbedding

def cosine_beta_schedule(steps, s=0.008):
    x = torch.linspace(0, steps, steps+1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionFill(nn.Module):
    def __init__(self, d_cond, d_embed, vocab_size, num_steps=1000):
        super().__init__()
        self.d_cond = d_cond
        self.d_embed = d_embed
        self.num_steps = num_steps

        betas = cosine_beta_schedule(num_steps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        self.step_emb = StepEmbedding(d_embed)
        self.net = nn.Sequential(
            nn.Linear(d_embed + d_cond, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed)
        )
        self.classifier = nn.Linear(d_embed, vocab_size)

    def add_noise(self, x0, t):            # x0:[B,L,d], t:[B,L]
        a = self.sqrt_alphas_cumprod[t]            # [B,L]
        am1 = self.sqrt_one_minus_alphas_cumprod[t]
        a = a.unsqueeze(-1)
        am1 = am1.unsqueeze(-1)
        noise = torch.randn_like(x0)
        return a * x0 + am1 * noise, noise

    def forward(self, x_noisy, cond, t, mask):
        # cond:[B,L,d_cond]; x_noisy:[B,L,d_embed]
        t_emb = self.step_emb(t)                   # [B,L,d_embed]
        h = torch.cat([x_noisy + t_emb, cond], dim=-1)
        eps = self.net(h)                          # [B,L,d_embed]
        return eps

    @torch.no_grad()
    def sample(self, z_t, mask, *, token_obs=None, steps=200):
        """
        条件采样（分钟级补全）
        z_t: [B,L,d] 条件
        mask: [B,L] True 为需要生成（缺失）的位置
        token_obs: [B,L] or None，观测到的 token 序列（缺失位可以是 mask_id）
        steps: 采样步数
        """
        device = z_t.device
        B, L, d = z_t.shape

        # 初始状态
        x = torch.randn(B, L, d, device=device)

        # 是否有观测强约束
        have_obs = token_obs is not None
        if have_obs:
            obs_mask = (~mask.bool()).unsqueeze(-1)                # [B,L,1] True=观测位
            x0_obs = self.emb(token_obs.to(device))                # [B,L,d]
            z0_obs = torch.randn_like(x0_obs)                      # 固定噪声

        # 时间与 alpha
        time_seq = torch.linspace(self.num_steps - 1, 0, steps, device=device).long()  # [steps]
        a_s = self.alphas_cumprod[time_seq]                                            # [steps]
        a_0 = torch.tensor(1.0, device=device, dtype=a_s.dtype)
        a_s_prev = torch.cat([a_s[1:], a_0.unsqueeze(0)])                               # [steps]

        for i in range(steps):
            t = time_seq[i]
            t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)
            a_t = a_s[i]

            # 观测位：强约束为 q(x_t|x0_obs)
            if have_obs:
                x_obs_t = torch.sqrt(a_t) * x0_obs + torch.sqrt(1.0 - a_t) * z0_obs    # [B,L,d]
                x = torch.where(obs_mask, x_obs_t, x)

            # 预测噪声并更新（DDIM 形式）
            eps = self.forward(x, z_t, t_tensor, mask)                                  # [B,L,d]
            a_t_prev = a_s_prev[i]
            x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t + 1e-8)       # [B,L,d]
            x = torch.sqrt(a_t_prev) * x0_pred + torch.sqrt(1.0 - a_t_prev) * eps      # [B,L,d]

        # 最终得到接近 x0 的表示，分类得到 token
        logits = self.classifier(x)                                                     # [B,L,vocab]
        return logits.argmax(dim=-1)                                                    # [B,L]

    @torch.no_grad()
    def sample_tokens(self, z_t, mask, token_obs=None, steps=200):
        """
        稳定接口：条件采样（分钟级补全）
        z_t: [B,L,d] 条件
        mask: [B,L] True=需要生成
        token_obs: [B,L] or None，观测 token（缺失位可为 mask_id）
        steps: 采样步数
        """
        device = z_t.device
        B, L, d = z_t.shape

        # 初始噪声状态
        x = torch.randn(B, L, d, device=device)

        # 观测强约束（把观测位沿扩散链强制跟随 q(x_t|x0_obs)）
        have_obs = token_obs is not None
        if have_obs:
            obs_mask = (~mask.bool()).unsqueeze(-1)           # [B,L,1]
            x0_obs = self.emb(token_obs.to(device))           # [B,L,d]
            z0_obs = torch.randn_like(x0_obs)                 # 固定噪声

        # 步长安排（从大 t 到 0）
        time_seq = torch.linspace(self.num_steps - 1, 0, steps, device=device).long()  # [steps]
        a_s = self.alphas_cumprod[time_seq]                                            # [steps]
        a_0 = torch.tensor(1.0, device=device, dtype=a_s.dtype)
        # 上一个 a（沿 time_seq 前进的“上一个”）
        a_s_prev = torch.cat([a_s[1:], a_0.unsqueeze(0)])                               # [steps]

        for i in range(steps):
            t = time_seq[i]
            t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)
            a_t = a_s[i]

            if have_obs:
                x_obs_t = torch.sqrt(a_t) * x0_obs + torch.sqrt(1.0 - a_t) * z0_obs    # [B,L,d]
                x = torch.where(obs_mask, x_obs_t, x)

            # 预测噪声并做一次 DDIM 更新
            eps = self.forward(x, z_t, t_tensor, mask)                                  # [B,L,d]
            a_t_prev = a_s_prev[i]
            x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t + 1e-8)       # [B,L,d]
            x = torch.sqrt(a_t_prev) * x0_pred + torch.sqrt(1.0 - a_t_prev) * eps      # [B,L,d]

        logits = self.classifier(x)                                                     # [B,L,vocab]
        return logits.argmax(dim=-1)                                                    # [B,L]