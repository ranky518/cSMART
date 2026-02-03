"""
DiffusionFill - 简化修复版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionFill(nn.Module):
    def __init__(self, d_cond, d_embed, vocab_size, num_steps=50):
        super().__init__()
        self.d_cond = d_cond
        self.d_embed = d_embed
        self.num_steps = num_steps
        
        # 噪声调度
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed)
        )
        
        # 条件融合层
        self.cond_proj = nn.Linear(d_cond, d_embed)
        
        # 去噪网络
        self.denoise_net = nn.Sequential(
            nn.Linear(d_embed * 2, d_embed * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_embed * 4, d_embed * 2),
            nn.GELU(),
            nn.Linear(d_embed * 2, d_embed)
        )
        
        # 分类器
        self.classifier = nn.Linear(d_embed, vocab_size)
        
        # 嵌入层（与Transformer共享）
        self.emb = None
        
        # 注册缓冲器
        self.register_buffer('betas_buf', self.betas)
        self.register_buffer('alphas_buf', self.alphas)
        self.register_buffer('alphas_cumprod_buf', self.alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod_buf', self.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod_buf', self.sqrt_one_minus_alphas_cumprod)
    
    def add_noise(self, x0, t):
        """添加噪声"""
        sqrt_alpha = self.sqrt_alphas_cumprod_buf[t].unsqueeze(-1)  # [B, L, 1]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod_buf[t].unsqueeze(-1)  # [B, L, 1]
        noise = torch.randn_like(x0)
        x_noisy = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return x_noisy, noise
    
    def forward(self, x_noisy, cond, t, mask=None):
    """
    前向传播
    x_noisy: [batch, seq_len, d_embed]
    cond: [batch, seq_len, d_cond]
    t: [batch, seq_len]
    mask: [batch, seq_len]
    """
    batch_size, seq_len_x, _ = x_noisy.shape
    batch_size_c, seq_len_c, _ = cond.shape
    
    # 添加调试信息
    print(f"DEBUG: x_noisy shape: {x_noisy.shape}")
    print(f"DEBUG: cond shape: {cond.shape}")
    print(f"DEBUG: t shape: {t.shape}")
    
    if seq_len_x != seq_len_c:
        print(f"ERROR: Sequence length mismatch! x_noisy: {seq_len_x}, cond: {seq_len_c}")
        # 尝试调整维度
        min_len = min(seq_len_x, seq_len_c)
        x_noisy = x_noisy[:, :min_len, :]
        cond = cond[:, :min_len, :]
        t = t[:, :min_len]
        if mask is not None:
            mask = mask[:, :min_len]
        print(f"Adjusted to min length: {min_len}")
    
    # 1. 时间嵌入
    t_norm = t.float() / self.num_steps
    t_emb = self.time_embed(t_norm.unsqueeze(-1))
    
    # 2. 条件投影
    cond_proj = self.cond_proj(cond)
    
    # 3. 合并特征
    print(f"DEBUG: x_noisy shape after adjustment: {x_noisy.shape}")
    print(f"DEBUG: cond_proj shape: {cond_proj.shape}")
    print(f"DEBUG: t_emb shape: {t_emb.shape}")
    
    h = torch.cat([x_noisy, cond_proj + t_emb], dim=-1)
    
    # 4. 去噪
    eps_pred = self.denoise_net(h)
    
    # 5. 可选：应用掩码
    if mask is not None:
        eps_pred = eps_pred * mask.unsqueeze(-1)
    
    return eps_pred
    
    @torch.no_grad()
    def sample(self, cond, mask=None, steps=None):
        """采样生成"""
        if steps is None:
            steps = self.num_steps
        
        batch_size, seq_len, _ = cond.shape
        
        # 初始噪声
        x_t = torch.randn(batch_size, seq_len, self.d_embed).to(cond.device)
        
        # 反向扩散过程
        for i in reversed(range(steps)):
            t = torch.full((batch_size, seq_len), i, device=cond.device)
            
            # 预测噪声
            eps_pred = self.forward(x_t, cond, t, mask)
            
            # 计算系数
            alpha_t = self.alphas_buf[i]
            alpha_cumprod_t = self.alphas_cumprod_buf[i]
            sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod_buf[i]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod_buf[i]
            
            # 估计x0
            x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * eps_pred) / sqrt_alpha_cumprod_t
            
            # 应用掩码
            if mask is not None:
                x0_pred = x0_pred * mask.unsqueeze(-1)
            
            # 计算下一步
            if i > 0:
                noise = torch.randn_like(x_t)
                beta_t = self.betas_buf[i]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sigma_t = torch.sqrt(beta_t)
                
                x_t = (1.0 / sqrt_alpha_t) * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * eps_pred) + sigma_t * noise
            else:
                x_t = x0_pred
        
        # 解码回token
        logits = self.classifier(x_t)
        tokens = logits.argmax(dim=-1)
        
        return tokens, x_t