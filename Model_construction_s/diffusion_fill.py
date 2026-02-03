import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionFill(nn.Module):
    def __init__(self, d_cond=512, d_embed=512, input_dim=2, num_steps=1000):
        super().__init__()
        self.d_embed = d_embed
        self.num_steps = num_steps
        self.input_dim = input_dim
        
        # Projection for x (lat, lon) -> latent
        self.input_proj = nn.Linear(input_dim, d_embed)
        
        # Conditional layers (merging transformer output z_t)
        self.cond_proj = nn.Linear(d_cond, d_embed)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed)
        )
        
        # Simple ResNet-like backbone for denoising
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed),
            nn.SiLU(),
            nn.Linear(d_embed, d_embed)
        )
        
        # Final head: Latent -> lat/lon
        self.output_head = nn.Linear(d_embed, input_dim)

        # Noise schedule
        beta = torch.linspace(1e-4, 0.02, num_steps)
        alpha = 1. - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        
        # Register buffers for alpha, beta, etc.
        self.register_buffer('betas', beta)
        self.register_buffer('alphas', alpha)
        self.register_buffer('alphas_cumprod', alpha_cumprod)
        self.register_buffer('alphas_cumprod_prev', F.pad(alpha_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alpha_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alpha))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', beta * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))

    def get_timestep_embedding(self, t, dim):
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        
        pad_dims = x0.ndim - t.ndim
        for _ in range(pad_dims):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        
        x_noisy = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_noisy, noise

    def forward(self, x_noisy, condition, t, mask):
        x_emb = self.input_proj(x_noisy)
        c_emb = self.cond_proj(condition)
        
        t_vec = t[:, 0] if t.dim() > 1 else t
        t_emb = self.get_timestep_embedding(t_vec, self.d_embed)
        t_emb = self.time_mlp(t_emb).unsqueeze(1) # [B, 1, D]
        
        h = x_emb + c_emb + t_emb
        h = self.net(h)
        pred_noise = self.output_head(h)
        return pred_noise

    @torch.no_grad()
    def sample(self, condition, mask_padding, mask_obs=None, traj_in=None):
        """
        Generate samples from pure noise given the condition.
        condition: [B, L, d_cond] (Transformer output)
        mask_padding: [B, L] (1=valid, 0=pad)
        """
        B, L, _ = condition.shape
        device = condition.device
        
        # 1. Start from pure noise
        x = torch.randn((B, L, self.input_dim), device=device)
        
        # 2. Iterate backwards
        for i in reversed(range(0, self.num_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self(x, condition, t, mask_padding)
            
            # Compute x_{t-1}
            # Formula: x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * eps) + sigma * z
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = self.sqrt_recip_alphas[i] * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
            x = x + torch.sqrt(beta) * noise
            
            # Inpainting Strategy (Optional but recommended):
            # If we trust the observed points in traj_in, we can enforce them at each step.
            # However, for pure generation testing based on condition, we can skip this 
            # or do simpler "replenishment".
            # Here we assume the Transformer condition is sufficient for the "filling" Logic.
            # If you want strict consistency:
            # if mask_obs is not None and traj_in is not None:
            #     # Get noisy version of ground truth at t-1
            #     if i > 0:
            #         noisy_true, _ = self.add_noise(traj_in, torch.full((B,), i-1, device=device))
            #     else:
            #         noisy_true = traj_in
            #     # Mix: keep known parts, let model generate unknown parts
            #     mask_obs_ex = mask_obs.unsqueeze(-1)
            #     x = noisy_true * mask_obs_ex + x * (1 - mask_obs_ex)

        return x