import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset_preprocess import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill import DiffusionFill  # 使用修复后的版本
import json

# ========== 配置消融模式 ==========
PREPROCESS_MODE = "LINEAR"
# ==================================

# ========== 1. 路径 ==========
base_dir = "/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master"
data_pkl = os.path.join(base_dir, "Model_pre/cluster_token_dataset.pkl")

out_dir = os.path.join("/home/yanglanqi/trajectory/geolife_clustering-master/Ablation/Data_preprocess/results", PREPROCESS_MODE)
os.makedirs(out_dir, exist_ok=True)

save_ckpt = os.path.join(out_dir, f"model_{PREPROCESS_MODE}.pt")

print(f"Start Training with Preprocess Mode: {PREPROCESS_MODE}")
print(f"Model will be saved to: {save_ckpt}")

# ========== 2. 基础常量 ==========
meta = pd.read_pickle(data_pkl)
base_vocab = int(meta['vocab_size'])
num_users = int(meta['num_users']) if 'num_users' in meta else int(max(s['user_id'] for s in meta['sequences']) + 1)

mask_id = base_vocab
pad_id  = base_vocab + 1
vocab_size_total = base_vocab + 2

cluster_centers = torch.tensor(meta['cluster_centers'], dtype=torch.float32)

# 超参数
d_model = 512
batch_size = 64
lr = 1e-4
epochs = 50
mask_ratio = 0.15
L = 96

if PREPROCESS_MODE == "NO_FILL":
    sample_step = 1
else:
    sample_step = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Haversine 距离 ---
def haversine_m(latlon_a, latlon_b):
    R = 6371000.0
    lat1 = torch.deg2rad(latlon_a[:, 0])
    lon1 = torch.deg2rad(latlon_a[:, 1])
    lat2 = torch.deg2rad(latlon_b[:, 0])
    lon2 = torch.deg2rad(latlon_b[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a.clamp(min=0.0, max=1.0)))
    return R * c

# ========== 3. 数据 ==========
dataset = MinuteTrajDataset(data_pkl, mask_ratio=mask_ratio, L=L, vocab_size=base_vocab,
                            span_mask=True, span_len_min=2, span_len_max=5,
                            sample_step=sample_step, preprocess_mode=PREPROCESS_MODE)

# 数据划分
test_ratio = 0.1
val_ratio = 0.1
n_total = len(dataset)
n_test = int(n_total * test_ratio)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val - n_test
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

# 保存索引
split_dir = os.path.join(out_dir, "split_indices")
os.makedirs(split_dir, exist_ok=True)
with open(os.path.join(split_dir, "test_indices.json"), "w") as f:
    json.dump(test_set.indices, f)

loader      = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader  = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# ========== 4. 模型初始化 ==========
transformer = TransformerCond(vocab_size=vocab_size_total, num_users=num_users, embed_dim=d_model,
                              nhead=8, num_layers=6, mask_id=pad_id, max_len=L).to(device)

diffusion = DiffusionFill(d_cond=d_model, d_embed=d_model, vocab_size=base_vocab, num_steps=50).to(device)
diffusion.emb = transformer.token_emb.embedding

# 参数优化
params = list(transformer.parameters()) + list(diffusion.parameters())
optimizer = torch.optim.AdamW(params, lr=lr)

print(f"Training Start: PadID={pad_id}, MaskID={mask_id}")

# ========== 5. 训练循环 ==========
for epoch in range(epochs):
    transformer.train()
    diffusion.train()
    
    for step, batch in enumerate(loader):
        token_id   = batch['token_id'].to(device)
        token_true = batch['token_true'].to(device)
        time_feat  = batch['time_feat'].to(device)
        user_id    = batch['user_id'].to(device)
        mask       = batch['mask'].to(device)
        
        # 获取条件特征
        z_t = transformer(token_id, time_feat, user_id)
        
        # 获取真实嵌入
        x0 = diffusion.emb(token_true)
        
        # 生成随机时间步
        batch_size, seq_len = x0.shape[:2]
        t = torch.randint(0, diffusion.num_steps, (batch_size, seq_len), device=device)
        
        # 添加噪声
        x_noisy, noise = diffusion.add_noise(x0, t)
        
        # 前向传播
        eps_pred = diffusion(x_noisy, z_t, t, mask)
        
        # DDPM损失
        loss_ddpm = F.mse_loss(eps_pred, noise, reduction='none')
        if mask is not None:
            loss_ddpm = (loss_ddpm * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
        else:
            loss_ddpm = loss_ddpm.mean()

        # 辅助分类损失
        a_t = diffusion.sqrt_alphas_cumprod_buf[t].unsqueeze(-1)
        am1_t = diffusion.sqrt_one_minus_alphas_cumprod_buf[t].unsqueeze(-1)
        x0_hat = (x_noisy - am1_t * eps_pred) / a_t
        logits = diffusion.classifier(x0_hat)
        
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = token_true.view(-1)
        flat_mask = mask.view(-1) if mask is not None else torch.ones_like(flat_targets, dtype=torch.bool)
        
        # 忽略 PAD 和 MASK
        valid = flat_mask & (flat_targets != mask_id) & (flat_targets != pad_id)
        
        if valid.any():
            loss_cls = F.cross_entropy(flat_logits[valid], flat_targets[valid])
            
            # 指标计算
            with torch.no_grad():
                pred_tokens = flat_logits.argmax(dim=-1)
                masked_preds = pred_tokens[valid]
                masked_targets = flat_targets[valid]
                correct = (masked_preds == masked_targets).sum().float()
                total = valid.sum().float()
                accuracy = correct / total
                
                pred_latlon = cluster_centers[masked_preds.clamp(max=base_vocab-1)].to(device)
                true_latlon = cluster_centers[masked_targets.clamp(max=base_vocab-1)].to(device)
                dists_m = haversine_m(true_latlon, pred_latlon)
                mae = dists_m.mean()
                acc20 = (dists_m <= 20.0).float().mean()
        else:
            loss_cls = torch.tensor(0.0, device=device)
            accuracy = torch.tensor(0.0, device=device)
            mae = torch.tensor(0.0, device=device)
            acc20 = torch.tensor(0.0, device=device)

        loss = loss_ddpm + 0.1 * loss_cls
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch:02d} | Step {step:04d} | Loss {loss.item():.4f} | "
                  f"Acc: {accuracy.item():.2%} | MAE: {mae.item():.2f}m | A@20: {acc20.item():.2%}")

    # ========== 6. 验证评估 ==========
    transformer.eval()
    diffusion.eval()
    
    val_metrics = {'loss': 0.0, 'acc': 0.0, 'mae': 0.0, 'acc20': 0.0, 'count': 0}
    
    with torch.no_grad():
        for batch in val_loader:
            token_id   = batch['token_id'].to(device)
            token_true = batch['token_true'].to(device)
            time_feat  = batch['time_feat'].to(device)
            user_id    = batch['user_id'].to(device)
            mask       = batch['mask'].to(device)

            z_t = transformer(token_id, time_feat, user_id)
            x0 = diffusion.emb(token_true)
            
            batch_size, seq_len = x0.shape[:2]
            t = torch.randint(0, diffusion.num_steps, (batch_size, seq_len), device=device)
            x_noisy, noise = diffusion.add_noise(x0, t)

            eps_pred = diffusion(x_noisy, z_t, t, mask)
            loss_ddpm = F.mse_loss(eps_pred, noise, reduction='none')
            if mask is not None:
                loss_ddpm = (loss_ddpm * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
            else:
                loss_ddpm = loss_ddpm.mean()

            a_t = diffusion.sqrt_alphas_cumprod_buf[t].unsqueeze(-1)
            am1_t = diffusion.sqrt_one_minus_alphas_cumprod_buf[t].unsqueeze(-1)
            x0_hat = (x_noisy - am1_t * eps_pred) / a_t
            logits = diffusion.classifier(x0_hat)

            flat_logits  = logits.view(-1, logits.size(-1))
            flat_targets = token_true.view(-1)
            flat_mask    = mask.view(-1) if mask is not None else torch.ones_like(flat_targets, dtype=torch.bool)
            valid_pos    = flat_mask & (flat_targets != mask_id) & (flat_targets != pad_id)

            if valid_pos.any():
                loss_cls = F.cross_entropy(flat_logits[valid_pos], flat_targets[valid_pos])
                
                pred_tokens = flat_logits.argmax(dim=-1)
                masked_preds = pred_tokens[valid_pos]
                masked_targets = flat_targets[valid_pos]
                correct = (masked_preds == masked_targets).sum().float()
                total = valid_pos.sum().float()
                acc = (correct / total).item()
                
                pred_latlon = cluster_centers[masked_preds.clamp(max=base_vocab-1)].to(device)
                true_latlon = cluster_centers[masked_targets.clamp(max=base_vocab-1)].to(device)
                dists_m = haversine_m(true_latlon, pred_latlon)
                mae_val = dists_m.mean().item()
                acc20_val = (dists_m <= 20.0).float().mean().item()
            else:
                loss_cls = torch.tensor(0.0, device=device)
                acc = 0.0
                mae_val = 0.0
                acc20_val = 0.0

            val_metrics['loss'] += (loss_ddpm.item() + 0.1 * loss_cls.item())
            val_metrics['acc'] += acc
            val_metrics['mae'] += mae_val
            val_metrics['acc20'] += acc20_val
            val_metrics['count'] += 1

    if val_metrics['count'] > 0:
        avg_loss = val_metrics['loss'] / val_metrics['count']
        avg_acc = val_metrics['acc'] / val_metrics['count']
        avg_mae = val_metrics['mae'] / val_metrics['count']
        avg_acc20 = val_metrics['acc20'] / val_metrics['count']
        
        print(f"[VAL] Epoch {epoch:02d} | Loss={avg_loss:.4f} | Acc={avg_acc:.2%} | "
              f"A@20={avg_acc20:.2%} | MAE={avg_mae:.2f}m")

# ========== 7. 保存模型 ==========
torch.save({
    'transformer': transformer.state_dict(),
    'diffusion': diffusion.state_dict(),
    'config': {
        'preprocess_mode': PREPROCESS_MODE,
        'vocab_size': base_vocab,
        'd_model': d_model,
        'mask_id': mask_id,
        'pad_id': pad_id
    }
}, save_ckpt)

print(f"模型已保存至 {save_ckpt}")