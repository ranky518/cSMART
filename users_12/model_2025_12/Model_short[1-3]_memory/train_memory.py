"""
记忆增强扩散模型训练脚本
Memory-Augmented Diffusion for Trajectory Imputation
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from datapreprocess import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill_memory import DiffusionFillWithMemory
from memory_bank import SpatioTemporalMemoryBank
import json

# ========== 1. 配置 ==========
data_pkl = "/home/yanglanqi/trajectory/zx_users_11/zx_user_11.pkl"
output_dir = "/home/yanglanqi/trajectory/zx_users_12/model_2025_12/Model_short[1-3]_memory"
os.makedirs(output_dir, exist_ok=True)
save_ckpt = os.path.join(output_dir, "model_memory.pt")

# ========== 2. 超参数 ==========
if not os.path.exists(data_pkl):
    raise FileNotFoundError(f"找不到数据集文件: {data_pkl}")

meta = pd.read_pickle(data_pkl)
base_vocab = int(meta['vocab_size'])
num_users = int(meta['num_users']) if 'num_users' in meta else int(max(s['user_id'] for s in meta['sequences']) + 1)
mask_id = base_vocab
vocab_size = base_vocab + 1
cluster_centers = torch.tensor(meta['cluster_centers'], dtype=torch.float32)

d_model = 512
batch_size = 64
lr = 1e-4
epochs = 50
L = 96
top_k_memory = 3  # 检索 Top-K 条历史轨迹
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}")
print(f"Vocab Size: {base_vocab}, Users: {num_users}")


# ========== 3. 距离函数 ==========
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


# ========== 4. 数据加载 ==========
dataset = MinuteTrajDataset(
    data_pkl, 
    mask_ratio=0.15, 
    L=L, 
    vocab_size=base_vocab,
    span_mask=True,
    span_len_min=1,
    span_len_max=3,
    span_count=1,
    sample_step=1
)

# 划分数据集
n_total = len(dataset)
n_test = int(n_total * 0.1)
n_val = int(n_total * 0.1)
n_train = n_total - n_val - n_test
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

# 保存划分索引
split_dir = os.path.join(output_dir, "SPLIT_DATASET")
os.makedirs(split_dir, exist_ok=True)
with open(os.path.join(split_dir, "train_indices.json"), "w") as f:
    json.dump(train_set.indices, f)
with open(os.path.join(split_dir, "val_indices.json"), "w") as f:
    json.dump(val_set.indices, f)
with open(os.path.join(split_dir, "test_indices.json"), "w") as f:
    json.dump(test_set.indices, f)

loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")


# ========== 5. 构建记忆库 ==========
print("Building Memory Bank from training data...")
memory_bank = SpatioTemporalMemoryBank(time_slots=6)
memory_bank.build_from_dataset(dataset)


# ========== 6. 记忆检索函数 ==========
def retrieve_memory_batch(batch, memory_bank, dataset, top_k=3, max_len=96):
    """
    为一个 batch 检索历史轨迹记忆
    
    Returns:
        memory_tokens: [B, K, L]
        memory_masks: [B, K, L]
    """
    B = batch['user_id'].shape[0]
    device = batch['user_id'].device
    
    memory_tokens = torch.zeros(B, top_k, max_len, dtype=torch.long, device=device)
    memory_masks = torch.zeros(B, top_k, max_len, dtype=torch.bool, device=device)
    
    for i in range(B):
        user_id = batch['user_id'][i].item()
        start_station = batch['start_station'][i].item()
        end_station = batch['end_station'][i].item()
        time_feat = batch['time_feat'][i]
        
        # 推断时段
        mid_time = time_feat[len(time_feat) // 2].item() if len(time_feat) > 0 else 0.5
        time_slot = memory_bank.get_time_slot(mid_time)
        weekday = batch['idx'][i].item() % 7  # 简化处理
        
        # 检索
        retrieved = memory_bank.retrieve(
            user_id, weekday, time_slot, 
            start_station, end_station, 
            top_k=top_k
        )
        
        # 填充到 tensor
        for j, entry in enumerate(retrieved):
            traj_tokens = entry['tokens'][:max_len]
            traj_len = len(traj_tokens)
            memory_tokens[i, j, :traj_len] = torch.tensor(traj_tokens, dtype=torch.long)
            memory_masks[i, j, :traj_len] = True
    
    return memory_tokens, memory_masks


# ========== 7. 模型初始化 ==========
transformer = TransformerCond(
    vocab_size=vocab_size, 
    num_users=num_users, 
    embed_dim=d_model,
    nhead=8, 
    num_layers=6, 
    mask_id=mask_id, 
    max_len=L
).to(device)

diffusion = DiffusionFillWithMemory(
    d_cond=d_model, 
    d_embed=d_model, 
    vocab_size=base_vocab,
    use_memory_gating=True
).to(device)

# 共享 token 嵌入
diffusion.emb = transformer.token_emb.embedding
diffusion.memory_encoder.token_emb = transformer.token_emb

# 参数收集
params = list(transformer.parameters())
param_ids = {id(p) for p in params}
for p in diffusion.parameters():
    if id(p) not in param_ids:
        params.append(p)

optimizer = torch.optim.AdamW(params, lr=lr)
print(f"Total Parameters: {sum(p.numel() for p in params):,}")


# ========== 8. 训练循环 ==========
for epoch in range(epochs):
    transformer.train()
    diffusion.train()
    
    for step, batch in enumerate(loader):
        token_id = batch['token_id'].to(device)
        token_true = batch['token_true'].to(device)
        time_feat = batch['time_feat'].to(device)
        user_id = batch['user_id'].to(device)
        mask = batch['mask'].to(device)
        
        # 检索记忆
        memory_tokens, memory_masks = retrieve_memory_batch(batch, memory_bank, dataset, top_k=top_k_memory, max_len=L)
        memory_tokens = memory_tokens.to(device)
        memory_masks = memory_masks.to(device)
        
        # Transformer 编码
        z_cond = transformer(token_id, time_feat, user_id)
        
        # 记忆编码
        z_memory = diffusion.encode_memory(memory_tokens, memory_masks, z_cond)
        
        # 扩散训练
        x0 = diffusion.emb(token_true)
        t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=device)
        x_noisy, noise = diffusion.add_noise(x0, t)
        
        eps_pred = diffusion(x_noisy, z_cond, z_memory, t, mask.float())
        
        # 损失
        loss_ddpm = (F.mse_loss(eps_pred, noise, reduction='none') * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
        
        # 辅助分类损失
        with torch.no_grad():
            a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
            am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x0_hat = (x_noisy - am1_t * eps_pred) / (a_t + 1e-6)
        
        logits = diffusion.classifier(x0_hat)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = token_true.view(-1)
        flat_mask = mask.view(-1)
        valid_pos = flat_mask & (flat_targets != mask_id) & (flat_targets < base_vocab)
        
        if valid_pos.any():
            loss_cls = F.cross_entropy(flat_logits[valid_pos], flat_targets[valid_pos], reduction='mean')
            
            with torch.no_grad():
                pred_tokens = flat_logits.argmax(dim=-1)
                masked_preds = pred_tokens[valid_pos]
                masked_targets = flat_targets[valid_pos]
                accuracy = (masked_preds == masked_targets).float().mean()
                
                pred_latlon = cluster_centers[masked_preds.clamp(max=base_vocab-1)].to(device)
                true_latlon = cluster_centers[masked_targets.clamp(max=base_vocab-1)].to(device)
                dists_m = haversine_m(true_latlon, pred_latlon)
                mae = dists_m.mean()
                rmse = torch.sqrt((dists_m ** 2).mean())
                
                acc20 = (dists_m <= 20.0).float().mean()
                acc50 = (dists_m <= 50.0).float().mean()
                acc100 = (dists_m <= 100.0).float().mean()
                acc200 = (dists_m <= 200.0).float().mean()
                acc500 = (dists_m <= 500.0).float().mean()
        else:
            loss_cls = torch.tensor(0.0, device=device)
            accuracy = mae = rmse = acc20 = acc50 = acc100 = acc200 = acc500 = torch.tensor(0.0, device=device)
        
        loss = loss_ddpm + 0.1 * loss_cls
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch:02d} | Step {step:04d} | "
                  f"loss={loss.item():.4f} | loss_cls={loss_cls.item():.4f} | "
                  f"Acc={accuracy.item():.2%} | A@20={acc20.item():.2%} | A@50={acc50.item():.2%} | "
                  f"A@100={acc100.item():.2%} | A@200={acc200.item():.2%} | A@500={acc500.item():.2%} | "
                  f"MAE={mae.item():.3f}m | RMSE={rmse.item():.3f}m")
    
    # ========== 验证 ==========
    transformer.eval()
    diffusion.eval()
    val_loss_sum = 0.0; val_loss_cls_sum = 0.0
    val_acc_sum = 0.0; val_mae_sum = 0.0; val_rmse_sum = 0.0
    val_acc20_sum = 0.0; val_acc50_sum = 0.0; val_acc100_sum = 0.0
    val_acc200_sum = 0.0; val_acc500_sum = 0.0
    val_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            token_id = batch['token_id'].to(device)
            token_true = batch['token_true'].to(device)
            time_feat = batch['time_feat'].to(device)
            user_id = batch['user_id'].to(device)
            mask = batch['mask'].to(device)
            
            memory_tokens, memory_masks = retrieve_memory_batch(batch, memory_bank, dataset, top_k=top_k_memory, max_len=L)
            memory_tokens = memory_tokens.to(device)
            memory_masks = memory_masks.to(device)
            
            z_cond = transformer(token_id, time_feat, user_id)
            z_memory = diffusion.encode_memory(memory_tokens, memory_masks, z_cond)
            
            x0 = diffusion.emb(token_true)
            t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=device)
            x_noisy, noise = diffusion.add_noise(x0, t)
            eps_pred = diffusion(x_noisy, z_cond, z_memory, t, mask.float())
            loss_ddpm = (F.mse_loss(eps_pred, noise, reduction='none') * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
            
            a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
            am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x0_hat = (x_noisy - am1_t * eps_pred) / (a_t + 1e-6)
            logits = diffusion.classifier(x0_hat)
            
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = token_true.view(-1)
            flat_mask = mask.view(-1)
            valid_pos = flat_mask & (flat_targets != mask_id) & (flat_targets < base_vocab)
            
            if valid_pos.any():
                loss_cls = F.cross_entropy(flat_logits[valid_pos], flat_targets[valid_pos], reduction='mean')
                pred_tokens = flat_logits.argmax(dim=-1)
                masked_preds = pred_tokens[valid_pos]
                masked_targets = flat_targets[valid_pos]
                acc = (masked_preds == masked_targets).float().mean().item()
                
                pred_latlon = cluster_centers[masked_preds.clamp(max=base_vocab-1)].to(device)
                true_latlon = cluster_centers[masked_targets.clamp(max=base_vocab-1)].to(device)
                dists_m = haversine_m(true_latlon, pred_latlon)
                mae = dists_m.mean().item()
                rmse = torch.sqrt((dists_m ** 2).mean()).item()
                
                acc20 = (dists_m <= 20.0).float().mean().item()
                acc50 = (dists_m <= 50.0).float().mean().item()
                acc100 = (dists_m <= 100.0).float().mean().item()
                acc200 = (dists_m <= 200.0).float().mean().item()
                acc500 = (dists_m <= 500.0).float().mean().item()
            else:
                 loss_cls = torch.tensor(0.0, device=device)
                 acc = 0.0; mae = 0.0; rmse = 0.0
                 acc20 = 0.0; acc50 = 0.0; acc100 = 0.0; acc200 = 0.0; acc500 = 0.0
            
            val_loss_sum += (loss_ddpm + 0.1 * loss_cls).item()
            val_loss_cls_sum += loss_cls.item()
            val_acc_sum += acc
            val_mae_sum += mae
            val_rmse_sum += rmse
            val_acc20_sum += acc20
            val_acc50_sum += acc50
            val_acc100_sum += acc100
            val_acc200_sum += acc200
            val_acc500_sum += acc500
            val_count += 1
            
    avg_val_loss = val_loss_sum / max(val_count, 1)
    avg_val_loss_cls = val_loss_cls_sum / max(val_count, 1)
    avg_val_acc = val_acc_sum / max(val_count, 1)
    avg_val_mae = val_mae_sum / max(val_count, 1)
    avg_val_rmse = val_rmse_sum / max(val_count, 1)
    avg_val_acc20 = val_acc20_sum / max(val_count, 1)
    avg_val_acc50 = val_acc50_sum / max(val_count, 1)
    avg_val_acc100 = val_acc100_sum / max(val_count, 1)
    avg_val_acc200 = val_acc200_sum / max(val_count, 1)
    avg_val_acc500 = val_acc500_sum / max(val_count, 1)
    
    print(f"[VAL] Epoch {epoch:02d} | val_loss={avg_val_loss:.4f} | val_loss_cls={avg_val_loss_cls:.4f} | "
          f"Acc={avg_val_acc:.2%} | A@20={avg_val_acc20:.2%} | A@50={avg_val_acc50:.2%} | "
          f"A@100={avg_val_acc100:.2%} | A@200={avg_val_acc200:.2%} | A@500={avg_val_acc500:.2%} | "
          f"MAE={avg_val_mae:.3f}m | RMSE={avg_val_rmse:.3f}m")

# ========== 9. 测试 ==========
# ... (类似验证逻辑)

# ========== 10. 保存 ==========
torch.save({
    'transformer': transformer.state_dict(),
    'diffusion': diffusion.state_dict()
}, save_ckpt)
print(f"模型已保存至 {save_ckpt}")
