import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill import DiffusionFill
import json  # 新增：保存索引

# ========== 1. 路径 ==========
data_pkl = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_pre/cluster_token_dataset.pkl"
os.makedirs("trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction", exist_ok=True)
save_ckpt = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction_mins/model_minutes.pt"

# ========== 2. 基础常量 ==========
meta = pd.read_pickle(data_pkl)
base_vocab = int(meta['vocab_size'])
num_users = int(meta['num_users']) if 'num_users' in meta else int(max(s['user_id'] for s in meta['sequences']) + 1)
mask_id = base_vocab
vocab_size = base_vocab + 1
# 新增：聚类中心 (用于 Token -> 经纬度 映射)
cluster_centers = torch.tensor(meta['cluster_centers'], dtype=torch.float32)  # [K,2] -> (lat, lon)
d_model = 512
batch_size = 64
lr = 1e-4
epochs = 50
mask_ratio = 0.15
L = 96
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 新增：Haversine 距离（米） ---
def haversine_m(latlon_a, latlon_b):
    # latlon_*: [N,2] (lat, lon) in degrees
    R = 6371000.0  # Earth radius (m)
    lat1 = torch.deg2rad(latlon_a[:, 0])
    lon1 = torch.deg2rad(latlon_a[:, 1])
    lat2 = torch.deg2rad(latlon_b[:, 0])
    lon2 = torch.deg2rad(latlon_b[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a.clamp(min=0.0, max=1.0)))
    return R * c  # [N]

# ========== 3. 数据 ==========
# 针对连续缺失 3-5 分钟的任务进行配置，启用 span_mask
dataset = MinuteTrajDataset(data_pkl, mask_ratio=mask_ratio, L=L, vocab_size=base_vocab,
                            span_mask=True, span_len_min=3, span_len_max=5)
# 划分训练/验证/测试（80%/10%/10%）
test_ratio = 0.1
val_ratio = 0.1
n_total = len(dataset)
n_test = int(n_total * test_ratio)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val - n_test
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

# 新增：保存划分索引到 SPLIT_DATASET
split_dir = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction_mins/SPLIT_DATASET"
os.makedirs(split_dir, exist_ok=True)
with open(os.path.join(split_dir, "train_indices.json"), "w") as f:
    json.dump(train_set.indices, f)
with open(os.path.join(split_dir, "val_indices.json"), "w") as f:
    json.dump(val_set.indices, f)
with open(os.path.join(split_dir, "test_indices.json"), "w") as f:
    json.dump(test_set.indices, f)
print(f"数据集划分索引已保存至 {split_dir}")

loader      = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
val_loader  = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

# ========== 4. 模型 & 共享嵌入 ==========
transformer = TransformerCond(vocab_size=vocab_size, num_users=num_users, embed_dim=d_model,
                              nhead=8, num_layers=6, mask_id=mask_id, max_len=L).to(device)
diffusion = DiffusionFill(d_cond=d_model, d_embed=d_model, vocab_size=base_vocab).to(device)  # 分类头输出基础簇数
# 共享 token 嵌入
diffusion.emb = transformer.token_emb.embedding

params = list(transformer.parameters()) + list(diffusion.parameters())
optimizer = torch.optim.AdamW(params, lr=lr)

# ========== 5. 训练循环 ==========
for epoch in range(epochs):
    transformer.train(); diffusion.train()
    for step, batch in enumerate(loader):
        token_id   = batch['token_id'].to(device)
        token_true = batch['token_true'].to(device)
        time_feat  = batch['time_feat'].to(device)
        user_id    = batch['user_id'].to(device)
        mask       = batch['mask'].to(device)

        # 条件
        z_t = transformer(token_id, time_feat, user_id)
        # 加噪
        x0 = diffusion.emb(token_true)
        t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=device)
        x_noisy, noise = diffusion.add_noise(x0, t)
        # 损失
        eps_pred = diffusion(x_noisy, z_t, t, mask)
        loss_ddpm = (F.mse_loss(eps_pred, noise, reduction='none') * mask.unsqueeze(-1)).sum() / mask.sum()

        # 可选分类辅助（用 x0_hat 近似重建）
        with torch.no_grad():
            a_t   = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)           # [B,L,1]
            am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) # [B,L,1]
            x0_hat = (x_noisy - am1_t * eps_pred) / a_t                      # [B,L,d]
        logits = diffusion.classifier(x0_hat)

        # 分类损失与准确率（仅针对掩码且非 mask_id 的位置）
        flat_logits  = logits.view(-1, logits.size(-1))   # [B*L, vocab]
        flat_targets = token_true.view(-1)                # [B*L]
        flat_mask    = mask.view(-1)                      # [B*L] bool
        valid_pos    = flat_mask & (flat_targets != mask_id)

        if valid_pos.any():
            loss_cls = F.cross_entropy(flat_logits[valid_pos], flat_targets[valid_pos], reduction='mean')
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # --- 计算分类准确率与 MAE/MSE（米） ---
        with torch.no_grad():
            if valid_pos.any():
                pred_tokens = flat_logits.argmax(dim=-1)          # [B*L]
                masked_preds = pred_tokens[valid_pos]             # ids
                masked_targets = flat_targets[valid_pos]          # ids
                # Accuracy
                correct = (masked_preds == masked_targets).sum().float()
                total = valid_pos.sum().float()
                accuracy = correct / total
                # Token -> (lat, lon)
                # 注意：cluster_centers 索引基于基础簇数 [0..base_vocab-1]
                pred_latlon = cluster_centers[masked_preds.clamp(max=base_vocab-1)].to(device)     # [N,2]
                true_latlon = cluster_centers[masked_targets.clamp(max=base_vocab-1)].to(device)   # [N,2]
                # MAE/MSE（米）
                dists_m = haversine_m(true_latlon, pred_latlon)                                  # [N]
                mae = dists_m.mean()                                                            # 平均绝对误差（米）
                mse = (dists_m ** 2).mean()                                                     # 均方误差（米^2）
            else:
                accuracy = torch.tensor(0.0, device=device)
                mae = torch.tensor(0.0, device=device)
                mse = torch.tensor(0.0, device=device)

        loss = loss_ddpm + 0.1 * loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch:02d} | Step {step:04d} | "
                  f"loss={loss.item():.4f} | loss_cls={loss_cls.item():.4f} | "
                  f"Acc={accuracy.item():.2%} | MAE(m)={mae.item():.3f} | MSE(m^2)={mse.item():.3f}")

    # ========== 6. 验证评估 ==========
    transformer.eval(); diffusion.eval()
    val_loss_sum = 0.0
    val_loss_cls_sum = 0.0
    val_acc_sum = 0.0
    val_mae_sum = 0.0
    val_mse_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for batch in val_loader:
            token_id   = batch['token_id'].to(device)
            token_true = batch['token_true'].to(device)
            time_feat  = batch['time_feat'].to(device)
            user_id    = batch['user_id'].to(device)
            mask       = batch['mask'].to(device)

            z_t = transformer(token_id, time_feat, user_id)
            x0 = diffusion.emb(token_true)
            t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=device)
            x_noisy, noise = diffusion.add_noise(x0, t)

            eps_pred = diffusion(x_noisy, z_t, t, mask)
            loss_ddpm = (F.mse_loss(eps_pred, noise, reduction='none') * mask.unsqueeze(-1)).sum() / mask.sum()

            a_t   = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
            am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x0_hat = (x_noisy - am1_t * eps_pred) / a_t
            logits = diffusion.classifier(x0_hat)

            flat_logits  = logits.view(-1, logits.size(-1))
            flat_targets = token_true.view(-1)
            flat_mask    = mask.view(-1)
            valid_pos    = flat_mask & (flat_targets != mask_id)

            if valid_pos.any():
                loss_cls = F.cross_entropy(flat_logits[valid_pos], flat_targets[valid_pos], reduction='mean')
                pred_tokens = flat_logits.argmax(dim=-1)
                masked_preds = pred_tokens[valid_pos]
                masked_targets = flat_targets[valid_pos]
                correct = (masked_preds == masked_targets).sum().float()
                total = valid_pos.sum().float()
                acc = (correct / total).item()
                pred_latlon = cluster_centers[masked_preds.clamp(max=base_vocab-1)].to(device)
                true_latlon = cluster_centers[masked_targets.clamp(max=base_vocab-1)].to(device)
                dists_m = haversine_m(true_latlon, pred_latlon)
                mae = dists_m.mean().item()
                mse = (dists_m ** 2).mean().item()
            else:
                loss_cls = torch.tensor(0.0, device=device)
                acc = 0.0
                mae = 0.0
                mse = 0.0

            val_loss_sum += (loss_ddpm + 0.1 * loss_cls).item()
            val_loss_cls_sum += loss_cls.item()
            val_acc_sum += acc
            val_mae_sum += mae
            val_mse_sum += mse
            val_count += 1

    avg_val_loss = val_loss_sum / max(val_count, 1)
    avg_val_loss_cls = val_loss_cls_sum / max(val_count, 1)
    avg_val_acc = val_acc_sum / max(val_count, 1)
    avg_val_mae = val_mae_sum / max(val_count, 1)
    avg_val_mse = val_mse_sum / max(val_count, 1)
    print(f"[VAL] Epoch {epoch:02d} | val_loss={avg_val_loss:.4f} | "
          f"val_loss_cls={avg_val_loss_cls:.4f} | val_acc={avg_val_acc:.2%} | "
          f"val_MAE(m)={avg_val_mae:.3f} | val_MSE(m^2)={avg_val_mse:.3f}")

# ========== 7. 测试评估 ==========
transformer.eval(); diffusion.eval()
test_loss_sum = 0.0
test_loss_cls_sum = 0.0
test_acc_sum = 0.0
test_mae_sum = 0.0
test_mse_sum = 0.0
test_count = 0
with torch.no_grad():
    for batch in test_loader:
        token_id   = batch['token_id'].to(device)
        token_true = batch['token_true'].to(device)
        time_feat  = batch['time_feat'].to(device)
        user_id    = batch['user_id'].to(device)
        mask       = batch['mask'].to(device)

        z_t = transformer(token_id, time_feat, user_id)
        x0 = diffusion.emb(token_true)
        t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=device)
        x_noisy, noise = diffusion.add_noise(x0, t)

        eps_pred = diffusion(x_noisy, z_t, t, mask)
        loss_ddpm = (F.mse_loss(eps_pred, noise, reduction='none') * mask.unsqueeze(-1)).sum() / mask.sum()

        a_t   = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
        am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x0_hat = (x_noisy - am1_t * eps_pred) / a_t
        logits = diffusion.classifier(x0_hat)

        flat_logits  = logits.view(-1, logits.size(-1))
        flat_targets = token_true.view(-1)
        flat_mask    = mask.view(-1)
        valid_pos    = flat_mask & (flat_targets != mask_id)

        if valid_pos.any():
            loss_cls = F.cross_entropy(flat_logits[valid_pos], flat_targets[valid_pos], reduction='mean')
            pred_tokens = flat_logits.argmax(dim=-1)
            masked_preds = pred_tokens[valid_pos]
            masked_targets = flat_targets[valid_pos]
            correct = (masked_preds == masked_targets).sum().float()
            total = valid_pos.sum().float()
            acc = (correct / total).item()
            pred_latlon = cluster_centers[masked_preds.clamp(max=base_vocab-1)].to(device)
            true_latlon = cluster_centers[masked_targets.clamp(max=base_vocab-1)].to(device)
            dists_m = haversine_m(true_latlon, pred_latlon)
            mae = dists_m.mean().item()
            mse = (dists_m ** 2).mean().item()
        else:
            loss_cls = torch.tensor(0.0, device=device)
            acc = 0.0
            mae = 0.0
            mse = 0.0

        test_loss_sum += (loss_ddpm + 0.1 * loss_cls).item()
        test_loss_cls_sum += loss_cls.item()
        test_acc_sum += acc
        test_mae_sum += mae
        test_mse_sum += mse
        test_count += 1

avg_test_loss = test_loss_sum / max(test_count, 1)
avg_test_loss_cls = test_loss_cls_sum / max(test_count, 1)
avg_test_acc = test_acc_sum / max(test_count, 1)
avg_test_mae = test_mae_sum / max(test_count, 1)
avg_test_mse = test_mse_sum / max(test_count, 1)
print(f"[TEST] avg_loss={avg_test_loss:.4f} | avg_loss_cls={avg_test_loss_cls:.4f} | "
      f"avg_acc={avg_test_acc:.2%} | avg_MAE(m)={avg_test_mae:.3f} | avg_MSE(m^2)={avg_test_mse:.3f}")

# ========== 8. 保存 ==========
torch.save({'transformer': transformer.state_dict(),
            'diffusion': diffusion.state_dict()}, save_ckpt)
print(f"模型已保存至 {save_ckpt}")