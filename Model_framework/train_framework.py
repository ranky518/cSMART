"""
消融实验训练脚本
对比：
  - TC-DPM (Ours): Transformer Context + Diffusion Generation
  - Transformer-Direct (Baseline): Transformer + 直接分类头
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset_frame import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill import DiffusionFill
import json
import numpy as np

torch.autograd.set_detect_anomaly(True)

# ========== 消融配置 ==========
ABLATION_DIRECT_CLASSIFIER = True  # True=Baseline, False=Ours(TC-DPM)
# ==============================

# 1. 路径配置
base_path = "/home/yanglanqi/trajectory/geolife_clustering-master"
data_pkl = os.path.join(base_path, "geolife_clustering-master/Model_pre/cluster_token_dataset.pkl")
suffix = "baseline_direct" if ABLATION_DIRECT_CLASSIFIER else "ours_diffusion"
out_dir = os.path.join(base_path, "Ablation/Model_framework/results", suffix)
os.makedirs(out_dir, exist_ok=True)
save_ckpt = os.path.join(out_dir, f"model_{suffix}.pt")

# 2. 数据配置
meta = pd.read_pickle(data_pkl)
base_vocab = int(meta['vocab_size'])
num_users = int(meta['num_users']) if 'num_users' in meta else int(max(s['user_id'] for s in meta['sequences']) + 1)
mask_id = base_vocab
pad_id = base_vocab + 1
vocab_size_total = base_vocab + 2  # base + mask + pad

cluster_centers = torch.tensor(meta['cluster_centers'], dtype=torch.float32)
d_model = 512
batch_size = 64
lr = 1e-4
epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"=" * 60)
print(f"Ablation Experiment: {suffix.upper()}")
print(f"Device: {device}")
print(f"Vocab Size: {base_vocab} (+mask+pad={vocab_size_total})")
print(f"=" * 60)


# 3. 距离函数
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


# 4. 数据加载
dataset = MinuteTrajDataset(data_pkl, mask_ratio=0.15, L=96, vocab_size=base_vocab,
                            span_mask=True, span_len_min=1, span_len_max=4, sample_step=10)

n_total = len(dataset)
n_test = int(n_total * 0.1)
n_val = int(n_total * 0.1)
n_train = n_total - n_val - n_test
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

# 保存划分
split_dir = os.path.join(out_dir, "data_split")
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


# 5. 模型初始化
transformer = TransformerCond(
    vocab_size=vocab_size_total, 
    num_users=num_users,
    embed_dim=d_model, 
    nhead=8, 
    num_layers=6,
    mask_id=mask_id, 
    max_len=96,
    use_direct_classifier=ABLATION_DIRECT_CLASSIFIER
).to(device)

if ABLATION_DIRECT_CLASSIFIER:
    diffusion = None
    params = list(transformer.parameters())
    print("Mode: Transformer-Direct (Baseline)")
else:
    diffusion = DiffusionFill(d_cond=d_model, d_embed=d_model, vocab_size=base_vocab).to(device)
    diffusion.emb = transformer.token_emb.embedding  # 共享嵌入
    params = list(transformer.parameters())
    param_ids = {id(p) for p in params}
    for p in diffusion.parameters():
        if id(p) not in param_ids:
            params.append(p)
    print("Mode: TC-DPM (Ours)")


# 权重初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

transformer.apply(init_weights)
if diffusion:
    diffusion.apply(init_weights)

optimizer = torch.optim.AdamW(params, lr=lr)
print(f"Total Parameters: {sum(p.numel() for p in params):,}")


# 6. 评估函数
def evaluate(model, diffusion, dataloader, device, cluster_centers, base_vocab, mask_id, pad_id, is_baseline):
    model.eval()
    if diffusion:
        diffusion.eval()
    
    total_loss = 0
    total_acc = 0
    total_mae = 0
    total_mse = 0
    total_acc20 = 0
    total_acc50 = 0
    total_acc100 = 0
    total_acc200 = 0
    total_acc500 = 0
    count = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            token_id = batch['token_id'].to(device).clamp(0, base_vocab + 1)
            token_true = batch['token_true'].to(device).clamp(0, base_vocab + 1)
            time_feat = batch['time_feat'].to(device)
            user_id = batch['user_id'].to(device)
            mask = batch['mask'].to(device)
            
            if is_baseline:
                logits = model(token_id, time_feat, user_id)
                logits = torch.clamp(logits, -100, 100)
            else:
                z_t = model(token_id, time_feat, user_id)
                x0 = diffusion.emb(token_true.clamp(0, base_vocab - 1))
                t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=device)
                x_noisy, noise = diffusion.add_noise(x0, t)
                eps_pred = diffusion(x_noisy, z_t, t, mask)
                
                a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
                am1 = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
                x0_hat = (x_noisy - am1 * eps_pred) / (a_t + 1e-6)
                x0_hat = torch.clamp(x0_hat, -10.0, 10.0)
                logits = diffusion.classifier(x0_hat)
            
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = token_true.view(-1)
            valid = mask.view(-1).bool() & (flat_targets != mask_id) & (flat_targets != pad_id) & (flat_targets < base_vocab)
            
            if valid.any():
                targets = flat_targets[valid]
                preds = flat_logits[valid, :base_vocab].argmax(dim=-1)
                
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())
                
                # Accuracy
                acc = (preds == targets).float().mean().item()
                
                # 距离计算
                pred_coords = cluster_centers[preds.clamp(0, base_vocab - 1).cpu()].to(device)
                true_coords = cluster_centers[targets.clamp(0, base_vocab - 1).cpu()].to(device)
                dists = haversine_m(true_coords, pred_coords)
                
                mae = dists.mean().item()
                mse = (dists ** 2).mean().item()
                acc20 = (dists <= 20).float().mean().item()
                acc50 = (dists <= 50).float().mean().item()
                acc100 = (dists <= 100).float().mean().item()
                acc200 = (dists <= 200).float().mean().item()
                acc500 = (dists <= 500).float().mean().item()
                
                total_acc += acc
                total_mae += mae
                total_mse += mse
                total_acc20 += acc20
                total_acc50 += acc50
                total_acc100 += acc100
                total_acc200 += acc200
                total_acc500 += acc500
                count += 1
    
    # 计算预测多样性（检验是否模型坍缩）
    unique_preds = len(set(all_preds)) if all_preds else 0
    unique_targets = len(set(all_targets)) if all_targets else 0
    
    return {
        'acc': total_acc / max(count, 1),
        'mae': total_mae / max(count, 1),
        'rmse': np.sqrt(total_mse / max(count, 1)),
        'acc20': total_acc20 / max(count, 1),
        'acc50': total_acc50 / max(count, 1),
        'acc100': total_acc100 / max(count, 1),
        'acc200': total_acc200 / max(count, 1),
        'acc500': total_acc500 / max(count, 1),
        'unique_preds': unique_preds,
        'unique_targets': unique_targets,
    }


# 7. 训练循环
best_val_acc = 0
history = []

for epoch in range(epochs):
    transformer.train()
    if diffusion:
        diffusion.train()
    
    epoch_loss = 0
    epoch_steps = 0
    
    for step, batch in enumerate(loader):
        token_id = batch['token_id'].to(device).clamp(0, vocab_size_total - 1)
        token_true = batch['token_true'].to(device).clamp(0, vocab_size_total - 1)
        time_feat = batch['time_feat'].to(device)
        user_id = batch['user_id'].to(device)
        mask = batch['mask'].to(device)
        
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if ABLATION_DIRECT_CLASSIFIER:
            # Baseline: 直接分类
            logits = transformer(token_id, time_feat, user_id)
            logits = torch.clamp(logits, -100, 100)
            
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = token_true.view(-1)
            valid = mask.view(-1).bool() & (flat_targets != mask_id) & (flat_targets != pad_id) & (flat_targets < base_vocab)
            
            if valid.any():
                loss = F.cross_entropy(flat_logits[valid, :base_vocab], flat_targets[valid])
        else:
            # Ours: Diffusion
            z_t = transformer(token_id, time_feat, user_id)
            if torch.isnan(z_t).any():
                continue
            
            x0 = diffusion.emb(token_true.clamp(0, base_vocab - 1))
            t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=device)
            x_noisy, noise = diffusion.add_noise(x0, t)
            eps_pred = diffusion(x_noisy, z_t, t, mask)
            
            mask_bool = mask.bool()
            if mask_bool.any():
                loss_ddpm = F.mse_loss(eps_pred[mask_bool], noise[mask_bool])
            else:
                loss_ddpm = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 辅助分类损失
            a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
            am1 = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x0_hat = (x_noisy - am1 * eps_pred) / (a_t + 1e-6)
            x0_hat = torch.clamp(x0_hat, -10.0, 10.0)
            logits = diffusion.classifier(x0_hat)
            
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = token_true.view(-1)
            valid = mask.view(-1).bool() & (flat_targets != mask_id) & (flat_targets != pad_id) & (flat_targets < base_vocab)
            
            loss_cls = torch.tensor(0.0, device=device)
            if valid.any():
                loss_cls = F.cross_entropy(flat_logits[valid], flat_targets[valid])
            
            loss = loss_ddpm + 0.1 * loss_cls
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_steps += 1
        
        if step % 100 == 0:
            print(f"Epoch {epoch:02d} | Step {step:04d} | Loss: {loss.item():.4f}")
    
    # 验证
    val_metrics = evaluate(transformer, diffusion, val_loader, device, cluster_centers, 
                          base_vocab, mask_id, pad_id, ABLATION_DIRECT_CLASSIFIER)
    
    print(f"[VAL] Epoch {epoch:02d} | "
          f"Acc={val_metrics['acc']:.2%} | "
          f"A@20={val_metrics['acc20']:.2%} | A@50={val_metrics['acc50']:.2%} | "
          f"A@100={val_metrics['acc100']:.2%} | A@200={val_metrics['acc200']:.2%} | A@500={val_metrics['acc500']:.2%} | "
          f"MAE={val_metrics['mae']:.2f}m | RMSE={val_metrics['rmse']:.2f}m | "
          f"UniqPred={val_metrics['unique_preds']}")
    
    history.append({
        'epoch': epoch,
        'train_loss': epoch_loss / max(epoch_steps, 1),
        **{f'val_{k}': v for k, v in val_metrics.items()}
    })
    
    if val_metrics['acc'] > best_val_acc:
        best_val_acc = val_metrics['acc']
        ckpt = {'transformer': transformer.state_dict()}
        if diffusion:
            ckpt['diffusion'] = diffusion.state_dict()
        torch.save(ckpt, save_ckpt)
        print(f"  >>> Best model saved (Acc={best_val_acc:.2%}) <<<")


# 8. 测试
print("\n" + "=" * 60)
print("Final Test Evaluation")
print("=" * 60)

ckpt = torch.load(save_ckpt)
transformer.load_state_dict(ckpt['transformer'])
if diffusion and 'diffusion' in ckpt:
    diffusion.load_state_dict(ckpt['diffusion'])

test_metrics = evaluate(transformer, diffusion, test_loader, device, cluster_centers,
                       base_vocab, mask_id, pad_id, ABLATION_DIRECT_CLASSIFIER)

print(f"[TEST] {suffix.upper()}")
print(f"  Accuracy: {test_metrics['acc']:.2%}")
print(f"  A@20m: {test_metrics['acc20']:.2%}")
print(f"  A@50m: {test_metrics['acc50']:.2%}")
print(f"  A@100m: {test_metrics['acc100']:.2%}")
print(f"  A@200m: {test_metrics['acc200']:.2%}")
print(f"  A@500m: {test_metrics['acc500']:.2%}")
print(f"  MAE: {test_metrics['mae']:.2f} m")
print(f"  RMSE: {test_metrics['rmse']:.2f} m")
print(f"  Unique Predictions: {test_metrics['unique_preds']} / {test_metrics['unique_targets']} targets")

# 保存结果
with open(os.path.join(out_dir, "test_results.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)
with open(os.path.join(out_dir, "training_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\nResults saved to: {out_dir}")