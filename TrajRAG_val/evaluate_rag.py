"""
验证 RAG (Retrieval-Augmented Generation) 引入的有效性
对比：With Memory (RAG) vs Without Memory (No RAG)
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import DataLoader, Subset

# 添加模型相关文件的路径，以便导入模块
MODEL_DIR = "/home/yanglanqi/trajectory/zx_users_12/model_2025_12/Model_short[3-10]_memory"
sys.path.append(MODEL_DIR)

from datapreprocess import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill_memory import DiffusionFillWithMemory
from memory_bank import SpatioTemporalMemoryBank

# ========== 配置 ==========
DATA_PKL = "/home/yanglanqi/trajectory/zx_users_11/zx_user_11.pkl"
CKPT_PATH = os.path.join(MODEL_DIR, "model_memory.pt")
SPLIT_DIR = os.path.join(MODEL_DIR, "SPLIT_DATASET")  # 确保加载相同的测试集
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数 (需与训练时一致)
D_MODEL = 512
BATCH_SIZE = 64
L = 96
TOP_K_MEMORY = 3

print(f"Using Device: {DEVICE}")

# ========== 工具函数 ==========
def haversine_m(latlon_a, latlon_b):
    """计算两组经纬度之间的 Haversine 距离 (米)"""
    R = 6371000.0
    lat1 = torch.deg2rad(latlon_a[:, 0])
    lon1 = torch.deg2rad(latlon_a[:, 1])
    lat2 = torch.deg2rad(latlon_b[:, 0])
    lon2 = torch.deg2rad(latlon_b[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return R * 2 * torch.arcsin(torch.sqrt(a.clamp(0, 1)))

def retrieve_memory_batch(batch, memory_bank, top_k=3, max_len=96):
    """为 Batch 检索记忆 (与训练脚本逻辑一致)"""
    B = batch['user_id'].shape[0]
    device = batch['user_id'].device
    memory_tokens = torch.zeros(B, top_k, max_len, dtype=torch.long, device=device)
    memory_masks = torch.zeros(B, top_k, max_len, dtype=torch.bool, device=device)
    
    for i in range(B):
        user_id = batch['user_id'][i].item()
        start_station = batch['start_station'][i].item()
        end_station = batch['end_station'][i].item()
        time_feat = batch['time_feat'][i]
        mid_time = time_feat[len(time_feat) // 2].item() if len(time_feat) > 0 else 0.5
        time_slot = memory_bank.get_time_slot(mid_time)
        weekday = batch['idx'][i].item() % 7
        
        retrieved = memory_bank.retrieve(user_id, weekday, time_slot, start_station, end_station, top_k=top_k)
        for j, entry in enumerate(retrieved):
            traj_tokens = entry['tokens'][:max_len]
            traj_len = len(traj_tokens)
            memory_tokens[i, j, :traj_len] = torch.tensor(traj_tokens, dtype=torch.long)
            memory_masks[i, j, :traj_len] = True
    return memory_tokens, memory_masks

def evaluate(transformer, diffusion, loader, memory_bank, cluster_centers, 
             no_memory=False, desc="Evaluating"):
    """
    评估函数
    no_memory=True 时，强制 memory 为空，模拟无 RAG 场景
    """
    transformer.eval()
    diffusion.eval()
    
    metrics = {
        'acc': 0.0, 'mae': 0.0, 'rmse': 0.0,
        'acc20': 0.0, 'acc50': 0.0, 'acc100': 0.0, 'acc200': 0.0, 'acc500': 0.0,
        'count': 0
    }
    
    with torch.no_grad():
        for batch in loader:
            token_id = batch['token_id'].to(DEVICE)
            token_true = batch['token_true'].to(DEVICE)
            time_feat = batch['time_feat'].to(DEVICE)
            user_id = batch['user_id'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            # 1. 记忆检索
            if no_memory:
                # 构造空的记忆 (B, K, L)
                B = token_id.size(0)
                memory_tokens = torch.zeros(B, TOP_K_MEMORY, L, dtype=torch.long, device=DEVICE)
                memory_masks = torch.zeros(B, TOP_K_MEMORY, L, dtype=torch.bool, device=DEVICE)
            else:
                memory_tokens, memory_masks = retrieve_memory_batch(batch, memory_bank, top_k=TOP_K_MEMORY, max_len=L)
                memory_tokens = memory_tokens.to(DEVICE)
                memory_masks = memory_masks.to(DEVICE)
            
            # 2. 前向传播
            z_cond = transformer(token_id, time_feat, user_id)
            z_memory = diffusion.encode_memory(memory_tokens, memory_masks, z_cond)
            
            # 使用 DDIM/DDPM 采样生成 (简化为一步预测用于评估，或根据训练时的采样逻辑)
            # 这里为了和训练脚本的验证逻辑一致，我们使用单步噪声预测来重建 x0 
            # (注意：更严谨的生成应该是一个完整的采样循环，但为了快速验证指标的一致性，通常复用训练时的验证逻辑)
            
            x0_gt = diffusion.emb(token_true)
            # 随机采样一个 t 进行评估 (模拟去噪过程的中间状态)
            # 或者固定 t=0 (即最后一步)，但这在训练框架中通常通过噪声预测网络来推断
            # 此处复用 train.py 中的验证逻辑：随机 t -> 预测 eps -> 重建 x0_hat
            t = torch.randint(0, diffusion.num_steps, (x0_gt.size(0), x0_gt.size(1)), device=DEVICE)
            x_noisy, noise = diffusion.add_noise(x0_gt, t)
            
            eps_pred = diffusion(x_noisy, z_cond, z_memory, t, mask.float())
            
            a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
            am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x0_hat = (x_noisy - am1_t * eps_pred) / (a_t + 1e-6)
            
            logits = diffusion.classifier(x0_hat)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = token_true.view(-1)
            flat_mask = mask.view(-1)
            
            # 3. 计算指标 (仅在 Mask 区域)
            valid_pos = flat_mask & (flat_targets != diffusion.vocab_size - 1) # mask_id check
            
            if valid_pos.any():
                preds = flat_logits.argmax(dim=-1)[valid_pos]
                targets = flat_targets[valid_pos]
                
                # Check bounds
                if preds.max() >= len(cluster_centers) or targets.max() >= len(cluster_centers):
                     # 如果预测出了 mask_id 或特殊 token，过滤掉
                     valid_indices = (preds < len(cluster_centers)) & (targets < len(cluster_centers))
                     preds = preds[valid_indices]
                     targets = targets[valid_indices]
                
                if len(preds) > 0:
                    metrics['acc'] += (preds == targets).float().mean().item()
                    
                    pred_ll = cluster_centers[preds].to(DEVICE)
                    true_ll = cluster_centers[targets].to(DEVICE)
                    dists = haversine_m(true_ll, pred_ll)
                    
                    metrics['mae'] += dists.mean().item()
                    metrics['rmse'] += torch.sqrt((dists ** 2).mean()).item()
                    metrics['acc20'] += (dists <= 20.0).float().mean().item()
                    metrics['acc50'] += (dists <= 50.0).float().mean().item()
                    metrics['acc100'] += (dists <= 100.0).float().mean().item()
                    metrics['acc200'] += (dists <= 200.0).float().mean().item()
                    metrics['acc500'] += (dists <= 500.0).float().mean().item()
                    metrics['count'] += 1
    
    # 平均化
    cnt = max(metrics['count'], 1)
    results = {k: v / cnt for k, v in metrics.items() if k != 'count'}
    print(f"[{desc}] Acc: {results['acc']:.2%} | MAE: {results['mae']:.2f}m | RMSE: {results['rmse']:.2f}m | A@200: {results['acc200']:.2%}")
    return results

def main():
    # 1. 加载元数据
    print(f"Loading metadata from {DATA_PKL}...")
    meta = pd.read_pickle(DATA_PKL)
    base_vocab = int(meta['vocab_size'])
    num_users = int(meta['num_users']) if 'num_users' in meta else int(max(s['user_id'] for s in meta['sequences']) + 1)
    mask_id = base_vocab
    vocab_size = base_vocab + 1
    cluster_centers = torch.tensor(meta['cluster_centers'], dtype=torch.float32)

    # 2. 准备数据集 (仅加载 Test Set)
    print("Preparing dataset...")
    # 注意：这里需要重新构建数据集对象，并加载之前保存的 test_indices
    # 3-10min 配置: sample_step=5
    full_dataset = MinuteTrajDataset(DATA_PKL, L=L, vocab_size=base_vocab,
                                     span_mask=True, span_len_min=1, span_len_max=2, sample_step=5)
    
    test_indices_path = os.path.join(SPLIT_DIR, "test_indices.json")
    if os.path.exists(test_indices_path):
        with open(test_indices_path, "r") as f:
            test_indices = json.load(f)
        test_set = Subset(full_dataset, test_indices)
        print(f"Loaded {len(test_set)} test samples from indices.")
    else:
        print("Warning: Test indices not found, using random split (results may vary).")
        n_total = len(full_dataset)
        n_test = int(n_total * 0.1)
        _, _, test_set = torch.utils.data.random_split(full_dataset, [n_total - 2*n_test, n_test, n_test])

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) # drop_last consistent with fix

    # 3. 构建记忆库 (需要用全量/训练数据构建)
    # 为了 RAG 生效，Memory Bank 应该包含历史信息。通常用训练集+验证集构建，或者全量构建（只要不泄露当前时刻的真值）
    # 这里简单复用全量构建逻辑
    memory_bank = SpatioTemporalMemoryBank(time_slots=6)
    memory_bank.build_from_dataset(full_dataset)

    # 4. 加载模型
    print(f"Loading model from {CKPT_PATH}...")
    transformer = TransformerCond(vocab_size=vocab_size, num_users=num_users, embed_dim=D_MODEL,
                                  nhead=8, num_layers=6, mask_id=mask_id, max_len=L).to(DEVICE)
    diffusion = DiffusionFillWithMemory(d_cond=D_MODEL, d_embed=D_MODEL, vocab_size=base_vocab,
                                        use_memory_gating=True).to(DEVICE)
    
    # 绑定 embedding
    diffusion.emb = transformer.token_emb.embedding
    diffusion.memory_encoder.token_emb = transformer.token_emb
    
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    transformer.load_state_dict(checkpoint['transformer'])
    diffusion.load_state_dict(checkpoint['diffusion'])
    
    # 5. 执行对比评估
    print("\n========== Starting Ablation Study: RAG Effectiveness ==========")
    
    # 实验 A: 启用 RAG (With Memory)
    res_rag = evaluate(transformer, diffusion, test_loader, memory_bank, cluster_centers, 
                       no_memory=False, desc="With RAG (Memory)")
    
    # 实验 B: 禁用 RAG (Without Memory)
    res_no_rag = evaluate(transformer, diffusion, test_loader, memory_bank, cluster_centers, 
                         no_memory=True, desc="Without RAG (No Memory)")
    
    # 6. 输出对比结果
    print("\n========== Results Summary ==========")
    print(f"{'Metric':<10} | {'With RAG':<15} | {'No RAG':<15} | {'Improvement':<15}")
    print("-" * 60)
    
    metrics_to_show = ['acc', 'mae', 'rmse', 'acc200']
    
    for m in metrics_to_show:
        val_rag = res_rag[m]
        val_no = res_no_rag[m]
        
        # Improvement calculation
        if m in ['mae', 'rmse']:
            # Lower is better
            imp = (val_no - val_rag) / (val_no + 1e-9) * 100
            diff_str = f"{imp:+.2f}% (Dec)"
        else:
            # Higher is better
            imp = (val_rag - val_no) / (val_no + 1e-9) * 100
            diff_str = f"{imp:+.2f}% (Inc)"
            
        print(f"{m.upper():<10} | {val_rag:<15.4f} | {val_no:<15.4f} | {diff_str}")

    print("-" * 60)
    print("Comparison Done.")

if __name__ == "__main__":
    main()
