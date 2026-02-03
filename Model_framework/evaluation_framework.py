"""
消融实验评估脚本
对比 TC-DPM (Ours) vs Transformer-Direct (Baseline)
"""
import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
from dataset_frame import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill import DiffusionFill
from collections import Counter

# ========== 配置 ==========
ABLATION_DIRECT_CLASSIFIER = False  # 与训练一致
# ==========================

base_path = "/home/yanglanqi/trajectory/geolife_clustering-master"
suffix = "baseline_direct" if ABLATION_DIRECT_CLASSIFIER else "ours_diffusion"

MODEL_PATH = os.path.join(base_path, "Ablation/Model_framework/results", suffix, f"model_{suffix}.pt")
DATA_PKL = os.path.join(base_path, "geolife_clustering-master/Model_pre/cluster_token_dataset.pkl")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def haversine_np(c1, c2):
    """NumPy 版本的 Haversine 距离"""
    R = 6371000.0
    r1, r2 = np.radians(c1), np.radians(c2)
    d = r2 - r1
    a = np.sin(d[:, 0] / 2) ** 2 + np.cos(r1[:, 0]) * np.cos(r2[:, 0]) * np.sin(d[:, 1] / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def evaluate():
    print(f"=" * 60)
    print(f"Evaluating: {suffix.upper()}")
    print(f"=" * 60)
    
    # 1. 加载数据
    meta = pd.read_pickle(DATA_PKL)
    centers = np.array(meta['cluster_centers'])
    base_vocab = int(meta['vocab_size'])
    num_users = int(meta['num_users']) if 'num_users' in meta else 1
    mask_id = base_vocab
    pad_id = base_vocab + 1
    vocab_size_total = base_vocab + 2
    
    dataset = MinuteTrajDataset(DATA_PKL, mask_ratio=0.15, L=96, vocab_size=base_vocab,
                                span_mask=True, span_len_min=1, span_len_max=4, sample_step=10)
    
    # 加载测试集索引
    split_path = os.path.join(base_path, "Ablation/Model_framework/results", suffix, "data_split/test_indices.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            test_indices = json.load(f)
        subset = Subset(dataset, test_indices)
    else:
        # 默认取最后 10%
        n_test = int(len(dataset) * 0.1)
        subset = Subset(dataset, range(len(dataset) - n_test, len(dataset)))
    
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    print(f"Test samples: {len(subset)}")
    
    # 2. 加载模型
    transformer = TransformerCond(
        vocab_size=vocab_size_total, 
        num_users=num_users,
        embed_dim=512, 
        nhead=8, 
        num_layers=6,
        mask_id=mask_id, 
        max_len=96,
        use_direct_classifier=ABLATION_DIRECT_CLASSIFIER
    ).to(DEVICE)
    
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    transformer.load_state_dict(ckpt['transformer'])
    transformer.eval()
    
    diffusion = None
    if not ABLATION_DIRECT_CLASSIFIER and 'diffusion' in ckpt:
        diffusion = DiffusionFill(d_cond=512, d_embed=512, vocab_size=base_vocab).to(DEVICE)
        diffusion.emb = transformer.token_emb.embedding
        diffusion.load_state_dict(ckpt['diffusion'])
        diffusion.eval()

    # 3. 评估
    all_preds = []
    all_targets = []
    all_dists = []
    
    with torch.no_grad():
        for batch in loader:
            token_id = batch['token_id'].to(DEVICE).clamp(0, vocab_size_total - 1)
            token_true = batch['token_true'].to(DEVICE).clamp(0, vocab_size_total - 1)
            time_feat = batch['time_feat'].to(DEVICE)
            user_id = batch['user_id'].to(DEVICE)
            mask = batch['mask'].to(DEVICE).bool()
            
            if not mask.any():
                continue
            
            if ABLATION_DIRECT_CLASSIFIER:
                logits = transformer(token_id, time_feat, user_id)
                pred_tokens = logits[:, :, :base_vocab].argmax(dim=-1)
            else:
                z_t = transformer(token_id, time_feat, user_id)
                x0 = diffusion.emb(token_true.clamp(0, base_vocab - 1))
                t = torch.randint(0, diffusion.num_steps, (x0.size(0), x0.size(1)), device=DEVICE)
                x_noisy, _ = diffusion.add_noise(x0, t)
                eps = diffusion(x_noisy, z_t, t, mask.float())
                
                a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
                am1 = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
                x0_hat = (x_noisy - am1 * eps) / (a_t + 1e-6)
                pred_tokens = diffusion.classifier(x0_hat).argmax(dim=-1)
            
            # 只计算 mask 位置
            pred_np = pred_tokens.cpu().numpy()
            true_np = token_true.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            valid = mask_np & (true_np != mask_id) & (true_np != pad_id) & (true_np < base_vocab)
            
            if valid.any():
                preds = pred_np[valid].clip(0, base_vocab - 1)
                targets = true_np[valid].clip(0, base_vocab - 1)
                
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.tolist())
                
                p_coords = centers[preds]
                t_coords = centers[targets]
                dists = haversine_np(p_coords, t_coords)
                all_dists.extend(dists.tolist())
    
    # 4. 计算指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_dists = np.array(all_dists)
    
    acc = (all_preds == all_targets).mean()
    mae = all_dists.mean()
    rmse = np.sqrt((all_dists ** 2).mean())
    acc20 = (all_dists <= 20).mean()
    acc50 = (all_dists <= 50).mean()
    acc100 = (all_dists <= 100).mean()
    acc200 = (all_dists <= 200).mean()
    acc500 = (all_dists <= 500).mean()
    
    # 多样性分析
    pred_counter = Counter(all_preds.tolist())
    target_counter = Counter(all_targets.tolist())
    unique_preds = len(pred_counter)
    unique_targets = len(target_counter)
    
    # Top-5 预测分布（检查是否坍缩）
    top5_preds = pred_counter.most_common(5)
    top5_ratio = sum([c for _, c in top5_preds]) / len(all_preds) if len(all_preds) > 0 else 0
    
    print(f"\n{'='*40}")
    print(f"Results for {suffix.upper()}")
    print(f"{'='*40}")
    print(f"  Accuracy:     {acc:.2%}")
    print(f"  Acc@20m:      {acc20:.2%}")
    print(f"  Acc@50m:      {acc50:.2%}")
    print(f"  Acc@100m:     {acc100:.2%}")
    print(f"  Acc@200m:     {acc200:.2%}")
    print(f"  Acc@500m:     {acc500:.2%}")
    print(f"  MAE:          {mae:.2f} m")
    print(f"  RMSE:         {rmse:.2f} m")
    print(f"{'='*40}")
    print(f"Diversity Analysis:")
    print(f"  Unique Preds:   {unique_preds}")
    print(f"  Unique Targets: {unique_targets}")
    print(f"  Top-5 Pred Ratio: {top5_ratio:.2%} (越低越多样)")
    print(f"  Top-5 Preds: {top5_preds}")
    
    # 保存结果
    results = {
        'model': suffix,
        'accuracy': float(acc),
        'acc20': float(acc20),
        'acc50': float(acc50),
        'acc100': float(acc100),
        'acc200': float(acc200),
        'acc500': float(acc500),
        'mae': float(mae),
        'rmse': float(rmse),
        'unique_preds': unique_preds,
        'unique_targets': unique_targets,
        'top5_pred_ratio': float(top5_ratio),
    }
    
    out_file = os.path.join(base_path, "Ablation/Model_framework/results", suffix, "eval_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    evaluate()
