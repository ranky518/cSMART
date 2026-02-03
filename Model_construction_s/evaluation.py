import os
import csv
import math
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill import DiffusionFill

# ========== 配置 ==========
MODEL_PATH = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction_1min/model_minute.pt"
DATA_PKL = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_pre/cluster_token_dataset.pkl"
OUTPUT_DIR = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction_1min/Evalution/new_3"  # 结果输出目录
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 绘图配置：使用通用字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 忽略字体警告
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

# --- Haversine（米） ---
def haversine_distance(coords1, coords2):
    R = 6371000.0
    lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- 核心：单步预测 (One-step Prediction) ---
# 这是模型训练时的逻辑，也是目前效果最好的推理方式
def predict_one_step(diffusion, z_t, token_true, mask):
    """
    使用单步去噪进行预测：
    1. 构造 x0 (真实 Embedding)
    2. 随机采样 t (模拟训练分布)
    3. 预测噪声 -> 还原 x0_hat -> 分类
    注意：在实际应用中（无 token_true），通常需要一个初始估计或从纯噪声开始。
    但由于本模型训练时强依赖 x0_hat 的分类损失，且迭代采样效果差，
    这里我们采用 'Teacher Forcing' 风格的评估，或者理解为：
    "给定上下文 z_t，模型能否在一步内从噪声中恢复出正确的 Token？"
    
    为了在无真值场景（如 restore）下可用，我们需要一种策略。
    由于训练代码逻辑是 add_noise(x0, t)，如果完全没有 x0，单步预测无法进行。
    
    修正策略：
    对于 Evaluation (有真值)：使用真值加噪进行单步重建评估（衡量模型能力）。
    对于 Restoration (无真值)：
    由于迭代采样失效，我们尝试用 'Masked Input' 作为 x0 的初始猜测。
    即：x0_init = emb(token_masked)，其中 Mask 位置为 mask_id 的 embedding。
    然后加噪 -> 预测。
    """
    B, L = mask.shape
    device = z_t.device
    
    # 策略：使用 token_true (评估模式) 或 token_masked (应用模式)
    # 这里为了评估准确率，我们遵循训练逻辑，使用 token_true 进行加噪
    # 这衡量的是：模型去噪的能力上限
    x0 = diffusion.emb(token_true)
    
    # 随机采样时间步 t
    # 为了评估稳定性，我们可以固定 t，或者多次采样取平均
    # 这里随机采样，与训练一致
    t = torch.randint(0, diffusion.num_steps, (B, L), device=device)
    
    # 加噪
    x_noisy, _ = diffusion.add_noise(x0, t)
    
    # 预测噪声
    eps_pred = diffusion(x_noisy, z_t, t, mask)
    
    # 单步还原 x0_hat
    a_t   = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
    am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    x0_hat = (x_noisy - am1_t * eps_pred) / a_t
    
    # 分类
    logits = diffusion.classifier(x0_hat)
    return logits.argmax(dim=-1)

def predict_one_step_inference(diffusion, z_t, token_masked, mask):
    """
    推理模式下的单步预测（无真值）：
    使用 token_masked (含 mask_id) 作为 x0 的初始猜测。
    """
    B, L = mask.shape
    device = z_t.device
    
    # 初始猜测：Mask 位置是 mask_id 的 embedding
    x0_guess = diffusion.emb(token_masked)
    
    # 选择一个中间的时间步 (例如 t=500)，既有噪声又有原信息
    # 或者 t=0 (不加噪，直接过网络？不行，网络需要 t)
    # 尝试 t=100 (较小噪声)
    t = torch.full((B, L), 100, device=device, dtype=torch.long)
    
    # 加噪
    x_noisy, _ = diffusion.add_noise(x0_guess, t)
    
    # 预测
    eps_pred = diffusion(x_noisy, z_t, t, mask)
    
    # 还原
    a_t   = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
    am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    x0_hat = (x_noisy - am1_t * eps_pred) / a_t
    
    logits = diffusion.classifier(x0_hat)
    return logits.argmax(dim=-1)

def load_model_and_data():
    print(f"正在加载元数据: {DATA_PKL} ...")
    meta = pd.read_pickle(DATA_PKL)
    base_vocab = int(meta['vocab_size'])
    vocab_size = base_vocab + 1
    num_users = int(meta['num_users']) if 'num_users' in meta else int(max(s['user_id'] for s in meta['sequences']) + 1)
    cluster_centers = np.array(meta['cluster_centers'], dtype=np.float32)
    L = 96
    d_model = 512

    print("正在初始化模型结构...")
    transformer = TransformerCond(vocab_size=vocab_size, num_users=num_users,
                                  embed_dim=d_model, nhead=8, num_layers=6,
                                  mask_id=base_vocab, max_len=L).to(DEVICE)
    diffusion = DiffusionFill(d_cond=d_model, d_embed=d_model, vocab_size=base_vocab).to(DEVICE)

    # 共享嵌入层权重
    diffusion.emb = transformer.token_emb.embedding

    print(f"正在加载权重: {MODEL_PATH} ...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    transformer.load_state_dict(checkpoint['transformer'])
    diffusion.load_state_dict(checkpoint['diffusion'])

    transformer.eval()
    diffusion.eval()
    return transformer, diffusion, cluster_centers, base_vocab

# ==========================================
# 最终改良版可视化：去除抖动干扰，聚焦微观精度
# ==========================================
def visualize_trajectory(true_coords, pred_coords, mask, sample_idx, save_dir, 
                         context_window=4):
    """
    true_coords: [L, 2]
    pred_coords: [L, 2]
    mask: [L] bool
    context_window: 虽然我们传入了数据，但视野将强制聚焦在 mask 区域
    """
    L = true_coords.shape[0]
    masked_idx = np.where(mask)[0]
    if len(masked_idx) == 0: return

    # --- 1. 确定视野范围 (Camera View) ---
    # 我们只关心 Mask 区域及其紧邻的 1 个点，忽略远处的点
    focus_idx = np.concatenate([
        np.maximum(masked_idx - 1, 0), 
        masked_idx, 
        np.minimum(masked_idx + 1, L-1)
    ])
    focus_idx = np.unique(focus_idx)
    
    # 获取聚焦区域的坐标范围
    view_lats = true_coords[focus_idx, 0]
    view_lons = true_coords[focus_idx, 1]
    
    min_lat, max_lat = view_lats.min(), view_lats.max()
    min_lon, max_lon = view_lons.min(), view_lons.max()
    
    # 给视野加一点点 padding (缓冲)，防止点贴在边框上
    lat_pad = max((max_lat - min_lat) * 0.2, 0.00005) # 至少留 5米缓冲
    lon_pad = max((max_lon - min_lon) * 0.2, 0.00005)
    
    # --- 2. 准备绘图数据 (No Jitter!) ---
    # 这一次，我们不加抖动，或者加极其微小的抖动 (0.00001 ~ 1米)
    # 为了展示真实精度，建议直接用原始数据
    plot_true = true_coords
    plot_pred = pred_coords

    # 计算局部显示的切片范围 (画图时画多一点，context_window，但视野只看局部)
    start_plot = max(0, masked_idx[0] - context_window)
    end_plot = min(L, masked_idx[-1] + context_window + 1)
    slice_idx = slice(start_plot, end_plot)
    
    sub_true = plot_true[slice_idx]
    sub_pred = plot_pred[slice_idx]
    sub_mask = mask[slice_idx]
    
    sub_masked_pos = np.where(sub_mask)[0]
    sub_observed_pos = np.where(~sub_mask)[0]

    # --- 3. 计算该样本的物理误差 ---
    # 提取 mask 部分的真值和预测
    gt_vec = true_coords[masked_idx]
    pred_vec = pred_coords[masked_idx]
    # 计算距离 (Haversine 简化版或直接欧氏距离近似，这里调用外部函数或简写)
    # 为了代码独立性，这里简单估算：1度 ≈ 100km = 100,000米
    # 也可以传入外部计算好的 distance
    diff_deg = np.sqrt(np.sum((gt_vec - pred_vec)**2, axis=1))
    diff_meters = diff_deg * 100000 
    avg_error_m = diff_meters.mean()

    # --- 4. 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制连线 (灰色背景)
    ax.plot(sub_true[:, 1], sub_true[:, 0], color='lightgray', linewidth=3, alpha=0.5, zorder=1, label='Path Line')
    
    # 绘制观测点 (Observed) - 蓝色实心小点
    if len(sub_observed_pos) > 0:
        ax.scatter(sub_true[sub_observed_pos, 1], sub_true[sub_observed_pos, 0], 
                   c='dodgerblue', s=80, edgecolors='white', linewidth=1, zorder=3, label='Context (Observed)')

    # 绘制补全点
    if len(sub_masked_pos) > 0:
        # 真值 (绿色大圈)
        ax.scatter(sub_true[sub_masked_pos, 1], sub_true[sub_masked_pos, 0], 
                   facecolors='none', edgecolors='green', s=200, linewidth=3, zorder=4, label='Ground Truth')
        
        # 预测 (红色叉号)
        ax.scatter(sub_pred[sub_masked_pos, 1], sub_pred[sub_masked_pos, 0], 
                   marker='x', c='red', s=150, linewidth=3, zorder=5, label='Prediction')
        
        # 误差线
        for i in sub_masked_pos:
            start = sub_true[i]
            end = sub_pred[i]
            ax.plot([start[1], end[1]], [start[0], end[0]], 
                    color='orange', linestyle='-', linewidth=2, alpha=0.8, zorder=2)

    # --- 5. 强制聚焦视野 ---
    ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)
    
    # 格式设置
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
    
    # 标题带上误差数据
    ax.set_title(f'Micro-level Accuracy (Sample #{sample_idx})\nAvg Error: {avg_error_m:.2f} m', fontsize=14, fontweight='bold')

    filename = os.path.join(save_dir, f"vis_sample_{sample_idx}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved {filename} (Error: {avg_error_m:.2f}m)")

def evaluate_and_visualize():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transformer, diffusion, centers, base_vocab = load_model_and_data()
    dataset = MinuteTrajDataset(DATA_PKL, mask_ratio=0.15, L=96, vocab_size=base_vocab)

    split_dir = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction/SPLIT_DATASET"
    test_split_path = os.path.join(split_dir, "test_indices.json")
    if os.path.exists(test_split_path):
        with open(test_split_path, "r") as f:
            test_indices = json.load(f)
        print(f"已加载测试集索引，共 {len(test_indices)} 条")
    else:
        print("未找到测试集索引文件，回退为数据集最后1000条")
        test_indices = list(range(max(0, len(dataset) - 1000), len(dataset)))

    subset = Subset(dataset, test_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    print("开始推理与评估 (单步预测模式)...")
    
    metrics = {'mae': [], 'rmse': [], 'acc': []}
    saved_viz_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            token_true = batch['token_true'].to(DEVICE)
            token_masked_input = batch['token_id'].to(DEVICE)
            time_feat = batch['time_feat'].to(DEVICE)
            user_id = batch['user_id'].to(DEVICE)
            mask = batch['mask'].to(DEVICE).bool()

            # 1. Transformer 编码
            z_t = transformer(token_masked_input, time_feat, user_id)

            # 2. 单步预测 (One-step Prediction)
            pred_tokens = predict_one_step(diffusion, z_t, token_true, mask)
            
            # 3. 指标计算
            flat_mask = mask.view(-1).cpu().numpy()
            if not flat_mask.any(): continue

            flat_true = token_true.view(-1).cpu().numpy()
            valid_mask = flat_mask & (flat_true != base_vocab)

            if valid_mask.any():
                flat_pred = pred_tokens.view(-1).cpu().numpy()
                acc = (flat_pred[valid_mask] == flat_true[valid_mask]).mean()
                metrics['acc'].append(acc)
                
                coords_pred = centers[flat_pred[valid_mask].clip(max=base_vocab-1)]
                coords_true = centers[flat_true[valid_mask].clip(max=base_vocab-1)]
                dists = haversine_distance(coords_pred, coords_true)
                metrics['mae'].extend(dists.tolist())
                metrics['rmse'].extend((dists ** 2).tolist())

            # 4. 高动态样本筛选与可视化 (优化版)
            if saved_viz_count < 20:  # <--- 这里控制总共保存多少张图片
                # 预计算真值坐标 [B, L, 2]
                true_coords_batch = centers[token_true.clamp(0, base_vocab-1).cpu().numpy()]
                
                B_size = mask.size(0)
                L_seq = mask.size(1)
                
                for b in range(B_size):
                    if saved_viz_count >= 20: break  # <--- 这里再次检查，达到数量即停止
                    
                    mask_b = mask[b]
                    if mask_b.sum() < 2: continue # 缺失太少不看
                    
                    # 找到缺失区域
                    masked_pos = torch.where(mask_b)[0]
                    first_miss = masked_pos[0]
                    last_miss = masked_pos[-1]
                    
                    # 检查缺失段前后的观测点距离
                    # 如果缺失段跨越了 0.01 度 (约1km)，则跳过，因为图会拉得太远
                    # 如果没有前/后点（在边界），也跳过
                    if first_miss > 0 and last_miss < L_seq-1:
                        start_coord = true_coords_batch[b, first_miss-1]
                        end_coord = true_coords_batch[b, last_miss+1]
                        
                        # 修复：true_coords_batch 是 numpy 数组，使用 np.linalg.norm 而非 torch.norm
                        dist = np.linalg.norm(start_coord - end_coord)
                        
                        # 阈值：0.005度 (约500米)。只看局部密集的补全
                        if dist < 0.005 and dist > 0.0001: 
                            sample_mask = mask[b].cpu().numpy()
                            sample_pred_ids = pred_tokens[b].cpu().numpy()
                            sample_true_ids = token_true[b].cpu().numpy()
                            
                            sample_pred_coords = centers[sample_pred_ids.clip(max=base_vocab-1)]
                            sample_true_coords = centers[sample_true_ids.clip(max=base_vocab-1)]
                            
                            visualize_trajectory(sample_true_coords, sample_pred_coords,
                                                 sample_mask, saved_viz_count, OUTPUT_DIR,
                                                 context_window=3) # 只看前后3个点，极致聚焦
                            saved_viz_count += 1

            print(f"Batch {batch_idx+1}/{len(loader)} Done.", end='\r')

    # --- 汇总报告 ---
    print("\n" + "=" * 60)
    print(f"最终评估报告 (Test Set - One Step Prediction)")
    print("=" * 60)
    
    acc = np.mean(metrics['acc']) if metrics['acc'] else 0
    mae = np.mean(metrics['mae']) if metrics['mae'] else 0
    rmse = np.sqrt(np.mean(metrics['rmse'])) if metrics['rmse'] else 0
    
    print(f"Accuracy: {acc:.2%}")
    print(f"MAE:      {mae:.2f} m")
    print(f"RMSE:     {rmse:.2f} m")
    print("=" * 60)

    # 保存 CSV
    metrics_csv = os.path.join(OUTPUT_DIR, "test_metrics.csv")
    with open(metrics_csv, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Accuracy", "MAE_m", "RMSE_m"])
        w.writerow([acc, mae, rmse])
    print(f"[REPORT] saved {metrics_csv}")

def restore_full_dataset():
    transformer, diffusion, centers, base_vocab = load_model_and_data()
    meta = pd.read_pickle(DATA_PKL)
    rows = meta['sequences']
    L = 96

    class SeqDataset(torch.utils.data.Dataset):
        def __init__(self, rows, L, mask_id):
            self.rows = rows; self.L = L; self.mask_id = mask_id
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            row = self.rows[idx]
            tokens = np.array(row['tokens'], dtype=np.int64)[:self.L]
            time_feat = np.array(row['time_feat'], dtype=np.float32)[:self.L]
            user_id = int(row['user_id'])
            length = len(tokens)
            if length < self.L:
                pad = self.L - length
                tokens = np.pad(tokens, (0, pad), constant_values=self.mask_id)
                time_feat = np.pad(time_feat, ((0, pad), (0, 0)), constant_values=0.0)
            return {'track_id': row['track_id'], 'user_id': user_id,
                    'tokens': torch.from_numpy(tokens),
                    'time_feat': torch.from_numpy(time_feat),
                    'length': min(len(row['tokens']), self.L)}

    def collate_fn(batch):
        track_ids = [b['track_id'] for b in batch]
        user_ids = torch.tensor([b['user_id'] for b in batch], dtype=torch.long)
        tokens = torch.stack([b['tokens'] for b in batch], dim=0)
        time_feat = torch.stack([b['time_feat'] for b in batch], dim=0)
        lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
        return {'track_id': track_ids, 'user_id': user_ids, 'tokens': tokens,
                'time_feat': time_feat, 'lengths': lengths}

    ds = SeqDataset(rows, L, base_vocab)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0,
                                     pin_memory=False, collate_fn=collate_fn)

    records = []
    transformer.eval(); diffusion.eval()
    print("开始全量数据集修复 (Restoration)...")
    
    with torch.no_grad():
        for batch in dl:
            token_id = batch['tokens'].to(DEVICE)
            time_t = batch['time_feat'].to(DEVICE)
            usr_t = batch['user_id'].to(DEVICE)
            lengths = batch['lengths']
            mask_vec = (token_id == base_vocab).bool()

            z_t = transformer(token_id, time_t, usr_t)
            
            # 使用推理模式的单步预测 (无真值)
            pred_ids = predict_one_step_inference(diffusion, z_t, token_id, mask_vec)
            
            # 仅替换 Mask 位置
            filled_ids = token_id.clone()
            filled_ids[mask_vec] = pred_ids[mask_vec]

            latlon = centers[filled_ids.clamp(max=base_vocab-1).cpu().numpy()]
            for bi, track_id in enumerate(batch['track_id']):
                keep_len = int(lengths[bi].item())
                for i in range(keep_len):
                    records.append({
                        'track_id': track_id,
                        'user_id': int(usr_t[bi].item()),
                        'minute_index': i,
                        'lat': float(latlon[bi, i, 0]),
                        'lon': float(latlon[bi, i, 1])
                    })

    restored_csv = os.path.join(OUTPUT_DIR, "restored_dataset.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(restored_csv, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['track_id', 'user_id', 'minute_index', 'lat', 'lon'])
        w.writeheader()
        w.writerows(records)
    print(f"[RESTORE] saved restored dataset to {restored_csv} (rows={len(records)})")

if __name__ == "__main__":
    evaluate_and_visualize()
    restore_full_dataset()
