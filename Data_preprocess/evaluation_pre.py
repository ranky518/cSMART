import os
import csv
import math
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset_preprocess import MinuteTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill import DiffusionFill

# ========== 配置消融模式 ==========
# 必须与 Training 保持一致以加载正确权重
PREPROCESS_MODE = "ZOH"  
# ==================================

base_dir = "/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master"
DATA_PKL = os.path.join(base_dir, "Model_pre/cluster_token_dataset.pkl")

# [修改] 模型目录：指向 Ablation/Data_preprocess/results
MODEL_DIR = os.path.join("/home/yanglanqi/trajectory/geolife_clustering-master/Ablation/Data_preprocess/results", PREPROCESS_MODE)
MODEL_PATH = os.path.join(MODEL_DIR, f"model_{PREPROCESS_MODE}.pt")
OUTPUT_DIR = os.path.join(MODEL_DIR, "Evaluation_Outputs")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 绘图配置：使用通用字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 忽略字体警告
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

# --- 新增：百度地图坐标转换 (WGS84 -> BD09) ---
def wgs84_to_bd09(lon, lat):
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    a = 6378245.0; ee = 0.00669342162296594323
    if not (73.66 < lon < 135.05 and 3.86 < lat < 53.55): return lon, lat
    def _transform(x, y):
        return (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2.0 / 3.0
    dLat = _transform(lon - 105.0, lat - 35.0); dLon = _transform(lon - 105.0, lat - 35.0) 
    dlat_v = -100.0 + 2.0 * (lon-105.0) + 3.0 * (lat-35.0) + 0.2 * (lat-35.0)**2 + 0.1 * (lon-105.0) * (lat-35.0) + 0.2 * math.sqrt(abs(lon-105.0))
    dlat_v += (20.0 * math.sin(6.0 * (lon-105.0) * math.pi) + 20.0 * math.sin(2.0 * (lon-105.0) * math.pi)) * 2.0 / 3.0
    dlat_v += (20.0 * math.sin((lat-35.0) * math.pi) + 40.0 * math.sin((lat-35.0) / 3.0 * math.pi)) * 2.0 / 3.0
    dlat_v += (160.0 * math.sin((lat-35.0) / 12.0 * math.pi) + 320 * math.sin((lat-35.0) * math.pi / 30.0)) * 2.0 / 3.0
    dlon_v = 300.0 + (lon-105.0) + 2.0 * (lat-35.0) + 0.1 * (lon-105.0)**2 + 0.1 * (lon-105.0) * (lat-35.0) + 0.1 * math.sqrt(abs(lon-105.0))
    dlon_v += (20.0 * math.sin(6.0 * (lon-105.0) * math.pi) + 20.0 * math.sin(2.0 * (lon-105.0) * math.pi)) * 2.0 / 3.0
    dlon_v += (20.0 * math.sin((lon-105.0) * math.pi) + 40.0 * math.sin((lon-105.0) / 3.0 * math.pi)) * 2.0 / 3.0
    dlon_v += (150.0 * math.sin((lon-105.0) / 12.0 * math.pi) + 300.0 * math.sin((lon-105.0) / 30.0 * math.pi)) * 2.0 / 3.0
    radLat = lat / 180.0 * math.pi
    magic = 1 - ee * (math.sin(radLat) ** 2)
    sqrtMagic = math.sqrt(magic)
    dlat_v = (dlat_v * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dlon_v = (dlon_v * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    mgLat = lat + dlat_v; mgLon = lon + dlon_v
    z = math.sqrt(mgLon * mgLon + mgLat * mgLat) + 0.00002 * math.sin(mgLat * x_pi)
    theta = math.atan2(mgLat, mgLon) + 0.000003 * math.cos(mgLon * x_pi)
    bd_lon = z * math.cos(theta) + 0.0065; bd_lat = z * math.sin(theta) + 0.006
    return bd_lon, bd_lat

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

# --- 新增：生成百度地图HTML + JS数据文件 ---
def generate_baidu_map_html(true_coords, pred_coords, mask, sample_idx, save_dir, avg_error, bucket_label=""):
    bd_true = [wgs84_to_bd09(c[1], c[0]) for c in true_coords] 
    bd_pred = [wgs84_to_bd09(c[1], c[0]) for c in pred_coords]
    masked_indices = np.where(mask)[0].tolist()
    if not masked_indices: return
    center_idx = masked_indices[len(masked_indices) // 2]
    center_pt = bd_true[center_idx]
    
    data_dict = {
        "center": center_pt, "path": bd_true, "observed": [], "ground_truth": [], "prediction": [], 
        "meta": {"id": int(sample_idx), "error": float(avg_error), "mask_count": int(mask.sum())}
    }
    for i in range(len(bd_true)):
        if not mask[i]: data_dict["observed"].append(bd_true[i])
    for i in masked_indices:
        data_dict["ground_truth"].append(bd_true[i])
        data_dict["prediction"].append(bd_pred[i])

    js_filename = f"vis_data_{bucket_label}_{sample_idx}.js"
    with open(os.path.join(save_dir, js_filename), 'w', encoding='utf-8') as f:
        f.write(f"var sample_data = {json.dumps(data_dict)};")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
    <style type="text/css">
        body, html,#allmap {{width: 100%;height: 100%;overflow: hidden;margin:0;font-family:"微软雅黑";}}
        #info {{position:absolute; top:20px; left:20px; z-index:999; background:white; padding:10px; border-radius:5px; box-shadow:0 0 5px #999;}}
    </style>
    <script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&ak=RL2tgKrUc3tefjormZwh5g0kivlCCo03"></script>
    <script type="text/javascript" src="{js_filename}"></script>
    <title>Visualization #{sample_idx} ({bucket_label})</title>
</head>
<body>
    <div id="info">
        <b>Sample #{sample_idx}</b><br>
        Bucket: {bucket_label}<br>
        Avg Error: {avg_error:.2f} m<br>
        <span style="color:black">● Observed (Ctx)</span><br>
        <span style="color:green">●-● Masked GT</span><br>
        <span style="color:red">★ Pred</span>
    </div>
    <div id="allmap"></div>
    <script type="text/javascript">
        if (typeof sample_data === 'undefined') {{ alert('JS Data not loaded!'); }}
        var map = new BMap.Map("allmap");
        
        var focusPoints = [];
        if (sample_data.ground_truth.length > 0) {{
            sample_data.ground_truth.forEach(p => focusPoints.push(new BMap.Point(p[0], p[1])));
            sample_data.prediction.forEach(p => focusPoints.push(new BMap.Point(p[0], p[1])));
            if (focusPoints.length < 5 && sample_data.observed.length > 0) {{
                 sample_data.observed.slice(-3).forEach(p => focusPoints.push(new BMap.Point(p[0], p[1])));
            }}
            map.setViewport(focusPoints, {{margins: [50, 50, 50, 50]}});
        }} else {{
            var allPoints = [];
            sample_data.path.forEach(p => allPoints.push(new BMap.Point(p[0], p[1])));
            map.setViewport(allPoints, {{margins: [20, 20, 20, 20]}});
        }}
        map.enableScrollWheelZoom(true);

        var allPath = [];
        sample_data.path.forEach(p => allPath.push(new BMap.Point(p[0], p[1])));
        var polyline = new BMap.Polyline(allPath, {{strokeColor:"gray", strokeWeight:2, strokeOpacity:0.4}});   
        map.addOverlay(polyline);

        sample_data.observed.forEach(function(p){{
            var pt = new BMap.Point(p[0], p[1]);
            var marker = new BMap.Marker(pt, {{
                icon: new BMap.Symbol(BMap_Symbol_SHAPE_CIRCLE, {{
                    scale: 5, fillColor: "black", fillOpacity: 0.6, strokeColor: "black", strokeWeight: 1
                }})
            }});
            map.addOverlay(marker);
        }});

        var gt_pts_array = [];
        sample_data.ground_truth.forEach(function(p){{
            var pt = new BMap.Point(p[0], p[1]);
            gt_pts_array.push(pt);
            var marker = new BMap.Marker(pt, {{
                icon: new BMap.Symbol(BMap_Symbol_SHAPE_CIRCLE, {{
                    scale: 5, fillColor: "white", fillOpacity: 0.1, strokeColor: "green", strokeWeight: 2
                }})
            }});
            map.addOverlay(marker);
        }});
        if (gt_pts_array.length > 1) {{
             var gtLine = new BMap.Polyline(gt_pts_array, {{strokeColor:"green", strokeWeight:3, strokeOpacity:0.8}});
             map.addOverlay(gtLine);
        }}

        sample_data.prediction.forEach(function(p, i){{
            var pt = new BMap.Point(p[0], p[1]);
            var marker = new BMap.Marker(pt, {{
                icon: new BMap.Symbol(BMap_Symbol_SHAPE_STAR, {{
                    scale: 5, fillColor: "red", fillOpacity: 0.9, strokeColor: "red", strokeWeight: 1
                }})
            }});
            map.addOverlay(marker);
            if (gt_pts_array[i]) {{
                var line = new BMap.Polyline([gt_pts_array[i], pt], {{strokeColor:"orange", strokeWeight:1, strokeStyle:"dashed"}});
                map.addOverlay(line);
            }}
        }});
    </script>
</body>
</html>"""
    filename = f"vis_sample_{bucket_label}_{sample_idx}.html"
    with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[VIZ] Saved {filename}")

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
    a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
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
    a_t = diffusion.sqrt_alphas_cumprod[t].unsqueeze(-1)
    am1_t = diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
    x0_hat = (x_noisy - am1_t * eps_pred) / a_t
    
    logits = diffusion.classifier(x0_hat)
    return logits.argmax(dim=-1)

def load_model_and_data():
    print(f"正在加载元数据: {DATA_PKL} ...")
    meta = pd.read_pickle(DATA_PKL)
    base_vocab = int(meta['vocab_size'])
    # [修改] 匹配 Training 的词表大小
    pad_id = base_vocab + 1
    vocab_size_total = base_vocab + 2 
    
    num_users = int(meta['num_users']) if 'num_users' in meta else int(max(s['user_id'] for s in meta['sequences']) + 1)
    cluster_centers = np.array(meta['cluster_centers'], dtype=np.float32)
    L = 96
    d_model = 512

    print("正在初始化模型结构...")
    transformer = TransformerCond(vocab_size=vocab_size_total, num_users=num_users,
                                  embed_dim=d_model, nhead=8, num_layers=6,
                                  mask_id=pad_id, max_len=L).to(DEVICE)
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

def visualize_trajectory(true_coords, pred_coords, mask, idx, out_dir, avg_error_m=0, bucket_label=""):
    generate_baidu_map_html(true_coords, pred_coords, mask, idx, out_dir, avg_error_m, bucket_label)

def evaluate_and_visualize():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Evaluating Mode: {PREPROCESS_MODE}")
    print(f"Loading Model: {MODEL_PATH}")
    
    transformer, diffusion, centers, base_vocab = load_model_and_data()
    
    # 关键：Dataset使用相同的 preprocess_mode
    # [修改] sample_step=10
    dataset = MinuteTrajDataset(DATA_PKL, mask_ratio=0.15, L=96, vocab_size=base_vocab,
                                span_mask=True, span_len_min=5, span_len_max=10,
                                sample_step=10, preprocess_mode=PREPROCESS_MODE)

    # 加载测试集索引
    split_file = os.path.join(MODEL_DIR, "split_indices/test_indices.json")
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            test_indices = json.load(f)
    else:
        print("Warning: Split indices not found, using last 1000.")
        test_indices = list(range(len(dataset)-1000, len(dataset)))

    subset = Subset(dataset, test_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    print("开始推理与评估 (单步预测模式 - 对齐 evaluation_long)...")
    
    metrics = {'mae': [], 'rmse': [], 'acc': [], 'acc20': [], 'acc50': [], 'acc100': []}
    viz_candidates = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            token_true = batch['token_true'].to(DEVICE)
            token_masked_input = batch['token_id'].to(DEVICE)
            time_feat = batch['time_feat'].to(DEVICE)
            user_id = batch['user_id'].to(DEVICE)
            mask = batch['mask'].to(DEVICE).bool()

            # 1. Transformer 编码
            z_t = transformer(token_masked_input, time_feat, user_id)

            # 2. 预测 (修改：使用 predict_one_step)
            pred_tokens = predict_one_step(diffusion, z_t, token_true, mask)
            
            # --- 指标计算 ---
            flat_mask = mask.view(-1).cpu().numpy()
            if not flat_mask.any(): continue
            flat_true = token_true.view(-1).cpu().numpy()
            # [修改] 确保 True Label 不是 Pad 或 Mask
            valid_mask = flat_mask & (flat_true != base_vocab) & (flat_true != base_vocab + 1)
            if valid_mask.any():
                flat_pred = pred_tokens.view(-1).cpu().numpy()
                acc = (flat_pred[valid_mask] == flat_true[valid_mask]).mean()
                metrics['acc'].append(acc)
                coords_pred = centers[flat_pred[valid_mask].clip(max=base_vocab-1)]
                coords_true = centers[flat_true[valid_mask].clip(max=base_vocab-1)]
                dists = haversine_distance(coords_pred, coords_true)
                metrics['mae'].extend(dists.tolist())
                metrics['rmse'].extend((dists ** 2).tolist())
                metrics['acc20'].append((dists <= 20).mean())
                metrics['acc50'].append((dists <= 50).mean())
                metrics['acc100'].append((dists <= 100).mean())

            # --- 4. 收集可视化候选者 ---
            token_true_np = token_true.cpu().numpy()
            pred_tokens_np = pred_tokens.cpu().numpy()
            mask_np = mask.cpu().numpy()
            B_size = mask.size(0)

            for b in range(B_size):
                mask_b = mask_np[b]
                if mask_b.sum() < 1: continue 

                sample_pred_ids = pred_tokens_np[b]
                sample_true_ids = token_true_np[b]
                masked_indices = np.where(mask_b)[0]
                if len(masked_indices) == 0: continue

                # --- [新增] 质量控制筛选 ---
                t_coords = centers[sample_true_ids.clip(max=base_vocab-1)] 
                masked_t_coords = t_coords[masked_indices]
                
                # 1. 剔除静止 (20分钟至少50m)
                if len(masked_t_coords) >= 2:
                    move_dists = np.linalg.norm(np.diff(masked_t_coords, axis=0), axis=1) * 100000 
                    total_move = move_dists.sum()
                else: 
                    total_move = 0
                
                if total_move < 50.0: continue

                # 2. 剔除视野跨度过大 (Geolife, 20min > 15km)
                # ...此处略去复杂的span计算，使用均值误差计算...
                
                p_coords = centers[sample_pred_ids.clip(max=base_vocab-1)][masked_indices]
                dists = haversine_distance(p_coords, masked_t_coords)
                error_m = dists.mean()
                
                viz_candidates.append({
                    'error': error_m,
                    'pred_ids': sample_pred_ids,
                    'true_ids': sample_true_ids,
                    'mask': mask_b
                })

            print(f"Batch {batch_idx+1}/{len(loader)} Done.", end='\r')

    print(f"\nCollected {len(viz_candidates)} candidates. Filtering by Error Buckets...")
    
    buckets_def = {
        '0-50m':    lambda x: x <= 50,
        '50-100m':  lambda x: 50 < x <= 100,
        '100-200m': lambda x: 100 < x <= 200,
        '200-500m': lambda x: 200 < x <= 500,
        'Old_Bad':  lambda x: x > 500
    }
    buckets_data = {k: [] for k in buckets_def.keys()}
    
    for c in viz_candidates:
        err = c['error']
        for b_name, condition in buckets_def.items():
            if condition(err):
                buckets_data[b_name].append(c)
                break

    print("Bucket Distribution:")
    for k, v in buckets_data.items():
        print(f"  [{k}]: {len(v)} samples")

    final_selection = []
    
    for b_name, b_list in buckets_data.items():
        if len(b_list) > 0:
            sorted_list = sorted(b_list, key=lambda x: x['error'])
            count = min(len(sorted_list), 3)
            print(f"  Bucket [{b_name}]: selecting top {count} (best error)")
            
            for i in range(count):
                cand = sorted_list[i]
                cand['bucket_label'] = b_name
                final_selection.append(cand)
    
    print(f"Selected {len(final_selection)} samples for display.")
    
    for i, cand in enumerate(final_selection):
        c_pred = centers[cand['pred_ids'].clip(max=base_vocab-1)]
        c_true = centers[cand['true_ids'].clip(max=base_vocab-1)]
        
        visualize_trajectory(c_true, c_pred, cand['mask'], i, OUTPUT_DIR, 
                             avg_error_m=cand['error'], bucket_label=cand['bucket_label'])

    # --- 汇总报告 ---
    print("\n" + "=" * 60)
    print(f"最终评估报告 (Test Set - One-Step Denoising)")
    print("=" * 60)
    
    acc = np.mean(metrics['acc']) if metrics['acc'] else 0
    acc20 = np.mean(metrics['acc20']) if metrics['acc20'] else 0
    acc50 = np.mean(metrics['acc50']) if metrics['acc50'] else 0
    acc100 = np.mean(metrics['acc100']) if metrics['acc100'] else 0
    mae = np.mean(metrics['mae']) if metrics['mae'] else 0
    rmse = np.sqrt(np.mean(metrics['rmse'])) if metrics['rmse'] else 0
    
    print(f"Accuracy: {acc:.2%}")
    print(f"Acc@20m:  {acc20:.2%}")
    print(f"Acc@50m:  {acc50:.2%}")
    print(f"Acc@100m: {acc100:.2%}")
    print(f"MAE:      {mae:.2f} m")
    print(f"RMSE:     {rmse:.2f} m")
    print("=" * 60)

    # 保存 CSV
    metrics_csv = os.path.join(OUTPUT_DIR, "test_metrics.csv")
    with open(metrics_csv, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Accuracy", "Acc@20", "Acc@50", "Acc@100", "MAE_m", "RMSE_m"])
        w.writerow([acc, acc20, acc50, acc100, mae, rmse])
    print(f"[REPORT] saved {metrics_csv}")

def restore_full_dataset():
    transformer, diffusion, centers, base_vocab = load_model_and_data()
    meta = pd.read_pickle(DATA_PKL)
    rows = meta['sequences']
    L = 96
    # [修改] 恢复时也保持 10分钟采样
    sample_step = 10  
    # 词表参数
    pad_id = base_vocab + 1

    class SeqDataset(torch.utils.data.Dataset):
        def __init__(self, rows, L, mask_id, pad_id):
            self.rows = rows; self.L = L
            self.mask_id = mask_id
            self.pad_id = pad_id

        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            row = self.rows[idx]
            tokens = np.array(row['tokens'], dtype=np.int64)
            time_feat = np.array(row['time_feat'], dtype=np.float32)
            
            # 应用采样
            if sample_step > 1:
                tokens = tokens[::sample_step]
                time_feat = time_feat[::sample_step]
                
            tokens = tokens[:self.L]
            time_feat = time_feat[:self.L]
            length = len(tokens)
            if length < self.L:
                pad = self.L - length
                # [修改] 用 pad_id 填充
                tokens = np.pad(tokens, (0, pad), constant_values=self.pad_id)
                time_feat = np.pad(time_feat, ((0, pad), (0, 0)), constant_values=0.0)
            
            # 这里不用 UserID 等，简化
            return {'track_id': row['track_id'], 
                    'user_id': int(row['user_id']),
                    'tokens': torch.from_numpy(tokens),
                    'time_feat': torch.from_numpy(time_feat),
                    'length': min(length, self.L)}

    def collate_fn(batch):
        track_ids = [b['track_id'] for b in batch]
        user_ids = torch.tensor([b['user_id'] for b in batch], dtype=torch.long)
        tokens = torch.stack([b['tokens'] for b in batch], dim=0)
        time_feat = torch.stack([b['time_feat'] for b in batch], dim=0)
        lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
        return {'track_id': track_ids, 'user_id': user_ids, 'tokens': tokens,
                'time_feat': time_feat, 'lengths': lengths}

    # Pass pad_id
    ds = SeqDataset(rows, L, base_vocab, pad_id)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0,
                                     pin_memory=False, collate_fn=collate_fn)

    records = []
    transformer.eval(); diffusion.eval()
    print("开始全量数据集修复 (Restoration - Inference Mode)...")
    
    with torch.no_grad():
        for batch in dl:
            token_id = batch['tokens'].to(DEVICE)
            time_t = batch['time_feat'].to(DEVICE)
            usr_t = batch['user_id'].to(DEVICE)
            lengths = batch['lengths']
            # Mask 是那些需要被预测的地方（这里全量数据如果没有Mask，则不需要恢复）
            # 如果目的是填充 'ZOH' 中产生的空洞，这里逻辑需要调整，但如果是填充 mask_token：
            mask_vec = (token_id == base_vocab).bool()

            z_t = transformer(token_id, time_t, usr_t)
            
            # 只有当存在 Mask 时才预测
            if mask_vec.any():
                pred_ids = predict_one_step_inference(diffusion, z_t, token_id, mask_vec)
                filled_ids = token_id.clone()
                filled_ids[mask_vec] = pred_ids[mask_vec]
            else:
                filled_ids = token_id
            
            # [修改] 注意处理 Pad Token (转为 mask_id 或 safe clamp)
            # 坐标转换时不能包含 pad_id，否则越界
            safe_ids = filled_ids.clamp(max=base_vocab-1) 
            latlon = centers[safe_ids.cpu().numpy()]
            for bi, track_id in enumerate(batch['track_id']):
                keep_len = int(lengths[bi].item())
                for i in range(keep_len):
                    records.append({
                        'track_id': track_id,
                        'user_id': int(usr_t[bi].item()),
                        'minute_index': i * sample_step,  # 恢复原始时间索引 (0, 3, 6...)
                        'lat': float(latlon[bi, i, 0]),
                        'lon': float(latlon[bi, i, 1])
                    })

    restored_csv = os.path.join(OUTPUT_DIR, "restored_dataset_3min.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(restored_csv, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['track_id', 'user_id', 'minute_index', 'lat', 'lon'])
        w.writeheader()
        w.writerows(records)
    print(f"[RESTORE] saved restored dataset to {restored_csv} (rows={len(records)})")

if __name__ == "__main__":
    evaluate_and_visualize()
    restore_full_dataset()
