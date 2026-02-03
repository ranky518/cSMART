import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import SecondTrajDataset
from transformer_cond import TransformerCond
from diffusion_fill import DiffusionFill

# ========== 1. Config & Paths ==========
data_pkl = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction_s/traj_5s_continuous.pkl"
save_dir = "trajectory/geolife_clustering-master/geolife_clustering-master/Model_construction_5s"
os.makedirs(save_dir, exist_ok=True)
save_ckpt = os.path.join(save_dir, "model_5s_continuous.pt")

# Hyperparams
d_model = 256
batch_size = 64
lr = 1e-3
epochs = 50
L = 96
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 2. Data Loading ==========
if not os.path.exists(data_pkl):
    raise FileNotFoundError(f"Please run plt_merge.py first to generate {data_pkl}")

with open(data_pkl, 'rb') as f:
    meta = pickle.load(f)
    norm_params = meta['norm_params']
    num_users = meta.get('num_users', 1)

dataset = SecondTrajDataset(data_pkl, L=L)
n_total = len(dataset)

# Explicit split to avoid rounding error
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ========== 3. Model ==========
transformer = TransformerCond(input_dim=2, num_users=num_users, embed_dim=d_model, max_len=L).to(device)
diffusion = DiffusionFill(d_cond=d_model, d_embed=d_model, input_dim=2).to(device)

optimizer = torch.optim.AdamW(list(transformer.parameters()) + list(diffusion.parameters()), lr=lr)

# --- Metrics Helper ---
def calc_haversine_error(pred_norm, true_norm, params):
    # Un-normalize from [-1, 1]
    # val_01 = (val_norm + 1) / 2
    # val_deg = val_01 * (max - min) + min
    
    lat_min, lat_max = params['lat_min'], params['lat_max']
    lon_min, lon_max = params['lon_min'], params['lon_max']
    
    # Clamp to ensure stability
    pred_lat_01 = (pred_norm[:, 0].clamp(-1, 1) + 1) / 2
    pred_lon_01 = (pred_norm[:, 1].clamp(-1, 1) + 1) / 2
    true_lat_01 = (true_norm[:, 0].clamp(-1, 1) + 1) / 2
    true_lon_01 = (true_norm[:, 1].clamp(-1, 1) + 1) / 2
    
    pred_lat = pred_lat_01 * (lat_max - lat_min) + lat_min
    pred_lon = pred_lon_01 * (lon_max - lon_min) + lon_min
    true_lat = true_lat_01 * (lat_max - lat_min) + lat_min
    true_lon = true_lon_01 * (lon_max - lon_min) + lon_min
    
    R = 6371000.0
    phi1, phi2 = torch.deg2rad(true_lat), torch.deg2rad(pred_lat)
    dphi = torch.deg2rad(pred_lat - true_lat)
    dlambda = torch.deg2rad(pred_lon - true_lon)
    
    a = torch.sin(dphi/2)**2 + torch.cos(phi1)*torch.cos(phi2)*torch.sin(dlambda/2)**2
    # Clamp 'a' to [0,1] to prevent NaN in sqrt
    c = 2 * torch.atan2(torch.sqrt(a.clamp(0, 1)), torch.sqrt((1-a).clamp(0, 1)))
    return R * c # meters

def estimate_x0(x_noisy, noise_pred, t, diffusion):
    # Estimate original clean signal x0 from Step t
    # x0 ~ (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)
    
    sq_alpha = diffusion.sqrt_alphas_cumprod[t] 
    sq_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[t]
    
    # Broadcast to match x_noisy [B, L, D]
    while sq_alpha.ndim < x_noisy.ndim:
        sq_alpha = sq_alpha.unsqueeze(-1)
        sq_one_minus = sq_one_minus.unsqueeze(-1)
        
    x0_pred = (x_noisy - sq_one_minus * noise_pred) / (sq_alpha + 1e-8)
    return x0_pred

# ========== 4. Training ==========
print("Starting Continuous Trajectory Training...")

for epoch in range(epochs):
    # --- TRAIN ---
    transformer.train(); diffusion.train()
    total_loss = 0
    train_count = 0
    
    for batch in loader:
        traj_in = batch['traj_in'].to(device)
        traj_true = batch['traj_true'].to(device)
        mask = batch['mask'].to(device)
        time_feat = batch['time_feat'].to(device)
        user_id = batch['user_id'].to(device)
        
        z_t = transformer(traj_in, time_feat, user_id)
        
        # Sample random t for each batch elt [B]
        t = torch.randint(0, diffusion.num_steps, (traj_true.size(0),), device=device)
        
        x_noisy, noise = diffusion.add_noise(traj_true, t)
        noise_pred = diffusion(x_noisy, z_t, t, mask)
        
        mask_expanded = mask.unsqueeze(-1)
        loss = F.mse_loss(noise_pred * mask_expanded, noise * mask_expanded, reduction='sum') / (mask.sum() + 1e-6)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        train_count += 1

    avg_train_loss = total_loss / max(train_count, 1)

    # --- VALIDATION ---
    transformer.eval(); diffusion.eval()
    val_loss_sum = 0
    val_mae_sum = 0
    val_mse_sum = 0
    val_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            traj_in = batch['traj_in'].to(device)
            traj_true = batch['traj_true'].to(device)
            mask = batch['mask'].to(device)
            time_feat = batch['time_feat'].to(device)
            user_id = batch['user_id'].to(device)

            z_t = transformer(traj_in, time_feat, user_id)
            t = torch.randint(0, diffusion.num_steps, (traj_true.size(0),), device=device)
            
            x_noisy, noise = diffusion.add_noise(traj_true, t)
            noise_pred = diffusion(x_noisy, z_t, t, mask)
            
            # Loss
            mask_expanded = mask.unsqueeze(-1)
            loss = F.mse_loss(noise_pred * mask_expanded, noise * mask_expanded, reduction='sum') / (mask.sum() + 1e-6)
            val_loss_sum += loss.item()
            
            # Reconstruction Metrics (Estimate x0)
            x0_hat = estimate_x0(x_noisy, noise_pred, t, diffusion)
            
            mask_bool = mask.bool()
            if mask_bool.any():
                dists = calc_haversine_error(x0_hat[mask_bool], traj_true[mask_bool], norm_params)
                val_mae_sum += dists.mean().item()
                val_mse_sum += (dists**2).mean().item()
            
            val_count += 1

    # Report
    avg_val_loss = val_loss_sum / max(val_count, 1)
    avg_val_mae = val_mae_sum / max(val_count, 1)
    avg_val_mse = val_mse_sum / max(val_count, 1)
    
    # Format matches requested table columns: 
    # avg_loss | avg_loss_cls | avg_acc | avg_MAE/m | avg_MSE/m^2
    print(f"Epoch {epoch:02d} [Val] | avg_loss={avg_val_loss:.4f} | avg_loss_cls=0.0000 | avg_acc=0.00% | avg_MAE/m={avg_val_mae:.3f} | avg_MSE/m^2={avg_val_mse:.3f}")

torch.save({'transformer': transformer.state_dict(), 'diffusion': diffusion.state_dict()}, save_ckpt)
print("Done.")