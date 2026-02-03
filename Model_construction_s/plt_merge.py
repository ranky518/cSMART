import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "plt_merge.csv") 
OUTPUT_PKL = os.path.join(BASE_DIR, "traj_5s_continuous.pkl")

L = 96  # Sequence length

def process_data():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        print(f"Please look for your CSV file. Current working dir: {os.getcwd()}")
        return

    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Normalize columns
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    # Map column names
    col_map = {}
    for c in df.columns:
        if 'lat' in c: col_map[c] = 'lat'
        elif 'lon' in c or 'lng' in c: col_map[c] = 'lon'
        elif 'time' in c or 'date' in c: col_map[c] = 'time'
        elif 'id' in c and 'user' not in c: col_map[c] = 'id'
        elif 'user' in c: col_map[c] = 'user_id'
        
    df = df.rename(columns=col_map)
    
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("Error: 'lat' or 'lon' columns missing.")
        return

    # --- 1. Filter Outliers (Critical for Geolocation Data) ---
    print(f"Original points: {len(df)}")
    
    # Simple hard constraints for Geolife (Mainly Beijing)
    # Beijing is roughly Lat 39-41, Lon 115-117.
    # We'll be slightly generous but exclude 0,0 or wild outliers.
    
    # Filter 1: Valid WGS84
    df = df[(df['lat'] > -90) & (df['lat'] < 90) & (df['lon'] > -180) & (df['lon'] < 180)]
    
    # Filter 2: Quantile Clipping (Remove top/bottom 0.1% outliers)
    lat_lower, lat_upper = df['lat'].quantile([0.001, 0.999])
    lon_lower, lon_upper = df['lon'].quantile([0.001, 0.999])
    
    print(f"Clipping Ranges -> Lat: [{lat_lower:.4f}, {lat_upper:.4f}], Lon: [{lon_lower:.4f}, {lon_upper:.4f}]")
    
    df = df[
        (df['lat'] >= lat_lower) & (df['lat'] <= lat_upper) &
        (df['lon'] >= lon_lower) & (df['lon'] <= lon_upper)
    ]
    print(f"Filtered points: {len(df)}")

    # --- 2. Calculate Normalization Params (Min-Max) ---
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    print(f"Final Bounds -> Lat: [{lat_min}, {lat_max}], Lon: [{lon_min}, {lon_max}]")
    
    norm_params = {
        'lat_min': lat_min, 'lat_max': lat_max,
        'lon_min': lon_min, 'lon_max': lon_max
    }
    
    # Normalize to range [-1, 1] for Diffusion
    def normalize(v, v_min, v_max):
        denom = (v_max - v_min)
        if denom == 0: denom = 1.0
        # [0, 1]
        val_01 = (v - v_min) / denom
        # [-1, 1]
        return val_01 * 2 - 1

    sequences = []
    
    # Group
    group_col = 'id' if 'id' in df.columns else 'user_id'
    if group_col not in df.columns:
        df['combined_id'] = 0
        group_col = 'combined_id'

    print("Segmenting trajectories...")
    for traj_id, group in tqdm(df.groupby(group_col)):
        if len(group) < L:
            continue
            
        group = group.sort_values('time') if 'time' in group.columns else group
        
        # Normalize
        lat_norm = normalize(group['lat'].values, lat_min, lat_max)
        lon_norm = normalize(group['lon'].values, lon_min, lon_max)
        coords_norm = np.stack([lat_norm, lon_norm], axis=1).astype(np.float32) # [T, 2]
        
        num_segments = len(coords_norm) // L
        for i in range(num_segments):
            start = i * L
            end = i * L + L
            
            seq = coords_norm[start:end]
            uid = group['user_id'].iloc[0] if 'user_id' in group.columns else 0
            
            sequences.append({
                'traj_norm': seq,          # [L, 2] in [-1, 1]
                'user_id': uid
            })

    output_data = {
        'sequences': sequences,
        'norm_params': norm_params,
        'num_users': int(df['user_id'].max()) + 1 if 'user_id' in df.columns else 1
    }

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"Processed {len(sequences)} sequences.")
    print(f"Saved to {OUTPUT_PKL}")

if __name__ == "__main__":
    process_data()