import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder

# --- 配置参数 ---
INPUT_FILE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/tracks_20.csv'
OUTPUT_PKL = 'trajectory/geolife_clustering-master/geolife_clustering-master/Model_pre/cluster_token_dataset.pkl'

# 虚拟基站的数量 (Vocab Size)
# 建议值：北京/昌平区域，2000-5000 比较合适。【可修改】
# 太少：精度低（格子太大）；太多：显存占用大，且每个格子数据稀疏。
NUM_CLUSTERS = 2000 

# 序列参数
WINDOW_SIZE = 256
STRIDE = 128

def cyclical_encoding(data, max_val):
    data_norm = 2 * np.pi * data / max_val
    return np.sin(data_norm), np.cos(data_norm)

def preprocess_with_clustering(input_path, output_path):
    print(f"正在读取 {input_path} ...")
    df = pd.read_csv(input_path)
    
    # 1. 准备数据
    print("正在整理经纬度数据...")
    # 提取所有点用于训练聚类器
    coords = df[['lat', 'lon']].values
    
    # 2. 执行 K-Means 聚类 (生成虚拟基站)
    print(f"开始 K-Means 聚类 (拟合 {NUM_CLUSTERS} 个虚拟基站)...")
    print("注意：数据量较大时，此步可能需要几分钟...")
    
    # 使用 MiniBatchKMeans 速度比标准 KMeans 快很多，效果在大数据下差不多
    kmeans = MiniBatchKMeans(
        n_clusters=NUM_CLUSTERS, 
        batch_size=4096, 
        random_state=42,
        n_init=3 # 尝试3次不同的初始化
    )
    
    # 训练并直接获取每个点的标签 (Token ID)
    # df['token_id'] 里存的就是 0 到 1999 的整数
    df['token_id'] = kmeans.fit_predict(coords)
    
    # 获取聚类中心 (即虚拟基站的经纬度)
    # Shape: [2000, 2]
    cluster_centers = kmeans.cluster_centers_
    
    print("聚类完成！")
    
    # 3. 基础时间特征处理
    print("正在处理时间特征...")
    # 原始精度：到秒；对齐到分钟
    df['dt'] = pd.to_datetime(df['date'] + ' ' + df['timestamp'])
    df['dt_min'] = df['dt'].dt.floor('T')                     # 按分钟向下取整
    df = df.sort_values(['user_id', 'dt_min']).reset_index(drop=True)

    # 周期编码基于分钟级时间
    df['hour'] = df['dt_min'].dt.hour
    df['day'] = df['dt_min'].dt.dayofweek
    df['hour_sin'], df['hour_cos'] = cyclical_encoding(df['hour'], 24.0)
    df['day_sin'], df['day_cos'] = cyclical_encoding(df['day'], 7.0)
    
    # 4. User ID 映射
    print("正在映射 User ID...")
    user_le = LabelEncoder()
    df['user_index'] = user_le.fit_transform(df['user_id'])
    num_users = len(user_le.classes_)
    
    # --- 新增：分钟级多数投票重采样 ---
    print("正在执行分钟级多数投票重采样...")
    # 为每个用户-分钟聚合，选择出现次数最多的 token_id 作为该分钟代表
    # 如果同一用户同一分钟内没有冲突（只有一个点），直接取该 token_id
    agg = (
        df.groupby(['user_id', 'dt_min'])['token_id']
          .agg(lambda x: x.value_counts().idxmax())
          .reset_index()
          .rename(columns={'token_id': 'token_min'})
    )
    # 合并回原表以便构建序列的时间特征（按分钟）
    df_min = (
        agg.merge(
            df[['user_id', 'dt_min', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'user_index']],
            on=['user_id', 'dt_min'],
            how='left'
        )
        .drop_duplicates(subset=['user_id', 'dt_min'])    # 每分钟一条
        .sort_values(['user_id', 'dt_min'])
        .reset_index(drop=True)
    )

    # --- 构建分钟级轨迹段（按用户的连续分钟） ---
    print(f"正在构建序列 (Window: {WINDOW_SIZE})...")
    # 连续分钟的定义：相邻 dt_min 差值为 1 分钟；遇到间隔则开启新段
    all_sequences = []
    for uid, grp in df_min.groupby('user_id'):
        grp = grp.sort_values('dt_min').reset_index(drop=True)
        # 切分为按分钟连续的段
        start_idx = 0
        for i in range(1, len(grp)):
            if (grp.loc[i, 'dt_min'] - grp.loc[i-1, 'dt_min']).total_seconds() != 60:
                # 断开，保存 [start_idx, i)
                segment = grp.iloc[start_idx:i]
                if len(segment) > 0:
                    # 使用 LabelEncoder 映射后的 user_index
                    uidx = int(segment['user_index'].iloc[0])
                    token_seq = segment['token_min'].to_numpy()
                    time_seq = segment[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].to_numpy()
                    seq_len = len(segment)
                    # 生成轨迹段 ID（用户_段序号_起始分钟索引）
                    track_id = f"{uid}_{segment.index[0]}"
                    if seq_len <= WINDOW_SIZE:
                        all_sequences.append({
                            'tokens': token_seq,
                            'time_feat': time_seq,
                            'user_id': uidx,
                            'length': seq_len,
                            'track_id': track_id
                        })
                    else:
                        for s in range(0, seq_len, STRIDE):
                            e = s + WINDOW_SIZE
                            sub_tokens = token_seq[s:e]
                            sub_time = time_seq[s:e]
                            if len(sub_tokens) >= 20:
                                all_sequences.append({
                                    'tokens': sub_tokens,
                                    'time_feat': sub_time,
                                    'user_id': uidx,
                                    'length': len(sub_tokens),
                                    'track_id': f"{track_id}_{s}"
                                })
                start_idx = i
        # 末段
        segment = grp.iloc[start_idx:len(grp)]
        if len(segment) > 0:
            uidx = int(segment['user_index'].iloc[0])
            token_seq = segment['token_min'].to_numpy()
            time_seq = segment[['hour_sin', 'hour_cos', 'day_sin', 'day_cos']].to_numpy()
            seq_len = len(segment)
            track_id = f"{uid}_{segment.index[0]}"
            if seq_len <= WINDOW_SIZE:
                all_sequences.append({
                    'tokens': token_seq,
                    'time_feat': time_seq,
                    'user_id': uidx,
                    'length': seq_len,
                    'track_id': track_id
                })
            else:
                for s in range(0, seq_len, STRIDE):
                    e = s + WINDOW_SIZE
                    sub_tokens = token_seq[s:e]
                    sub_time = time_seq[s:e]
                    if len(sub_tokens) >= 20:
                        all_sequences.append({
                            'tokens': sub_tokens,
                            'time_feat': sub_time,
                            'user_id': uidx,
                            'length': len(sub_tokens),
                            'track_id': f"{track_id}_{s}"
                        })

    # 6. 保存结果
    save_data = {
        'sequences': all_sequences,
        'vocab_size': NUM_CLUSTERS,         # 词表大小
        'cluster_centers': cluster_centers, # [2000, 2] 聚类中心坐标，用于 Embedding 初始化
        'num_users': num_users
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print("=" * 50)
    print("预处理完成！(分钟级多数投票)")
    print(f"虚拟基站数 (K): {NUM_CLUSTERS}")
    print(f"序列样本数: {len(all_sequences)}")
    print(f"聚类中心数据已保存，可用于可视化或模型初始化。")
    print(f"结果文件: {output_path}")
    print("=" * 50)

    # 预览前 10 条样例（分钟级）
    print("样例预览（前 10 条）：")
    for i, s in enumerate(all_sequences[:10]):
        print(f"\n=== Sample {i} ===")
        print(f"track_id: {s['track_id']}, user_id: {s['user_id']}, length: {s['length']}")
        tokens = np.array(s['tokens'])
        time_feat = np.array(s['time_feat'])
        print(f"tokens shape: {tokens.shape}, head: {tokens[:10].tolist()}")
        print(f"time_feat shape: {time_feat.shape}, head(3): {time_feat[:3].tolist()}")

if __name__ == "__main__":
    preprocess_with_clustering(INPUT_FILE, OUTPUT_PKL)