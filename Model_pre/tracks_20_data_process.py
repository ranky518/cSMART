import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# --- 配置参数 ---
INPUT_FILE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/tracks_20.csv'
OUTPUT_PKL = 'trajectory/geolife_clustering-master/geolife_clustering-master/Model_pre/Data_Model_pre.pkl'

# 序列长度设置
# 你的统计显示 95% < 190，我们可以稍微给点余量设为 256
# 256 是 2 的幂，对 GPU 训练稍微友好一点
WINDOW_SIZE = 256 
STRIDE = 128  # 滑动步长，对于长轨迹，每隔 128 个点切一段 (50% 重叠)

def cyclical_encoding(data, max_val):
    """
    将时间特征转换为 Sin/Cos 编码，保留周期性
    """
    data_norm = 2 * np.pi * data / max_val
    return np.sin(data_norm), np.cos(data_norm)

def preprocess_data(input_path, output_path):
    print(f"正在读取 {input_path} ...")
    df = pd.read_csv(input_path)
    
    # 1. 基础处理：合并时间，排序
    print("正在处理时间与排序...")
    df['dt'] = pd.to_datetime(df['date'] + ' ' + df['timestamp'])
    df = df.sort_values(['track_id', 'dt']).reset_index(drop=True)
    
    # 2. 坐标归一化 (Min-Max Scaling)
    # 这一步非常重要，否则模型很难收敛
    print("正在归一化经纬度...")
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()
    
    # 保存归一化参数，方便以后把预测结果反归一化回真实经纬度
    scale_params = {
        'min_lat': min_lat, 'max_lat': max_lat,
        'min_lon': min_lon, 'max_lon': max_lon
    }
    
    df['norm_lat'] = (df['lat'] - min_lat) / (max_lat - min_lat)
    df['norm_lon'] = (df['lon'] - min_lon) / (max_lon - min_lon)
    
    # 3. 时间特征提取与周期性编码
    print("正在进行时间特征编码 (Sin/Cos)...")
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek # 0=Monday, 6=Sunday
    
    # Hour (周期 24)
    df['hour_sin'], df['hour_cos'] = cyclical_encoding(df['hour'], 24.0)
    # Day of Week (周期 7)
    df['day_sin'], df['day_cos'] = cyclical_encoding(df['day_of_week'], 7.0)
    
    # 4. 用户 ID 映射 (Label Encoding)
    print("正在映射 User ID...")
    le = LabelEncoder()
    df['user_index'] = le.fit_transform(df['user_id'])
    user_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    num_users = len(user_mapping)
    
    # 5. 构建序列 (核心步骤)
    print(f"正在构建序列 (Window Size: {WINDOW_SIZE}, Stride: {STRIDE})...")
    
    # 按 track_id 分组
    grouped = df.groupby('track_id')
    
    all_sequences = []
    
    total_tracks = len(grouped)
    processed_count = 0
    
    for track_id, group in grouped:
        # 提取需要的特征列
        # Shape: [Length, Features]
        # 特征顺序: [Lat, Lon, Hour_Sin, Hour_Cos, Day_Sin, Day_Cos, User_Index]
        # 注意：User_Index 是静态的，但在序列模型中通常也作为每个时间步的输入，或者单独处理
        # 这里我们把它拼在序列里，dataset处理时再拆分
        
        feature_cols = ['norm_lat', 'norm_lon', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        seq_features = group[feature_cols].values
        user_idx = group['user_index'].iloc[0] # 该轨迹的用户ID
        
        seq_len = len(group)
        
        # --- 滑动窗口切分逻辑 ---
        if seq_len <= WINDOW_SIZE:
            # 如果轨迹短于窗口，直接保留 (后续在 Dataset 里做 Padding)
            # 我们保存原始数据，不在这里做 padding，节省存储空间
            all_sequences.append({
                'features': seq_features,      # (L, 6)
                'user_id': user_idx,           # int
                'length': seq_len,
                'track_id': track_id
            })
        else:
            # 如果轨迹过长，进行切分
            for start in range(0, seq_len, STRIDE):
                end = start + WINDOW_SIZE
                # 截取片段
                sub_seq = seq_features[start:end]
                # 如果最后一段太短（例如小于 20），丢弃，防止噪音
                if len(sub_seq) >= 20: 
                    all_sequences.append({
                        'features': sub_seq,
                        'user_id': user_idx,
                        'length': len(sub_seq),
                        'track_id': f"{track_id}_sub_{start}" # 标记一下是切分出来的
                    })
        
        processed_count += 1
        if processed_count % 5000 == 0:
            print(f"已处理 {processed_count}/{total_tracks} 条原始轨迹...")

    # 6. 保存所有结果
    save_data = {
        'sequences': all_sequences,     # 核心数据列表
        'scale_params': scale_params,   # 归一化参数 (反归一化用)
        'num_users': num_users,         # 用户总数 (定义 Embedding 层大小用)
        'user_mapping': user_mapping,   # 用户ID映射表
        'feature_names': feature_cols   # 特征名字
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print("=" * 50)
    print("预处理完成！")
    print(f"原始轨迹段数: {total_tracks}")
    print(f"切分后序列数: {len(all_sequences)}")
    print(f"用户总数: {num_users}")
    print(f"归一化范围: Lat[{min_lat}, {max_lat}], Lon[{min_lon}, {max_lon}]")
    print(f"结果已保存至: {output_path}")
    print("=" * 50)
    print("输出文件说明:")
    print("sequences: 包含列表，每个元素是一个 dict {'features': np.array, 'user_id': int, ...}")
    print("features 维度: (Length, 6)。6列分别为: norm_lat, norm_lon, h_sin, h_cos, d_sin, d_cos")

if __name__ == "__main__":
    preprocess_data(INPUT_FILE, OUTPUT_PKL)



# 数据以一种结构化的字典列表形式存储，而不是简单地把ID作为一个数字拼接在经纬度后面
# 轨迹特征处理：
# features (维度 6) 通过一个 Linear 层 映射到高维（例如维度 128）。
# 用户 ID 处理：
# user_id (整数 5) 输入到一个 Embedding 层（可以理解为一个查表操作）。
# 表里存着 User 5 的“人物画像向量”（也是维度 128）。这个向量是模型自己学出来的，代表 User 5 的个性（比如他喜欢走直线，还是喜欢去特定区域）。
# 融合 (Fusion)：
# 将 轨迹向量 和 用户向量 相加（Element-wise Add）或者 拼接（Concat）。
# 相加的逻辑：相当于给这条轨迹打上了“User 5 风格”的底色。