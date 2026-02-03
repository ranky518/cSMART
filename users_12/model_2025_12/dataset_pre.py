import os
import glob
import pandas as pd
import numpy as np
import pickle
# from sklearn.cluster import MiniBatchKMeans  <-- 不需要聚类了
from tqdm import tqdm

# ========== 配置 ==========
BASE_DIR = "/home/yanglanqi/trajectory/zx_users_12"
RAW_DATA_DIR = os.path.join(BASE_DIR, "result_by_user")
FILTERED_LIST_PATH = os.path.join(BASE_DIR, "analysis_results/high_quality_user_ids.txt")

# 输出路径
OUTPUT_DIR = "/home/yanglanqi/trajectory/zx_users_12/model_2025_12/preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "zx_station_dataset.pkl")

# 参数设置
# NUM_CLUSTERS = 20000  # <--- 该参数已废弃，词表大小将由实际基站数量自动决定
SAMPLE_RATE_MIN = 1   # 采样率 (分钟)
SEQ_LEN = 96          # 序列长度

def load_qualified_users():
    """加载筛选后的高质量用户ID列表"""
    if not os.path.exists(FILTERED_LIST_PATH):
        raise FileNotFoundError(f"找不到筛选列表: {FILTERED_LIST_PATH}，请先运行 analysis.py")
    
    with open(FILTERED_LIST_PATH, 'r') as f:
        user_ids = [line.strip() for line in f if line.strip()]
    
    print(f"已加载 {len(user_ids)} 个筛选用户ID。")
    return set(user_ids)

def build_location_vocab(file_list):
    """
    第一遍扫描：扫描所有文件，提取唯一的 (lat, lon) 坐标作为 Token
    替代原先的 K-Means 聚类
    """
    print("开始构建基站词表 (Scanning unique locations)...")
    unique_locs = set()
    
    for fp in tqdm(file_list, desc="Scanning Locations"):
        try:
            # 只读取经纬度列
            df = pd.read_csv(fp, usecols=['c_lat', 'c_lng'])
            
            # [关键] 简单保留6位小数，防止浮点数微小差异导致同一个基站被识别为两个
            df = df.round({'c_lat': 6, 'c_lng': 6})
            
            # 转换为元组并添加到集合中去重
            locs = list(zip(df['c_lat'], df['c_lng']))
            unique_locs.update(locs)
                
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            
    # 构建词表: 将集合转为有序列表，并建立映射
    # loc_to_id: {(lat, lon): id}
    sorted_locs = sorted(list(unique_locs)) # 排序保证 ID 的确定性
    loc_to_id = {loc: idx for idx, loc in enumerate(sorted_locs)}
    
    # 词表中心 (Token -> LatLon)，保持格式兼容，依然命名为 cluster_centers
    vocab_centers = np.array(sorted_locs, dtype=np.float32)
    
    print(f"基站词表构建完成。共发现 {len(vocab_centers)} 个唯一基站坐标。")
    return loc_to_id, vocab_centers

def process_and_tokenize(file_list, loc_to_id):
    """
    第二遍扫描：将经纬度转换为 Token 序列 (查表法)
    """
    print("开始数据转换 (LatLon -> TokenID)...")
    dataset_sequences = []
    
    # 建立 user_id 的映射 (原始ID字符串 -> 数字ID 0~N)
    user_str_to_int = {}
    
    global_track_id = 0
    
    for fp in tqdm(file_list, desc="Tokenizing"):
        try:
            uid_str = os.path.basename(fp).split('.')[0]
            
            # 读取数据
            df = pd.read_csv(fp)
            df['c_event_time'] = pd.to_datetime(df['c_event_time'])
            
            # [新增] 关键修改：去除重复时间戳，保留最后一条 (或第一条)
            df = df.drop_duplicates(subset=['c_event_time'], keep='last')
            
            df = df.sort_values('c_event_time')
            
            # [关键] 同样保留6位小数，以匹配词表中的 Key
            df = df.round({'c_lat': 6, 'c_lng': 6})
            
            # 映射 User ID
            if uid_str not in user_str_to_int:
                user_str_to_int[uid_str] = len(user_str_to_int)
            uid_int = user_str_to_int[uid_str]
            
            # 设置时间索引
            df = df.set_index('c_event_time')
            times = df.index
            
            # 提取坐标列并查表映射为 Token ID
            lat_col = df['c_lat'].values
            lng_col = df['c_lng'].values
            
            # 列表推导式查表，比 DataFrame apply 更快
            tokens = [loc_to_id.get((r_lat, r_lng), -1) for r_lat, r_lng in zip(lat_col, lng_col)]
            tokens = np.array(tokens)
            
            # 检查是否有未匹配到的坐标 (-1)
            if np.any(tokens == -1):
                # 这种情况理论上不应发生，除非 set 和 list 处理 float 方式有微小差异
                print(f"Warning: Found unknown locations in {fp}, skipping unavailable points.")
                # 简单处理：过滤掉 -1 的点（或者报错）
                valid_mask = tokens != -1
                tokens = tokens[valid_mask]
                times = times[valid_mask]

            if len(tokens) < SEQ_LEN:
                continue

            # 切分轨迹 (Segmenting)
            # 策略：简单的滑动窗口，步长为 SEQ_LEN / 2
            step_size = SEQ_LEN // 2
            
            for i in range(0, len(tokens) - SEQ_LEN + 1, step_size):
                segment_tokens = tokens[i : i + SEQ_LEN]
                seg_times = times[i : i + SEQ_LEN]
                
                # 构造时间特征
                minutes_of_day = seg_times.hour * 60 + seg_times.minute
                time_feat = (minutes_of_day / 1440.0).to_numpy(dtype=np.float32)
                
                dataset_sequences.append({
                    'user_id': uid_int,
                    'track_id': global_track_id,
                    'tokens': segment_tokens.astype(np.int64), # Token ID 序列
                    'time_feat': time_feat,                    # 时间特征
                    'orig_uid': uid_str                        # 保留原始ID方便追溯
                })
                global_track_id += 1
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")

    return dataset_sequences, len(user_str_to_int)

def main():
    # 1. 获取目标文件列表
    qualified_ids = load_qualified_users()
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    target_files = [f for f in all_files if os.path.basename(f).split('.')[0] in qualified_ids]
    
    print(f"共待处理文件: {len(target_files)} (总文件数: {len(all_files)})")
    
    if len(target_files) == 0:
        print("错误：没有找到匹配的文件，请检查路径和ID列表。")
        return

    # 2. 构建基站词表 (替代原 K-Means)
    loc_to_id, vocab_centers = build_location_vocab(target_files)
    vocab_size = len(vocab_centers)
    
    # 3. 转换数据
    sequences, num_users = process_and_tokenize(target_files, loc_to_id)
    
    print(f"生成样本总数: {len(sequences)}")
    print(f"涉及用户总数: {num_users}")
    print(f"最终基站词表大小 (Vocab Size): {vocab_size}")

    # 4. 保存结果
    meta_data = {
        'vocab_size': vocab_size,      # 动态大小，不再是固定的 20000
        'num_users': num_users,
        'cluster_centers': vocab_centers, # 实际上是 exact base station locations
        'sequences': sequences
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(meta_data, f)
        
    print(f"\n[完成] 预处理数据已保存至: {OUTPUT_PKL}")
    print("注意：训练时 vocab_size 将自动根据基站数量调整，请确保模型 Embedding 层能适配。")

if __name__ == "__main__":
    main()
