import os
import glob
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# ========== 配置 ==========
BASE_DIR = "/home/yanglanqi/trajectory/zx_users_11"
RAW_DATA_DIR = os.path.join(BASE_DIR, "result_by_users")

# 输出路径
OUTPUT_PKL = os.path.join(BASE_DIR, "zx_user_11.pkl")

# 参数设置
SEQ_LEN = 96          # 序列长度

def find_all_csv_files(base_dir):
    """
    递归查找所有CSV文件
    目录结构: result_by_users/c_msisdn=X/part-*.csv
    """
    pattern = os.path.join(base_dir, "c_msisdn=*", "*.csv")
    files = glob.glob(pattern)
    print(f"Found {len(files)} CSV files")
    return files

def extract_user_id_from_path(filepath):
    """
    从路径中提取用户ID
    路径格式: .../c_msisdn=0/part-xxx.csv
    """
    # 获取父目录名
    parent_dir = os.path.basename(os.path.dirname(filepath))
    # 提取 c_msisdn= 后面的值
    if parent_dir.startswith("c_msisdn="):
        return parent_dir.split("=")[1]
    return parent_dir

def build_location_vocab(file_list):
    """
    第一遍扫描：扫描所有文件，提取唯一的 (lat, lon) 坐标作为 Token
    """
    print("=" * 60)
    print("Step 1: 构建基站词表 (Scanning unique locations)...")
    print("=" * 60)
    unique_locs = set()
    
    for fp in tqdm(file_list, desc="Scanning Locations"):
        try:
            # 读取CSV文件
            df = pd.read_csv(fp)
            
            # 检查必要的列是否存在
            if 'c_lat' not in df.columns or 'c_lng' not in df.columns:
                print(f"Warning: Missing lat/lng columns in {fp}")
                print(f"  Available columns: {df.columns.tolist()}")
                continue
            
            # 保留6位小数，防止浮点数微小差异
            df = df.round({'c_lat': 6, 'c_lng': 6})
            
            # 转换为元组并添加到集合中去重
            locs = list(zip(df['c_lat'], df['c_lng']))
            unique_locs.update(locs)
                
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            
    # 构建词表: 将集合转为有序列表，并建立映射
    sorted_locs = sorted(list(unique_locs))
    loc_to_id = {loc: idx for idx, loc in enumerate(sorted_locs)}
    
    # 词表中心 (Token -> LatLon)
    vocab_centers = np.array(sorted_locs, dtype=np.float32)
    
    print(f"\n基站词表构建完成:")
    print(f"  - 唯一基站数量: {len(vocab_centers)}")
    if len(vocab_centers) > 0:
        print(f"  - 经度范围: [{vocab_centers[:, 1].min():.6f}, {vocab_centers[:, 1].max():.6f}]")
        print(f"  - 纬度范围: [{vocab_centers[:, 0].min():.6f}, {vocab_centers[:, 0].max():.6f}]")
    
    return loc_to_id, vocab_centers

def process_and_tokenize(file_list, loc_to_id):
    """
    第二遍扫描：将经纬度转换为 Token 序列
    """
    print("\n" + "=" * 60)
    print("Step 2: 数据转换 (LatLon -> TokenID)...")
    print("=" * 60)
    
    dataset_sequences = []
    user_str_to_int = {}
    global_track_id = 0
    
    # 统计信息
    total_points = 0
    valid_sequences = 0
    skipped_short = 0
    
    for fp in tqdm(file_list, desc="Tokenizing"):
        try:
            # 提取用户ID
            uid_str = extract_user_id_from_path(fp)
            
            # 读取数据
            df = pd.read_csv(fp)
            
            # 检查必要的列
            required_cols = ['c_lat', 'c_lng', 'c_event_time']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in {fp}")
                continue
            
            # 解析时间戳
            df['c_event_time'] = pd.to_datetime(df['c_event_time'])
            
            # 去除重复时间戳
            df = df.drop_duplicates(subset=['c_event_time'], keep='last')
            df = df.sort_values('c_event_time')
            
            # 保留6位小数以匹配词表
            df = df.round({'c_lat': 6, 'c_lng': 6})
            
            total_points += len(df)
            
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
            
            # 查表映射
            tokens = np.array([loc_to_id.get((lat, lng), -1) 
                              for lat, lng in zip(lat_col, lng_col)])
            
            # 过滤无效点
            valid_mask = tokens != -1
            if not valid_mask.all():
                num_invalid = (~valid_mask).sum()
                print(f"Warning: {num_invalid} unknown locations in {fp}")
                tokens = tokens[valid_mask]
                times = times[valid_mask]

            if len(tokens) < SEQ_LEN:
                skipped_short += 1
                continue

            # 切分轨迹 (滑动窗口)
            step_size = SEQ_LEN // 2
            
            for i in range(0, len(tokens) - SEQ_LEN + 1, step_size):
                segment_tokens = tokens[i : i + SEQ_LEN]
                seg_times = times[i : i + SEQ_LEN]
                
                # 构造时间特征 (一天中的分钟数 / 1440)
                minutes_of_day = seg_times.hour * 60 + seg_times.minute
                time_feat = (minutes_of_day / 1440.0).to_numpy(dtype=np.float32)
                
                dataset_sequences.append({
                    'user_id': uid_int,
                    'track_id': global_track_id,
                    'tokens': segment_tokens.astype(np.int64),
                    'time_feat': time_feat,
                    'orig_uid': uid_str
                })
                global_track_id += 1
                valid_sequences += 1
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n处理完成:")
    print(f"  - 总轨迹点数: {total_points}")
    print(f"  - 生成序列数: {valid_sequences}")
    print(f"  - 跳过(太短): {skipped_short}")
    print(f"  - 用户数量: {len(user_str_to_int)}")
    
    return dataset_sequences, len(user_str_to_int)

def main():
    print("=" * 60)
    print("ZX Users 11 数据集预处理")
    print("=" * 60)
    
    # 1. 查找所有CSV文件
    all_files = find_all_csv_files(RAW_DATA_DIR)
    
    if len(all_files) == 0:
        print("错误：没有找到CSV文件！")
        print(f"请检查目录: {RAW_DATA_DIR}")
        return
    
    # 显示样例文件
    print(f"\n样例文件路径:")
    for f in all_files[:3]:
        print(f"  {f}")
    if len(all_files) > 3:
        print(f"  ... 还有 {len(all_files) - 3} 个文件")

    # 2. 构建基站词表
    loc_to_id, vocab_centers = build_location_vocab(all_files)
    vocab_size = len(vocab_centers)
    
    if vocab_size == 0:
        print("错误：没有找到有效的基站坐标！")
        return
    
    # 3. 转换数据
    sequences, num_users = process_and_tokenize(all_files, loc_to_id)
    
    if len(sequences) == 0:
        print("警告：没有生成任何序列！")
        print("可能原因：所有轨迹长度都小于 SEQ_LEN =", SEQ_LEN)

    # 4. 保存结果
    print("\n" + "=" * 60)
    print("Step 3: 保存结果")
    print("=" * 60)
    
    meta_data = {
        'vocab_size': vocab_size,
        'num_users': num_users,
        'cluster_centers': vocab_centers,  # 基站坐标 (lat, lng)
        'sequences': sequences
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(meta_data, f)
        
    print(f"\n[完成] 数据已保存至: {OUTPUT_PKL}")
    print(f"\n数据集统计:")
    print(f"  - 基站词表大小 (vocab_size): {vocab_size}")
    print(f"  - 用户数量 (num_users): {num_users}")
    print(f"  - 序列数量: {len(sequences)}")
    if len(sequences) > 0:
        print(f"  - 序列长度: {SEQ_LEN}")
        print(f"\n样例序列:")
        seq = sequences[0]
        print(f"  user_id: {seq['user_id']}")
        print(f"  track_id: {seq['track_id']}")
        print(f"  orig_uid: {seq['orig_uid']}")
        print(f"  tokens shape: {seq['tokens'].shape}")
        print(f"  time_feat shape: {seq['time_feat'].shape}")
        print(f"  tokens[:10]: {seq['tokens'][:10]}")

if __name__ == "__main__":
    main()
