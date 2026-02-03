#speed、user_id、date、time_interval
import pandas as pd
import numpy as np
import os

# --- 参数配置 ---
INPUT_FILE = '/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master/narrow_down/changping.csv'
OUTPUT_FILE = '/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/feet_tracks.csv'

# 阈值设置
MAX_SPEED_KMH = 12.0  # 最大步行速度 (km/h)
MAX_SPEED_MS = MAX_SPEED_KMH / 3.6  # 转换为 m/s (约 3.33 m/s)

# 时间中断阈值 (秒): 如果两点间隔超过此时间(例如20分钟)，即使距离很近，也视为轨迹中断
# 这是为了防止用户在一个地方停留太久后继续走，被强行连成一段
MAX_TIME_GAP_SEC = 20 * 60 

def haversine_np(lon1, lat1, lon2, lat2):
    """
    使用 Numpy 向量化计算 Haversine 距离 (单位: 米)
    输入可以是数值，也可以是 Pandas Series (列)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # 地球半径 (米)
    return c * r

def process_tracks(input_path, output_path):
    print(f"正在读取数据: {input_path} ...")
    # 读取 CSV
    df = pd.read_csv(input_path)
    
    # 1. 数据预处理
    print("正在预处理数据 (合并时间, 排序)...")
    # 合并日期和时间，转为 datetime 对象
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['timestamp'])
    
    # 确保数据是严格按 用户 -> 时间 排序的
    df = df.sort_values(by=['user_id', 'datetime']).reset_index(drop=True)

    # 2. 向量化计算 下一点 (Next Point) 的信息
    # shift(-1) 表示把整列向上移一格，这样第 i 行就能看到第 i+1 行的数据
    print("正在计算距离与速度...")
    
    df['next_lat'] = df['lat'].shift(-1)
    df['next_lon'] = df['lon'].shift(-1)
    df['next_time'] = df['datetime'].shift(-1)
    df['next_user'] = df['user_id'].shift(-1)
    df['next_date'] = df['date'].shift(-1)

    # 计算距离 (米)
    # 注意：最后一行 shift 后是 NaN，计算结果也是 NaN，不影响逻辑
    df['dist_m'] = haversine_np(df['lon'], df['lat'], df['next_lon'], df['next_lat'])

    # 计算时间差 (秒)
    df['time_diff_s'] = (df['next_time'] - df['datetime']).dt.total_seconds()

    # 计算速度 (m/s)
    # 处理除以零的情况：如果时间差为0，速度设为0 (或者设为无穷大，视情况而定，这里设为0防止报错)
    df['speed_ms'] = np.where(df['time_diff_s'] > 0, df['dist_m'] / df['time_diff_s'], 0)

    # 3. 识别断点 (Cut Points)
    # 如果满足以下任意条件，则说明当前点 i 和点 i+1 之间断开了：
    # 1. 用户变了
    # 2. 日期变了
    # 3. 速度超过阈值 (非步行)
    # 4. 时间间隔过大 (数据丢失或停留)
    
    print(f"正在根据速度阈值 ({MAX_SPEED_KMH} km/h) 切分轨迹...")
    
    # 初始化断点标记 (False 表示相连，True 表示断开)
    # 默认所有点之间都是连着的
    df['is_cut'] = False
    
    # 条件1: 用户不同
    df.loc[df['user_id'] != df['next_user'], 'is_cut'] = True
    
    # 条件2: 日期不同 (跨天)
    df.loc[df['date'] != df['next_date'], 'is_cut'] = True
    
    # 条件3: 速度过快 (根据你的要求，这部分即去除 bus/taxi 等)
    df.loc[df['speed_ms'] > MAX_SPEED_MS, 'is_cut'] = True
    
    # 条件4: 时间间隔过大 (可选，优化轨迹质量)
    df.loc[df['time_diff_s'] > MAX_TIME_GAP_SEC, 'is_cut'] = True

    # 4. 生成 Segment ID (临时轨迹段 ID)
    # 逻辑：cumsum() 会对 True (1) 进行累加。每次遇到 True，ID 就会 +1，从而开启新的一段
    # 我们需要 shift(1) 是因为 'is_cut' 描述的是 i 和 i+1 之间断开，所以 i+1 应该是新段的开始
    df['segment_id'] = df['is_cut'].shift(1).fillna(False).astype(int).cumsum()

    # 5. 过滤与重命名
    print("正在生成 track_id 并清理过短轨迹...")

    # 统计每一段有多少个点
    seg_counts = df.groupby('segment_id').size()
    
    # 只保留点数 >= 3 的轨迹
    # (只有2个点的轨迹往往是一个孤立的向量，对于步行分析意义不大，可根据需求改为 >=2)
    valid_segs = seg_counts[seg_counts >= 3].index
    
    # 筛选数据
    clean_df = df[df['segment_id'].isin(valid_segs)].copy()
    
    if clean_df.empty:
        print("警告：没有符合条件的轨迹数据！")
        return

    # 生成最终的 track_id (例如 000_1, 000_2)
    # 利用 groupby 和 cumcount 生成组内序号
    # dense_rank 确保序号是连续的 (1, 2, 3...)
    
    # 首先我们要给每个 segment 一个针对该用户的排名
    # 我们取每个 segment 的第一行来代表该 segment
    seg_meta = clean_df.drop_duplicates(subset=['segment_id'])[['segment_id', 'user_id', 'datetime']].sort_values(['user_id', 'datetime'])
    
    # 在元数据层面上，计算每个用户下的 segment 序号
    seg_meta['user_track_idx'] = seg_meta.groupby('user_id').cumcount() + 1
    
    # 将序号映射回主表
    seg_map = dict(zip(seg_meta['segment_id'], seg_meta['user_track_idx']))
    clean_df['track_suffix'] = clean_df['segment_id'].map(seg_map)
    
    # 拼接字符串
    clean_df['track_id'] = clean_df['user_id'].astype(str) + '_' + clean_df['track_suffix'].astype(str)

    # 6. 保存结果
    # 保留所需的列
    output_cols = ['track_id', 'user_id', 'lat', 'lon', 'date', 'timestamp']
    clean_df[output_cols].to_csv(output_path, index=False, encoding='utf-8')
    
    # 统计信息
    original_points = len(df)
    kept_points = len(clean_df)
    num_tracks = clean_df['track_id'].nunique()
    
    print("=" * 50)
    print("处理完成！")
    print(f"原始点数: {original_points}")
    print(f"清洗后点数: {kept_points} (移除率: {(1 - kept_points/original_points):.2%})")
    print(f"生成步行轨迹段数: {num_tracks}")
    print(f"结果已保存至: {output_path}")
    print("=" * 50)

if __name__ == '__main__':
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
    else:
        process_tracks(INPUT_FILE, OUTPUT_FILE)