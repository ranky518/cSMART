import pandas as pd
import numpy as np

# filepath: /home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master/Model_pre/verify_distance.py
# 替换为你的 csv 实际路径
CSV_PATH = "/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/tracks_20.csv"

def haversine_np(lon1, lat1, lon2, lat2):
    """
    计算两点间哈弗辛距离（单位：米）
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km * 1000

def analyze_1min_distance():
    print("Loading data...")
    # 读取 CSV (假设列名为 latitude, longitude, ordered_time_str, user_id 等)
    # 根据你提供的 csv 截图，看起来有 user_id, latitude, longitude, ordered_time_str
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        return

    print(f"Total rows: {len(df)}")
    
    # 确保时间列是 datetime
    df['time'] = pd.to_datetime(df['timestamp'])
    
    # 排序
    df = df.sort_values(by=['user_id', 'time'])
    
    # 计算时间差 (秒)
    df['dt'] = df.groupby('user_id')['time'].diff().dt.total_seconds()
    
    # 计算位移 (米)
    # Shift latitude/longitude to get previous point
    df['prev_lat'] = df.groupby('user_id')['lat'].shift(1)
    df['prev_lon'] = df.groupby('user_id')['lon'].shift(1)
    
    # 只计算非空行
    mask = df['prev_lat'].notna()
    valid_df = df[mask].copy()
    
    valid_df['dist_m'] = haversine_np(
        valid_df['prev_lon'].values, valid_df['prev_lat'].values,
        valid_df['lon'].values, valid_df['lat'].values
    )
    
    # --- 筛选 1分钟 间隔的数据 ---
    # 你的数据如果已经规整化为 1分钟一条，那么 dt 应该大部分是 60s
    # 这里我们放宽一点范围 55s - 65s 视为 1分钟采样点
    target_df = valid_df[(valid_df['dt'] >= 55) & (valid_df['dt'] <= 65)]
    
    if len(target_df) == 0:
        print("Warning: No exact 1-minute intervals found. Checking average interval...")
        print(f"Average Delta Time: {valid_df['dt'].mean():.2f}s")
        # 如果数据不是严格1分钟对齐的，我们取所有移动状态下的平均速度推算
        moving_df = valid_df[valid_df['dist_m'] > 10] # 排除静止
        avg_speed_mps = (moving_df['dist_m'] / moving_df['dt']).mean()
        print(f"Est. 1-min displacement based on avg speed: {avg_speed_mps * 60:.2f} meters")
        return

    print(f"\nAnalyzing {len(target_df)} pairs of 1-minute intervals...")
    
    # 统计分布
    distances = target_df['dist_m']
    
    print("-" * 40)
    print(f"1-Minute Displacement Statistics:")
    print(f"Min:   {distances.min():.2f} m")
    print(f"Mean:  {distances.mean():.2f} m")
    print(f"Median:{distances.median():.2f} m")
    print(f"Max:   {distances.max():.2f} m")
    print("-" * 40)
    
    base_station_min = 100
    base_station_max = 300
    
    in_range = ((distances >= base_station_min) & (distances <= base_station_max)).sum()
    less_range = (distances < base_station_min).sum()
    more_range = (distances > base_station_max).sum()
    
    print(f"In Station Range ({base_station_min}-{base_station_max}m): {in_range} ({in_range/len(distances):.1%})")
    print(f"Under Station Range (<{base_station_min}m - Low Speed):  {less_range} ({less_range/len(distances):.1%})")
    print(f"Over Station Range (>{base_station_max}m - High Speed):   {more_range} ({more_range/len(distances):.1%})")

    # 简单的速度推断
    # 步行 (<85m/min), 骑行/慢车 (85-400m/min), 汽车 (>400m/min)
    walk = (distances < 85).sum()
    bike = ((distances >= 85) & (distances < 400)).sum()
    car  = (distances >= 400).sum()
    
    print("-" * 40)
    print("Transport Mode Inference (Approx.):")
    print(f"Walk (<5km/h): {walk/len(distances):.1%}")
    print(f"Bike/Traffic (5-24km/h): {bike/len(distances):.1%}")
    print(f"Car (>24km/h): {car/len(distances):.1%}")

if __name__ == "__main__":
    analyze_1min_distance()