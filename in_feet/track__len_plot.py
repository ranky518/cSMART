import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置 ---
INPUT_FILE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/feet_tracks.csv'
OUTPUT_IMAGE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/track_length_distribution.png'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def haversine_np(lon1, lat1, lon2, lat2):
    """ 计算两点间距离 (米) """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return c * r

def analyze_lengths(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    print("正在读取数据...")
    df = pd.read_csv(file_path)
    
    # 确保按 track_id 和 时间 排序，否则计算距离会乱
    # 合并日期时间以便正确排序
    df['dt'] = pd.to_datetime(df['date'] + ' ' + df['timestamp'])
    df = df.sort_values(['track_id', 'dt']).reset_index(drop=True)

    print("正在计算轨迹长度...")
    
    # --- 向量化计算每一步的距离 ---
    # 1. 获取前一个点的坐标和 track_id
    df['prev_lat'] = df['lat'].shift(1)
    df['prev_lon'] = df['lon'].shift(1)
    df['prev_track'] = df['track_id'].shift(1)
    
    # 2. 计算当前点与上一点的距离
    df['step_dist'] = haversine_np(df['lon'], df['lat'], df['prev_lon'], df['prev_lat'])
    
    # 3. 如果 track_id 变了（新的一段轨迹开始），则该步距离设为 0
    df.loc[df['track_id'] != df['prev_track'], 'step_dist'] = 0
    
    # 4. 第一行一定是 0
    df.loc[0, 'step_dist'] = 0
    
    # 5. 按 track_id 分组求和
    track_stats = df.groupby('track_id')['step_dist'].sum()
    
    # --- 统计分析 ---
    count = len(track_stats)
    mean_len = track_stats.mean()
    median_len = track_stats.median()
    max_len = track_stats.max()
    min_len = track_stats.min()
    total_len = track_stats.sum() / 1000 # km

    print("=" * 40)
    print("【轨迹长度统计结果】")
    print(f"总轨迹段数: {count}")
    print(f"平均长度: {mean_len:.2f} 米")
    print(f"中位数长度: {median_len:.2f} 米")
    print(f"最短轨迹: {min_len:.2f} 米")
    print(f"最长轨迹: {max_len:.2f} 米")
    print(f"所有轨迹总里程: {total_len:.2f} 千米")
    print("-" * 40)
    
    # 分位数查看
    print("分位数统计:")
    print(track_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print("=" * 40)

    # --- 绘图 ---
    print("正在绘制分布图...")
    plt.figure(figsize=(12, 10))

    # 子图 1: 频数直方图 (Histogram)
    plt.subplot(2, 2, 1)
    # 这里的 bins 设为 50，范围根据 99% 的数据来定，避免极值把图拉得太长
    limit_99 = track_stats.quantile(0.99)
    data_for_hist = track_stats[track_stats <= limit_99]
    
    plt.hist(data_for_hist, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'轨迹长度分布直方图 (前99%的数据)\n过滤掉了 > {limit_99:.0f}米的极值')
    plt.xlabel('长度 (米)')
    plt.ylabel('轨迹数量')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 子图 2: 箱线图 (Boxplot) - 用于看整体分布和离群点
    plt.subplot(2, 2, 2)
    plt.boxplot(track_stats, vert=False, patch_artist=True, 
                boxprops=dict(facecolor="lightgreen"))
    plt.title('轨迹长度箱线图 (含离群值)')
    plt.xlabel('长度 (米)')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # 子图 3: 累计分布函数 (CDF)
    plt.subplot(2, 2, 3)
    sorted_data = np.sort(track_stats)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    plt.plot(sorted_data, yvals, color='orange', linewidth=2)
    plt.title('累计分布函数 (CDF)')
    plt.xlabel('长度 (米)')
    plt.ylabel('累计比例 (0~1)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 在 CDF 上标出中位数
    plt.axvline(median_len, color='red', linestyle='--', alpha=0.6)
    plt.text(median_len, 0.5, f' 中位数: {median_len:.0f}m', color='red')

    # 子图 4: 对数坐标直方图 (如果长尾效应很明显)
    plt.subplot(2, 2, 4)
    plt.hist(track_stats, bins=50, color='purple', edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.title('轨迹长度分布 (Y轴对数坐标)')
    plt.xlabel('长度 (米)')
    plt.ylabel('轨迹数量 (Log Scale)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"图表已保存至: {OUTPUT_IMAGE}")
    # plt.show()

if __name__ == "__main__":
    analyze_lengths(INPUT_FILE)