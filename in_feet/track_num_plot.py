import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置 ---
INPUT_FILE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/feet_tracks.csv'
OUTPUT_IMAGE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/track_point_counts.png'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_counts(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    print("正在读取数据...")
    df = pd.read_csv(file_path)

    # 1. 统计每个 track_id 的点数
    print("正在统计点数...")
    # value_counts() 默认按数量降序排列，sort_index() 将其按 track_id 排序（这里我们不需要按id排，保持数值排序即可）
    counts = df['track_id'].value_counts()
    
    # 2. 计算基本统计量
    total_tracks = len(counts)
    min_pts = counts.min()
    max_pts = counts.max()
    mean_pts = counts.mean()
    median_pts = counts.median()
    
    # 计算分位数 (Percentiles)
    p10 = counts.quantile(0.10)
    p25 = counts.quantile(0.25)
    p75 = counts.quantile(0.75)
    p90 = counts.quantile(0.90)
    p95 = counts.quantile(0.95)
    p99 = counts.quantile(0.99)

    print("=" * 40)
    print("【轨迹点数统计分析】")
    print(f"轨迹段总数 (Tracks): {total_tracks}")
    print("-" * 40)
    print(f"最少点数 (Min): {min_pts}")
    print(f"最多点数 (Max): {max_pts}")
    print(f"平均点数 (Mean): {mean_pts:.2f}")
    print(f"中位数点数 (Median): {median_pts:.0f}")
    print("-" * 40)
    print("【分布详情】")
    print(f"10% 的轨迹点数少于: {p10:.0f}")
    print(f"25% 的轨迹点数少于: {p25:.0f}")
    print(f"50% 的轨迹点数少于: {median_pts:.0f}")
    print(f"75% 的轨迹点数少于: {p75:.0f}")
    print(f"90% 的轨迹点数少于: {p90:.0f}")
    print(f"95% 的轨迹点数少于: {p95:.0f}")
    print(f"99% 的轨迹点数少于: {p99:.0f}")
    print("=" * 40)

    # --- 3. 绘图 ---
    print("正在绘制图表...")
    plt.figure(figsize=(14, 8))

    # 子图1: 频数直方图 (Histogram) - 关注主体部分
    # 我们限制显示范围在 95% 分位数以内，不然极值会让柱子挤在一起看不清
    plt.subplot(2, 2, 1)
    limit_x = p95
    data_core = counts[counts <= limit_x]
    plt.hist(data_core, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'点数分布直方图 (前95%的数据, <= {limit_x:.0f}点)')
    plt.xlabel('包含的点数')
    plt.ylabel('轨迹数量')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 子图2: 箱线图 (Boxplot) - 看整体和离群值
    plt.subplot(2, 2, 2)
    plt.boxplot(counts, vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.title('点数分布箱线图 (含离群值)')
    plt.xlabel('包含的点数')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # 子图3: 累计分布函数 (CDF) - 决策神器
    plt.subplot(2, 2, 3)
    sorted_counts = np.sort(counts)
    yvals = np.arange(len(sorted_counts)) / float(len(sorted_counts) - 1)
    plt.plot(sorted_counts, yvals, color='orange', linewidth=2)
    
    # 辅助线：比如我们想看有多少轨迹少于 10 个点
    threshold_low = 10
    percent_low = (counts < threshold_low).mean()
    plt.axvline(threshold_low, color='red', linestyle='--', alpha=0.5)
    plt.text(threshold_low, 0.5, f' x={threshold_low}\n {percent_low:.1%}的数据在此左侧', color='red', fontsize=9)
    
    # 限制CDF的X轴显示范围，方便看清楚前面的变化
    plt.xlim(0, p95 * 1.2) 
    plt.title('累计分布函数 CDF (局部放大)')
    plt.xlabel('包含的点数')
    plt.ylabel('累计比例 (0~1)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 子图4: 对数坐标全貌
    plt.subplot(2, 2, 4)
    plt.hist(counts, bins=100, color='purple', alpha=0.7)
    plt.yscale('log')
    plt.title('点数分布全貌 (Y轴对数坐标)')
    plt.xlabel('包含的点数')
    plt.ylabel('轨迹数量 (Log)')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"图表已保存至: {OUTPUT_IMAGE}")
    print("建议参考【分布详情】中的分位数来设定你的过滤阈值。")

if __name__ == "__main__":
    analyze_counts(INPUT_FILE)