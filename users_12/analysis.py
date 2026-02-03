import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 配置 (优化路径管理) ==========
BASE_DIR = "/home/yanglanqi/trajectory/zx_users_12"  # 根目录
DATA_DIR = os.path.join(BASE_DIR, "result_by_user")

# 将结果统一放到一个文件夹下，方便下载查看
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
OUTPUT_REPORT = os.path.join(RESULTS_DIR, "user_stats_report.csv")
FILTERED_LIST = os.path.join(RESULTS_DIR, "qualified_users_list.csv")

# 创建输出目录
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"结果输出目录已创建: {RESULTS_DIR}")

def analyze_users():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"找到 {len(csv_files)} 个用户文件。开始分析...")

    stats_list = []

    for file_path in csv_files:
        try:
            # 读取CSV，假设格式如 0.csv 所示
            # c_msisdn,c_lat,c_lng,c_event_time
            df = pd.read_csv(file_path)
            
            if df.empty:
                continue

            # 转换时间列
            df['c_event_time'] = pd.to_datetime(df['c_event_time'])
            
            # 基础统计
            user_id = os.path.basename(file_path).split('.')[0]
            num_points = len(df)
            
            # 时间跨度
            min_time = df['c_event_time'].min()
            max_time = df['c_event_time'].max()
            duration_days = (max_time - min_time).total_seconds() / (3600 * 24)
            
            # 采样频率估算 (中位数间隔)
            # 先排序
            df = df.sort_values('c_event_time')
            time_diffs = df['c_event_time'].diff().dropna().dt.total_seconds()
            median_interval_sec = time_diffs.median() if not time_diffs.empty else 0
            
            # 有效天数 (有多少天是有数据的)
            active_days = df['c_event_time'].dt.date.nunique()

            stats_list.append({
                'user_id': user_id,
                'num_points': num_points,
                'start_time': min_time,
                'end_time': max_time,
                'duration_days': duration_days,
                'active_days': active_days,
                'median_interval_sec': median_interval_sec
            })
            
        except Exception as e:
            print(f"处理文件 {file_path} 出错: {e}")

    # 转为 DataFrame
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values(by='num_points', ascending=False)
    
    # 保存原始统计报告
    stats_df.to_csv(OUTPUT_REPORT, index=False)
    print(f"统计报告已保存至: {OUTPUT_REPORT}")

    # ========== 统计可视化与分析 ==========
    print("\n========== 数据分布概况 ==========")
    print(stats_df.describe())

    # 绘制分布图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(stats_df['num_points'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Data Points per User')
    plt.xlabel('Number of Points')
    plt.ylabel('User Count')
    plt.yscale('log') # 对数坐标，因为通常长尾分布严重

    plt.subplot(1, 3, 2)
    plt.hist(stats_df['active_days'], bins=50, color='salmon', edgecolor='black')
    plt.title('Distribution of Active Days')
    plt.xlabel('Active Days')
    
    plt.subplot(1, 3, 3)
    # 过滤掉极端异常值以便绘图
    valid_intervals = stats_df[stats_df['median_interval_sec'] < 3600]['median_interval_sec']
    plt.hist(valid_intervals, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Median Sampling Interval (sec, < 1h)')
    plt.xlabel('Seconds')

    plt.tight_layout()
    viz_path = os.path.join(PLOT_DIR, "user_stats_dist.png")
    plt.savefig(viz_path)
    print(f"分布图已保存至: {viz_path}")

    # ========== 建议的筛选策略 ==========
    # 制定策略逻辑：
    # 1. 活跃天数：至少多少天的数据才有研究周期性规律的价值？通常建议 > 7天 或 > 30天
    # 2. 数据点量：每天至少有多少个点？假设每天10个点，30天即300点。
    # 3. 采样间隔：间隔太大数据过于稀疏，无法做轨迹恢复或预测。
    
    # 根据需求调整：筛选出数据点 >= 10000 的用户
    min_active_days = 5     # 保持基础活跃天数要求
    min_points = 10000      # [修改] 改为10000，提取头部约51%的用户
    
    qualified_users = stats_df[
        (stats_df['active_days'] >= min_active_days) & 
        (stats_df['num_points'] >= min_points)
    ]

    percentage = len(qualified_users) / len(stats_df) * 100 if len(stats_df) > 0 else 0

    print("\n========== 执行筛选 ==========")
    print(f"应用条件: 活跃天数 >= {min_active_days} 天 且 总数据点数 >= {min_points}")
    print(f"符合条件的用户数: {len(qualified_users)} / {len(stats_df)} ({percentage:.2f}%)")
    
    # 导出筛选后的用户列表
    qualified_users.to_csv(FILTERED_LIST, index=False)
    print(f"\n[完成] 详细用户列表(CSV)已保存至: {FILTERED_LIST}")

    # [新增] 额外保存一份纯 UserID 列表 (txt)，方便后续脚本读取
    id_list_path = os.path.join(RESULTS_DIR, "high_quality_user_ids.txt")
    with open(id_list_path, 'w') as f:
        for uid in qualified_users['user_id']:
            f.write(f"{uid}\n")
    print(f"[完成] 纯用户ID列表(TXT)已保存至: {id_list_path}")
    
    print(f"提示: 您可以直接下载整个文件夹 '{RESULTS_DIR}' 到本地。")

    # 分档统计，帮助你做决策
    print("\n--- 分档统计参考 ---")
    thresholds = [100, 500, 1000, 5000, 10000]
    print(f"{'Min Points':<15} | {'User Count':<15} | {'Ratio':<15}")
    print("-" * 50)
    for th in thresholds:
        count = len(stats_df[stats_df['num_points'] >= th])
        ratio = count / len(stats_df) * 100
        print(f">= {th:<13} | {count:<15} | {ratio:.2f}%")

if __name__ == "__main__":
    analyze_users()
