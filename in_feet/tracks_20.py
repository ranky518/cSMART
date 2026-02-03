import pandas as pd
import numpy as np
import os

# --- 配置参数 ---
INPUT_FILE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/feet_tracks.csv'
OUTPUT_FILE = 'trajectory/geolife_clustering-master/geolife_clustering-master/in_feet/tracks_20.csv' # 输出给 Transformer 用的文件
MIN_POINTS_THRESHOLD = 20               # 你的筛选阈值

def filter_tracks(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    print(f"正在读取 {input_path} ...")
    df = pd.read_csv(input_path)
    
    # 1. 统计每个 track_id 的点数
    print("正在计算每条轨迹的点数...")
    track_counts = df['track_id'].value_counts()
    
    # 2. 筛选符合条件的 track_id
    # 保留点数 >= 20 的轨迹
    valid_track_ids = track_counts[track_counts >= MIN_POINTS_THRESHOLD].index
    
    # 从原始数据中提取这些轨迹
    filtered_df = df[df['track_id'].isin(valid_track_ids)].copy()
    
    # 确保按 track_id 和 时间 排序 (这对 Transformer 的序列输入至关重要)
    # 合并时间列用于排序
    filtered_df['dt'] = pd.to_datetime(filtered_df['date'] + ' ' + filtered_df['timestamp'])
    filtered_df = filtered_df.sort_values(['track_id', 'dt']).reset_index(drop=True)
    
    # 删除临时辅助列 'dt'，保持文件干净
    filtered_df = filtered_df.drop(columns=['dt'])

    if filtered_df.empty:
        print("警告: 筛选后没有剩余数据！请检查阈值是否过高。")
        return

    # --- 3. 统计筛选后的数据特征 (为 Transformer 参数做准备) ---
    original_tracks = len(track_counts)
    final_tracks = filtered_df['track_id'].nunique()
    original_points = len(df)
    final_points = len(filtered_df)
    
    # 计算筛选后的轨迹长度分布，辅助设定 max_seq_len
    final_counts = filtered_df['track_id'].value_counts()
    p90_len = final_counts.quantile(0.90)
    p95_len = final_counts.quantile(0.95)
    p99_len = final_counts.quantile(0.99)
    max_len = final_counts.max()

    print("=" * 50)
    print(f"【筛选结果统计 (阈值 >= {MIN_POINTS_THRESHOLD} 点)】")
    print("-" * 50)
    print(f"轨迹段数量: {original_tracks} -> {final_tracks} (保留了 {final_tracks/original_tracks:.1%})")
    print(f"轨迹点总数: {original_points} -> {final_points} (保留了 {final_points/original_points:.1%})")
    print("-" * 50)
    print("【Transformer 模型参数建议】")
    print("为了设置 Dataset 的 max_seq_len (Padding标准):")
    print(f"  - 90% 的轨迹长度在 {p90_len:.0f} 以内")
    print(f"  - 95% 的轨迹长度在 {p95_len:.0f} 以内 (推荐参考)")
    print(f"  - 99% 的轨迹长度在 {p99_len:.0f} 以内")
    print(f"  - 最长轨迹长度: {max_len}")
    print("=" * 50)

    # 4. 保存
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"清洗后的数据已保存至: {output_path}")

if __name__ == "__main__":
    filter_tracks(INPUT_FILE, OUTPUT_FILE)