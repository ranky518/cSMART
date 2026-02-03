import os
import pandas as pd
import numpy as np

def analyze_user_trajectory_intervals(file_path, time_column='c_event_time'):
    """
    从单个CSV文件中分析用户轨迹点之间的时间间隔。

    参数:
    file_path (str): CSV文件的路径。
    time_column (str): 表示时间的列名。

    返回:
    list of pandas.Timedelta or None: 所有时间间隔的列表，如果无法计算则返回None。
    """
    try:
        df = pd.read_csv(file_path)
        if time_column not in df.columns:
            print(f"警告: 文件 {file_path} 中缺少 '{time_column}' 列。跳过此文件。")
            return None

        if len(df) < 2:
            print(f"警告: 文件 {file_path} 的记录少于2条，无法计算间隔。跳过此文件。")
            return None

        # 转换时间列为datetime对象并排序
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(by=time_column)

        # 计算连续轨迹点之间的时间差
        intervals = df[time_column].diff().dropna()
        
        return intervals.tolist()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。")
        return None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def analyze_all_users_from_dir(base_dir, time_column='c_event_time'):
    """
    统计指定目录结构下所有用户轨迹点之间的时间间隔。

    参数:
    base_dir (str): 包含用户子目录的根目录路径。
    time_column (str): CSV中表示时间的列名。

    返回:
    list: 所有用户的所有时间间隔的列表。
    """
    all_intervals = []
    if not os.path.isdir(base_dir):
        print(f"错误: 目录 '{base_dir}' 不存在。")
        return []

    # 遍历根目录下的所有用户子目录
    user_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"找到 {len(user_dirs)} 个用户目录。")

    for user_dir_name in user_dirs:
        user_dir_path = os.path.join(base_dir, user_dir_name)
        csv_files = [f for f in os.listdir(user_dir_path) if f.endswith('.csv')]
        if csv_files:
            # 假设每个用户目录只处理一个CSV文件
            csv_file_path = os.path.join(user_dir_path, csv_files[0])
            intervals = analyze_user_trajectory_intervals(csv_file_path, time_column)
            if intervals:
                all_intervals.extend(intervals)
        else:
            print(f"警告: 目录 {user_dir_path} 中没有找到CSV文件。")
    
    return all_intervals

def main():
    """
    主函数，设置目录并执行统计分析。
    """
    data_directory = '/home/yanglanqi/trajectory/zx_users_11/result_by_users'
    
    # 'c_event_time' 应为记录轨迹点时间的列
    time_column = 'c_event_time'
    
    # 计算所有用户的所有轨迹点时间间隔
    all_intervals = analyze_all_users_from_dir(
        data_directory, 
        time_column=time_column
    )
    
    print(f"\n已成功分析 {len(all_intervals)} 个轨迹点时间间隔。")

    # 计算并打印时间间隔的统计信息
    if all_intervals:
        # 将Timedelta对象转换为总秒数以便计算
        intervals_in_seconds = [interval.total_seconds() for interval in all_intervals]
        
        # 筛选出一天内的间隔（一天 = 24 * 3600 = 86400秒）
        one_day_seconds = 24 * 60 * 60
        intervals_within_a_day_seconds = [s for s in intervals_in_seconds if s <= one_day_seconds]
        
        if intervals_within_a_day_seconds:
            print(f"\n在 {len(intervals_within_a_day_seconds)} 个一天内的间隔中进行统计:")
            
            min_interval = pd.to_timedelta(min(intervals_within_a_day_seconds), unit='s')
            max_interval = pd.to_timedelta(max(intervals_within_a_day_seconds), unit='s')
            avg_interval_seconds = np.mean(intervals_within_a_day_seconds)
            avg_interval = pd.to_timedelta(avg_interval_seconds, unit='s')
            median_interval_seconds = np.median(intervals_within_a_day_seconds)
            median_interval = pd.to_timedelta(median_interval_seconds, unit='s')
            
            print("\n一天内轨迹点时间间隔的统计 (采样频率分析):")
            print(f"  最短间隔 (最高采样频率): {min_interval}")
            print(f"  最长间隔 (最低采样频率): {max_interval}")
            print(f"  平均间隔: {avg_interval}")
            print(f"  间隔中位数: {median_interval}")
        else:
            print("未能找到任何一天内的时间间隔。")

        # 筛选出半天内的间隔（半天 = 12 * 3600 = 43200秒）
        half_a_day_seconds = 12 * 60 * 60
        intervals_within_half_a_day_seconds = [s for s in intervals_in_seconds if s <= half_a_day_seconds]
        
        if intervals_within_half_a_day_seconds:
            print(f"\n在 {len(intervals_within_half_a_day_seconds)} 个半天内的间隔中进行统计:")
            
            min_interval = pd.to_timedelta(min(intervals_within_half_a_day_seconds), unit='s')
            max_interval = pd.to_timedelta(max(intervals_within_half_a_day_seconds), unit='s')
            avg_interval_seconds = np.mean(intervals_within_half_a_day_seconds)
            avg_interval = pd.to_timedelta(avg_interval_seconds, unit='s')
            median_interval_seconds = np.median(intervals_within_half_a_day_seconds)
            median_interval = pd.to_timedelta(median_interval_seconds, unit='s')
            
            print("\n半天内轨迹点时间间隔的统计 (采样频率分析):")
            print(f"  最短间隔 (最高采样频率): {min_interval}")
            print(f"  最长间隔 (最低采样频率): {max_interval}")
            print(f"  平均间隔: {avg_interval}")
            print(f"  间隔中位数: {median_interval}")
        else:
            print("未能找到任何半天内的时间间隔。")
    else:
        print("未能计算任何轨迹点的时间间隔。")


if __name__ == '__main__':
    main()
