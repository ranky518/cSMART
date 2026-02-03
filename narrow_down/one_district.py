import os
import io
import csv
import glob

# --- 1. 定义筛选范围 (基于topN.py之前的运行结果) ---
MIN_LNG, MAX_LNG = 116.2012, 116.4963
MIN_LAT, MAX_LAT = 39.8189, 40.0003

def is_in_target_area(lat, lon):
    """
    判断点是否在目标区域 (Changping/Beijing Core) 内
    """
    return (MIN_LAT <= lat <= MAX_LAT) and (MIN_LNG <= lon <= MAX_LNG)

def read_plt_file(file_path):
    """
    读取单个.plt轨迹文件
    返回：列表，每个元素为 [lat, lon, date, timestamp]
    """
    points = []
    try:
        with io.open(file_path, 'r', encoding='utf-8') as f:
            # 这里的逻辑稍微优化，避免读取整个文件到内存
            for i, line in enumerate(f):
                if i < 6: # 跳过前6行元数据
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 7:
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        # 提前在这里做筛选，如果不符合范围，直接丢弃，不占用内存
                        if is_in_target_area(lat, lon):
                            date = parts[5]
                            timestamp = parts[6]
                            points.append([lat, lon, date, timestamp])
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    return points

def process_and_save_trajectories(data_dir, output_csv_path, skip_labeled_users=True):
    """
    遍历数据，筛选点，并直接写入 CSV 文件
    """
    # 获取所有用户文件夹
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在 -> {data_dir}")
        return

    user_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    user_dirs.sort()
    
    print(f"找到 {len(user_dirs)} 个用户文件夹，准备开始处理...")
    print(f"目标区域: Lat({MIN_LAT}~{MAX_LAT}), Lng({MIN_LNG}~{MAX_LNG})")
    print(f"输出文件: {output_csv_path}")

    # 统计计数器
    total_points_saved = 0
    users_processed_count = 0
    users_skipped_label = 0
    
    # 打开 CSV 文件准备写入 (使用 'w' 模式，newline='' 防止空行)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        # 写入表头
        writer.writerow(['user_id', 'lat', 'lon', 'date', 'timestamp'])

        for user_id in user_dirs:
            user_path = os.path.join(data_dir, user_id)
            
            # 1. 检查 labels.txt
            labels_file = os.path.join(user_path, 'labels.txt')
            if skip_labeled_users and os.path.exists(labels_file):
                users_skipped_label += 1
                # print(f"用户 {user_id} 存在 labels.txt，跳过") # 调试时可打开
                continue

            # 2. 查找 Trajectory 文件夹
            trajectory_dir = os.path.join(user_path, 'Trajectory')
            if not os.path.exists(trajectory_dir):
                continue
            
            # 获取所有 plt 文件
            plt_files = glob.glob(os.path.join(trajectory_dir, '*.plt'))
            if not plt_files:
                continue

            users_processed_count += 1
            user_valid_points = 0
            
            print(f"正在处理用户 {user_id} ({len(plt_files)} 个文件)...", end="\r")

            # 3. 处理每个 PLT 文件
            for plt_file in plt_files:
                # read_plt_file 内部已经做了 is_in_target_area 的筛选
                valid_points = read_plt_file(plt_file)
                
                if valid_points:
                    # 批量写入 CSV
                    rows_to_write = [[user_id] + p for p in valid_points]
                    writer.writerows(rows_to_write)
                    
                    count = len(valid_points)
                    user_valid_points += count
                    total_points_saved += count
            
            # (可选) 打印该用户贡献了多少点，不需要可以注释掉
            # if user_valid_points > 0:
            #     print(f"用户 {user_id}: 保存了 {user_valid_points} 个点")

    print("\n" + "=" * 50)
    print("处理完成！")
    print("=" * 50)
    print(f"跳过带标签用户数: {users_skipped_label}")
    print(f"实际处理用户数: {users_processed_count}")
    print(f"共保存轨迹点数: {total_points_saved}")
    print(f"结果已保存至: {output_csv_path}")

if __name__ == "__main__":
    # 配置路径
    base_data_dir = "/home/yanglanqi/trajectory/Data_128"
    
    # 结果保存路径 (保存在当前脚本同目录下)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(current_dir, "changping.csv")
    
    # 开始执行
    process_and_save_trajectories(base_data_dir, output_csv, skip_labeled_users=True)