
import pandas as pd
import numpy as np

# 替换为你的文件路径
FILE_PATH = '/home/yanglanqi/trajectory/zx_users_12/model_2025_12/preprocessed_data/zx_station_dataset.pkl'

def inspect_users():
    print(f"正在加载: {FILE_PATH} ...")
    data = pd.read_pickle(FILE_PATH)
    
    seqs = data['sequences']
    print(f"总计轨迹条数: {len(seqs)}\n")

    # --- 1. 统计每个用户有多少条轨迹 ---
    user_counts = {}
    for item in seqs:
        uid = item['user_id']
        user_counts[uid] = user_counts.get(uid, 0) + 1
    
    print(f"用户 ID 范围: {min(user_counts.keys())} ~ {max(user_counts.keys())}")
    print(f"包含用户数量: {len(user_counts)}")
    print("-" * 50)

    # --- 2. 抽样展示特定用户 (例如每隔 10 个用户抽一个) ---
    target_users = [0, 10, 20, 30, 40, 50] 
    found_users = set()
    
    print("【多用户抽样展示】")
    for item in seqs:
        uid = item['user_id']
        
        # 如果是目标用户，且还没展示过
        if uid in target_users and uid not in found_users:
            tokens = item['tokens']
            
            # 判断是否静止 (Token是否全一样)
            is_static = np.all(tokens == tokens[0])
            status = "静止 (Static)" if is_static else "移动中 (Moving)"
            
            print(f"User ID: {uid:02d} | Track ID: {item['track_id']} | 状态: {status}")
            print(f"  Token 预览 (前15): {tokens[:15]} ...")
            if not is_static:
                # 如果是移动的，打印一下中间不同的部分看看
                unique_tokens = np.unique(tokens)
                print(f"  包含的不同 Token 数: {len(unique_tokens)} 个 (如: {unique_tokens[:10]}...)")
            
            print("-" * 30)
            found_users.add(uid)
            
        if len(found_users) >= len(target_users):
            break

if __name__ == "__main__":
    inspect_users()