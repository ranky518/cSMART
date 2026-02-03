import os
import json
import matplotlib.pyplot as plt

# --- 配置区域 ---
MIN_COUNT = 1000  # 筛选阈值
# 设置中文字体，防止图片坐标中文乱码 (Windows一般用SimHei, Linux/Mac可能需要换)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def parse_js_file(file_path):
    """
    专门解析格式为 var points = [...]; 的 JS 文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # 移除 'var points =' 前缀
    if content.startswith("var points ="):
        content = content[len("var points ="):].strip()
    
    # 移除末尾的分号 ';'
    if content.endswith(";"):
        content = content[:-1].strip()
        
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        print(f"解析 JSON 出错: {e}")
        return []

def plot_histogram(data, save_path):
    """
    统计 count 数量并绘制直方图
    """
    # 1. 提取 count 列表
    counts = [p.get('count', 0) for p in data]
    
    # 2. 定义区间 bins 和 标签 labels
    # 含义: 0-500, 500-1000, 1000-1500, ...
    bins = [0, 500, 1000, 1500, 2000, 3000, float('inf')]
    labels = ['<500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '>3000']
    
    # 3. 统计每个区间的落点数量
    hist_counts = [0] * len(labels)
    for c in counts:
        for i in range(len(bins) - 1):
            # 判断 count 是否在当前区间 [low, high)
            if bins[i] <= c < bins[i+1]:
                hist_counts[i] += 1
                break
    
    print(f"直方图统计分布: {dict(zip(labels, hist_counts))}")

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, hist_counts, color='skyblue', edgecolor='black')
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, str(int(height)), 
                 ha='center', va='bottom')

    plt.xlabel('Count 数量区间')
    plt.ylabel('聚类中心个数')
    plt.title('聚类中心 Count 分布统计')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 保存
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"直方图已保存至: {save_path}")

def save_filtered_js(data, save_path):
    """
    筛选 count > 1000 的数据并保存为 narrow.js
    """
    # 筛选逻辑：count > 1000
    filtered_data = [p for p in data if p.get('count', 0) > MIN_COUNT]
    
    count_diff = len(data) - len(filtered_data)
    print(f"原始数据: {len(data)} 条 -> 筛选后(>{MIN_COUNT}): {len(filtered_data)} 条 (剔除了 {count_diff} 条)")

    # 构造 JS 文件内容
    js_content = "var points = " + json.dumps(filtered_data, ensure_ascii=False, indent=4) + ";"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"筛选后的数据已保存至: {save_path}")

def process_heatmap_data(input_js_path):
    """
    主处理函数：整合解析、画图、筛选保存
    """
    # 1. 确定输出文件的路径（基于输入文件的目录）
    base_dir = os.path.dirname(input_js_path)
    output_image_path = os.path.join(base_dir, "narrow_down.png") # 直方图路径
    output_js_path = os.path.join(base_dir, "narrow.js")          # 筛选结果路径

    if not os.path.exists(input_js_path):
        print(f"错误：找不到文件 {input_js_path}")
        return

    # 2. 解析数据
    print("正在解析数据...")
    data = parse_js_file(input_js_path)
    if not data:
        return

    # 3. 绘制并保存直方图
    print("正在绘制直方图...")
    plot_histogram(data, output_image_path)

    # 4. 筛选并保存 JS
    print("正在导出筛选结果...")
    save_filtered_js(data, output_js_path)

if __name__ == "__main__":
    # 输入文件路径
    base_dir = "/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master"
    input_file = os.path.join(base_dir, "new_heatmap_points.js")
    
    # 执行
    process_heatmap_data(input_file)