import os
import json
import matplotlib.pyplot as plt

# --- 1. 配置参数 ---
# 只显示 count 大于此值的点，防止低热力点干扰视线
# 建议设置为 500 或 1000，只看核心区域
FILTER_THRESHOLD = 1000 

# 设置中文字体 (防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def parse_js_file(file_path):
    """ 解析 JS 文件中的 JSON 数据 """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # 清理 JS 前后缀
    if content.startswith("var points ="):
        content = content[len("var points ="):].strip()
    if content.endswith(";"):
        content = content[:-1].strip()
        
    try:
        return json.loads(content)
    except Exception as e:
        print(f"解析失败: {e}")
        return []

def plot_scatter_map(data, output_path):
    """ 绘制经纬度散点图 """
    # 1. 筛选数据
    filtered_data = [p for p in data if p.get('count', 0) >= FILTER_THRESHOLD]
    
    if not filtered_data:
        print(f"没有找到 count >= {FILTER_THRESHOLD} 的数据。")
        return

    # 2. 提取坐标和 count
    lngs = [p['lng'] for p in filtered_data]
    lats = [p['lat'] for p in filtered_data]
    counts = [p['count'] for p in filtered_data]

    # 3. 开始绘图
    plt.figure(figsize=(12, 10)) # 画布大一点，方便看清
    
    # 绘制散点
    # c=counts: 颜色根据 count 变化
    # cmap='Reds': 使用红色渐变 (浅红->深红)
    # s=50: 点的大小
    # alpha=0.8: 透明度，防止重叠完全遮挡
    sc = plt.scatter(lngs, lats, c=counts, cmap='Reds', s=50, alpha=0.8, edgecolors='grey', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Count 热力值')

    # 4. 坐标轴与网格设置 (关键步骤)
    plt.xlabel('经度 (Longitude)')
    plt.ylabel('纬度 (Latitude)')
    plt.title(f'高热力区域分布散点图 (Count >= {FILTER_THRESHOLD})')
    
    # 强制让 X 和 Y 轴比例一致，避免地图拉伸变形
    # plt.axis('equal') 
    # 注：如果你发现图太扁或太高，可以注释掉上面这行 plt.axis('equal')

    # 禁用科学计数法 (让坐标显示为 116.35 而不是 1.16e2)
    plt.ticklabel_format(useOffset=False, style='plain')

    # 开启主网格 (实线) 和 次网格 (虚线)
    plt.grid(which='major', linestyle='-', linewidth='0.8', color='gray', alpha=0.6)
    plt.minorticks_on() # 开启次刻度
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)

    # 5. 保存与展示
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"散点图已保存至: {output_path}")
    
    # --- 顺便打印一下极值范围，辅助你判断 ---
    print("-" * 30)
    print(f"【数据统计范围 (Count >= {FILTER_THRESHOLD})】")
    print(f"经度范围 (X轴): {min(lngs):.4f} ~ {max(lngs):.4f}")
    print(f"纬度范围 (Y轴): {min(lats):.4f} ~ {max(lats):.4f}")
    print("-" * 30)

if __name__ == "__main__":
    # 路径设置
    base_dir = "/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master"
    input_file = os.path.join(base_dir, "new_heatmap_points.js")
    output_image = os.path.join(base_dir, "hotspot_scatter.png") # 输出文件名
    
    # 运行
    data = parse_js_file(input_file)
    if data:
        plot_scatter_map(data, output_image)