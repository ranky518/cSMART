#!/usr/bin/env python3
# 文件：/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master/Model_pre/quick_view.py
import sys, pickle, pprint

pkl_path = sys.argv[1] if len(sys.argv) > 1 else \
           "/home/yanglanqi/trajectory/geolife_clustering-master/geolife_clustering-master/Model_pre/Data_Model_pre.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("===== 反序列化成功，类型：{} =====".format(type(data)))

# 1. 如果是 dict
if isinstance(data, dict):
    keys = list(data.keys())
    print("共 {} 个键：{}".format(len(keys), keys))
    for k in keys[:5]:           # 只拿前 5 个键
        print("\n--- 键 {} 的前 5 条样本 ---".format(repr(k)))
        v = data[k]
        # 修正：区分可切片 vs 不可切片
        if hasattr(v, '__getitem__') and hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
            try:
                preview = list(v.items())[:5] if isinstance(v, dict) else v[:5]
            except TypeError:      # 遇到 set 等仍可能失败
                preview = v
            pprint.pprint(preview)
        else:
            pprint.pprint(v)

# 2. 如果是 list / tuple
elif isinstance(data, (list, tuple)):
    print("共 {} 条记录，展示前 5 条：".format(len(data)))
    pprint.pprint(data[:5])

# 3. 如果是 pandas.DataFrame / Series
elif hasattr(data, "head"):
    print("形状：", data.shape)
    print(data.head(10))

# 4. 如果是 numpy.ndarray
elif hasattr(data, "dtype") and hasattr(data, "shape"):
    print("形状：{}, dtype：{}".format(data.shape, data.dtype))
    print("前 5 个元素：", data.flat[:5])

# 5. 其他
else:
    pprint.pprint(data)