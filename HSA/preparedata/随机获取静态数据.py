import numpy as np
import pandas as pd
from scipy.stats import qmc
from HSA.tool.tool import getwritefilepath

# 1. 定义设计参数的上下限 (参考论文 Table 2)
# 顺序: [A_in, A_out, E_in, E_out, P_Tr_in, P_Tr_out, P_mut_max]
lower_bounds = np.array([2, 2, 1, 1, 0.8, 0.8, 1])
upper_bounds = np.array([20, 20, 20, 20, 4, 4, 5])

# 2. 设置样本数量 (论文训练集规模为 2000)
num_samples = 10000

# 3. 初始化 LHS 采样器 (维度 d=7) 拉丁超立方抽样函数
sampler = qmc.LatinHypercube(d=len(lower_bounds))
sample = sampler.random(n=num_samples)

# 4. 将 [0, 1] 空间的采样值缩放到物理边界范围
xd_data = qmc.scale(sample, lower_bounds, upper_bounds)

# 5. 封装为 DataFrame
columns = [
    'A_in：1000平方', 'A_out：1000平方',
    'E_in：MWh', 'E_out：MWh',
    'P_Tr_max_in：MW', 'P_Tr_max_out：MW',
    'P_mut_max：MW'
]
df_xd = pd.DataFrame(xd_data, columns=columns)

# 保存数据供后续 EMS 计算使用
dir = "../data/静态数据/"
name = "10000个静态数据.csv"
path = getwritefilepath(__file__, dir, name)
df_xd.to_csv(path, index=False, encoding="utf-8")
print("已成功生成 2000 组 LHS 采样数据并保存至 2000个静态数据.csv")
