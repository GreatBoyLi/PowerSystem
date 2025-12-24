import pandas as pd
import numpy as np
from HSA.preparedata.convex_apt import run_optimal_ems
import os

# ==========================================
# 1. 定义成本参数 (参考论文 Table 2 的假定值)
# ==========================================
# 需要根据论文 Table 2 填入真实的单价
UNIT_PRICES = {
    'c_pv': 178200,  # RMB/1000平方米 )
    'c_es': 400000,  # RMB/MWh
    'c_tr': 80000,  # RMB/MW   变压器的单位应该是VA，一般W是VA*80%，这里按论文中的MW算
    'c_mut': 5000000  # RMB/MW (互助设备)
}


# ==========================================
# 2. 计算 f1: 初始投资成本 (CAPEX)
# ==========================================
def calculate_f1_capex(row):
    """
    输入: 一行 x_d 数据 (Series)
    输出: f1 投资成本
    """
    # 提取变量 (根据之前的 CSV 列名)
    A_in, A_out = row['A_in：1000平方'], row['A_out：1000平方']
    E_in, E_out = row['E_in：MWh'], row['E_out：MWh']
    P_Tr_in, P_Tr_out = row['P_Tr_max_in：MW'], row['P_Tr_max_out：MW']
    P_mut = row['P_mut_max：MW']

    # 简单的加权求和
    #  假设 A 是 1000m^2, E 是 MWh, P 是 MW，单价也是这个单位，直接相乘即可，不用换算
    cost_pv = (A_in + A_out) * UNIT_PRICES['c_pv']
    cost_es = (E_in + E_out) * UNIT_PRICES['c_es']
    cost_tr = (P_Tr_in + P_Tr_out) * UNIT_PRICES['c_tr']
    cost_mut = P_mut * UNIT_PRICES['c_mut']

    return cost_pv + cost_es + cost_tr + cost_mut


# ==========================================
# 3. 计算 f2: 运行成本期望值 (OPEX)
# ==========================================
def calculate_f2_expectation(row, profile_data_list, prices):
    """
    输入: 一行 x_d 数据, 10个随机剖面, 价格数据
    输出: f2 (10个场景的平均运行成本)
    """
    total_cost = 0
    valid_count = 0
    file_count = 0

    # 循环 10 个随机剖面
    for profile_pd in profile_data_list:
            # 3. 读取CSV文件
            file_count += 1
            # 调用之前的 EMS 求解函数 (run_optimal_ems)
            cost = run_optimal_ems(row.values, profile_pd, prices)
            if cost != np.inf:
                total_cost += cost
                valid_count += 1
            else:
                # 如果某个场景无解(Infeasible)，通常给一个巨大的惩罚值
                return 1e9, False  # cost, is_feasible

    # 如果 10 个场景都可行，返回平均值
    if valid_count == file_count:
        f2_value = total_cost / valid_count
        return f2_value, True
    else:
        return 1e9, False


if "__main__" == __name__:
    # ==========================================
    # 4. 主程序: 批量处理 2000 个样本
    # ==========================================

    # A. 读取数据
    profile_dir = "../data/随机剖面/"
    static_file = "data/静态数据/2000个静态数据.csv"
    price_file = "../data/电价/电价.csv"

    # --- 优化关键：预先读取剖面数据 ---
    print("正在预读取剖面数据到内存...")
    profile_data_list = []
    # 假设 profile_dir 下全是需要的 csv
    file_list = [f for f in os.listdir(profile_dir) if f.lower().endswith('.csv')]

    for file_name in file_list:
        file_path = os.path.join(profile_dir, file_name)
        df = pd.read_csv(file_path)
        profile_data_list.append(df)

    print(f"成功预加载 {len(profile_data_list)} 个剖面场景。")


    df_xd = pd.read_csv(static_file)
    prices_df = pd.read_csv(price_file)

    # B. 初始化结果列
    f1_list = []
    f2_list = []
    feasibility_list = []

    print(f"开始计算 {len(df_xd)} 个样本的 f1 和 f2...")

    for index, row in df_xd.iterrows():
        # 1. 算 f1 (瞬间完成)
        f1 = calculate_f1_capex(row)

        # 2. 算 f2 (比较慢，因为要解 10 次凸优化)
        f2, is_feasible = calculate_f2_expectation(row, profile_data_list, prices_df)

        f1_list.append(f1)
        f2_list.append(f2)
        feasibility_list.append(1 if not is_feasible else 0)  # 无解的标签为1，有解的标签为0

        if index % 50 == 0:
            print(f"已处理 {index + 1}/{len(df_xd)} 个样本...")

    # C. 保存结果
    df_xd['f1_investment'] = f1_list
    df_xd['f2_operation'] = f2_list
    df_xd['feasible'] = feasibility_list

    # 保存为带标签的训练集
    train_data_dir = "../data/train/"
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    train_file = os.path.join(train_data_dir, 'training_dataset_final.csv')
    df_xd.to_csv(train_file, index=False)
    print("计算完成，训练集已生成！")
