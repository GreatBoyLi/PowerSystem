import os

# 让 OMP, MKL 等库只用一个线程，防止在外层多进程时打架
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cvxpy as cp
import numpy as np
import pandas as pd


def run_optimal_ems(xd, profile, prices):
    """
    xd: 设计参数 [A_in, E_in, P_Tr_in, A_out, E_out, P_Tr_out, P_mut_max] 
    profile: 包含 'pv_unit', 'load_in', 'load_out' 的 DataFrame (672步长) 
    prices: 包含 'real_time', 'day_ahead', 'cancel_penalty' 的字典
    """
    Nt = 672  # 168h / 0.25h 
    dt = 0.25  #

    # 解析 xd 
    A_in, E_in, P_Tr_in = xd[0], xd[2], xd[4]
    A_out, E_out, P_Tr_out = xd[1], xd[3], xd[5]
    P_mut_max = xd[6]  #

    # --- 定义决策变量 (Table 1)  ---
    # 这个初始化是一个变量，没有具体的值，相当于方程中的x
    f_gah = cp.Variable((Nt, 2), nonneg=True)  # 日前购电率
    f_gcan = cp.Variable((Nt, 2), nonneg=True)  # 日前取消率 
    f_grt = cp.Variable((Nt, 2), nonneg=True)  # 实时购电率 
    f_pv = cp.Variable((Nt, 2), nonneg=True)  # PV 削减率 
    f_mut = cp.Variable(Nt)  # 互助功率比例 [-1, 1] 
    P_ES = cp.Variable((Nt, 2))  # 储能功率 
    SOC = cp.Variable((Nt + 1, 2))  # SOC 状态 

    # --- 目标函数 (Eq. 12 & 13) ---
    total_cost = 0
    for i in range(2):
        p_tr = P_Tr_in if i == 0 else P_Tr_out
        # pv_raw = profile['pv_unit'].values * (A_in if i == 0 else A_out)
        A = (A_in if i == 0 else A_out) * 1000  # 将A_in,A_out的单位由1000平方米，转换为平方米
        pv_raw = cp.multiply(profile['pv_unit'].values, A) / 1000  # 单位为kW

        # 成本项：PV浪费 + 实时购电 + 日前购电 + 取消惩罚
        term_pv = dt * cp.sum(cp.multiply(f_pv[:, i], pv_raw))
        term_grid = dt * cp.sum(
            cp.multiply(prices['real_time'].values, f_grt[:, i]) * p_tr +
            cp.multiply(prices['day_ahead'].values, f_gah[:, i]) * p_tr +
            cp.multiply(prices['cancel_penalty'].values, f_gcan[:, i]) * p_tr
        )
        total_cost += term_pv + term_grid

    objective = cp.Minimize(total_cost)

    # --- 约束条件 (Table 1)  ---
    constraints = []
    for i in range(2):
        p_tr = P_Tr_in if i == 0 else P_Tr_out
        pv_raw = profile['pv_unit'].values * (A_in if i == 0 else A_out)
        p_load = profile['load_in'].values if i == 0 else profile['load_out'].values
        sign_mut = -1 if i == 0 else 1  # 进城流出(-), 出城流入(+)

        # 获取当前站点的储能容量 E (MWh)
        E_cap = E_in if i == 0 else E_out

        # 1. 动态功率平衡 (Eq. 17) 
        constraints += [
            cp.multiply((f_gah[:, i] - f_gcan[:, i] + f_grt[:, i]), p_tr) +
            cp.multiply((1 - f_pv[:, i]), pv_raw) +
            P_ES[:, i] + cp.multiply(cp.multiply(sign_mut, f_mut), P_mut_max) == p_load
        ]

        # 2. SOC 演化与能量平衡 (Eq. 14, 15) 
        # 简化内阻模型，近似 P_ES = V_oc * I_ES
        E = E_in if i == 0 else E_out
        constraints += [
            SOC[1:, i] == SOC[:-1, i] - cp.multiply(P_ES[:, i], dt) / E,
            SOC[0, i] == 0.6,  # 
            SOC[-1, i] == 0.6,  # 
            SOC[:, i] >= 0.2, SOC[:, i] <= 0.8  # 电池安全边界 
        ]

        # ==========================================
        # 3. 新增: 充放电倍率约束 (Eq. 9)
        # 对应论文: -1 <= I_ES / Q_N <= 2
        # 转化为: -E <= P_ES <= 2E
        # ==========================================
        constraints += [
            P_ES[:, i] <= 2 * E_cap,  # 放电不超过 2C
            P_ES[:, i] >= -1 * E_cap  # 充电不超过 1C (注意负号)
        ]

        # 4. 变量边界限制
        constraints += [
            f_gah[:, i] <= 1, f_grt[:, i] <= 1,
            f_gcan[:, i] <= f_gah[:, i],
            f_gah[:, i] + f_grt[:, i] <= 1,
            f_pv[:, i] <= 1
        ]

    constraints += [f_mut >= -1, f_mut <= 1]

    # 求解器调用
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    return prob.value if prob.status == 'optimal' else np.inf


if "__main__" == __name__:
    profile_dir = "data/随机剖面/"
    static_file = "data/静态数据/2000个静态数据.csv"
    price_file = "data/电价/电价.csv"

    static_np = pd.read_csv(static_file).to_numpy()
    price_pd = pd.read_csv(price_file)

    for i in range(len(static_np)):
        static_data = static_np[i]
        for file_name in os.listdir(profile_dir):
            if file_name.endswith(".csv") or file_name.endswith(".CSV"):  # 兼容大写后缀
                # 拼接完整文件路径（避免路径分隔符问题，推荐用os.path.join）
                file_path = os.path.join(profile_dir, file_name)
                profile_pd = None
                try:
                    # 3. 读取CSV文件
                    profile_pd = pd.read_csv(file_path)
                except Exception as e:
                    print(f"读取文件 {file_name} 失败：{str(e)}\n")
                if profile_pd is not None:
                    a = run_optimal_ems(static_data, profile_pd, price_pd)
                    print(a)
