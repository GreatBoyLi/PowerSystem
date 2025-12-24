import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from 训练数据准备 import calculate_f1_capex, calculate_f2_expectation
import os


# 假设这些函数已经在其他地方定义好了
# from your_module import calculate_f1_capex, calculate_f2_expectation

# --- 1. 将循环体封装成一个独立的函数 ---
def process_single_row(row, profile_data_list, prices_df):
    """
    处理单行数据的函数，必须放在主程序块之外或能够被独立调用
    """
    # 1. 算 f1
    f1 = calculate_f1_capex(row)

    # 2. 算 f2
    f2, is_feasible = calculate_f2_expectation(row, profile_data_list, prices_df)

    # 返回结果元组
    return f1, f2, (1 if not is_feasible else 0)


# --- 主程序 ---
if __name__ == '__main__':  # Windows下使用多进程必须加这行判断

    # A. 读取数据
    profile_dir = "data/随机剖面/"
    static_file = "data/静态数据/10000个静态数据.csv"
    price_file = "data/电价/电价.csv"

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

    print(f"开始并行计算 {len(df_xd)} 个样本...")

    # 获取CPU核心数，留一个核心给系统，避免死机
    num_cores = multiprocessing.cpu_count() - 2

    # B. 并行计算
    # joblib 会自动处理多进程分配
    # n_jobs: 进程数，-1表示使用所有CPU
    # backend: 'loky' 是默认且稳健的后端
    results = Parallel(n_jobs=num_cores, backend='loky', verbose=10)(
        delayed(process_single_row)(row, profile_data_list, prices_df)
        for _, row in df_xd.iterrows()
    )

    # results 是一个列表，格式为 [(f1, f2, feasible), (f1, f2, feasible), ...]

    # C. 解包结果并保存
    # 使用 zip(*results) 将列表解压为三个独立的元组
    f1_list, f2_list, feasibility_list = zip(*results)

    # D. 保存结果
    df_xd['f1_investment'] = f1_list
    df_xd['f2_operation'] = f2_list
    df_xd['feasible'] = feasibility_list

    # 保存为带标签的训练集
    train_data_dir = "data/train/"
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    train_file = os.path.join(train_data_dir, 'training_dataset_final.csv')
    df_xd.to_csv(train_file, index=False)
    print("计算完成，训练集已生成！")
