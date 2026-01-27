import pandas as pd
import os
from GPTPV.utils.config import load_config


def preprocess_real_station_xlsx(input_file, output_data_file, output_stats_file):
    print(f"🧹 正在读取 Excel 文件: {input_file}")

    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 '{input_file}'")
        return

    try:
        # --- 核心修改：直接使用 read_excel ---
        # header=0: 第一行是列名
        # skiprows=[1]: 跳过第二行（那个中文描述信息）
        # engine='openpyxl': 专门读取 .xlsx
        df = pd.read_excel(input_file, header=0, skiprows=[1], engine='openpyxl')
        print("   ✅ Excel 读取成功！")

    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return

    # --- 后续清洗逻辑 (保持不变) ---

    # 1. 检查列数
    if df.shape[1] < 2:
        print("❌ 数据列数不足，预期至少2列（时间, 功率）")
        return

    # 2. 重命名列 (强制英文列名)
    print(f"   原始列名: {df.columns.tolist()}")
    df.columns = ['Timestamp', 'Real_Power']

    # 3. 格式转换
    print("🔄 执行数据清洗...")
    # 转时间格式
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    # 删除无效时间行
    df = df.dropna(subset=['Timestamp'])
    df = df.set_index('Timestamp')
    df = df.sort_index()

    # 转数值格式 (非数字变为 NaN，然后填 0)
    df['Real_Power'] = pd.to_numeric(df['Real_Power'], errors='coerce')
    df = df.fillna(0)

    # 4. 计算统计量
    mean_val = df['Real_Power'].mean()
    std_val = df['Real_Power'].std()

    # 防止标准差为 0
    if std_val == 0:
        std_val = 1.0
        print("⚠️ 警告: 数据标准差为0 (可能是全0数据)")

    print(f"📊 统计结果 - 均值: {mean_val:.4f}, 标准差: {std_val:.4f}")

    # 5. 保存统计量 (反归一化用)
    stats_df = pd.DataFrame({'mean': [mean_val], 'std': [std_val]})
    stats_df.to_csv(output_stats_file, index=False)

    # 6. 归一化并保存训练数据
    df_norm = (df - mean_val) / std_val
    df_norm.to_csv(output_data_file)

    print(f"✅ 处理完毕！")
    print(f"   -> 训练数据: {output_data_file}")
    print(f"   -> 统计参数: {output_stats_file}")


if __name__ == "__main__":
    config = load_config()

    # 配置输入输出文件名
    raw_file = config["file_paths"]["lllmy_raw_file"]  # 原始文件
    clean_file = config["file_paths"]["lllmy_clean_file"]  # 给 Dataset 用的文件
    stats_file = config["file_paths"]["lllmy_stats_file"]  # 保存均值方差的文件

    # 执行
    preprocess_real_station_xlsx(raw_file, clean_file, stats_file)
