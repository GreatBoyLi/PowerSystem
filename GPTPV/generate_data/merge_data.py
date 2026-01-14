import pandas as pd
import os
from GPTPV.utils.config import load_config


def merge_datasets(config):
    # --- 配置路径 ---

    SATELLITE_CSV = config["file_paths"]["himawari_output"]  # 卫星 GHI 数据
    ERA5_CSV = config["file_paths"]["era5_output"]  # ERA5 温度/风速/降水 数据

    # 输出文件
    OUTPUT_MERGED_CSV = config["file_paths"]["merged_data_output"]

    print("🔄 1. 读取卫星辐射数据 (GHI)...")
    if not os.path.exists(SATELLITE_CSV):
        print(f"❌ 找不到文件: {SATELLITE_CSV}")
        return
    df_sat = pd.read_csv(SATELLITE_CSV, index_col="Timestamp", parse_dates=True)
    print(f"   -> 卫星数据形状: {df_sat.shape}")

    print("🔄 2. 读取 ERA5 气象数据 (Temp, Wind, Precip)...")
    if not os.path.exists(ERA5_CSV):
        print(f"❌ 找不到文件: {ERA5_CSV}")
        return
    df_era = pd.read_csv(ERA5_CSV, index_col="Timestamp", parse_dates=True)
    print(f"   -> ERA5数据形状: {df_era.shape}")

    print("🔗 3. 执行数据合并 (Inner Join)...")
    # 以卫星数据的时间轴为基准，取交集
    # 论文中提到数据是 15-min interval [cite: 295]
    df_merged = df_sat.join(df_era, how='inner')

    # 检查是否有空值
    if df_merged.isnull().values.any():
        print("⚠️ 警告: 合并后的数据存在缺失值，正在使用向前填充(ffill)处理...")
        df_merged = df_merged.ffill().bfill()

    print(f"✅ 合并完成！最终数据形状: {df_merged.shape}")
    print(f"   时间范围: {df_merged.index.min()} 到 {df_merged.index.max()}")

    # 保存
    df_merged.to_csv(OUTPUT_MERGED_CSV)
    print(f"💾 数据已保存至: {OUTPUT_MERGED_CSV}")


if __name__ == "__main__":
    config_file = "../config/config.yaml"
    config = load_config(config_file)

    merge_datasets(config)