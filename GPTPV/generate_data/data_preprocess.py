import xarray as xr
import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 1. 输入文件路径 (刚才合并好的那个文件)
NC_FILE = "../data/era5/era5_shanxi_2020_01.nc"

# 2. 论文研究的目标地点 (例如：山西太原某地)
# 如果论文是研究整个山西的平均情况，请把 USE_SPECIFIC_LOCATION 改为 False
USE_SPECIFIC_LOCATION = True
TARGET_LAT = 37.87  # 太原纬度
TARGET_LON = 112.55  # 太原经度

# 3. 输出文件名
OUTPUT_CSV = "../data/era5/dataset_for_paper.csv"


# ===========================================

def preprocess_era5(nc_path):
    print(f"🔄 正在读取: {nc_path}")
    ds = xr.open_dataset(nc_path)

    # -------------------------------------------------------
    # 步骤 1: 空间处理 (Spatial Selection)
    # -------------------------------------------------------
    if USE_SPECIFIC_LOCATION:
        # 方法 A: 提取离目标经纬度最近的网格点 (Nearest Neighbor)
        print(f"📍 正在提取坐标 ({TARGET_LAT}, {TARGET_LON}) 最近的格点数据...")
        ds_local = ds.sel(latitude=TARGET_LAT, longitude=TARGET_LON, method='nearest')
    else:
        # 方法 B: 计算整个区域的平均值 (Regional Mean)
        print("🌍 正在计算区域平均值...")
        ds_local = ds.mean(dim=['latitude', 'longitude'])

    # 转为 Pandas DataFrame (时间序列)
    df = ds_local.to_dataframe().reset_index()

    # 清理多余索引
    cols_to_drop = ['number', 'expver', 'latitude', 'longitude']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # -------------------------------------------------------
    # 步骤 2: 单位换算 (Unit Conversion)
    # -------------------------------------------------------
    print("🧮 正在进行物理量换算...")

    # 1. 气温: K -> ℃
    df['T2m_C'] = df['t2m'] - 273.15

    # 2. 降水: m -> mm
    # 注意：ERA5的tp是累积量，如果是负数需置0 (数值计算误差)
    df['Precip_mm'] = df['tp'] * 1000
    df['Precip_mm'] = df['Precip_mm'].apply(lambda x: max(x, 0))

    # -------------------------------------------------------
    # 步骤 3: 风速与风向计算 (由 U/V 分量推导)
    # -------------------------------------------------------
    # 风速 = sqrt(u^2 + v^2)
    df['WindSpeed_m_s'] = np.sqrt(df['u10'] ** 2 + df['v10'] ** 2)

    # 风向 (角度 0-360)
    # 气象学风向定义：从哪里吹来。
    # 计算公式通常用 arctan2(u, v) * 180 / pi
    # 这里加 180 是为了将数学方向转为气象方向
    df['WindDir_deg'] = (180 + (180 / np.pi) * np.arctan2(df['u10'], df['v10'])) % 360

    # -------------------------------------------------------
    # 步骤 4: 时间特征提取 (Feature Extraction)
    # -------------------------------------------------------
    # 很多论文的模型(如LSTM, Random Forest)需要显式的时间特征
    df['Month'] = df['valid_time'].dt.month
    df['Day'] = df['valid_time'].dt.day
    df['Hour'] = df['valid_time'].dt.hour

    # 季节性特征 (正弦/余弦编码，处理时间的周期性，论文常用技巧)
    # 比如 23点 和 0点 很近，但数字很远，用 sin/cos 可以解决
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    # -------------------------------------------------------
    # 步骤 5: 整理与保存
    # -------------------------------------------------------
    # 选取最终需要的列
    final_cols = [
        'valid_time',
        'T2m_C', 'Precip_mm', 'WindSpeed_m_s', 'WindDir_deg',  # 物理特征
        'Month', 'Day', 'Hour', 'Hour_sin', 'Hour_cos'  # 时间特征
    ]

    df_final = df[final_cols].copy()

    # 重命名列以符合论文常见格式 (可选)
    df_final.rename(columns={'valid_time': 'Time'}, inplace=True)

    print("===== 📊 数据集预览 =====")
    print(df_final.head())

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 数据预处理完成！已保存至: {OUTPUT_CSV}")
    print("   您可以直接将此文件导入 PyTorch/TensorFlow/Matlab 进行模型训练。")


if __name__ == "__main__":
    if os.path.exists(NC_FILE):
        preprocess_era5(NC_FILE)
    else:
        print(f"❌ 找不到文件: {NC_FILE}")