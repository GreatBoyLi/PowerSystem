import xarray as xr
import pandas as pd
import numpy as np
import os
from GPTPV.utils.config import load_config

config_file = "../config/config.yaml"
config = load_config(config_file)

ERA5_FILE = config["file_paths"]["era5_dir"]
OUTPUT_CSV = config["file_paths"]["era5_output"]
REAL_STATIONS = config["stations"]["real_stations"]
POINTS_PER_STATION = config["stations"]["virtual_points_per_station"]


# ===========================================

def extract_and_broadcast_era5():
    if not os.path.exists(ERA5_FILE):
        print(f"❌ 找不到文件: {ERA5_FILE}")
        return

    print(f"🔄 正在读取 ERA5 文件: {ERA5_FILE}")
    ds = xr.open_dataset(ERA5_FILE)

    # 1. 预处理数据 (单位换算)
    print("🧮 正在进行物理量计算与单位换算...")

    # 气温 K -> C
    temp_c = ds['t2m'] - 273.15

    # 降水 m -> mm (并将负数置0)
    precip_mm = ds['tp'] * 1000
    precip_mm = precip_mm.where(precip_mm >= 0, 0)

    # 风速 (u, v) -> speed
    wind_speed = np.sqrt(ds['u10'] ** 2 + ds['v10'] ** 2)

    # 2. 初始化字典存储所有列数据（核心优化：避免循环插列）
    # 先存入时间索引，后续所有列都存在这个字典里
    time_index = pd.to_datetime(ds.valid_time.values)
    data_dict = {"Timestamp": time_index}  # 时间列作为基础

    print("🚀 正在提取并分发数据...")

    # 3. 遍历 5 个真实电站中心
    for station in REAL_STATIONS:
        s_name = station['name']
        s_lat = station['lat']
        s_lon = station['lon']

        print(f"   -> 处理电站: {s_name} ({s_lat}, {s_lon})")

        # --- A. 提取该电站中心点的 ERA5 数据 ---
        # 使用 nearest 方法找到最近的 ERA5 网格
        # 因为 ERA5 分辨率粗，周围几公里的虚拟点其实都在这个网格里
        t_val = temp_c.sel(latitude=s_lat, longitude=s_lon, method='nearest').values
        p_val = precip_mm.sel(latitude=s_lat, longitude=s_lon, method='nearest').values
        w_val = wind_speed.sel(latitude=s_lat, longitude=s_lon, method='nearest').values

        # --- B. 广播给该电站旗下的所有虚拟点 (P0 - P19) ---
        for i in range(POINTS_PER_STATION):
            # 构建列名 (例如 Station_1_P0_Temp)
            # 这里的命名格式要与你之后合并数据时的预期一致
            base_col = f"{s_name}_P{i}"
            data_dict[f"{base_col}_Temp"] = t_val
            data_dict[f"{base_col}_Wind"] = w_val
            data_dict[f"{base_col}_Precip"] = p_val

    # 4. 一次性构建DataFrame（关键：避免碎片化）
    final_df = pd.DataFrame(data_dict)
    final_df = final_df.set_index("Timestamp")  # 设置时间为索引

    print("===== 📊 数据预览 (前5行, 前6列) =====")
    print(final_df.iloc[:, :6].head())

    # 4. 保存
    final_df.to_csv(OUTPUT_CSV)
    print(f"\n✅ 处理完成！数据已保存至: {OUTPUT_CSV}")
    print("💡 说明：由于 ERA5 分辨率较低(~30km)，同一电站下的虚拟点共享相同的气象数据是合理的。")


if __name__ == "__main__":
    extract_and_broadcast_era5()
