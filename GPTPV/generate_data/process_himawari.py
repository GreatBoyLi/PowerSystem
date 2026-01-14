import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from GPTPV.utils.config import load_config


# ===========================================

def get_spatial_indices(sample_file):
    """
    只运行一次：计算 "哪些像素点" 是我们需要提取的。
    返回一个列表，包含 100 个 (lat_idx, lon_idx) 坐标对以及经纬度数值。
    """
    print(f"🌍 正在计算空间索引，使用模板文件: {sample_file}")
    ds = xr.open_dataset(sample_file)

    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # 生成网格坐标矩阵
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    all_selected_indices = []

    for station in REAL_STATIONS:
        # 计算该电站到所有像素点的距离平方 (欧氏距离近似)
        dist_sq = (lat_grid - station['lat']) ** 2 + (lon_grid - station['lon']) ** 2

        # 找到距离最近的 N 个点
        flat_indices = np.argpartition(dist_sq.ravel(), POINTS_PER_STATION)[:POINTS_PER_STATION]
        y_indices, x_indices = np.unravel_index(flat_indices, dist_sq.shape)

        for y, x in zip(y_indices, x_indices):
            all_selected_indices.append({
                "station": station['name'],
                "lat_val": float(lats[y]),  # 确保转换为python float
                "lon_val": float(lons[x]),
                "lat_idx": y,
                "lon_idx": x
            })

    print(f"✅ 空间选点完成！共选中 {len(all_selected_indices)} 个虚拟站点。")
    return all_selected_indices


def save_station_coordinates(indices_list, save_path):
    """
    新增功能：将筛选出的虚拟站点经纬度保存为 CSV
    """
    print(f"💾 正在保存虚拟站点坐标至: {save_path}")

    coord_data = []

    # 遍历列表，构造与 process_temporal_data 中完全一致的 Station ID
    for i, info in enumerate(indices_list):
        # 这里的命名逻辑必须与下面处理时间序列时的逻辑保持一致：StationName_P{0...19}
        station_id = f"{info['station']}_P{i % POINTS_PER_STATION}"

        coord_data.append({
            "Station_ID": station_id,
            "Real_Station_Ref": info['station'],  # 归属的真实电站
            "Latitude": info['lat_val'],
            "Longitude": info['lon_val']
        })

    df_coords = pd.DataFrame(coord_data)
    df_coords.to_csv(save_path, index=False)
    print(f"✅ 坐标保存成功！")


def process_temporal_data(target_indices, date_list):
    """
    遍历每一天、每个文件，提取 SWR 数据
    """
    results = []

    for current_date in date_list:
        date_str = current_date.strftime("%Y-%m-%d")
        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")
        dd = current_date.strftime("%d")

        print(f"\n📅 正在提取数据: {date_str}")

        time_mapping = {
            "00": "00",
            "20": "15",
            "30": "30",
            "50": "45"
        }

        for hour in tqdm(range(24), desc="Hour Loop"):
            hh = f"{hour:02d}"
            hour_dir = os.path.join(DATA_DIR, f"{yyyy}{mm}", dd, hh)

            if not os.path.exists(hour_dir):
                continue

            files = sorted(os.listdir(hour_dir))

            for f_name in files:
                if not f_name.endswith(".nc") or "02401_02401" not in f_name:
                    continue

                try:
                    time_part = f_name.split("_")[2]
                    minute_str = time_part[2:]
                except:
                    continue

                if minute_str not in time_mapping:
                    continue

                target_minute = time_mapping[minute_str]

                full_path = os.path.join(hour_dir, f_name)
                try:
                    ds = xr.open_dataset(full_path)
                    swr_data = ds['SWR'].values
                    timestamp = pd.Timestamp(f"{date_str} {hh}:{target_minute}:00")
                    row_data = {"Timestamp": timestamp}

                    for i, idx_info in enumerate(target_indices):
                        val = swr_data[idx_info['lat_idx'], idx_info['lon_idx']]
                        if np.isnan(val) or val < 0:
                            val = 0.0

                        # 命名列名: Station_1_P0 (逻辑必须与 save_station_coordinates 一致)
                        col_name = f"{idx_info['station']}_P{i % POINTS_PER_STATION}"
                        row_data[col_name] = val

                    results.append(row_data)
                    ds.close()

                except Exception as e:
                    print(f"读取错误 {f_name}: {e}")

    return pd.DataFrame(results)


def main(config):
    # 1. 找样板文件
    sample_files = glob.glob(f"{DATA_DIR}/*/*/*/*.nc")
    if not sample_files:
        print("❌ 目录下没有找到任何 .nc 文件！")
        return

    valid_sample = None
    for f in sample_files:
        if "02401_02401" in f:
            valid_sample = f
            break

    if not valid_sample:
        print("❌ 没找到 02401_02401 规格的文件！")
        return

    # 2. 计算空间索引
    spatial_indices = get_spatial_indices(valid_sample)

    # --- 新增步骤：保存经纬度 ---
    save_station_coordinates(spatial_indices, coord_output_file)
    # -------------------------

    # 3. 处理时间序列
    dates = pd.date_range(START_DATE, END_DATE)
    df = process_temporal_data(spatial_indices, dates)

    # 4. 排序和保存数据
    if not df.empty:
        df = df.sort_values("Timestamp").set_index("Timestamp")
        df = df.bfill()

        print("\n===== 数据预览 =====")
        print(df.head())

        df.to_csv(output_file)
        print(f"\n✅ SWR数据处理完成！已保存至: {output_file}")
    else:
        print("⚠️ 没有提取到任何数据。")


if __name__ == "__main__":
    config_file = "../config/config.yaml"
    config = load_config(config_file)

    DATA_DIR = config["file_paths"]["himawari_dir"]
    output_file = config["file_paths"]["himawari_output"]

    # 新增：定义坐标保存的文件路径
    coord_output_file = config["file_paths"]["output_coord_csv"]

    # 2. 模拟论文中的 "5个真实电站" 坐标
    REAL_STATIONS = config["stations"]["real_stations"]

    # 3. 每个电站选多少个虚拟点？
    POINTS_PER_STATION = config["stations"]["virtual_points_per_station"]

    # 4. 要处理的日期范围
    START_DATE = config["dates"]["start_date"]
    END_DATE = config["dates"]["end_date"]

    main(config)
