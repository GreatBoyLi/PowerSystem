import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
from GPTPV.utils.config import load_config

config_file = "../config/config.yaml"
config = load_config(config_file)

DATA_DIR = config["file_paths"]["himawari_dir"]

# 2. 模拟论文中的 "5个真实电站" 坐标 (以山西太原 37.8, 112.5 为中心)
# 我们在中心附近随机散布 5 个点
REAL_STATIONS = config["stations"]["real_stations"]

# 3. 每个电站选多少个虚拟点？ (论文说5个站共100个点 -> 每个站20个)
POINTS_PER_STATION = config["stations"]["virtual_points_per_station"]

# 4. 要处理的日期范围
START_DATE = config["dates"]["start_date"]
END_DATE = config["dates"]["end_date"]  # 先试跑一天


# ===========================================

def get_spatial_indices(sample_file):
    """
    只运行一次：计算 "哪些像素点" 是我们需要提取的。
    返回一个列表，包含 100 个 (lat_idx, lon_idx) 坐标对。
    """
    print(f"🌍 正在计算空间索引，使用模板文件: {sample_file}")
    ds = xr.open_dataset(sample_file)

    # 获取经纬度网格
    # 注意：Himawari数据的 lat 可能是从大到小排列的，lon 是从小到大
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # 生成网格坐标矩阵 (用于计算距离)
    # 这步可能会消耗一点内存，但在 5km 精度下完全没问题
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    all_selected_indices = []

    for station in REAL_STATIONS:
        # 计算该电站到所有像素点的距离平方 (欧氏距离近似)
        dist_sq = (lat_grid - station['lat']) ** 2 + (lon_grid - station['lon']) ** 2

        # 找到距离最近的 N 个点的扁平索引 (flat index)
        # argpartition 比 argsort 快，只找出前 N 个，不严格排序
        flat_indices = np.argpartition(dist_sq.ravel(), POINTS_PER_STATION)[:POINTS_PER_STATION]

        # 将扁平索引转回 (y, x) 二维索引
        # unravel_index 会返回 (lat_indices, lon_indices)
        y_indices, x_indices = np.unravel_index(flat_indices, dist_sq.shape)

        # 存起来
        for y, x in zip(y_indices, x_indices):
            all_selected_indices.append({
                "station": station['name'],
                "lat_val": lats[y],
                "lon_val": lons[x],
                "lat_idx": y,
                "lon_idx": x
            })

    print(f"✅ 空间选点完成！共选中 {len(all_selected_indices)} 个虚拟站点。")
    return all_selected_indices


def process_temporal_data(target_indices, date_list):
    """
    遍历每一天、每个文件，提取 SWR 数据
    """
    results = []  # 存放最终数据

    for current_date in date_list:
        date_str = current_date.strftime("%Y-%m-%d")
        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")
        dd = current_date.strftime("%d")

        print(f"\n📅 正在提取数据: {date_str}")

        # 遍历 24 小时 x 6 个时刻 (00, 10, 20, 30, 40, 50)
        # 论文要求的逻辑：
        # 00 -> 00
        # 10 -> 扔掉
        # 20 -> 15 (Mapping)
        # 30 -> 30
        # 40 -> 扔掉
        # 50 -> 45 (Mapping)

        # 我们先只关心我们需要的时间点: 00, 20, 30, 50
        # 对应的目标分钟: 00, 15, 30, 45
        time_mapping = {
            "00": "00",
            "20": "15",
            "30": "30",
            "50": "45"
        }

        for hour in tqdm(range(24), desc="Hour Loop"):  # 遍历小时
            hh = f"{hour:02d}"
            hour_dir = os.path.join(DATA_DIR, f"{yyyy}{mm}", dd, hh)

            if not os.path.exists(hour_dir):
                continue

            # 获取该小时下的所有 .nc 文件
            files = sorted(os.listdir(hour_dir))

            for f_name in files:
                if not f_name.endswith(".nc") or "02401_02401" not in f_name:
                    continue

                # 解析文件名中的分钟 (H08_..._0420_...)
                # 文件名格式: H08_20200101_0420_...
                # 分钟在第 3 段 (index 2) 的后两位
                try:
                    time_part = f_name.split("_")[2]  # "0420"
                    minute_str = time_part[2:]  # "20"
                except:
                    continue

                # 按照论文规则筛选：只要 00, 20, 30, 50
                if minute_str not in time_mapping:
                    continue

                target_minute = time_mapping[minute_str]  # 转换成 00, 15, 30, 45

                # 打开文件提取数据
                full_path = os.path.join(hour_dir, f_name)
                try:
                    ds = xr.open_dataset(full_path)
                    swr_data = ds['SWR'].values  # 读取整个矩阵 (为了速度，一次读入内存)
                    # 注意：有些文件里SWR可能有 scale_factor，xarray会自动处理

                    # 构造这一行数据的时间戳
                    timestamp = pd.Timestamp(f"{date_str} {hh}:{target_minute}:00")

                    row_data = {"Timestamp": timestamp}

                    # 循环提取那 100 个点的值
                    for i, idx_info in enumerate(target_indices):
                        val = swr_data[idx_info['lat_idx'], idx_info['lon_idx']]
                        # 处理 NaN 和 负值 (夜间)
                        if np.isnan(val) or val < 0:
                            val = 0.0

                        # 命名列名: Station_1_Point_0
                        col_name = f"{idx_info['station']}_P{i % POINTS_PER_STATION}"
                        row_data[col_name] = val

                    results.append(row_data)
                    ds.close()

                except Exception as e:
                    print(f"读取错误 {f_name}: {e}")

    return pd.DataFrame(results)


def main():
    # 1. 找一个存在的 .nc 文件做模板，计算空间索引
    # 自动搜索目录下第一个符合条件的文件
    sample_files = glob.glob(f"{DATA_DIR}/*/*/*/*.nc")
    if not sample_files:
        print("❌ 目录下没有找到任何 .nc 文件，请检查路径！")
        return

    # 找一个 5km 的文件
    valid_sample = None
    for f in sample_files:
        if "02401_02401" in f:
            valid_sample = f
            break

    if not valid_sample:
        print("❌ 没找到 02401_02401 规格的文件！")
        return

    # 2. 计算空间索引 (这一步只做一次)
    spatial_indices = get_spatial_indices(valid_sample)

    # 3. 处理时间序列
    dates = pd.date_range(START_DATE, END_DATE)
    df = process_temporal_data(spatial_indices, dates)

    # 4. 排序和去重
    if not df.empty:
        df = df.sort_values("Timestamp").set_index("Timestamp")

        # 5. 填补缺失值 (论文规则：missing value filled with next moment)
        # ffill是向前填，bfill是向后(next moment)填
        df = df.bfill()

        print("\n===== 数据预览 =====")
        print(df.head())

        output_file = "virtual_pv_data_shanxi.csv"
        df.to_csv(output_file)
        print(f"\n✅ 处理完成！数据已保存至: {output_file}")
    else:
        print("⚠️ 没有提取到任何数据，请检查日期范围或文件名。")


if __name__ == "__main__":
    main()
