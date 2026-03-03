import xarray as xr
import numpy as np
import os
import pandas as pd
from utils.config import load_config
from joblib import Parallel, delayed
import multiprocessing


def process_one_directory(daily_dir, save_dir, target_lat, target_lon, crop_size):
    """
    处理一个日期目录下的所有 .nc 文件
    """
    if not os.path.exists(daily_dir):
        print(f"⚠️ 目录不存在，跳过: {daily_dir}")
        return

    # 遍历目录下的文件
    for file in os.listdir(daily_dir):
        # 【关键】过滤掉非 .nc 文件（比如 .ipynb_checkpoints 或文件夹）
        if not file.endswith(".nc"):
            continue

        full_file_path = os.path.join(daily_dir, file)

        try:
            # 1. 打开数据集 (decode_timedelta=True 消除警告)
            # 使用 engine='netcdf4' 显式指定引擎更稳健
            ds = xr.open_dataset(full_file_path, decode_timedelta=True, engine='netcdf4')

            # 2. 找到最近的中心点索引
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # 找到最近点的索引
            lat_idx = (np.abs(lats - target_lat)).argmin()
            lon_idx = (np.abs(lons - target_lon)).argmin()

            # 计算切片范围
            half = crop_size // 2

            # 增加边界检查，防止索引越界报错
            lat_start = max(0, lat_idx - half)
            lat_end = min(len(lats), lat_idx + half)
            lon_start = max(0, lon_idx - half)
            lon_end = min(len(lons), lon_idx + half)

            lat_slice = slice(lat_start, lat_end)
            lon_slice = slice(lon_start, lon_end)

            # 3. 提取 Band 13 数据
            crop_data = ds['tbb_13'].isel(latitude=lat_slice, longitude=lon_slice)

            # 检查裁剪后的形状是否符合预期 (96, 96)
            if crop_data.shape != (crop_size, crop_size):
                print(f"⚠️ {file} 裁剪尺寸异常 {crop_data.shape}，跳过")
                ds.close()
                continue

            # 4. 直接保存为 .npy
            file_name = file.replace(".nc", "_crop.npy")
            save_path = os.path.join(save_dir, file_name)

            # .values 提取为 numpy 数组
            np.save(save_path, crop_data.values.astype(np.float32))  # 转为float32节省空间
            # print(f"✅ 保存: {file_name}")

            ds.close()  # 记得关闭文件释放内存

        except Exception as e:
            print(f"❌ 处理失败 {file}: {e}")


def process_single_day(current_date, base_read_path, base_save_path, target_lat, target_lon, crop_size):
    """
    封装单日处理任务，供多进程调用
    """
    yyyy = current_date.strftime("%Y")
    mm = current_date.strftime("%m")
    dd = current_date.strftime("%d")
    yyyymm = f"{yyyy}{mm}"

    daily_read_path = os.path.join(base_read_path, yyyymm, dd)
    daily_save_path = os.path.join(base_save_path, yyyymm, dd)

    if not os.path.exists(daily_save_path):
        os.makedirs(daily_save_path, exist_ok=True)

    print(f"🚀 开始多进程任务: {yyyy}-{mm}-{dd}")
    process_one_directory(daily_read_path, daily_save_path, target_lat, target_lon, crop_size)
    return f"Done: {yyyy}-{mm}-{dd}"


if __name__ == "__main__":
    # 加载配置
    config = load_config("../config/config.yaml")

    # 提取参数
    TARGET_LAT = config["stations"]["lat"]
    TARGET_LON = config["stations"]["lon"]
    CROP_SIZE = config["statellite"]["crop_size"]
    BASE_SATELLITE_PATH = config["file_paths"]["satellite_path"]
    BASE_SAVE_DIR = config["file_paths"]["crop_statellite_path"]

    dates = pd.date_range(start=config["dates"]["start_date"],
                          end=config["dates"]["end_date"], freq='D')

    # --- 并行执行核心部分 ---
    # n_jobs=-1 使用全部核心；如果内存小，建议改为 n_jobs=4 或 8
    print(f"🛰️ 卫星数据裁剪开始，总日期数: {len(dates)}")

    # 获取CPU核心数，留一个核心给系统，避免死机
    num_cores = multiprocessing.cpu_count() - 10

    Parallel(n_jobs=num_cores, verbose=10)(
        delayed(process_single_day)(
            d, BASE_SATELLITE_PATH, BASE_SAVE_DIR, TARGET_LAT, TARGET_LON, CROP_SIZE
        ) for d in dates
    )

    print("✅ 所有任务已圆满完成！")
