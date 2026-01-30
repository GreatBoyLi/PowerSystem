import xarray as xr
import numpy as np
import os
import pandas as pd
from MultiModal.utils.config import load_config


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
            print(f"✅ 保存: {file_name}")

            ds.close()  # 记得关闭文件释放内存

        except Exception as e:
            print(f"❌ 处理失败 {file}: {e}")


if __name__ == "__main__":
    config_file = "../config/config.yaml"
    config = load_config(config_file)

    # 从配置加载参数
    TARGET_LAT = config["stations"]["lat"]
    TARGET_LON = config["stations"]["lon"]
    CROP_SIZE = config["statellite"]["crop_size"]

    # 【修复重点 1】这里只获取基础路径，不要在循环里覆盖它
    BASE_SATELLITE_PATH = config["file_paths"]["satellite_path"]
    BASE_SAVE_DIR = config["file_paths"]["crop_statellite_path"]

    start_date = config["dates"]["start_date"]
    end_date = config["dates"]["end_date"]

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    for current_date in dates:
        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")
        dd = current_date.strftime("%d")
        yyyymm = f"{yyyy}{mm}"

        print(f"\n📅 处理日期: {yyyy}-{mm}-{dd}")

        # 【修复重点 2】使用临时变量 daily_path，绝对不要修改 BASE_SATELLITE_PATH
        # 原代码：file_path = os.path.join(file_path, ...) 会导致路径无限变长
        daily_read_path = os.path.join(BASE_SATELLITE_PATH, yyyymm, dd)
        daily_save_path = os.path.join(BASE_SAVE_DIR, yyyymm, dd)

        if not os.path.exists(daily_save_path):
            os.makedirs(daily_save_path)

        # 调用处理函数
        process_one_directory(daily_read_path, daily_save_path, TARGET_LAT, TARGET_LON, CROP_SIZE)