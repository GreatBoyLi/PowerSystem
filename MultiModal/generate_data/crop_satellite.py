import xarray as xr
import numpy as np
import os
import pandas as pd
from MultiModal.utils.config import load_config


def process_one_file(file_path, save_dir):
    for file in os.listdir(file_path):
        full_file_path = os.path.join(file_path, file)
        try:
            # 1. 打开数据集
            ds = xr.open_dataset(full_file_path, decode_timedelta=True)

            # JAXA L1 Gridded 数据通常有 'latitude' 和 'longitude' 坐标变量
            # 如果没有直接坐标，需要根据起始经纬度和分辨率计算索引
            # 这里假设是标准 Gridded 格式，直接利用 sel 方法最快

            # 2. 找到最近的中心点并裁剪
            # method='nearest' 会自动找最近的像素
            # slice 用于切片，注意 latitude 通常是从北到南（大到小），需要小心顺序

            # ⚠️ 注意：为了保证正好是 96x96，建议先找中心点索引，再按索引切片
            # 获取经纬度数组
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # 找到最近点的索引 (欧氏距离最小)
            lat_idx = (np.abs(lats - TARGET_LAT)).argmin()
            lon_idx = (np.abs(lons - TARGET_LON)).argmin()

            # 计算切片范围 (半宽 48)
            half = CROP_SIZE // 2
            lat_slice = slice(lat_idx - half, lat_idx + half)
            lon_slice = slice(lon_idx - half, lon_idx + half)

            # 3. 提取 Band 13 数据
            # 假设变量名是 'tbb_13'，根据你之前的 print 确认
            crop_data = ds['tbb_13'].isel(latitude=lat_slice, longitude=lon_slice)

            # 4. 检查尺寸是否正确 (边缘情况可能小于 96)
            if crop_data.shape != (CROP_SIZE, CROP_SIZE):
                print(f"⚠️ 裁剪尺寸不对 {crop_data.shape}，可能靠边了，跳过")
                return

            # 5. 保存裁剪后的小文件 (比如存为 .npy 或小的 .nc)
            # 推荐存为 .npy 方便后续做数据集
            file_name = os.path.basename(full_file_path).replace(".nc", "_crop.npy")
            save_path = os.path.join(save_dir, file_name)

            # 这一步将数据加载到内存并保存，极大地减小了体积
            np.save(save_path, crop_data.values)
            print(f"✅ 已裁剪并保存: {file_name}")

        except Exception as e:
            print(f"❌ 处理失败 {file_path}: {e}")


# 示例调用
# process_one_file("你的文件路径.nc", "保存目录")


if __name__ == "__main__":
    config_file = "../config/config.yaml"
    config = load_config(config_file)

    # 配置论文中的参数
    TARGET_LAT = config["stations"]["lat"]
    TARGET_LON = config["stations"]["lon"]
    CROP_SIZE = config["statellite"]["crop_size"]
    file_path = config["file_paths"]["satellite_path"]

    start_date = config["dates"]["start_date"]
    end_date = config["dates"]["end_date"]

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    for current_date in dates:
        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")
        dd = current_date.strftime("%d")
        yyyymm = f"{yyyy}{mm}"

        print(f"\n📅 处理日期: {yyyy}-{mm}-{dd}")
        file_path = os.path.join(file_path, yyyymm, dd)
        save_dir = config["file_paths"]["crop_statellite_path"]
        save_dir = os.path.join(save_dir, yyyymm, dd)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        process_one_file(file_path, save_dir)
