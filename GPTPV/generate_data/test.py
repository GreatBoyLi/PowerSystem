import zipfile
import xarray as xr
import os

file_path = "../data/era5/era5_shanxi_2020_02.nc"  # 那个伪装的文件

# 1. 确认它是 ZIP
if zipfile.is_zipfile(file_path):
    print("📦 检测到这是一个 ZIP 压缩包！正在解压...")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # 查看压缩包里有什么文件
        file_list = zip_ref.namelist()
        print(f"压缩包内包含: {file_list}")

        # 找到里面的 nc 文件 (通常只有一个)
        nc_file_name = [f for f in file_list if f.endswith('.nc')][0]

        # 解压到当前文件夹
        zip_ref.extract(nc_file_name, path="../data/era5/")

        real_nc_path = os.path.join("../data/era5/", nc_file_name)
        print(f"✅ 解压成功: {real_nc_path}")

        # 2. 读取真正的 NC 文件
        ds = xr.open_dataset(real_nc_path)
        print("\n--- 读取成功 ---")
        print(ds)
else:
    print("这不是 ZIP 文件，继续排查其他问题。")