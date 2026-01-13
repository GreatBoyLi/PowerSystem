import xarray as xr
import os

# 1. 找一个你刚下载好的文件路径
# 请修改为你实际的路径
file_path = "../data/himawari/202001/01/04/H08_20200101_0400_RFL021_FLDK.02401_02401.nc"
# 注意：尽量找一个中午的文件（比如0400 UTC = 北京时间12:00），这样能看到非零的辐射值

# 2. 检查文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 找不到文件: {file_path}")
    print("请手动修改 file_path 变量，指向一个你硬盘里真实存在的 .nc 文件")
else:
    try:
        # 3. 打开文件
        ds = xr.open_dataset(file_path)

        print("===== 📄 文件基础信息 =====")
        print(ds)
        print("\n===== 🌍 坐标信息 =====")
        a = ds.coords['longitude'].values
        b = ds.coords['latitude'].values
        print("经度 (lon) 示例:", ds.coords['longitude'].values[0:5])  # 名字可能是 longitude 或 lon
        print("纬度 (lat) 示例:", ds.coords['latitude'].values[0:5])  # 名字可能是 latitude 或 lat

        print("\n===== ☀️ 变量列表 =====")
        for var_name in ds.data_vars:
            print(f"变量名: {var_name}, 维度: {ds[var_name].dims}")

    except Exception as e:
        print(f"打开失败: {e}")