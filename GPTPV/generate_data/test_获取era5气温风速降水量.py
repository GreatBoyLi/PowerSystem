import xarray as xr
import os

# 修改为你刚下载的那个文件名
file_path = "../data/era5/era5_shanxi_2020_01.nc"

if os.path.exists(file_path):
    try:
        ds = xr.open_dataset(file_path)
        print("===== 📄 ERA5 文件信息 =====")
        print(ds)
        print("\n===== 🌍 变量检查 =====")
        # 通常变量名是 't2m' (气温) 和 'tp' (降水)
        # 也有可能是 '2t' 或 'total_precipitation'，取决于下载方式
        for var in ds.data_vars:
            print(f"变量: {var} | 维度: {ds[var].dims} | 单位: {ds[var].attrs.get('units', '未知')}")

        # 检查一下具体数值（看看是不是开尔文）
        if 't2m' in ds:
            sample_temp = ds['t2m'].values[0, 0, 0]
            print(f"\n🌡️ 样本气温值: {sample_temp:.2f} (如果是 270 左右，说明是开尔文)")

    except Exception as e:
        print(f"❌ 打开失败: {e}")
else:
    print("找不到文件，请检查路径。")