import cdsapi
import os
import zipfile
import xarray as xr  # 新增：用于合并数据

# 保存路径
SAVE_DIR = "../data/era5/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 启动客户端
c = cdsapi.Client()


def download_era5_month(year, month):
    """
    下载指定年月的 ERA5 hourly 数据。
    逻辑：下载ZIP -> 解压所有NC文件 -> (如果多个)合并为一个NC -> 清理临时文件
    """
    # 1. 定义文件名
    final_nc_name = f"era5_shanxi_{year}_{month:02d}.nc"
    final_nc_path = os.path.join(SAVE_DIR, final_nc_name)

    temp_zip_name = f"era5_shanxi_{year}_{month:02d}.zip"
    temp_zip_path = os.path.join(SAVE_DIR, temp_zip_name)

    # 2. 检查最终文件是否已存在
    if os.path.exists(final_nc_path):
        print(f"✅ 最终文件已存在，跳过: {final_nc_path}")
        return

    print(f"⬇️ 正在请求 ERA5 数据: {year}-{month:02d} ...")

    try:
        # 3. 下载数据 (保存为 .zip)
        # CDS API 即使指定 netcdf，对于混合变量（瞬时+累积）也会返回 ZIP
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_temperature',  # 瞬时值
                    'total_precipitation',  # 累积值 (通常导致被分包)
                    '10m_u_component_of_wind',  # 瞬时值
                    '10m_v_component_of_wind',  # 瞬时值
                ],
                'year': str(year),
                'month': f"{month:02d}",
                'day': [str(d).zfill(2) for d in range(1, 32)],  # 自动生成 01-31
                'time': [f"{h:02d}:00" for h in range(24)],  # 自动生成 00:00-23:00
                'area': [41, 110, 34, 115],  # 北, 西, 南, 东
            },
            temp_zip_path)

        # 4. 解压并处理
        print(f"📦 下载完成，正在解压: {temp_zip_name} ...")

        extracted_nc_files = []  # 记录解压出来的临时文件路径

        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            # 找出所有的 .nc 文件 (可能有一个 data.nc，也可能有 data.nc 和 data_1.nc)
            nc_members = [f for f in all_files if f.endswith('.nc')]

            if not nc_members:
                raise Exception("错误：压缩包里没找到任何 .nc 文件！")

            # 解压所有 nc 文件
            for member in nc_members:
                zip_ref.extract(member, path=SAVE_DIR)
                extracted_nc_files.append(os.path.join(SAVE_DIR, member))

        # 5. 合并或重命名逻辑
        if len(extracted_nc_files) == 1:
            # Case A: 只有一个文件，直接重命名
            print("🧩 压缩包内仅包含一个文件，直接重命名...")
            # 如果目标存在先删除（防止rename报错）
            if os.path.exists(final_nc_path):
                os.remove(final_nc_path)
            os.rename(extracted_nc_files[0], final_nc_path)

        else:
            # Case B: 包含多个文件 (说明气温和降水被分开了)，需要合并
            print(f"🧩 检测到 {len(extracted_nc_files)} 个分块文件，正在用 xarray 合并...")

            datasets = []
            try:
                # 读取所有临时文件
                for f in extracted_nc_files:
                    datasets.append(xr.open_dataset(f))

                # 合并 (compat='override' 忽略微小的坐标差异)
                combined_ds = xr.merge(datasets, compat='override')

                # 保存为最终文件
                # 提示：engine='netcdf4' 确保兼容性，encoding用于压缩（可选）
                combined_ds.to_netcdf(final_nc_path, engine='netcdf4')
                print("✅ 合并并保存成功！")

            except Exception as merge_err:
                raise Exception(f"合并过程中出错: {merge_err}")
            finally:
                # 务必关闭文件句柄，否则无法删除临时文件 (Windows常见问题)
                for ds in datasets:
                    ds.close()

            # 删除解压出来的临时分块文件 (如 data.nc, data_1.nc)
            for f in extracted_nc_files:
                if os.path.exists(f):
                    os.remove(f)

        # 6. 删除原始 zip 文件
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

        print(f"🎉 处理完毕: {final_nc_name}")

    except Exception as e:
        print(f"❌ 下载或处理出错 ({year}-{month:02d}): {e}")
        # 出错时保留 zip 以便排查，或者也可以选择在这里删除
        # if os.path.exists(temp_zip_path): os.remove(temp_zip_path)


if __name__ == "__main__":
    # 下载 2020 年全年的数据
    for month in range(1, 13):
        download_era5_month(2020, month)