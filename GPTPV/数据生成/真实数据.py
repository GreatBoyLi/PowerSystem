import xarray as xr
import pandas as pd


def download_jaxa_himawari_data(year, month, region='FULL'):
    """
    下载JAXA处理的Himawari-8数据

    参数：
    year: 年份，如2020
    month: 月份，如1
    region: 'FULL'（全圆盘）或'SEA'（东南亚）

    返回：
    xarray Dataset对象
    """
    # JAXA P-Tree数据URL模板
    base_url = "https://www.eorc.jaxa.jp/ptree/userguide/data"

    # 云产品和辐射产品URL
    # 具体URL结构需要根据产品类型调整
    # 示例：云量产品
    url = f"{base_url}/CLD/{year}/{month:02d}/..."

    # 使用xarray直接打开NetCDF文件
    try:
        ds = xr.open_dataset(url)
        return ds
    except Exception as e:
        print(f"下载失败: {e}")
        return None


# 处理辐射数据
def process_himawari_radiation(ds, lat_range, lon_range):
    """
    提取指定区域的辐射数据
    """
    # 选择区域
    ds_region = ds.sel(latitude=slice(*lat_range),
                       longitude=slice(*lon_range))

    # 提取GHI（假设变量名为'ghi'或'surface_solar_radiation_downwards'）
    if 'surface_solar_radiation_downwards' in ds_region:
        ghi = ds_region['surface_solar_radiation_downwards']
    elif 'ghi' in ds_region:
        ghi = ds_region['ghi']
    else:
        raise KeyError("未找到GHI变量")

    # 转换为DataFrame
    ghi_df = ghi.to_dataframe().reset_index()

    return ghi_df
