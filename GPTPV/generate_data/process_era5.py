import xarray as xr
import numpy as np
import pandas as pd
import os

# ================= 配置区域 =================
# 1. ERA5 文件路径 (你刚下载的那个)
ERA5_FILE = "./data/era5/era5_shanxi_2020_01.nc"

# 2. 之前生成的 "100个虚拟站点" 的坐标
# ⚠️ 注意：为了演示，这里我直接定义生成逻辑。
# 在实际项目中，建议你读取上一步生成的 csv 里的坐标，或者复用 get_spatial_indices 的结果
# 这里我们再次模拟生成这 100 个点的坐标 (Lat, Lon)
# 假设我们只处理第 1 个电站的第 1 个点作为演示
TARGET_POINTS = [
    {"id": "Station_1_P0", "lat": 37.80, "lon": 112.50},
    # ... 实际应该有 100 个点
]


# ===========================================

def process_era5_data(nc_file, target_points):
    print(f"🔄 正在处理: {nc_file}")
    ds = xr.open_dataset(nc_file)

    # 1. 准备坐标网格 (ERA5)
    # ERA5 的 lat 也是从大到小，lon 从小到大
    era_lats = ds['latitude'].values
    era_lons = ds['longitude'].values

    results_temp = {}  # 存气温
    results_precip = {}  # 存降水

    # 2. 空间匹配：为每个虚拟站点找最近的 ERA5 网格
    for pt in target_points:
        # 计算距离 (简单的绝对值差，找下标)
        # abs(数组 - 目标值).argmin() 返回最近值的索引
        lat_idx = np.abs(era_lats - pt['lat']).argmin()
        lon_idx = np.abs(era_lons - pt['lon']).argmin()

        # 提取该网格的所有时间数据
        # t2m = 2米气温, tp = 总降水
        # ⚠️ 注意变量名可能是 't2m' 或 '2t', 'tp' 或 'total_precipitation'，请根据上一步"体检"结果修改
        raw_temp = ds['t2m'][:, lat_idx, lon_idx].to_pandas()  # 转成 Pandas Series
        raw_precip = ds['tp'][:, lat_idx, lon_idx].to_pandas()

        # === 数据清洗与单位换算 ===

        # A. 气温处理
        # 单位：开尔文 -> 摄氏度
        temp_c = raw_temp - 273.15
        # 时间插值：1小时 -> 15分钟
        # resample('15T') 会生成空行，interpolate('linear') 会填补
        temp_15min = temp_c.resample('15min').interpolate(method='linear')

        # B. 降水处理
        # 单位：米 -> 毫米 (x1000)
        # 逻辑：论文说 "Daily precipitation was calculated by summing 1 h cumulants"
        # 所以我们要先算日总和，然后把这个数字“广播”给当天的所有 15分钟时刻
        precip_mm = raw_precip * 1000
        daily_precip = precip_mm.resample('D').sum()  # 算出每天的总降水

        # 把日降水映射回 15分钟数据 (向前填充)
        # 例如：1月1日全天的 precip 都是 1月1日的总和
        precip_15min = daily_precip.reindex(temp_15min.index, method='ffill')

        # 存入字典
        results_temp[pt['id']] = temp_15min
        results_precip[pt['id']] = precip_15min

    ds.close()

    # 3. 合并成 DataFrame
    df_temp = pd.DataFrame(results_temp)
    df_precip = pd.DataFrame(results_precip)

    # 给列名加后缀区分
    df_temp.columns = [f"{c}_Temp" for c in df_temp.columns]
    df_precip.columns = [f"{c}_Precip" for c in df_precip.columns]

    # 横向合并
    final_df = pd.concat([df_temp, df_precip], axis=1)
    return final_df


if __name__ == "__main__":
    if os.path.exists(ERA5_FILE):
        df_era5 = process_era5_data(ERA5_FILE, TARGET_POINTS)

        print("\n===== ✅ 处理结果预览 =====")
        print(df_era5.head())
        print(f"\n数据形状: {df_era5.shape}")

        # 保存一下看看
        df_era5.to_csv("era5_processed_sample.csv")
        print("已保存至 era5_processed_sample.csv")
    else:
        print("请先下载数据！")