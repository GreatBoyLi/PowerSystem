import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
from GPTPV.utils.config import load_config

config_file = "../config/config.yaml"
config = load_config(config_file)

# --- 配置修改 ---
# 1. 这里现在指向包含 nc 文件的文件夹路径
ERA5_DIR = config["file_paths"]["era5_dir"]
OUTPUT_CSV = config["file_paths"]["era5_output"]
REAL_STATIONS = config["stations"]["real_stations"]
POINTS_PER_STATION = config["stations"]["virtual_points_per_station"]

# 2. 获取时间范围
START_DATE = pd.to_datetime(config["dates"]["start_date"])
# 给结束日期加上 23小时59分59秒，确保覆盖一整天
END_DATE = pd.to_datetime(config["dates"]["end_date"]) + pd.Timedelta(hours=23, minutes=59, seconds=59)


def get_needed_year_months(start_dt, end_dt):
    """返回 (year, month) 的列表，包含 start 到 end 涉及的所有月份"""
    result = []
    curr = start_dt
    while curr <= end_dt:
        result.append(curr.strftime("%Y_%m"))
        # 移动到下个月第一天
        if curr.month == 12:
            curr = pd.Timestamp(year=curr.year + 1, month=1, day=1)
        else:
            curr = pd.Timestamp(year=curr.year, month=curr.month + 1, day=1)

    # 去重（虽以上逻辑不会重复，但为了保险）并返回
    return sorted(list(set(result)))


# ===========================================

def get_relevant_era5_files(data_dir, start_date, end_date):
    """
    根据开始和结束日期，筛选出需要读取的 .nc 文件列表。
    文件名格式假设为: era5_shanxi_YYYY_MM.nc
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "era5_shanxi_*.nc")))
    selected_files = []

    # 生成我们需要覆盖的年月列表 (例如: 2020-01, 2020-02)
    # 使用 'MS' (Month Start) 频率生成
    needed_periods = get_needed_year_months(start_date, end_date)

    # 如果时间范围在一个月内 (例如 1月5日到1月10日)，date_range可能为空，手动补上
    if len(needed_periods) == 0:
        needed_periods = [start_date.strftime("%Y_%m")]
        # 如果跨月但没满一个月(例如1月31到2月1日)，需要把结束月也加上
        if start_date.strftime("%Y_%m") != end_date.strftime("%Y_%m"):
            needed_periods.append(end_date.strftime("%Y_%m"))

    print(f"📅 需要寻找的月份: {list(needed_periods)}")

    for f_path in all_files:
        f_name = os.path.basename(f_path)
        # 简单粗暴匹配：只要文件名包含 "2020_01" 这种字符串就选中
        for period in needed_periods:
            if period in f_name:
                selected_files.append(f_path)
                break

    return sorted(selected_files)


def extract_and_broadcast_era5():
    # 1. 筛选文件
    relevant_files = get_relevant_era5_files(ERA5_DIR, START_DATE, END_DATE)

    if not relevant_files:
        print(f"❌ 在 {ERA5_DIR} 下未找到匹配 {START_DATE} 到 {END_DATE} 的文件！")
        return

    print(f"📂 将加载以下 {len(relevant_files)} 个文件:")
    for f in relevant_files:
        print(f"   - {os.path.basename(f)}")

    # 2. 使用 open_mfdataset 同时打开多个文件并自动合并时间维度
    print("🔄 正在加载并合并数据集...")
    # chunks参数有助于处理大文件，防止内存溢出
    try:
        ds = xr.open_mfdataset(relevant_files, combine='by_coords', chunks={'time': 500})
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 3. 标准化时间列名 (防止有的文件叫 valid_time 有的叫 time)
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})

    # 4. 🎯 核心步骤：时间切片 (Time Slicing)
    # 这一步只保留 config 中配置的时间段
    print(f"✂️ 正在裁切时间范围: {START_DATE} -> {END_DATE}")
    try:
        ds_sliced = ds.sel(time=slice(START_DATE, END_DATE))
    except Exception as e:
        print(f"❌ 时间裁切失败，请检查nc文件内的时间格式。错误: {e}")
        return

    if ds_sliced.time.size == 0:
        print("⚠️ 裁切后数据为空！请检查 Start/End Date 是否在文件的时间范围内。")
        return

    # 5. 时间重采样与插值 (1h -> 15min) [保留之前的核心逻辑]
    print("⏳ 正在执行 15分钟 频率的插值处理...")
    ds_15min = ds_sliced.resample(time='15min').interpolate('linear')

    # 6. 手动构造最后 3 个时间点 (23:15, 23:30, 23:45)
    # 获取最后一个时间点 (23:00) 的数据
    last_frame = ds_15min.isel(time=-1)

    # 生成需要补充的时间戳
    last_time = ds_15min.time.values[-1]
    tail_times = pd.date_range(start=last_time + pd.Timedelta(minutes=15), periods=3, freq='15min')

    # 7. 创建尾部数据 (复制 23:00 的值)
    # 我们通过遍历 timestamps，把 last_frame 赋予新的时间坐标
    tail_list = []
    for t in tail_times:
        # 复制数据，并扩展维度赋予新的时间
        # expand_dims 配合 assign_coords 是 xarray 标准增加时间步的方法
        new_frame = last_frame.expand_dims(time=1).assign_coords(time=[t])
        tail_list.append(new_frame)

    # 8. 拼接 (Concat)
    # 将 原本的数据 + 3个新的尾巴 拼起来
    ds_final = xr.concat([ds_15min] + tail_list, dim='time')

    # 将变量名指回 ds_15min 以便后续代码不用改
    ds_15min = ds_final

    # 9. 物理量计算
    print("🧮 正在进行物理量计算...")
    temp_c = ds_15min['t2m'] - 273.15
    precip_mm = ds_15min['tp'] * 1000
    precip_mm = precip_mm.where(precip_mm >= 0, 0)
    wind_speed = np.sqrt(ds_15min['u10'] ** 2 + ds_15min['v10'] ** 2)

    print("📥 正在将计算结果加载至内存 (Persisting data)...")
    # load() 会强制触发计算并把结果存入内存，之后的 .sel 就会是毫秒级的纯内存操作
    # 如果没有这步，则会在 .values 处触发计算
    temp_c = temp_c.load()
    precip_mm = precip_mm.load()
    wind_speed = wind_speed.load()

    # 10. 提取与广播
    # 为了避免内存溢出，如果数据量特别大，这里可以考虑先 load() 进内存
    # 或者直接进行计算 (xarray 是懒加载的)
    print("🚀 正在提取并分发数据...")

    # 确保时间索引是 pandas datetime
    time_index = pd.to_datetime(ds_15min.time.values)
    data_dict = {"Timestamp": time_index}

    for station in REAL_STATIONS:
        s_name = station['name']
        s_lat = station['lat']
        s_lon = station['lon']

        # 提取
        t_val = temp_c.sel(latitude=s_lat, longitude=s_lon, method='nearest').values
        p_val = precip_mm.sel(latitude=s_lat, longitude=s_lon, method='nearest').values
        w_val = wind_speed.sel(latitude=s_lat, longitude=s_lon, method='nearest').values

        for i in range(POINTS_PER_STATION):
            base_col = f"{s_name}_P{i}"
            data_dict[f"{base_col}_Temp"] = t_val
            data_dict[f"{base_col}_Wind"] = w_val
            data_dict[f"{base_col}_Precip"] = p_val

    # 8. 保存
    final_df = pd.DataFrame(data_dict)
    final_df = final_df.set_index("Timestamp")

    print(f"===== 📊 数据预览 (时间范围: {final_df.index.min()} 到 {final_df.index.max()}) =====")
    print(final_df.iloc[-10:, :3].head())

    final_df.to_csv(OUTPUT_CSV, header=True)
    print(f"\n✅ 处理完成！已保存至: {OUTPUT_CSV}")


if __name__ == "__main__":
    extract_and_broadcast_era5()
