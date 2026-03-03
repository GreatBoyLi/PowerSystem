import pandas as pd
import pvlib
import os
from utils.config import load_config


# ===========================================

def process_timeseries(config):
    # 您的 CSV 文件路径 (请确保文件在正确位置)
    CSV_PATH = config["file_paths"]["pv_file"]

    # 结果保存路径
    OUTPUT_PATH = config["file_paths"]["series_file"]

    # 论文指定的实验时间段
    START_DATE = config["dates"]["start_date"]
    END_DATE = config["dates"]["end_date"]

    # 论文指定的电站参数 (BP Solar, Alice Springs) [cite: 285]
    LATITUDE = config["stations"]["lat"]
    LONGITUDE = config["stations"]["lon"]
    ALTITUDE = config["stations"]["altitude"]  # 海拔 (米)
    CAPACITY = config["stations"]["capacity"]  # 装机容量 5.0 kW

    print(f"🚀 开始处理时间序列数据: {CSV_PATH}")

    # 1. 读取数据
    # DKA 数据通常日期格式规范，直接 parse_dates 即可
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'], index_col='timestamp')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 筛选时间范围 (2020-01-01 到 2022-10-08)
    # 注意：包含结束日期的当天
    mask = (df.index >= START_DATE) & (df.index <= f"{END_DATE} 23:59:59")
    df = df.loc[mask]

    if df.empty:
        print("⚠️ 警告：筛选后数据为空，请检查 CSV 中的时间列是否正确！")
        return
    print(f"   筛选后数据量 (5min): {len(df)} 条")

    # 3. 提取核心列并重采样 (5min -> 15min)
    # 我们只需要 Active_Power，其他气象数据如果是实测的也可以保留作为参考，但模型输入主要用计算值
    # resample('15T').mean() 会计算每15分钟的平均功率
    df_15min = df[['Active_Power']].resample('15min').mean()

    # 简单的线性插值填充少量缺失值
    df_15min = df_15min.interpolate(method='linear', limit=4)
    print(f"   重采样后数据量 (15min): {len(df_15min)} 条")

    # ========================================================
    # 🔑 关键修复：赋予时区信息
    # ========================================================
    # 告诉 pandas，这些时间是 "Australia/Darwin" 的当地时间
    # ambiguous='NaT' 处理夏令时切换时的重叠时间
    try:
        # 如果原本没有时区，加上时区
        if df_15min.index.tz is None:
            df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT',
                                                        nonexistent='shift_forward')
        else:
            # 如果原本有时区（比如UTC），转为当地时间
            df_15min.index = df_15min.index.tz_convert('Australia/Darwin')
    except Exception as e:
        print(f"⚠️ 时区转换警告: {e}")
        # 备用方案：强制指定
        df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT')

    print("   ✅ 已修正为爱丽丝泉当地时间 (ACST)")

    # 4. 计算天文学特征 (Zenith & Clear-sky GHI)
    # 这是论文明确要求的两个额外输入特征 [cite: 128, 129]
    print("   正在计算太阳天顶角和晴空辐照度...")

    # 定义地理位置
    location = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE, tz='Australia/Darwin')

    # pvlib 计算需要带时区的时间索引
    # DKA 数据通常是本地时间 (Alice Springs 是 ACST, UTC+9.5)
    # 这里我们简单处理，假设 index 就是本地时间，直接用来计算太阳位置
    times = df_15min.index

    # 4.1 计算太阳位置 (包含 Zenith)
    solpos = location.get_solarposition(times)
    df_15min['Solar_Zenith'] = solpos['zenith'].values

    # 4.2 计算晴空辐照度 (Clear-sky GHI)
    # 使用 Ineichen 模型 (它是标准且效果很好的晴空模型)
    cs = location.get_clearsky(times, model='ineichen')
    df_15min['Clear_Sky_GHI'] = cs['ghi'].values

    # 5. 数据清洗：剔除夜间数据
    # "night data were removed (theta_z > 85)"
    print(f"   清洗前: {len(df_15min)}")

    # 保留 Zenith <= 85 的行
    df_clean = df_15min  #[df_15min['Solar_Zenith'] <= 85].copy()   不筛选了

    # 6. 归一化 (Normalization) [cite: 305]
    # 论文中提到对卫星图做了归一化，通常对功率也需要做归一化以便训练
    # 归一化公式: Power_Norm = Power / Installed_Capacity
    df_clean['Power_Norm'] = df_clean['Active_Power'] / CAPACITY

    # 某些时候功率可能微小负值（逆变器待机），修正为0
    df_clean['Power_Norm'] = df_clean['Power_Norm'].clip(lower=0)
    df_clean['Clear_Sky_GHI'] = df_clean['Clear_Sky_GHI'].clip(lower=0)

    print(f"   清洗后 (剔除夜间): {len(df_clean)}")
    print(f"   ✅ 最终特征列: {list(df_clean.columns)}")

    # 7. 保存结果
    # 确保目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH)
    print(f"💾 处理完成！文件已保存至: {OUTPUT_PATH}")

    # 打印前几行看看
    print("\n数据预览:")
    print(df_clean[['Active_Power', 'Power_Norm', 'Solar_Zenith', 'Clear_Sky_GHI']].head())


if __name__ == "__main__":
    # 加载配置
    config = load_config("../config/config.yaml")

    process_timeseries(config)
