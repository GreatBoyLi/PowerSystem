import pandas as pd
import numpy as np
import pvlib
from tqdm import tqdm
import warnings
from GPTPV.utils.config import load_config
import os

# 忽略警告
warnings.filterwarnings("ignore")


def generate_station_params(station_ids, coords_df, config):
    """
    根据论文 Table I 生成虚拟站点参数
    """
    PARAM_CONFIG = config["parameter1"]

    print("🎲 正在基于真实坐标生成站点参数...")
    np.random.seed(42)
    params_list = []

    if coords_df.index.name != "Station_ID":
        coords_df = coords_df.set_index("Station_ID")

    for sid in station_ids:
        try:
            lat = coords_df.loc[sid, "Latitude"]
            lon = coords_df.loc[sid, "Longitude"]
        except KeyError:
            continue

        azi_offset = np.random.uniform(*PARAM_CONFIG["azimuth_range"])
        azimuth = 180 + azi_offset
        tilt = lat + np.random.uniform(*PARAM_CONFIG["tilt_offset"])
        capacity = np.random.uniform(*PARAM_CONFIG["capacity_range"])

        init_eff = np.random.uniform(*PARAM_CONFIG["efficiency_range"])
        sensitivity = np.random.uniform(*PARAM_CONFIG["sensitivity_range"])
        years = np.random.uniform(*PARAM_CONFIG["years_range"])
        cleaning_interval = np.random.randint(
            PARAM_CONFIG["cleaning_days_range"][0],
            PARAM_CONFIG["cleaning_days_range"][1] + 1
        )

        params_list.append({
            "station_id": sid,
            "latitude": lat,
            "longitude": lon,
            "azimuth": azimuth,
            "tilt": tilt,
            "capacity_kw": capacity,
            "initial_efficiency": init_eff,
            "sensitivity": sensitivity,
            "operation_years": years,
            "cleaning_interval": cleaning_interval
        })

    return pd.DataFrame(params_list).set_index("station_id")


def calculate_soiling_factor(precip_series, cleaning_interval, daily_loss_rate=0.002):
    """计算动态积灰因子"""
    n_steps = len(precip_series)
    soiling_factors = np.ones(n_steps)
    dates = precip_series.index.date
    start_date = dates[0]
    current_dirt = 0.0
    steps_per_day = 96
    loss_per_step = daily_loss_rate / steps_per_day
    precip_values = precip_series.values
    last_clean_day_idx = 0

    for i in range(1, n_steps):
        current_dirt += loss_per_step
        if precip_values[i] > 1.0:  # Rain reset
            current_dirt = 0.0
            last_clean_day_idx = (dates[i] - start_date).days

        current_day_idx = (dates[i] - start_date).days
        if (current_day_idx - last_clean_day_idx) >= cleaning_interval:  # Manual clean
            current_dirt = 0.0
            last_clean_day_idx = current_day_idx

        soiling_factors[i] = max(0.8, 1.0 - current_dirt)

    return soiling_factors


def run_simulation(config):
    # --- 配置路径 ---
    INPUT_WEATHER_CSV = config["file_paths"]["merged_data_output"]
    INPUT_COORDS_CSV = config["file_paths"]["output_coord_csv"]
    OUTPUT_POWER_CSV = config["file_paths"]["output_power_csv"]
    OUTPUT_PARAMS_CSV = config["file_paths"]["output_params_csv"]
    OUTPUT_STATS_CSV = config["file_paths"]["output_stats_csv"]

    if not os.path.exists(INPUT_WEATHER_CSV):
        print(f"❌ 找不到气象数据: {INPUT_WEATHER_CSV}")
        return
    if not os.path.exists(INPUT_COORDS_CSV):
        print(f"❌ 找不到坐标文件: {INPUT_COORDS_CSV}")
        return

    print("📖 读取数据...")
    df_weather = pd.read_csv(INPUT_WEATHER_CSV, index_col="Timestamp", parse_dates=True)
    df_coords = pd.read_csv(INPUT_COORDS_CSV)

    all_cols = df_weather.columns
    station_ids = [c for c in all_cols if not c.endswith(('_Temp', '_Wind', '_Precip'))]
    print(f"🔎 识别到 {len(station_ids)} 个虚拟站点。")

    # --- 生成参数 ---
    params_df = generate_station_params(station_ids, df_coords, config)
    params_df.to_csv(OUTPUT_PARAMS_CSV)

    # --- 太阳位置 ---
    mean_lat = params_df["latitude"].mean()
    mean_lon = params_df["longitude"].mean()
    site_loc = pvlib.location.Location(mean_lat, mean_lon, tz='Asia/Shanghai')
    print("🌞 计算太阳位置...")
    solpos = site_loc.get_solarposition(df_weather.index)

    # --- 🔧 加载组件和逆变器库 ---
    print("🔧 加载 PV 库...")
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    # 1. 查找光伏组件
    module_candidates = [col for col in sandia_modules.columns if 'Canadian_Solar' in col]
    module_name = module_candidates[0] if module_candidates else sandia_modules.columns[0]
    module = sandia_modules[module_name]
    print(f"✅ 组件模型: {module_name}")

    # 2. 查找逆变器
    inverter_candidates = [col for col in cec_inverters.columns if 'Enphase' in col and 'M250' in col]
    inverter_name = inverter_candidates[0] if inverter_candidates else cec_inverters.columns[0]
    inverter = cec_inverters[inverter_name]
    print(f"✅ 逆变器模型: {inverter_name}")

    results_df = pd.DataFrame(index=df_weather.index)

    # 预计算大气参数
    pressure = pvlib.atmosphere.alt2pres(800)  # 海拔800米
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['zenith'])
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)

    print("🚀 开始多站点物理仿真...")
    for sid in tqdm(station_ids):
        if sid not in params_df.index: continue

        # A. 准备输入
        ghi = df_weather[sid]
        temp = df_weather[f"{sid}_Temp"]
        wind = df_weather[f"{sid}_Wind"]
        try:
            precip = df_weather[f"{sid}_Precip"]
        except KeyError:
            precip = pd.Series(0, index=df_weather.index)
        p = params_df.loc[sid]

        # B. 物理计算链（从辐射到直流功率）
        # 1. 分解辐照：把水平面总辐照（GHI）分解为直射（DNI）和散射（DHI）
        irrad = pvlib.irradiance.erbs(ghi, solpos['zenith'], df_weather.index)
        # 2. 计算入射角（AOI）：太阳光线与光伏板表面的夹角
        aoi = pvlib.irradiance.aoi(p['tilt'], p['azimuth'], solpos['zenith'], solpos['azimuth'])
        # 3. 计算光伏板表面总辐照（POA）：直射+散射+反射辐照
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=p['tilt'], surface_azimuth=p['azimuth'],
            dni=irrad['dni'], ghi=ghi, dhi=irrad['dhi'],
            solar_zenith=solpos['zenith'], solar_azimuth=solpos['azimuth']
        )
        # 4. 计算光伏电池温度（不是环境温度）
        cell_temp = pvlib.temperature.faiman(poa['poa_global'], temp, wind)
        # 5. 计算有效辐照（考虑大气、入射角衰减）
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            poa['poa_direct'], poa['poa_diffuse'], airmass_abs, aoi, module
        )

        # 6. 计算直流（DC）功率（基于Sandia组件模型）
        dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temp, module)

        # C. 逆变器 AC 功率计算 (关键修复点)
        # -----------------------------------------------------------
        # 错误修复：cecinverter 数据库使用 sandia 逆变器方程
        # 使用 pvlib.inverter.sandia 而不是 .cec
        # -----------------------------------------------------------
        ac_single = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)

        # 计算系统缩放倍数（匹配装机容量）
        module_rated_power = module['Impo'] * module['Vmpo']
        n_modules = (p['capacity_kw'] * 1000) / module_rated_power

        # 总 AC 功率（单组件功率 × 组件数量）
        ac_total = ac_single * n_modules

        # D. 应用各类损耗（老化、积灰、初始效率）
        aging_factor = 1.0 - (p['sensitivity'] * p['operation_years'])  # 老化损耗
        soiling_factor = calculate_soiling_factor(precip, p['cleaning_interval'])  # 积灰损耗
        # 最终交流功率 = 总AC功率 × 初始效率 × 老化因子 × 积灰因子
        final_ac = ac_total * p['initial_efficiency'] * aging_factor * soiling_factor
        # 数据清洗：填充空值、过滤负功率
        final_ac = final_ac.fillna(0)
        final_ac[final_ac < 0] = 0
        results_df[sid] = final_ac

    # --- 归一化与保存 ---
    print("📊 执行 Z-Score 归一化...")
    stats = results_df.agg(['mean', 'std'])
    stats.to_csv(OUTPUT_STATS_CSV)

    std_safe = stats.loc['std'].replace(0, 1)
    df_norm = (results_df - stats.loc['mean']) / std_safe

    df_norm.to_csv(OUTPUT_POWER_CSV)
    print(f"✅ 仿真完成！结果保存至: {OUTPUT_POWER_CSV}")


if __name__ == "__main__":
    config_file = "../config/config.yaml"
    config = load_config(config_file)
    run_simulation(config)
