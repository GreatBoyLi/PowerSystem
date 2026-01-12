"""
基于物理模型生成虚拟光伏功率数据的完整代码
参考论文：An Ultra-Short-Term Distributed Photovoltaic Power Forecasting Method Based on GPT
作者：Hengqi Zhang, Jie Yang, Siyuan Fan, Hua Geng, Changkun Shao
IEEE Transactions on Sustainable Energy, 2025

本代码实现论文中描述的虚拟光伏数据生成方法，使用pvlib库基于物理模型生成15分钟间隔的功率数据
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib import pvsystem, modelchain, location
from pvlib.iotools import get_pvgis_tmy
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class VirtualPVDataGenerator:
    """
    虚拟光伏数据生成器

    根据论文方法，基于物理模型生成虚拟分布式光伏功率数据
    主要步骤：
    1. 确定虚拟光伏电站位置（地理网格）
    2. 获取/生成气象数据（GHI、温度、降水等）
    3. 随机化光伏系统参数
    4. 使用pvlib模型链计算功率
    5. 数据后处理与保存
    """

    def __init__(self, start_date='2020-01-01', end_date='2020-12-31',
                 time_resolution='15min', random_seed=42):
        """
        初始化生成器

        参数：
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
        time_resolution: 时间分辨率，如'15min', '1H'
        random_seed: 随机种子，保证可重复性
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.time_resolution = time_resolution
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # 生成时间索引
        self.time_index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=time_resolution,
            tz='UTC'
        )

        # 论文中提到的PV参数范围（根据论文Ta   ble I推断）
        self.pv_param_ranges = {
            # 倾角：0-90度，实际系统中常见范围15-45度
            'tilt': {'min': 0, 'max': 90, 'unit': 'degrees'},
            # 方位角：-180到180度，北半球通常朝南（0度）
            'azimuth': {'min': -180, 'max': 180, 'unit': 'degrees'},
            # 安装容量：1-1000 kW，分布式光伏典型范围
            'capacity': {'min': 1, 'max': 1000, 'unit': 'kW'},
            # 运行年数：0-20年
            'operation_years': {'min': 0, 'max': 20, 'unit': 'years'},
            # 初始效率：0.15-0.25
            'initial_efficiency': {'min': 0.15, 'max': 0.25, 'unit': 'pu'},
            # 温度系数：-0.004到-0.003 /°C
            'temperature_coeff': {'min': -0.004, 'max': -0.003, 'unit': '/°C'},
            # 清洁天数：每月0-4次（转化为每年0-48次）
            'cleaning_days_per_year': {'min': 0, 'max': 48, 'unit': 'days/year'}
        }

    def generate_geographic_grid(self, center_lat=37.5, center_lon=112.5,
                                 radius_km=100, resolution_km=10):
        """
        生成地理网格点（虚拟光伏电站位置）

        参数：
        center_lat, center_lon: 中心点经纬度（山西太原附近）
        radius_km: 生成网格的半径（公里）
        resolution_km: 网格分辨率（公里）

        返回：
        DataFrame包含latitude, longitude列
        """
        # 地球近似半径 (km)
        R = 6371

        # 1度纬度大约对应的公里数
        km_per_lat_deg = 111
        # 1度经度对应的公里数 (取决于纬度)
        km_per_lon_deg = 111 * np.cos(np.radians(center_lat))

        # 计算需要生成的经纬度范围 (Delta)
        delta_lat = radius_km / km_per_lat_deg
        delta_lon = radius_km / km_per_lon_deg

        # 计算步长 (Step)
        step_lat = resolution_km / km_per_lat_deg
        step_lon = resolution_km / km_per_lon_deg

        # 生成纬度和经度数组
        lats = np.arange(center_lat - delta_lat, center_lat + delta_lat, step_lat)
        lons = np.arange(center_lon - delta_lon, center_lon + delta_lon, step_lon)

        # 生成网格并展平
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        locations = pd.DataFrame({
            'station_id': [f'PV_{i:03d}' for i in range(len(lat_grid.flatten()))],
            'latitude': lat_grid.flatten(),
            'longitude': lon_grid.flatten()
        })

        print(f"生成了 {len(locations)} 个虚拟光伏电站位置")
        return locations

    def generate_meteorological_data(self, locations, use_synthetic=True):
        """
        生成气象数据

        参数：
        locations: 位置DataFrame
        use_synthetic: 是否使用合成数据（如无真实数据）

        返回：
        dict，键为station_id，值为包含气象数据的DataFrame
        """
        n_stations = len(locations)
        meteo_data = {}

        if use_synthetic:
            print("使用合成气象数据...")
            for idx, row in locations.iterrows():
                station_id = row['station_id']
                lat, lon = row['latitude'], row['longitude']

                # 生成合成气象数据（基于典型日模式）
                df_meteo = self._generate_synthetic_meteo(lat, lon)
                meteo_data[station_id] = df_meteo

        else:
            print("使用真实气象数据（需要网络连接）...")
            # 这里可以集成真实数据源，如PVGIS、NSRDB等
            for idx, row in locations.iterrows():
                try:
                    station_id = row['station_id']
                    lat, lon = row['latitude'], row['longitude']

                    # 使用PVGIS获取典型气象年数据
                    # 注意：需要网络连接，且PVGIS API有调用限制
                    tmy_data, _, _ = get_pvgis_tmy(
                        latitude=lat,
                        longitude=lon,
                        start=2010,
                        end=2020
                    )

                    # 重采样到15分钟间隔
                    tmy_data = tmy_data.resample('15min').interpolate()
                    meteo_data[station_id] = tmy_data

                except Exception as e:
                    print(f"获取站点 {station_id} 气象数据失败: {e}")
                    # 回退到合成数据
                    df_meteo = self._generate_synthetic_meteo(lat, lon)
                    meteo_data[station_id] = df_meteo

        return meteo_data

    def _generate_synthetic_meteo(self, lat, lon):
        """
        生成合成气象数据（当没有真实数据时使用）

        基于地理位置和季节模式生成合理的GHI、温度、降水数据
        """
        n_timesteps = len(self.time_index)

        # 创建基础DataFrame
        df = pd.DataFrame(index=self.time_index)

        # 1. 生成日照时角（影响GHI的日变化）
        # 简单模拟：使用正弦函数模拟日变化，考虑季节变化
        hours_of_day = self.time_index.hour + self.time_index.minute / 60.0
        day_of_year = self.time_index.dayofyear

        # 日变化幅度（正午最大）
        day_variation = np.sin((hours_of_day - 12) * np.pi / 12)
        day_variation[day_variation < 0] = 0  # 夜晚为0

        # 季节变化（夏季更高）
        season_variation = 1 + 0.3 * np.sin((day_of_year - 172) * 2 * np.pi / 365)

        # 2. 生成GHI（全球水平辐照度，W/m²）
        # 典型范围：夜晚0，正午可达1000 W/m²
        ghi_base = 800  # 最大GHI
        ghi_noise = np.random.normal(0, 50, n_timesteps)  # 随机噪声
        ghi = ghi_base * day_variation * season_variation + ghi_noise
        ghi[ghi < 0] = 0
        ghi[ghi > 1200] = 1200

        # 添加随机云层影响
        cloud_prob = np.random.random(n_timesteps)
        cloud_mask = cloud_prob < 0.2  # 20%的时间有云
        ghi[cloud_mask] = ghi[cloud_mask] * np.random.uniform(0.1, 0.5, sum(cloud_mask))

        df['ghi'] = ghi

        # 3. 生成温度（°C）
        # 日变化模式：夜间低，午后高
        temp_base = 15  # 平均温度
        temp_daily_amp = 10  # 日温差
        temp_season_amp = 15  # 季节温差

        temp = (temp_base +
                temp_daily_amp * day_variation * 0.8 +
                temp_season_amp * np.sin((day_of_year - 172) * 2 * np.pi / 365) +
                np.random.normal(0, 2, n_timesteps))

        # 纬度修正（高纬度温度更低）
        lat_effect = (45 - abs(lat)) / 45 * 10  # 赤道最热，极地最冷
        temp = temp - lat_effect

        df['temp_air'] = temp

        # 4. 生成风速（m/s）
        wind_speed = np.random.weibull(2, n_timesteps) * 3  # Weibull分布，典型风速
        df['wind_speed'] = wind_speed

        # 5. 生成降水（mm/day，论文中使用日总降水量）
        # 降水概率随季节变化
        rain_prob = 0.05 + 0.1 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
        rain_events = np.random.random(n_timesteps) < rain_prob

        # 日降水量（mm）
        daily_precip = np.zeros(n_timesteps)
        for i in range(0, n_timesteps, 96):  # 96个15分钟=24小时
            if i + 96 <= n_timesteps:
                if np.any(rain_events[i:i + 96]):
                    # 生成一个降水日
                    rain_amount = np.random.exponential(5)  # 平均5mm
                    # 将日降水量分配到有降水的时刻
                    rain_moments = np.where(rain_events[i:i + 96])[0]
                    if len(rain_moments) > 0:
                        daily_precip[i:i + 96][rain_moments] = rain_amount / len(rain_moments)

        df['precipitation'] = daily_precip

        return df

    def generate_random_pv_parameters(self, n_stations):
        """
        为每个虚拟光伏电站生成随机参数

        返回：
        DataFrame，每行对应一个电站的参数
        """
        pv_params_list = []

        for i in range(n_stations):
            params = {
                'station_id': f'PV_{i:03d}',
                # 倾角：实际系统常见范围15-45度
                'tilt': np.random.uniform(15, 45),
                # 方位角：北半球通常朝南（0度），有一定偏差
                'azimuth': np.random.uniform(-30, 30),
                # 安装容量（kW）：分布式光伏典型范围
                'capacity_kw': np.random.uniform(10, 500),
                # 运行年数
                'operation_years': np.random.uniform(0, 10),
                # 初始效率
                'initial_efficiency': np.random.uniform(0.18, 0.22),
                # 温度系数
                'temperature_coeff': np.random.uniform(-0.004, -0.003),
                # 每年清洁次数
                'cleaning_days_per_year': np.random.randint(0, 24),
                # 其他参数
                'modules_per_string': np.random.randint(5, 15),
                'strings_per_inverter': np.random.randint(1, 5),
                # 根据论文，还包括清洗日、初始效率、灵敏度等
                'soiling_loss': np.random.uniform(0.02, 0.05),  # 污秽损失
                'mismatch_loss': np.random.uniform(0.02, 0.03),  # 失配损失
                'wiring_loss': np.random.uniform(0.01, 0.02),  # 线损
            }
            pv_params_list.append(params)

        return pd.DataFrame(pv_params_list)

    def calculate_pv_power(self, location_row, pv_params, meteo_data):
        """
        使用pvlib计算单个光伏电站的功率

        参数：
        location_row: 包含latitude, longitude的Series
        pv_params: 光伏参数Series
        meteo_data: 气象数据DataFrame

        返回：
        power_series: 功率时间序列（kW）
        """
        try:
            # 1. 创建位置对象
            site = location.Location(
                latitude=location_row['latitude'],
                longitude=location_row['longitude'],
                tz='UTC',
                altitude=200,  # 默认海拔200m
                name=location_row['station_id']
            )

            # 2. 定义光伏系统参数
            # 使用Canadian Solar CS6K-280M模块作为示例
            sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
            cec_modules = pvlib.pvsystem.retrieve_sam('CecMod')

            # 选择模块（随机选择或固定）
            module_name = 'Canadian_Solar_CS6K_280M'
            if module_name in sandia_modules.columns:
                module = sandia_modules[module_name]
            else:
                # 使用默认模块
                module = {
                    'pdc0': 280,  # 标称功率(W)
                    'V_mp': 31.4,
                    'I_mp': 8.92,
                    'V_oc': 38.9,
                    'I_sc': 9.45,
                    'alpha_sc': 0.0005,
                    'beta_oc': -0.0031,
                    'gamma_pdc': -0.004,
                    'cells_in_series': 60,
                }

            # 逆变器参数
            inverter_name = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_'
            sandia_inverters = pvlib.pvsystem.retrieve_sam('sandiainverter')
            if inverter_name in sandia_inverters.columns:
                inverter = sandia_inverters[inverter_name]
            else:
                inverter = {
                    'pdc0': 250,  # 标称功率(W)
                    'pdc': 250,
                    'pnt': 0.5,
                    'vac': 208,
                    'vdcmax': 60,
                }

            # 3. 创建PVSystem对象
            system = pvsystem.PVSystem(
                modules_per_string=int(pv_params['modules_per_string']),
                strings_per_inverter=int(pv_params['strings_per_inverter']),
                inverter=inverter,
                module_parameters=module,
                temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'],
                inverter_parameters=inverter,
                racking_model='open_rack',
                albedo=0.2,  # 地面反射率
            )

            # 4. 创建ModelChain
            mc = modelchain.ModelChain(
                system,
                site,
                aoi_model='physical',
                spectral_model='no_loss',
                temperature_model='sapm',
                losses_model='no_loss'
            )

            # 5. 准备气象数据
            weather = pd.DataFrame({
                'ghi': meteo_data['ghi'],
                'dni': meteo_data['ghi'] * 0.7,  # 估算DNI
                'dhi': meteo_data['ghi'] * 0.3,  # 估算DHI
                'temp_air': meteo_data['temp_air'],
                'wind_speed': meteo_data['wind_speed']
            }, index=meteo_data.index)

            # 6. 运行模型链
            mc.run_model(weather)

            # 7. 获取交流功率（kW）
            ac_power_kw = mc.results.ac / 1000.0  # 转换为kW

            # 8. 应用系统损失
            total_loss = (pv_params['soiling_loss'] +
                          pv_params['mismatch_loss'] +
                          pv_params['wiring_loss'])

            # 考虑清洁影响（清洁后效率提高）
            cleaning_effect = 1 - (pv_params['soiling_loss'] *
                                   (1 - pv_params['cleaning_days_per_year'] / 365))

            # 考虑老化（运行年数影响）
            aging_factor = 1 - (pv_params['operation_years'] * 0.005)  # 每年0.5%衰减

            # 应用所有损失和修正
            final_power = (ac_power_kw *
                           (1 - total_loss) *
                           cleaning_effect *
                           aging_factor *
                           (pv_params['capacity_kw'] / 280)  # 容量缩放
                           )

            # 确保功率非负且不超过容量
            final_power[final_power < 0] = 0
            final_power[final_power > pv_params['capacity_kw']] = pv_params['capacity_kw']

            return final_power

        except Exception as e:
            print(f"计算电站 {location_row['station_id']} 功率时出错: {e}")
            # 返回零序列作为备用
            return pd.Series(0, index=self.time_index, name='power_kw')

    def generate_virtual_pv_data(self, n_stations=10, center_lat=37.5, center_lon=112.5):
        """
        主函数：生成虚拟光伏数据

        参数：
        n_stations: 虚拟电站数量
        center_lat, center_lon: 中心点经纬度

        返回：
        dict包含：
            - locations: 地理位置DataFrame
            - pv_parameters: 光伏参数DataFrame
            - meteo_data: 气象数据dict
            - power_data: 功率数据DataFrame（电站×时间）
        """
        print("=" * 60)
        print("开始生成虚拟光伏功率数据")
        print(f"时间范围: {self.start_date.date()} 到 {self.end_date.date()}")
        print(f"时间分辨率: {self.time_resolution}")
        print(f"虚拟电站数量: {n_stations}")
        print("=" * 60)

        # 1. 生成地理位置
        print("\n1. 生成虚拟光伏电站位置...")
        # 先生成网格，然后随机选择n_stations个点
        all_locations = self.generate_geographic_grid(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=50,  # 50公里半径
            resolution_km=20  # 20公里分辨率
        )

        # 随机选择n_stations个位置
        if len(all_locations) > n_stations:
            selected_indices = np.random.choice(
                len(all_locations),
                size=n_stations,
                replace=False
            )
            locations = all_locations.iloc[selected_indices].reset_index(drop=True)
        else:
            locations = all_locations

        # 2. 生成光伏参数
        print("\n2. 生成随机光伏参数...")
        pv_parameters = self.generate_random_pv_parameters(n_stations)

        # 3. 生成气象数据
        print("\n3. 生成气象数据...")
        meteo_data = self.generate_meteorological_data(locations, use_synthetic=True)

        # 4. 计算每个电站的功率
        print("\n4. 计算光伏功率...")
        power_data = pd.DataFrame(index=self.time_index)

        for idx, (_, location_row) in enumerate(locations.iterrows()):
            station_id = location_row['station_id']
            print(f"  计算电站 {station_id} ({idx + 1}/{n_stations})...")

            # 获取该电站的参数和气象数据
            pv_params = pv_parameters.iloc[idx]
            station_meteo = meteo_data[station_id]

            # 计算功率
            power_series = self.calculate_pv_power(location_row, pv_params, station_meteo)
            power_data[station_id] = power_series

        # 5. 数据后处理
        print("\n5. 数据后处理...")
        # 添加时间特征
        power_data['year'] = power_data.index.year
        power_data['month'] = power_data.index.month
        power_data['day'] = power_data.index.day
        power_data['hour'] = power_data.index.hour
        power_data['minute'] = power_data.index.minute

        # 计算总功率
        power_columns = [col for col in power_data.columns if col.startswith('PV_')]
        power_data['total_power_kw'] = power_data[power_columns].sum(axis=1)

        # 6. 保存数据
        print("\n6. 保存数据...")
        result = {
            'locations': locations,
            'pv_parameters': pv_parameters,
            'meteo_data': meteo_data,
            'power_data': power_data,
            'time_index': self.time_index,
            'config': {
                'start_date': str(self.start_date.date()),
                'end_date': str(self.end_date.date()),
                'time_resolution': self.time_resolution,
                'n_stations': n_stations,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'random_seed': self.random_seed
            }
        }

        return result

    def save_to_files(self, result, output_dir='./virtual_pv_data'):
        """
        将生成的数据保存到文件

        参数：
        result: generate_virtual_pv_data返回的结果
        output_dir: 输出目录
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # 保存配置
        config_file = os.path.join(output_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(result['config'], f, indent=2)

        # 保存地理位置
        locations_file = os.path.join(output_dir, 'locations.csv')
        result['locations'].to_csv(locations_file, index=False)

        # 保存光伏参数
        params_file = os.path.join(output_dir, 'pv_parameters.csv')
        result['pv_parameters'].to_csv(params_file, index=False)

        # 保存功率数据
        power_file = os.path.join(output_dir, 'power_data.csv')
        result['power_data'].to_csv(power_file)

        # 保存气象数据（每个电站单独保存）
        meteo_dir = os.path.join(output_dir, 'meteo_data')
        os.makedirs(meteo_dir, exist_ok=True)

        for station_id, meteo_df in result['meteo_data'].items():
            meteo_file = os.path.join(meteo_dir, f'{station_id}_meteo.csv')
            meteo_df.to_csv(meteo_file)

        print(f"数据已保存到目录: {output_dir}")
        print(f"  - 配置: {config_file}")
        print(f"  - 位置: {locations_file}")
        print(f"  - 参数: {params_file}")
        print(f"  - 功率: {power_file}")
        print(f"  - 气象: {meteo_dir}/")

    def visualize_results(self, result, sample_days=7):
        """
        可视化生成的数据

        参数：
        result: 生成的数据结果
        sample_days: 显示多少天的数据
        """
        power_data = result['power_data']
        locations = result['locations']

        # 选择最近sample_days天的数据
        end_date = power_data.index[-1]
        start_date = end_date - timedelta(days=sample_days)
        mask = (power_data.index >= start_date) & (power_data.index <= end_date)
        sample_data = power_data[mask]

        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('虚拟光伏数据生成结果', fontsize=16)

        # 1. 总功率时间序列
        ax1 = axes[0, 0]
        ax1.plot(sample_data.index, sample_data['total_power_kw'], 'b-', linewidth=1)
        ax1.set_title(f'总光伏功率 ({sample_days}天)')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('功率 (kW)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 2. 单个电站功率示例
        ax2 = axes[0, 1]
        station_columns = [col for col in power_data.columns if col.startswith('PV_')]
        sample_station = station_columns[0] if station_columns else 'total_power_kw'
        ax2.plot(sample_data.index, sample_data[sample_station], 'g-', linewidth=1)
        ax2.set_title(f'示例电站 {sample_station} 功率')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('功率 (kW)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 3. 地理位置分布
        ax3 = axes[1, 0]
        scatter = ax3.scatter(locations['longitude'], locations['latitude'],
                              c=result['pv_parameters']['capacity_kw'],
                              cmap='viridis', s=50, alpha=0.7)
        ax3.set_title('虚拟光伏电站地理位置')
        ax3.set_xlabel('经度')
        ax3.set_ylabel('纬度')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='容量 (kW)')

        # 4. 容量分布直方图
        ax4 = axes[1, 1]
        capacities = result['pv_parameters']['capacity_kw']
        ax4.hist(capacities, bins=20, edgecolor='black', alpha=0.7)
        ax4.set_title('光伏电站容量分布')
        ax4.set_xlabel('容量 (kW)')
        ax4.set_ylabel('频数')
        ax4.grid(True, alpha=0.3)

        # 5. 日功率曲线（平均）
        ax5 = axes[2, 0]
        power_data['hour_decimal'] = power_data['hour'] + power_data['minute'] / 60.0
        daily_profile = power_data.groupby('hour_decimal')['total_power_kw'].mean()
        ax5.plot(daily_profile.index, daily_profile.values, 'r-', linewidth=2)
        ax5.set_title('平均日功率曲线')
        ax5.set_xlabel('小时')
        ax5.set_ylabel('平均功率 (kW)')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 24)

        # 6. 月总发电量
        ax6 = axes[2, 1]
        monthly_energy = power_data.groupby('month')['total_power_kw'].sum() * 0.25  # 15分钟间隔，乘以0.25小时
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax6.bar(range(1, 13), monthly_energy, alpha=0.7)
        ax6.set_title('月总发电量')
        ax6.set_xlabel('月份')
        ax6.set_ylabel('发电量 (kWh)')
        ax6.set_xticks(range(1, 13))
        ax6.set_xticklabels(months, rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('./virtual_pv_data/virtual_pv_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()


# 使用示例
def main():
    """主函数示例"""
    # 1. 初始化生成器
    generator = VirtualPVDataGenerator(
        start_date='2020-01-01',
        end_date='2020-03-31',  # 3个月数据用于演示
        time_resolution='15min',
        random_seed=42
    )

    # 2. 生成虚拟光伏数据
    result = generator.generate_virtual_pv_data(
        n_stations=10,  # 生成10个虚拟电站
        center_lat=37.5,  # 山西太原附近
        center_lon=112.5
    )

    # 3. 保存数据
    generator.save_to_files(result, output_dir='./virtual_pv_data')

    # 4. 可视化结果
    generator.visualize_results(result, sample_days=7)

    # 5. 打印统计信息
    print("\n" + "=" * 60)
    print("数据生成完成！统计信息：")
    print("=" * 60)

    power_data = result['power_data']
    power_columns = [col for col in power_data.columns if col.startswith('PV_')]

    print(f"时间范围: {power_data.index[0]} 到 {power_data.index[-1]}")
    print(f"时间点数: {len(power_data)}")
    print(f"虚拟电站数: {len(power_columns)}")
    print(f"总发电量: {power_data['total_power_kw'].sum() * 0.25:.1f} kWh")  # 15分钟间隔

    for i, station in enumerate(power_columns[:3]):  # 显示前3个电站信息
        station_power = power_data[station]
        print(f"\n电站 {station}:")
        print(f"  最大功率: {station_power.max():.2f} kW")
        print(f"  平均功率: {station_power.mean():.2f} kW")
        print(f"  容量因子: {station_power.mean() / result['pv_parameters'].iloc[i]['capacity_kw']:.3f}")


if __name__ == "__main__":
    main()
