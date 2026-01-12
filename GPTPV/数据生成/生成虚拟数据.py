import pandas as pd
import matplotlib.pyplot as plt
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# ==========================================
# 1. 设置虚拟电站的地理位置 (对应论文 Section V-A)
# ==========================================
# 假设我们在中国山西 (论文中提到的地点)
latitude = 37.8
longitude = 112.5
tz = 'Asia/Shanghai'
altitude = 800 # 海拔

location = Location(latitude, longitude, tz, altitude, name='Shanxi_Virtual_Station')

# ==========================================
# 2. 准备/模拟气象数据 (对应论文 Section V-B)
# ==========================================
# 论文中提到时间分辨率是 15分钟 (freq='15min')
times = pd.date_range(start='2024-06-01', end='2024-06-03', freq='15min', tz=tz)

# A. 获取“晴天”理论辐照度 (模拟 GHI)
# 在实际项目中，这里应该读取你的 NWP (数值天气预报) 或历史气象数据
cs = location.get_clearsky(times)

# B. 创建气象 DataFrame
# 论文提到输入主要是：GHI (总辐射), Air Temp (气温), Wind Speed (风速)
weather_data = pd.DataFrame(index=times)
weather_data['ghi'] = cs['ghi']  # 使用晴天模型作为 GHI
weather_data['dni'] = cs['dni']  # 直射
weather_data['dhi'] = cs['dhi']  # 散射
weather_data['temp_air'] = 25.0  # 假设气温 25度 (实际应为时序数据)
weather_data['wind_speed'] = 2.0 # 假设风速 2m/s

# ==========================================
# 3. 定义光伏系统参数 (物理建模的核心)
# ==========================================
# 这里定义组件的参数，模拟一个真实的单晶硅组件
module_parameters = {
    'pdc0': 300,  # 标准测试条件下的功率 (300W)
    'gamma_pdc': -0.004, # 温度系数
    'b': 0.05,
    'a_c': 500, # 经验参数
    'alpha_sc': 0.001,
    'I_L_ref': 9.0,
    'I_o_ref': 2e-10,
    'R_s': 0.3,
    'R_sh_ref': 500,
    'Adjust': 2.5
}

# 温度模型参数 (让模型知道温度高了效率会降)
temp_model_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# 定义支架安装方式 (固定支架，朝南，倾角30度)
mount = FixedMount(surface_tilt=30, surface_azimuth=180)

# 定义阵列
array = Array(mount=mount,
              module_parameters=module_parameters,
              temperature_model_parameters=temp_model_params)

# 定义逆变器 (简单假设效率 96%)
inverter_parameters = {
    'pdc0': 3000, # 逆变器额定直流功率
    'eta_inv_nom': 0.96 # 额定效率
}

# 组装系统
system = PVSystem(arrays=[array],
                  inverter_parameters=inverter_parameters)

# ==========================================
# 4. 运行模型链 (The Model Chain)
# ==========================================
# 这一步就是论文说的 "map weather data to virtual PV power data"
mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss')

# 开始计算
# 也就是输入气象数据 -> 经过一系列物理公式 -> 输出功率
mc.run_model(weather_data)

# ==========================================
# 5. 结果展示
# ==========================================
virtual_power = mc.results.ac  # 获取交流侧功率 (AC Power)

print("生成的数据前5行：")
print(virtual_power.head())

# 简单的可视化
plt.figure(figsize=(10, 5))
virtual_power.plot()
plt.title('Generated Virtual PV Power (Model Chain Output)')
plt.ylabel('Power (W)')
plt.xlabel('Time')
plt.grid(True)
plt.show()

# 保存为 CSV (这就变成了你的 Training Data)
# virtual_power.to_csv('virtual_pv_data.csv')