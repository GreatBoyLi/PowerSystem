import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_grid_points(center_lat, center_lon, radius_km, resolution_km):
    """
    生成以中心点为中心，指定半径和分辨率的密集网格点。

    参数:
    center_lat (float): 中心纬度
    center_lon (float): 中心经度
    radius_km (float): 覆盖半径 (公里)
    resolution_km (float): 网格间距 (公里)

    返回:
    pd.DataFrame: 包含 'latitude', 'longitude' 的表格
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

    # 生成纬度和经度数组 (使用 numpy 的 arange)
    lats = np.arange(center_lat - delta_lat, center_lat + delta_lat, step_lat)
    lons = np.arange(center_lon - delta_lon, center_lon + delta_lon, step_lon)

    # 生成网格 (Meshgrid)
    # 这会生成所有可能的组合点
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 展平为列表
    points = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten()
    })

    return points


# ==========================================
# 实战演示
# ==========================================

# 1. 设定中心点 (比如论文中的山西某地)
center_lat = 37.8
center_lon = 112.5

# 2. 设定参数
# 半径 50公里，每隔 2公里 取一个点 (分辨率)
df_grid = generate_grid_points(center_lat, center_lon, radius_km=50, resolution_km=2)

print(f"生成的虚拟站点数量: {len(df_grid)} 个")
print(df_grid.head())

# ==========================================
# 3. 可视化检查
# ==========================================
plt.figure(figsize=(8, 8))
plt.scatter(df_grid['longitude'], df_grid['latitude'], s=1, alpha=0.5, label='Virtual Stations')
plt.scatter(center_lon, center_lat, color='red', s=100, marker='*', label='Target Station (Center)')

plt.title(f'Generated Grid Points ({len(df_grid)} Stations)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.axis('equal')  # 保持比例，防止地图变形
plt.show()

# 4. 保存坐标列表，下一步就可以拿去查天气数据了
# df_grid.to_csv('virtual_station_coordinates.csv', index=False)