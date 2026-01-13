import cdsapi
import os

# 保存路径
SAVE_DIR = "../data/era5/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 启动客户端
c = cdsapi.Client()


def download_era5_month(year, month):
    """
    下载指定年月的 ERA5 hourly 数据
    """
    filename = os.path.join(SAVE_DIR, f"era5_shanxi_{year}_{month:02d}.nc")

    if os.path.exists(filename):
        print(f"✅ 文件已存在，跳过: {filename}")
        return

    print(f"⬇️ 正在请求 ERA5 数据: {year}-{month:02d} ... (这可能需要排队)")

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',  # 依然使用 .nc 格式

            # 变量选择 (根据论文需求)
            'variable': [
                '2m_temperature',  # 气温
                'total_precipitation',  # 降水
                '10m_u_component_of_wind', # 建议加上风速，即便论文摘要没提
                '10m_v_component_of_wind',
            ],

            # 时间选择 (ERA5 是每小时数据)
            'year': str(year),
            'month': f"{month:02d}",
            'day': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
                '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24',
                '25', '26', '27', '28', '29', '30', '31',
            ],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
            ],

            # 区域选择 (North, West, South, East)
            # 这是一个包含山西的大矩形框，避免下载全球数据浪费时间
            'area': [
                41, 110, 34, 115,
            ],
        },
        filename)
    print(f"✅ 下载完成: {filename}")


if __name__ == "__main__":
    # 下载 2020 年全年的数据
    for month in range(1, 13):
        download_era5_month(2020, month)
