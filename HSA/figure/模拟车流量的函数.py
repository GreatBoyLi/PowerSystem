import numpy as np
import matplotlib.pyplot as plt

# 定义时间轴 (0-24小时)
hour = np.linspace(0, 24, 1000)

# 计算高斯峰值
y = np.exp(-(hour - 9)**2 / 6)
sigma_traffic = y * 0.2
y = np.maximum(0, np.random.normal(y, sigma_traffic))

# 绘图
plt.figure(figsize=(10, 4))
plt.plot(hour, y, label='Gaussian Peak (9:00 AM)', color='blue', lw=2)
plt.fill_between(hour, y, alpha=0.2, color='blue') # 阴影部分模拟流量分布
plt.title('Visualization of Traffic Peak Modeling')
plt.xlabel('Hour of Day')
plt.ylabel('Weight Factor (0-1)')
plt.xticks(np.arange(0, 25, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()