import numpy as np
import matplotlib.pyplot as plt

def generate_china_market_prices(Nt=672, delta_t=0.25):
    """
    基于中国山东电力现货市场(2024)特征生成价格数据
    单位: RMB/kWh
    """
    # 1. 构建典型的"鸭子曲线"基础形状 (24小时)
    hours = np.arange(0, 24, delta_t)
    base_curve = np.zeros_like(hours)

    for i, h in enumerate(hours):
        if 0 <= h < 6:          # 夜间平段
            base_curve[i] = 0.35
        elif 6 <= h < 9:        # 早高峰爬坡
            base_curve[i] = 0.60
        elif 9 <= h < 15:       # 午间光伏深谷 (关键特征)
            # 模拟山东市场的"负电价"或极低电价
            base_curve[i] = 0.05 + 0.1 * np.cos(np.pi * (h - 12) / 3)
        elif 15 <= h < 17:      # 下午回升
            base_curve[i] = 0.50
        elif 17 <= h < 21:      # 晚高峰 (最高价)
            base_curve[i] = 1.10
        else:                   # 21-24 回落
            base_curve[i] = 0.40

    # 2. 生成一周的数据 (Day-Ahead)
    # 加入每日的随机扰动，模拟工作日/周末差异
    da_prices = []
    for day in range(7):
        # 每日基础波动 +/- 10%
        daily_factor = np.random.uniform(0.9, 1.1)
        # 加上高斯噪声
        noise = np.random.normal(0, 0.02, len(base_curve))
        daily_price = base_curve * daily_factor + noise
        da_prices.append(np.maximum(daily_price, -0.05)) # 允许少量负电价，但不低于-0.05

    da_prices = np.concatenate(da_prices)

    # 3. 生成实时价格 (Real-Time)
    # 实时价格通常比日前价格波动更大 (Volatility)
    # 模拟逻辑: RT = DA + 较大的随机波动
    rt_noise = np.random.normal(0, 0.15, Nt) # 波动幅度大
    rt_prices = da_prices + rt_noise

    # 实时价格可能出现更极端的峰谷
    rt_prices = np.clip(rt_prices, -0.1, 1.5) # 限制在合理区间 [-0.1, 1.5] RMB

    # 4. 生成取消/偏差惩罚 (Cancel Penalty)
    # 对应中国市场的"两个细则"偏差考核费用
    # 假设为固定惩罚费率，约为峰值电价的 10%-15%
    cancel_penalty = np.ones(Nt) * 0.15 # 0.15 RMB/kWh 的固定惩罚成本

    # 封装返回
    prices = {
        'day_ahead': da_prices,       # (672,)
        'real_time': rt_prices,       # (672,)
        'cancel_penalty': cancel_penalty # (672,)
    }
    return prices, hours

# --- 执行生成并可视化 ---
price_data, day_hours = generate_china_market_prices()

# 绘图展示前24小时对比
plt.figure(figsize=(10, 5))
plt.plot(day_hours, price_data['day_ahead'][:96], 'b-', label='Day-Ahead Price (RMB/kWh)')
plt.plot(day_hours, price_data['real_time'][:96], 'r--', alpha=0.6, label='Real-Time Price (RMB/kWh)')
plt.axhline(y=0.15, color='g', linestyle=':', label='Penalty Cost')
plt.title("Synthesized China Spot Market Prices (Shandong Typical Profile)")
plt.xlabel("Hour of Day")
plt.ylabel("Price (RMB/kWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 打印部分数据验证
print("Price Statistics (RMB/kWh):")
print(f"DA Mean: {np.mean(price_data['day_ahead']):.3f}, Max: {np.max(price_data['day_ahead']):.3f}, Min: {np.min(price_data['day_ahead']):.3f}")
print(f"RT Mean: {np.mean(price_data['real_time']):.3f}, Max: {np.max(price_data['real_time']):.3f}, Min: {np.min(price_data['real_time']):.3f}")