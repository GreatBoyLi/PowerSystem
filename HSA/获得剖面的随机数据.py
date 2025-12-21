import numpy as np
import pandas as pd


class HSADataGenerator:
    def __init__(self):
        # 论文核心参数设定
        self.delta_t = 0.25  # 步长 15 分钟 
        self.total_hours = 168  # 总时长 1 周 
        self.Nt = int(self.total_hours / self.delta_t)  # 总步数 = 672

        # 充电行为参数 (参考论文 2.1 节) 
        self.alpha1 = 0.33  # EV 渗透率
        self.alpha2 = 0.11  # 充电需求率
        self.n_ch_max = 50  # 充电桩的数量
        self.p_fast = 0.5  # 快充选择概率
        self.P_F = 0.120  # 快充功率 120kW (单位: MW)
        self.P_S = 0.060  # 慢充功率 60kW (单位: MW)
        self.P_BL = 0.05  # 假设基础负荷 50kW

    def get_diurnal_traffic_mu(self, t):
        """模拟图 1(c) 的双峰交通流均值"""
        hour = (t * self.delta_t) % 24
        # 构造早晚高峰 (9:00 和 18:00)
        mu = 50 + 250 * (np.exp(-(hour - 9) ** 2 / 6) + np.exp(-(hour - 18) ** 2 / 8))
        return mu

    def get_diurnal_pv_mu(self, t):
        """模拟光伏出力"""
        # 1. 获取该时刻的动态均值 mu_PV(t)
        # 只有白天有出力，参考 Fig. 1(d)
        hour = (t * self.delta_t) % 24
        if 6 <= hour <= 18:
            # 这里的 300 是 Fig. 1(d) 中的峰值参考
            mu_pv_t = 300 * np.sin(np.pi * (hour - 6) / 12)

            # 2. 获取该时刻的动态标准差 sigma_PV(t)
            # 论文指出 sigma 也是动态的
            sigma_pv_t = mu_pv_t * 0.15  # 假设 15% 的气象扰动系数

            # 3. 进行正态分布抽样
            sample_val = np.random.normal(mu_pv_t, sigma_pv_t)
            return max(0, sample_val)  # 功率不能为负

        return 0

    def generate_single_profile(self):
        """生成一个 672 步长的完整剖面 """
        profile = {
            'pv_unit': [],  # 单位面积 PV 功率 (W/m2)
            'ch_fast_in': [],  # 进城站快充数量
            'ch_slow_in': [],  # 进城慢充数量
            'load_in': [],  # 进城站负荷 (MW)
            'ch_fast_out': [],  # 出城站快充数量
            'ch_slow_out': [],  # 出城站慢数量
            'load_out': []  # 出城站负荷 (MW)
        }

        for t in range(self.Nt):
            # 1. 抽样单位面积光伏 
            pv_val = self.get_diurnal_pv_mu(t)
            profile['pv_unit'].append(pv_val)

            # 2. 抽样交通量 N ~ N(mu, sigma) 
            mu_traffic = self.get_diurnal_traffic_mu(t)
            sigma_traffic = mu_traffic * 0.2

            # 分别为进城和出城抽样 (增加随机差异)
            n_in = max(0, np.random.normal(mu_traffic, sigma_traffic))
            n_out = max(0, np.random.normal(mu_traffic * 0.85, sigma_traffic))

            # 3. 计算充电负荷 (二项分布模拟个体行为) 
            for n_total, key in zip([n_in, n_out], [['ch_fast_in', 'ch_slow_in', 'load_in'],
                                                    ['ch_fast_out', 'ch_slow_out', 'load_out']]):
                n_ch = np.random.binomial(int(n_total), self.alpha1 * self.alpha2)
                if n_ch > self.n_ch_max:
                    n_fast = np.random.binomial(n_ch, self.p_fast)
                    profile[key[0]].append(n_fast)
                    n_slow = self.n_ch_max - n_fast
                    profile[key[1]].append(n_slow)
                else:
                    n_fast = np.random.binomial(n_ch, self.p_fast)
                    profile[key[0]].append(n_fast)
                    n_slow = 0
                    profile[key[1]].append(0)
                # 总负荷 = 快充 + 慢充 + 基础负荷 
                total_load = n_fast * self.P_F + n_slow * self.P_S + self.P_BL
                profile[key[2]].append(total_load)

        return pd.DataFrame(profile)


# --- 执行生成 ---
generator = HSADataGenerator()
# 生成 10 个随机剖面用于计算期望值 f2 
num_scenarios = 10
scenarios = [generator.generate_single_profile() for _ in range(num_scenarios)]
# 存储数据
count = 0
for scenario in scenarios:
    name = f"data/随机剖面/剖面数据_{count}"
    scenario.to_csv(name, index=False, encoding="utf-8")
    count += 1

# 打印第一个剖面的前 5 行验证结果
print(f"生成的步长总数: {len(scenarios[0])}")  # 应输出 672
print(scenarios[0].head())
