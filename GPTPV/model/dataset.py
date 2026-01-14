import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from GPTPV.utils.config import load_config


class PVForecastDataset(Dataset):
    def __init__(self, csv_file, input_len=112, output_len=16, mode='train', train_ratio=0.8):
        """
        参数:
        - csv_file: 刚才生成的 normalized power csv 路径
        - input_len: 输入序列长度 (论文取 112) [cite: 377]
        - output_len: 预测序列长度 (论文取 16)
        - mode: 'train' 或 'val' (划分训练集和验证集)
        """
        self.input_len = input_len
        self.output_len = output_len

        # 1. 读取数据
        print(f"📂 Loading dataset from {csv_file}...")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # 2. 提取时间戳特征 (用于 Time Embedding)
        # 论文公式 (6): T_k = [e_d, e_w, e_m, e_y]
        # 我们预先计算好所有时间步的特征
        timestamps = df.index

        # Day of Year (归一化到 -0.5 ~ 0.5)
        day_of_year = timestamps.dayofyear.values
        e_y = (day_of_year - 1) / 365.0 - 0.5

        # Day of Month
        day_of_month = timestamps.day.values
        e_m = (day_of_month - 1) / 30.0 - 0.5

        # Day of Week
        day_of_week = timestamps.dayofweek.values
        e_w = (day_of_week) / 6.0 - 0.5

        # Hour of Day (注意：数据是15min间隔，论文公式是 Hour number)
        # 这里我们用 (hour + minute/60) 精度更高，或者严格按论文只取 hour
        hour_of_day = timestamps.hour.values
        e_d = (hour_of_day) / 23.0 - 0.5

        # 拼接时间特征: (Time_Steps, 4)
        self.time_features = np.stack([e_d, e_w, e_m, e_y], axis=1).astype(np.float32)

        # 3. 处理功率数据 (Time_Steps, Num_Stations)
        self.data = df.values.astype(np.float32)

        # 4. 生成滑窗索引 (Samples)
        # 并不是简单的切片，因为我们有 100 个站点。
        # 我们把所有站点的数据视为“独立的样本”，但在时间轴上滑动。
        # 总样本数 = (时间步数 - window_size + 1) * 站点数

        n_timestamps, n_stations = self.data.shape
        total_window = input_len + output_len

        # 划分训练/验证集 (按时间切分)
        split_idx = int(n_timestamps * train_ratio)

        self.samples = []

        if mode == 'train':
            # 遍历时间轴 (直到 split_idx)
            for t in range(split_idx - total_window):
                # 遍历所有站点
                for s in range(n_stations):
                    self.samples.append((t, s))
        else:
            # 验证集
            for t in range(split_idx, n_timestamps - total_window):
                for s in range(n_stations):
                    self.samples.append((t, s))

        print(f"✅ {mode.upper()} Dataset created. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取当前样本的起始时间和站点索引
        start_t, station_idx = self.samples[idx]

        # 1. 切分时间窗口
        mid_t = start_t + self.input_len
        end_t = mid_t + self.output_len

        # 2. 获取功率数据 (Power Value)
        # Input: [0, 112]
        x_seq = self.data[start_t: mid_t, station_idx]
        # Target: [112, 128]
        y_seq = self.data[mid_t: end_t, station_idx]

        # 3. 获取时间特征 (Time Embedding Input)
        # 注意：Transformer 需要知道 Input 和 Output 对应的时间
        x_time = self.time_features[start_t: mid_t, :]  # Encoder 用的时间
        y_time = self.time_features[mid_t: end_t, :]  # Decoder 用的时间

        # 4. 扩展维度以适配模型输入 (seq_len, 1) -> 因为是单变量
        return {
            "x_seq": torch.tensor(x_seq).unsqueeze(-1),  # Encoder Input (112, 1)
            "y_seq": torch.tensor(y_seq).unsqueeze(-1),  # Target (16, 1)
            "x_time": torch.tensor(x_time),  # Encoder Time (112, 4)
            "y_time": torch.tensor(y_time)  # Decoder Time (16, 4)
        }


# --- 测试代码 ---
if __name__ == "__main__":
    config_file = "../config/config.yaml"
    config = load_config(config_file)
    # 假设你的 CSV 路径
    csv_path = config["file_paths"]["output_power_csv"]

    # 创建 Dataset
    train_ds = PVForecastDataset(csv_path, mode='train')

    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # 取一个 Batch 看看长什么样
    batch = next(iter(train_loader))
    print("\n📦 Batch Data Shapes:")
    print(f"Encoder Input (Power): {batch['x_seq'].shape}")  # 预期: [32, 112, 1]
    print(f"Target Output (Power): {batch['y_seq'].shape}")  # 预期: [32, 16, 1]
    print(f"Encoder Time Feats:    {batch['x_time'].shape}")  # 预期: [32, 112, 4]