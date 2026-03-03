import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from utils.config import load_config


class SatellitePVDataset(Dataset):
    def __init__(self, csv_path, satellite_dir,
                 input_seq_len=16, output_seq_len=4,
                 mode='train', split_ratio=0.8):
        """
        Args:
            csv_path: 处理好的 CSV 路径
            satellite_dir: .npy 文件所在的根目录
            input_seq_len: 输入序列长度 (4小时 = 16个点)
            output_seq_len: 预测序列长度 (1小时 = 4个点)
            mode: 'train', 'val', or 'test'
        """
        self.input_len = input_seq_len
        self.output_len = output_seq_len
        self.satellite_dir = satellite_dir

        # 1. 读取 CSV (已经是剔除夜间数据的)
        self.df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        self.df = self.df.sort_index()

        # 2. 划分数据集
        n = len(self.df)
        train_end = int(n * split_ratio)
        val_end = int(n * (split_ratio + 0.1))

        if mode == 'train':
            self.data = self.df.iloc[:train_end]
        elif mode == 'val':
            self.data = self.df.iloc[train_end:val_end]
        else:  # test
            self.data = self.df.iloc[val_end:]

        # 3. 打印数据量信息
        print(f"[{mode}] 数据集加载: {len(self.data)} 行")

        # 计算最大可用索引
        # 我们需要取 [i, i+input_len+output_len] 这么长的一段
        # 所以 i 的最大值是 len(data) - (input + output)
        self.max_idx = len(self.data) - (self.input_len + self.output_len)

    def __len__(self):
        # 只要长度够切一个窗口，就是有效样本
        # 简单拼接模式下，样本数 ≈ 总行数
        return max(0, self.max_idx + 1)

    def __getitem__(self, idx):
        # 简单拼接逻辑：直接按行号取，不管时间戳是否连续
        # -------------------------------------------------------
        # 窗口定义：
        # 历史窗口: data[idx : idx + input_len]
        # 未来窗口: data[idx + input_len : idx + input_len + output_len]

        # 1. 切片索引
        hist_start = idx
        hist_end = idx + self.input_len
        future_end = hist_end + self.output_len

        # 2. 获取数值特征 (Power, Zenith, GHI)
        features = ['Power_Norm', 'Clear_Sky_GHI', 'Solar_Zenith']
        x_numeric = self.data.iloc[hist_start:hist_end][features].values

        # 3. 获取预测目标 (未来功率)
        y_power = self.data.iloc[hist_end:future_end]['Power_Norm'].values

        # 4. 获取图像数据
        # 根据对应行的时间戳去读文件
        hist_timestamps = self.data.index[hist_start:hist_end]

        images = []
        for ts in hist_timestamps:
            # 文件名格式: sat_15min_20200101_0600.npy
            file_name = f"sat_15min_{ts.strftime('%Y%m%d_%H%M')}.npy"

            yyyy = ts.strftime("%Y")
            mm = ts.strftime("%m")
            dd = ts.strftime("%d")
            yyyymm = f"{yyyy}{mm}"
            file_path = os.path.join(self.satellite_dir, yyyymm, dd, file_name)

            if os.path.exists(file_path):
                # 读取并归一化
                img = np.load(file_path).astype(np.float32)
                # 假设亮温范围 175-340K
                img = (img - 175.0) / (340.0 - 175.0)
                # 增加 Channel 维度: (H, W) -> (1, H, W)
                img = np.expand_dims(img, axis=0)
            else:
                # print(f"致命错误：找不到卫星云图文件！请检查路径或时间戳是否匹配: {file_path}")
                # 缺图填充 (全0 或 均值)
                img = np.zeros((1, 96, 96), dtype=np.float32)

            images.append(img)

        # 堆叠 -> (Seq, C, H, W)
        x_images = np.stack(images, axis=0)

        return {
            'x_images': torch.from_numpy(x_images).float(),  # Shape: (16, 1, 96, 96)
            'x_numeric': torch.from_numpy(x_numeric).float(),  # Shape: (16, 3)
            'y': torch.from_numpy(y_power).float()  # Shape: (4,)
        }


if __name__ == "__main__":

    # 加载配置
    config = load_config("../config/config.yaml")
    # 测试代码
    csv_file = config["file_paths"]["series_file"]
    sat_dir = config["file_paths"]["aligned_satellite_path"]

    if os.path.exists(csv_file):
        ds = SatellitePVDataset(csv_file, sat_dir, mode='train')
        print(f"Dataset 长度: {len(ds)}")
        if len(ds) > 0:
            sample = ds[0]
            print(f"Input Image: {sample['x_images'].shape}")
            print(f"Input Numeric: {sample['x_numeric'].shape}")
            print(f"Target: {sample['y'].shape}")

            # 检查一下时间戳看看是不是真的拼接了
            print("\n检查第 100 个样本的时间戳 (展示简单拼接效果):")
            # 随便取一段跨天的数据展示
            # 注意：这里的 index 只是演示，具体是否跨天取决于数据
            idx = 45  # 假设这里刚好跨天
            subset = ds.df.iloc[idx: idx + 20]
            print(subset.index)
    else:
        print("请先生成 CSV 文件")
