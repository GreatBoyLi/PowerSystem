import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 导入你的模型和数据集定义
from GPTPV.model.dataset import PVForecastDataset
from GPTPV.model.model import PVGPT
from GPTPV.utils.config import load_config

# ⚠️ 这里的模型参数必须与你【训练时】设置的一模一样！
# 如果你训练时改成了 d_model=128，这里也要改，否则报错
MODEL_CONFIG = {
    'd_model': 28,
    'nhead': 4,
    'num_encoder_layers': 1,
    'num_decoder_layers': 1
}


def load_stats(stats_file):
    """读取均值和方差，用于反归一化"""
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"找不到统计文件: {stats_file}，无法还原真实功率！")

    df = pd.read_csv(stats_file)
    mean_val = df['mean'].iloc[0]
    std_val = df['std'].iloc[0]
    print(f"📊 加载统计参数: Mean={mean_val:.4f}, Std={std_val:.4f}")
    return mean_val, std_val


def predict_full_year(config):
    # 真实数据的 CSV (清洗后的)
    DATA_CSV_PATH = config["file_paths"]["lllmy_clean_file"]  # 给 Dataset 用的文件
    # 统计参数 CSV (用于反归一化)
    STATS_CSV_PATH = config["file_paths"]["lllmy_stats_file"]  # 保存均值方差的文件
    # 训练好的模型权重
    weights_dir = config["file_paths"]["weights_dir"]
    weights_name = "best_pretrained_model.pth"
    MODEL_PATH = os.path.join(weights_dir, weights_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- 1. 加载数据 ---
    # 技巧：设置 train_ratio=0，这样所有数据都会被 Dataset 划分为 'val' 模式
    # 从而实现对整个文件（全年）的遍历
    print(f"📂 读取全年数据: {DATA_CSV_PATH}")
    full_ds = PVForecastDataset(DATA_CSV_PATH, mode='val', train_ratio=0.0)
    # shuffle=False 非常重要！必须按时间顺序预测
    full_loader = DataLoader(full_ds, batch_size=512, shuffle=False, num_workers=4)

    # --- 2. 加载模型 ---
    print(f"🧠 加载模型: {MODEL_PATH}")
    model = PVGPT(**MODEL_CONFIG).to(device)

    # 加载权重 (处理可能的 DataParallel keys)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # 如果权重里的 key 有 'module.' 前缀 (多卡训练产生)，需要去掉
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # --- 3. 加载统计量 ---
    real_mean, real_std = load_stats(STATS_CSV_PATH)

    # --- 4. 开始预测 ---
    preds = []
    truths = []

    print("🔮 开始进行全年预测...")
    with torch.no_grad():
        for batch in tqdm(full_loader):
            x_seq = batch['x_seq'].to(device)
            y_seq = batch['y_seq'].to(device)
            x_time = batch['x_time'].to(device)
            y_time = batch['y_time'].to(device)

            tgt_input = torch.zeros_like(y_seq).to(device)

            # 模型推理
            output = model(x_seq, tgt_input, x_time, y_time)

            # --- 关键步骤：取预测序列的第一个点拼接 ---
            # output shape: [Batch, 16, 1] -> 取 [:, 0, 0]
            # 这样拼起来就是连续的时间序列
            batch_pred = output[:, 0, 0].cpu().numpy()
            batch_true = y_seq[:, 0, 0].cpu().numpy()

            preds.extend(batch_pred)
            truths.extend(batch_true)

    # 转为 numpy 数组
    preds = np.array(preds)
    truths = np.array(truths)

    # --- 5. 反归一化 (Inverse Normalization) ---
    print("🔄 执行反归一化...")
    preds_kw = preds * real_std + real_mean
    truths_kw = truths * real_std + real_mean

    # 修正负值 (光伏功率不能为负)
    preds_kw[preds_kw < 0] = 0
    truths_kw[truths_kw < 0] = 0

    return truths_kw, preds_kw


def plot_results(truths, preds):
    """绘制并保存图像（含趋势曲线）"""
    print("🎨 正在绘图...")

    # --- 新增：计算滑动平均趋势线 ---
    # 将 numpy 数组转换为 pandas Series 以便计算
    s_true = pd.Series(truths)
    s_pred = pd.Series(preds)

    # 设定滑动窗口大小
    # 数据是15分钟间隔，一天有 96 个点
    # 这里的 window=96*7 表示计算“7天移动平均线”，能很好地展示季节性趋势
    window_size = 96 * 7

    # 计算均值 (center=True 让曲线对齐中间)
    ma_true = s_true.rolling(window=window_size, center=True).mean()
    ma_pred = s_pred.rolling(window=window_size, center=True).mean()

    # 1. 绘制全年概览图 (背景+趋势线)
    plt.figure(figsize=(15, 6))

    # A. 绘制原始数据 (背景)
    # 降低 alpha 透明度，让它看起来像淡色的背景“柱状图”
    plt.plot(truths, label='Actual (Raw)', color='green', alpha=0.25, linewidth=0.5)
    plt.plot(preds, label='Predicted (Raw)', color='red', alpha=0.25, linewidth=0.5)

    # B. 绘制趋势曲线 (前景)
    # 加粗 linewidth，使用更深的颜色
    plt.plot(ma_true, label='Actual Trend (7-Day Avg)', color='darkgreen', linewidth=2.5)
    plt.plot(ma_pred, label='Predicted Trend (7-Day Avg)', color='darkred', linewidth=2.5)

    plt.title('Full Year PV Power Prediction with Trend Lines')
    plt.xlabel('Time Steps (15min intervals)')
    plt.ylabel('Power (kW)')
    plt.legend(loc='upper right')  # 图例放右上角
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_full_year.png', dpi=300)
    print("✅ 全年预测图（含趋势线）已保存: prediction_full_year.png")

    # 2. 绘制局部细节图 (保持不变，或也可以加上局部趋势)
    start_idx = 2000
    end_idx = 2400  # 400个点 ≈ 4天

    if len(truths) > end_idx:
        plt.figure(figsize=(15, 6))
        # 局部图通常不需要平滑曲线，因为我们要看具体的拟合细节
        plt.plot(range(start_idx, end_idx), truths[start_idx:end_idx], label='Actual', color='green', marker='.',
                 markersize=4)
        plt.plot(range(start_idx, end_idx), preds[start_idx:end_idx], label='Prediction', color='red', linestyle='--',
                 linewidth=2)

        plt.title('Zoomed-in Detail (4 Days)')
        plt.xlabel('Time Steps')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_detail.png', dpi=300)
        print("✅ 局部细节图已保存: prediction_detail.png")


if __name__ == "__main__":
    # --- 1. 配置参数 ---
    # 请确保这些路径与你实际文件位置一致
    config = load_config()

    # 执行预测
    y_true, y_pred = predict_full_year(config)

    # 绘图
    plot_results(y_true, y_pred)

    # 可选：计算一下误差指标
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"\n📊 最终误差指标 (kW):")
    print(f"   RMSE: {rmse:.4f} kW")
    print(f"   MAE : {mae:.4f} kW")
