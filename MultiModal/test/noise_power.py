import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from utils.merics import evaluate_metrics
from utils.config import load_config
from MultiModal.model.dataset import SatellitePVDataset
from MultiModal.model.MultiModalPVNet import MultiModalPVNet

def inject_ts_noise(nums, variance=0.0, drop_rate=0.0):
    """
    精准给光伏功率数值序列 (x_numeric) “下毒”
    nums shape: (Batch, Seq_Len, Features)
    特征顺序: ['Power_Norm', 'Clear_Sky_GHI', 'Solar_Zenith']
    """
    # 克隆一份数据，防止修改原始数据
    noisy_nums = nums.clone()

    # ⚠️ 核心细节：我们只给第 0 个特征 (历史光伏功率 Power_Norm) 下毒！
    # 因为 GHI 和天顶角是天文公式算出来的理论值，现实中绝不会受传感器损坏影响。
    power_seq = noisy_nums[:, :, 0]

    # ☠️ 毒药 1：高斯白噪声 (Gaussian Noise)
    if variance > 0.0:
        std = variance ** 0.5
        noise = torch.randn_like(power_seq) * std
        power_seq = power_seq + noise

    # ☠️ 毒药 2：随机丢包 (Random Drop / Missing Data)
    if drop_rate > 0.0:
        # 生成一个与 power_seq 形状相同的 0-1 均匀分布随机矩阵
        # 如果随机数小于 drop_rate，说明这个点“断网”了，强制填 0
        drop_mask = torch.rand_like(power_seq) < drop_rate
        power_seq[drop_mask] = 0.0

    # 数据归一化保护：防止加了噪声后超出物理意义的界限 (0~1)
    noisy_nums[:, :, 0] = torch.clamp(power_seq, 0.0, 1.0)

    return noisy_nums


def run_robustness_test(model, test_loader, device, test_type="noise"):
    """
    运行抗噪测试并绘制曲线
    test_type: "noise" (测高斯噪声) 或 "drop" (测丢包率)
    """
    model.eval()

    if test_type == "noise":
        levels = [0.0, 0.01, 0.05, 0.1, 0.2]  # 噪声方差
        title = "Robustness Against Gaussian Noise (Sensor Jitter)"
        xlabel = "Noise Variance ($\sigma^2$)"
    else:
        levels = [0.0, 0.1, 0.2, 0.3, 0.5]  # 丢包率 (10% 到 50% 的数据丢失)
        title = "Robustness Against Missing Data (Communication Drop)"
        xlabel = "Missing Data Rate"

    print(f"\n🚀 开始执行 {title} 测试...")
    rmse_results = []

    with torch.no_grad():
        for level in levels:
            all_daytime_preds = []
            all_daytime_targets = []

            for batch in test_loader:
                imgs = batch['x_images'].to(device)
                nums = batch['x_numeric'].to(device)
                targets = batch['y'].to(device)
                zeniths = batch['y_zenith'].to(device)

                # 💉 精准注射毒药
                if test_type == "noise":
                    noisy_nums = inject_ts_noise(nums, variance=level, drop_rate=0.0)
                else:
                    noisy_nums = inject_ts_noise(nums, variance=0.0, drop_rate=level)

                # 模型推理 (注意：此时喂给模型的是被毒化的数值特征)
                preds, _, _ = model(imgs, noisy_nums)

                # 🛡️ 戴上面具：只考核白天预测得准不准
                mask = zeniths <= 85.0
                if mask.sum() > 0:
                    all_daytime_preds.append(preds[mask].cpu())
                    all_daytime_targets.append(targets[mask].cpu())

            if len(all_daytime_preds) > 0:
                all_daytime_preds = torch.cat(all_daytime_preds, dim=0)
                all_daytime_targets = torch.cat(all_daytime_targets, dim=0)

                # 调用之前的评估函数 (假设你已经定义了 evaluate_metrics)
                metrics = evaluate_metrics(all_daytime_preds, all_daytime_targets)
                rmse = metrics['RMSE']
                print(f"干扰强度 {level:<4} | RMSE: {rmse:.8f} | MAE: {metrics['MAE']:.8f} | R: {metrics['R(%)']:.6f}%")
                rmse_results.append(rmse)
            else:
                rmse_results.append(0)

    # 📈 自动画图
    plt.figure(figsize=(8, 5))
    plt.plot(levels, rmse_results, marker='s', color='#1f77b4', linewidth=2.5, markersize=8, label='MultiModal (Ours)')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('RMSE (Root Mean Square Error)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(levels)

    save_path = f"robustness_{test_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 曲线图已保存至: {save_path}")


# ================= 测试执行入口 =================
if __name__ == "__main__":
    # 硬件设置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {DEVICE}")
    # 加载配置
    config = load_config("../config/config.yaml")
    # 路径设置
    CSV_PATH = config["file_paths"]["series_file"]
    SAT_DIR = config["file_paths"]["aligned_satellite_path"]
    SAVE_DIR = "../checkpoints/"

    # 1. 实例化测试集 Loader (用你之前修改好的 Dataset)
    test_dataset = SatellitePVDataset(CSV_PATH, SAT_DIR, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. 重新初始化模型，并加载你刚训练好的最强权重！
    model = MultiModalPVNet(visual_dim=64, ts_dim=64, output_seq_len=4).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_mae_model.pth")))

    # 3. 召唤抗噪评估函数
    # 注意：确保 evaluate_metrics 函数已经被导入或定义在上方
    run_robustness_test(model, test_loader, DEVICE, test_type="")