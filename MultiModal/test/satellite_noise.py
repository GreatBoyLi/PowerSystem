import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from utils.merics import evaluate_metrics
from utils.config import load_config
from MultiModal.model.dataset import SatellitePVDataset
from MultiModal.model.MultiModalPVNet import MultiModalPVNet


def add_gaussian_noise(images, variance):
    """
    给卫星云图添加指定方差的高斯噪声 (Gaussian Noise)
    """
    if variance == 0.0:
        return images

    # 标准差是方差的平方根
    std = variance ** 0.5
    # 生成与原图形状相同的高斯噪声
    noise = torch.randn_like(images) * std
    # 将噪声叠加到原图上
    noisy_images = images + noise

    # 🌟 关键：因为我们的图片之前归一化到了 [0, 1] 左右
    # 加了噪声可能会越界，所以要把数值强行截断在合理范围内，防止网络输入崩溃
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

    return noisy_images


def evaluate_noise_robustness(model, test_loader, device, variances=[0.0, 0.01, 0.05, 0.1, 0.2]):
    """
    在不同的噪声方差下，评估模型的抗干扰能力
    """
    print("🛡️ 开始进行 Noise Robustness Evaluation (抗噪鲁棒性测试)...")
    model.eval()

    # 用于保存各个噪声级别下的 RMSE 结果
    rmse_results = []

    with torch.no_grad():
        for var in variances:
            all_daytime_preds = []
            all_daytime_targets = []

            for batch in test_loader:
                imgs = batch['x_images'].to(device)
                nums = batch['x_numeric'].to(device)
                targets = batch['y'].to(device)
                zeniths = batch['y_zenith'].to(device)

                # ☠️ 1. 给卫星云图“下毒”
                noisy_imgs = add_gaussian_noise(imgs, var)

                # 2. 模型进行推理
                preds, _, _ = model(noisy_imgs, nums)

                # 3. 🛡️ 戴上面具：剔除夜间数据 (天顶角 > 85°)
                # 我们只关心白天加了噪声后，模型预测得准不准
                mask = zeniths <= 86.0

                # 提取白天的预测值和真实值
                if mask.sum() > 0:
                    all_daytime_preds.append(preds[mask].cpu())
                    all_daytime_targets.append(targets[mask].cpu())

            # 拼接整个测试集的白天数据
            if len(all_daytime_preds) > 0:
                all_daytime_preds = torch.cat(all_daytime_preds, dim=0)
                all_daytime_targets = torch.cat(all_daytime_targets, dim=0)

                # 调用我们之前写的终极评估函数
                metrics = evaluate_metrics(all_daytime_preds, all_daytime_targets)
                rmse = metrics['RMSE']
                print(f"噪声方差: {var:<4} | 纯白天 RMSE: {rmse:.8f} | R: {metrics['R(%)']:.6f}%")
                rmse_results.append(rmse)
            else:
                print(f"噪声方差: {var:<4} | 没有有效的白天数据！")
                rmse_results.append(0)

    # ==========================================
    # 📈 自动绘制抗噪折线图
    # ==========================================
    plt.figure(figsize=(8, 5))
    # 画出带标记的红色实线
    plt.plot(variances, rmse_results, marker='o', color='#d62728', linewidth=2.5, markersize=8, label='MultiModal-DCCA')

    plt.title('Noise Robustness Evaluation', fontsize=14, fontweight='bold')
    plt.xlabel('Gaussian Noise Variance ($/\sigma^2$)', fontsize=12)
    plt.ylabel('RMSE (Root Mean Square Error)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # 限制 x 轴刻度，使其和 variances 列表对齐
    plt.xticks(variances)

    save_path = "noise_robustness_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n🎉 抗噪评估完成！极其精美的曲线图已保存至: {save_path}")


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
    evaluate_noise_robustness(model, test_loader, DEVICE, variances=[0.0, 0.01, 0.05, 0.1, 0.2])
