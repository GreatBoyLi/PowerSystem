import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # 导入绘图库

from GPTPV.model.dataset import PVForecastDataset
from GPTPV.model.model import PVGPT
from GPTPV.utils.config import load_config

import os

# 在代码开头设置，指定第3张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# --- 配置 ---
config_file = "../config/config.yaml"
config = load_config(config_file)
CSV_PATH = config["file_paths"]["output_power_csv"]
weights_dir = config["file_paths"]["weights_dir"]
weights_name = "best_pretrained_model.pth"
MODEL_SAVE_PATH = os.path.join(weights_dir, weights_name)

# 超参数
BATCH_SIZE = 1024
LR = 0.00005
EPOCHS = 20


def plot_loss_curve(train_losses, val_losses):
    """画出训练和验证的 Loss 曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')  # 保存图像
    print("📉 Loss 曲线已保存至 training_loss_curve.png")
    plt.close()


def plot_validation_results(model, val_loader, device):
    """画出验证集的预测对比图 (随机取4个样本)"""
    model.eval()

    # 获取一个 Batch 的数据
    batch = next(iter(val_loader))
    x_seq = batch['x_seq'].to(device)
    y_seq = batch['y_seq'].to(device)
    x_time = batch['x_time'].to(device)
    y_time = batch['y_time'].to(device)

    tgt_input = torch.zeros_like(y_seq).to(device)

    with torch.no_grad():
        # 预测
        pred = model(x_seq, tgt_input, x_time, y_time)

    # 转回 CPU 方便画图
    y_true = y_seq.cpu().numpy()
    y_pred = pred.cpu().numpy()

    # 画图 (画4个子图展示不同样本)
    plt.figure(figsize=(15, 10))
    for i in range(4):  # 展示 Batch 中的前4个样本
        plt.subplot(2, 2, i + 1)

        # y_true[i] shape is (16, 1), flatten to (16,)
        plt.plot(y_true[i].flatten(), label='Ground Truth', marker='o', color='green')
        plt.plot(y_pred[i].flatten(), label='Prediction', marker='x', color='red', linestyle='--')

        plt.title(f'Validation Sample {i + 1}')
        plt.xlabel('Time Steps (Future 4 hours)')
        plt.ylabel('Normalized Power')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('validation_results.png')
    print("🖼️ 验证集预测对比图已保存至 validation_results.png")
    plt.close()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 准备数据
    train_ds = PVForecastDataset(CSV_PATH, mode='train', train_ratio=0.8)
    val_ds = PVForecastDataset(CSV_PATH, mode='val', train_ratio=0.8)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # 验证集 shuffle=True 是为了画图时能随机看到不同的样本，不影响验证指标
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. 初始化模型
    # model = PVGPT(d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dropout=0.3).to(device)
    # 模型已经过拟合，把模型的参数减少 尝试 d_model=64 或 128
    model = PVGPT(d_model=28, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dropout=0.3).to(device)  # 参数1

    # model = PVGPT(d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dropout=0.3).to(device)  # 参数2
    #
    # model = PVGPT(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dropout=0.3).to(device)  # 参数3
    #
    # model = PVGPT(d_model=256, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dropout=0.3).to(device)  # 参数4

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    # --- 新增：记录每个 epoch 的 loss ---
    train_loss_history = []
    val_loss_history = []

    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        for batch in loop:
            x_seq = batch['x_seq'].to(device)
            y_seq = batch['y_seq'].to(device)
            x_time = batch['x_time'].to(device)
            y_time = batch['y_time'].to(device)

            tgt_input = torch.zeros_like(y_seq).to(device)

            optimizer.zero_grad()
            output = model(x_seq, tgt_input, x_time, y_time)
            loss = criterion(output, y_seq)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # 4. 验证循环
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_seq = batch['x_seq'].to(device)
                y_seq = batch['y_seq'].to(device)
                x_time = batch['x_time'].to(device)
                y_time = batch['y_time'].to(device)

                tgt_input = torch.zeros_like(y_seq).to(device)

                output = model(x_seq, tgt_input, x_time, y_time)
                loss = criterion(output, y_seq)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # --- 记录 Loss ---
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 5. 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Model saved to {MODEL_SAVE_PATH}")

    # --- 训练结束后：画 Loss 曲线 ---
    plot_loss_curve(train_loss_history, val_loss_history)

    # --- 训练结束后：加载最佳模型并可视化预测效果 ---
    print("🔄 Loading best model for visualization...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    plot_validation_results(model, val_loader, device)


if __name__ == "__main__":
    train()
