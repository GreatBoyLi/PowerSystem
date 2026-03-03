import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # 【新增 1】导入绘图库

# 导入我们自己写的模块
from MultiModal.model.dataset import SatellitePVDataset
from MultiModal.model.model import MultiModalPVNet
# from MultiModal.model.model_new import MultiModalPVNet
from MultiModal.utils.config import load_config

# ================= 配置区域 (Hyperparameters) =================
# 加载配置
config = load_config("../config/config.yaml")

# 路径设置
CSV_PATH = config["file_paths"]["series_file"]
SAT_DIR = config["file_paths"]["aligned_satellite_path"]
SAVE_DIR = "../checkpoints/"

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10

# 硬件设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {DEVICE}")


# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        imgs = batch['x_images'].to(device)
        nums = batch['x_numeric'].to(device)
        targets = batch['y'].to(device)

        optimizer.zero_grad()
        preds = model(imgs, nums)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['x_images'].to(device)
            nums = batch['x_numeric'].to(device)
            targets = batch['y'].to(device)

            preds = model(imgs, nums)
            loss = criterion(preds, targets)
            running_loss += loss.item()

    return running_loss / len(loader)


# 【新增 2】绘图函数
def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 保存图片
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭画布释放内存
    print(f"📈 损失曲线图已保存至: {save_path}")


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("📂 正在加载数据集...")
    if not os.path.exists(CSV_PATH) or not os.path.exists(SAT_DIR):
        print(f"❌ 错误: 找不到数据文件。请检查路径:\n CSV: {CSV_PATH}\n SAT: {SAT_DIR}")
        return

    train_dataset = SatellitePVDataset(CSV_PATH, SAT_DIR, mode='train')
    val_dataset = SatellitePVDataset(CSV_PATH, SAT_DIR, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ 数据集加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    model = MultiModalPVNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience_counter = 0

    # 【新增 3】初始化列表用于存储 Loss
    train_loss_history = []
    val_loss_history = []

    print(f"🔥 开始训练 (Epochs: {NUM_EPOCHS})")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        # 【新增 4】记录每一轮的 Loss
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   🏆 New Best Model Saved! (Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            break

    print("-" * 60)
    print("🎉 训练结束！最佳模型已保存在:", os.path.join(SAVE_DIR, "best_model.pth"))

    # 【新增 5】调用绘图函数
    plot_save_path = os.path.join(SAVE_DIR, "loss_curve.png")
    plot_loss_curve(train_loss_history, val_loss_history, plot_save_path)


if __name__ == "__main__":
    main()