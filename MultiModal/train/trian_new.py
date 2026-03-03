import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # 【新增 1】导入绘图库
import torch.nn.functional as F

# 导入我们自己写的模块
from MultiModal.model.dataset import SatellitePVDataset
from MultiModal.model.MultiModalPVNet import MultiModalPVNet
from utils.config import load_config
from utils.merics import evaluate_metrics

# ================= 配置区域 (Hyperparameters) =================
# 加载配置
config = load_config("../config/config.yaml")

# 路径设置
CSV_PATH = config["file_paths"]["series_file"]
SAT_DIR = config["file_paths"]["aligned_satellite_path"]
SAVE_DIR = "../checkpoints/"

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10

# 硬件设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {DEVICE}")


# ============================================================

class DCCALoss(nn.Module):
    """
    终极稳定版特征对齐损失 (基于跨模态互相关矩阵)
    平替传统 DCCA，彻底根除 linalg.eigh 导致的 NaN 梯度爆炸问题。
    """

    def __init__(self, lambd=5e-3):
        super(DCCALoss, self).__init__()
        # 用于平衡对角线和非对角线惩罚的系数
        self.lambd = lambd

    def forward(self, H1, H2):
        batch_size = H1.size(0)

        # 🛡️ 终极防御：如果由于某种原因只传进来 1 个样本，直接返回 0 的 Loss，不参与对齐计算
        if batch_size <= 1:
            return torch.tensor(0.0, device=H1.device, requires_grad=True)

        # 加入 unbiased=False，防止除以 N-1 造成的各种隐患
        H1_norm = (H1 - H1.mean(dim=0)) / (H1.std(dim=0, unbiased=False) + 1e-8)
        H2_norm = (H2 - H2.mean(dim=0)) / (H2.std(dim=0, unbiased=False) + 1e-8)

        # 2. 计算跨模态互相关矩阵 (Cross-Correlation Matrix)
        # C 的大小为 (Feature_Dim, Feature_Dim)
        C = torch.matmul(H1_norm.t(), H2_norm) / batch_size

        # 3. 构建目标：让互相关矩阵 C 尽量接近单位矩阵 (Identity Matrix)
        # 这意味着：
        # - 第 i 个视觉特征和第 i 个数值特征高度相关 (对角线为 1)
        # - 第 i 个视觉特征和第 j 个数值特征互不干扰 (非对角线为 0)
        c_diff = (C - torch.eye(C.size(0), device=C.device)).pow(2)

        # 提取对角线部分的误差 (迫使模态对齐)
        on_diag_loss = torch.diag(c_diff).sum()

        # 提取非对角线部分的误差 (去除特征内部冗余)
        off_diag_loss = c_diff.sum() - on_diag_loss

        # 总损失
        loss = on_diag_loss + self.lambd * off_diag_loss
        return loss


class PaperDCCALoss(nn.Module):
    """
    完全按照论文 公式 (21) - (26) 实现的 DCCA 损失函数
    """

    def __init__(self, r1=1e-3, r2=1e-3, eps=1e-8):
        super(PaperDCCALoss, self).__init__()
        # 论文明确指出的正则化常数 r1 和 r2 均设为 10^-3
        self.r1 = r1
        self.r2 = r2
        self.eps = eps  # 用于防止底层特征分解时出现数学崩溃的极小值保护

    def forward(self, H1, H2):
        # n 即为公式里的样本大小 (Batch Size) [cite: 251]
        n = H1.size(0)

        # 为了防止只有 1 个样本时下面除以 (n-1) 报错
        if n <= 1:
            return torch.tensor(0.0, device=H1.device, requires_grad=True)

        # ==========================================
        # 公式 (21)：去均值中心化 [cite: 251]
        # ==========================================
        # E_bar = E - 1/n * E * I
        H1_bar = H1 - H1.mean(dim=0, keepdim=True)
        H2_bar = H2 - H2.mean(dim=0, keepdim=True)

        # ==========================================
        # 公式 (24) - (26)：计算协方差矩阵
        # ==========================================
        # 注意：在 PyTorch 中，(Batch, Dim) 的矩阵乘法转置方式与论文表达略有不同，
        # H1_bar.t() @ H1_bar 等价于论文中特征维度展开的 E_bar * E_bar'

        # 公式 (25): Σ_11 = 1/(n-1) * E1_bar * E1_bar' + r1 * I [cite: 258]
        Sigma11 = torch.matmul(H1_bar.t(), H1_bar) / (n - 1)
        Sigma11 = Sigma11 + self.r1 * torch.eye(Sigma11.size(0), device=H1.device)

        # 公式 (26): Σ_22 = 1/(n-1) * E2_bar * E2_bar' + r2 * I
        # 备注：原论文写的是 1/(n-2) [cite: 260]，这是统计学无偏估计的笔误，工程上均统一为 n-1
        Sigma22 = torch.matmul(H2_bar.t(), H2_bar) / (n - 1)
        Sigma22 = Sigma22 + self.r2 * torch.eye(Sigma22.size(0), device=H2.device)

        # 公式 (24): Σ_12 = 1/(n-1) * E1_bar * E2_bar' [cite: 256]
        Sigma12 = torch.matmul(H1_bar.t(), H2_bar) / (n - 1)

        # ==========================================
        # 核心难点：计算 Σ_11^(-1/2) 和 Σ_22^(-1/2)
        # ==========================================
        # 通过特征值分解：Σ = V * diag(L) * V^T  ==>  Σ^(-1/2) = V * diag(L^(-1/2)) * V^T
        L1, V1 = torch.linalg.eigh(Sigma11)
        L2, V2 = torch.linalg.eigh(Sigma22)

        # 钳制特征值，防止出现负数或 0 导致后续开根号变成 NaN
        L1 = torch.clamp(L1, min=self.eps)
        L2 = torch.clamp(L2, min=self.eps)

        # 计算负二分之一次方矩阵
        Sigma11_inv_sqrt = torch.matmul(V1, torch.matmul(torch.diag(L1 ** -0.5), V1.t()))
        Sigma22_inv_sqrt = torch.matmul(V2, torch.matmul(torch.diag(L2 ** -0.5), V2.t()))

        # ==========================================
        # 公式 (23)：构建典型相关矩阵 T_t [cite: 253]
        # ==========================================
        # T_t = Σ_11^(-1/2) * Σ_12 * Σ_22^(-1/2)
        T_t = torch.matmul(Sigma11_inv_sqrt, torch.matmul(Sigma12, Sigma22_inv_sqrt))

        # ==========================================
        # 公式 (22)：计算相关性 (迹范数 / 核范数) [cite: 250]
        # ==========================================
        # 迹范数 ||T_t||_tr 等于矩阵 T_t 的所有奇异值之和
        # 使用 SVD (奇异值分解) 求出奇异值 S
        U, S, V = torch.linalg.svd(T_t)

        # 相关性 = 奇异值之和
        corr = torch.sum(S)

        # ==========================================
        # DCCA Loss 输出
        # ==========================================
        # 因为优化器是求最小值，我们要最大化相关性，所以取负数 (公式 19 的逻辑)
        return -corr


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        imgs = batch['x_images'].to(device)
        nums = batch['x_numeric'].to(device)
        targets = batch['y'].to(device)

        optimizer.zero_grad()
        preds, v_feat, t_feat = model(imgs, nums)
        # 计算多目标 Loss
        loss_mse = criterion_mse(preds, targets)  # 预测准不准
        loss_dcca = criterion_dcca(v_feat, t_feat)  # 模态对不对齐
        # loss = criterion(preds, targets)
        # 论文总损失公式
        loss = loss_mse + lambda_c * loss_dcca
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item(), loss_mse=loss_mse.item(), loss_dcca=loss_dcca.item())

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    # 🌟 新增：用于收集整个验证集的预测值和真实值
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['x_images'].to(device)
            nums = batch['x_numeric'].to(device)
            targets = batch['y'].to(device)

            preds, v_feat, t_feat = model(imgs, nums)

            # 计算 Loss
            loss_mse = criterion_mse(preds, targets)
            loss_dcca = criterion_dcca(v_feat, t_feat)
            loss = loss_mse + lambda_c * loss_dcca
            running_loss += loss.item()

            # 🌟 新增：把数据转移到 CPU 并存入列表，防止撑爆显存
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # 🌟 新增：将列表中的 Tensor 在第 0 维度（Batch 维度）拼接起来
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 🌟 新增：调用评估函数，计算四个指标
    metrics = evaluate_metrics(all_preds, all_targets)

    # 修改返回值，现在把 loss 和 指标字典 一起返回
    return running_loss / len(loader), metrics


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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ 数据集加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    model = MultiModalPVNet(
        visual_dim=64,  # 视觉特征维度
        ts_dim=64,  # 数值特征维度
        output_seq_len=4  # 预测未来4个时间步
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 分别初始化四个指标的历史最佳记录
    # RMSE, MAE, MAPE 是越小越好，所以初始值设为正无穷大
    best_rmse = float('inf')
    best_mae = float('inf')
    best_mape = float('inf')
    # R (相关系数) 是越大越好，所以初始值设为负无穷大
    best_r = -float('inf')

    patience_counter = 0

    # 【新增 3】初始化列表用于存储 Loss
    train_loss_history = []
    val_loss_history = []

    print(f"🔥 开始训练 (Epochs: {NUM_EPOCHS})")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        print(val_metrics)

        # 【新增 4】记录每一轮的 Loss
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 获取当前 Epoch 的各项指标
        current_rmse = val_metrics['RMSE']
        current_mae = val_metrics['MAE']
        current_mape = val_metrics['MAPE(%)']
        current_r = val_metrics['R(%)']

        # 设置一个标志位，只要有任何一个指标破纪录了，就重置早停计数器
        any_improvement = False

        # 🏆 1. 评判 RMSE (越小越好)
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_rmse_model.pth"))
            print(f"   ⭐ [RMSE 冠军] 创新低: {best_rmse:.4f}，模型已保存！")

        # 🏆 2. 评判 MAE (越小越好)
        if current_mae < best_mae:
            best_mae = current_mae
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_mae_model.pth"))
            print(f"   ⭐ [MAE  冠军] 创新低: {best_mae:.4f}，模型已保存！")

        # 🏆 3. 评判 MAPE (越小越好)
        if current_mape < best_mape:
            best_mape = current_mape
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_mape_model.pth"))
            print(f"   ⭐ [MAPE 冠军] 创新低: {best_mape:.2f}%，模型已保存！")

        # 🏆 4. 评判 R (越大越好)
        if current_r > best_r:
            best_r = current_r
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_r_model.pth"))
            print(f"   🚀 [R 相关性冠军] 创新高: {best_r:.2f}%，模型已保存！")

        # 早停机制 (Early Stopping) 逻辑更新：
        # 只要这四个指标中有一个还在变好，我们就继续给模型机会
        if any_improvement:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⏳ 所有四项指标均未提升 ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            break

    print("-" * 60)
    print("🎉 训练结束！最佳模型已保存在:", os.path.join(SAVE_DIR, "best_model.pth"))

    # 【新增 5】调用绘图函数
    plot_save_path = os.path.join(SAVE_DIR, "loss_curve.png")
    plot_loss_curve(train_loss_history, val_loss_history, plot_save_path)


if __name__ == "__main__":
    criterion_mse = nn.MSELoss()
    criterion_dcca = DCCALoss()
    lambda_c = 0.01  # 论文中的平衡权重 \lambda_C，通常取 0.01 到 0.1
    main()
