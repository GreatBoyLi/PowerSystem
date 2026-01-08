import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os
from HSA.tool.tool import getreadfilepath

# 在代码开头设置，指定第3张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ==========================================
# 1. 配置与设备检测
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 300  # 回归任务通常需要更多 Epoch 来拟合细节
HIDDEN_DIM = 256  # 回归任务可能需要稍微宽一点的网络


# ==========================================
# 2. 数据准备 (关键步骤)
# ==========================================
class EMSRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 3. 定义回归模型 (Regression MLP)
# ==========================================
class CostPredictor(nn.Module):
    def __init__(self, input_dim):
        super(CostPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),  # 回归任务中间层依然用 ReLU
            nn.Dropout(0.1),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM // 2, 1)
            # !!! 注意: 最后一层没有任何激活函数 (Linear output) !!!
            # 这样才能输出任意范围的数值
        )

    def forward(self, x):
        return self.net(x)


if "__main__" == __name__:

    # 读取 CSV
    name = "../data/train/training_dataset_final.csv"
    path = getreadfilepath(__file__, name)
    df = pd.read_csv(path)

    # --- 关键: 筛选可行样本 ---
    print(f"原始样本数: {len(df)}")
    df_feasible = df[df['feasible'] == 0].copy()
    print(f"可行样本数 (用于训练预测器): {len(df_feasible)}")

    if len(df_feasible) < 100:
        print("警告: 可行样本太少，神经网络可能无法训练！建议检查 LHS 采样范围或 EMS 约束。")

    # 定义特征和目标
    feature_cols = [
        'A_in：1000平方', 'A_out：1000平方',
        'E_in：MWh', 'E_out：MWh',
        'P_Tr_max_in：MW', 'P_Tr_max_out：MW',
        'P_mut_max：MW'
    ]

    X = df_feasible[feature_cols].values
    y = df_feasible['f2_operation'].values

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 归一化 (Input Scaling) ---
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    # --- 标签归一化 (Target Scaling) ---
    # f2 的值可能在 1000~50000 之间，直接训练梯度不稳定
    # 我们把 y 也缩放到 0 附近
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # 保存 Scalers (预测时两个都需要！)
    joblib.dump(scaler_x, 'predictor_scaler_x.pkl')
    joblib.dump(scaler_y, 'predictor_scaler_y.pkl')
    print("Scalers saved.")

    # 创建 DataLoader
    train_dataset = EMSRegressionDataset(X_train_scaled, y_train_scaled)
    test_dataset = EMSRegressionDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CostPredictor(input_dim=7).to(device)

    # 损失函数: 均方误差 (MSE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 4. 训练循环
    # ==========================================
    print("Start Training Predictor...")
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)  # 计算 MSE

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], MSE Loss: {avg_loss:.6f}")

    # ==========================================
    # 5. 评估与可视化 (反归一化)
    # ==========================================
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    # 拼接结果
    preds_scaled = np.concatenate(preds_list)
    targets_scaled = np.concatenate(targets_list)

    # --- 关键: 反归一化 (Inverse Transform) ---
    # 将神经网络输出的 scaled 值还原回真实的 RMB 成本
    preds_real = scaler_y.inverse_transform(preds_scaled)
    targets_real = scaler_y.inverse_transform(targets_scaled.reshape(-1, 1))

    # 计算真实物理量纲下的指标
    r2 = r2_score(targets_real, preds_real)
    mae = mean_absolute_error(targets_real, preds_real)

    print("-" * 30)
    print(f"Test Set R2 Score: {r2:.4f} (越接近 1 越好)")
    print(f"Mean Absolute Error: {mae:.2f} RMB")
    print("-" * 30)

    # 保存模型
    torch.save(model.state_dict(), 'cost_predictor.pth')
    print("Model saved as cost_predictor.pth")

    # 绘图: 预测值 vs 真实值
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_real, preds_real, alpha=0.5, s=10)
    plt.plot([targets_real.min(), targets_real.max()], [targets_real.min(), targets_real.max()], 'r--', lw=2)
    plt.xlabel('True Cost (f2)')
    plt.ylabel('Predicted Cost (f2)')
    plt.title(f'Prediction Accuracy (R2={r2:.3f})')
    plt.grid(True)
    plt.savefig('predictor_accuracy.png')
    print("Accuracy plot saved.")

    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 8))
    plt.plot(loss_history)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('predictor_loss.png')
    print("Loss curve saved as predictor_loss.png")
