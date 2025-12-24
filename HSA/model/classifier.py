import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # 用于保存归一化参数
import matplotlib.pyplot as plt
import os
from HSA.tool.tool import getreadfilepath

# 在代码开头设置，指定第3张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ==========================================
# 1. 配置与设备检测
# ==========================================
# 检测是否有 NVIDIA GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 超参数设置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 200
HIDDEN_DIM = 128  # 隐藏层神经元数量


# ==========================================
# 2. 数据准备
# ==========================================
class EMSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 变成 (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 读取之前生成的 CSV
name = "../data/train/training_dataset_final.csv"
path = getreadfilepath(__file__, name)
df = pd.read_csv(path)

# 定义输入特征 (7维 X_d)
feature_cols = [
    'A_in：1000平方', 'A_out：1000平方',
    'E_in：MWh', 'E_out：MWh',
    'P_Tr_max_in：MW', 'P_Tr_max_out：MW',
    'P_mut_max：MW'
]

X = df[feature_cols].values
y = df['feasible'].values  # 0: 可靠, 1: 不可行

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# !!! 关键步骤: 数据标准化 (Standardization) !!!
# 神经网络对输入数据的尺度非常敏感，必须进行归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存 scaler 供后续预测时使用 (这步很重要！)
joblib.dump(scaler, 'classifier_scaler.pkl')
print("Scaler saved as classifier_scaler.pkl")

# 创建 DataLoader
train_dataset = EMSDataset(X_train_scaled, y_train)
test_dataset = EMSDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 3. 定义神经网络模型 (DNN Classifier)
# ==========================================
class FeasibilityClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FeasibilityClassifier, self).__init__()
        # 论文中通常使用 3-4 层全连接层
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),  # BN层加速收敛
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM // 2, 1)  # 输出层 (输出 Logits)
            # 注意：这里不加 Sigmoid，因为我们将使用 BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)


# 初始化模型并移动到 GPU
model = FeasibilityClassifier(input_dim=7).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 结合了 Sigmoid + BCELoss，数值更稳定
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. 训练循环
# ==========================================
print("Start Training...")
loss_history = []

for epoch in range(EPOCHS):
    model.train()  # 切换到训练模式
    running_loss = 0.0

    for inputs, labels in train_loader:
        # 将数据移动到 GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# ==========================================
# 5. 模型评估与保存
# ==========================================
model.eval()  # 切换到评估模式
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # 将 Logits 转换为概率，再转换为 0/1 预测
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 保存模型权重
torch.save(model.state_dict(), 'feasibility_classifier.pth')
print("Model saved as feasibility_classifier.pth")

# 绘制 Loss 曲线
plt.plot(loss_history)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.savefig('classifier_loss.png')
print("Loss curve saved as classifier_loss.png")
