import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    """
    辅助模块：将 CNN 应用于时间序列的每一帧
    输入: (Batch, Sequence, Channel, Height, Width)
    输出: (Batch, Sequence, Output_Dim)
    """

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # x.shape: (Batch, Seq, C, H, W)
        batch_size, seq_len, c, h, w = x.size()

        # 合并 Batch 和 Seq 维度 -> (Batch * Seq, C, H, W)
        # 这样 CNN 就会把每一帧当成独立的图片处理
        c_in = x.view(batch_size * seq_len, c, h, w)

        # 经过 CNN
        c_out = self.module(c_in)

        # 恢复维度 -> (Batch, Seq, Output_Dim)
        # 假设 CNN 输出是 (Batch*Seq, Feature_Dim)
        r_out = c_out.view(batch_size, seq_len, -1)
        return r_out


class SimpleCNN(nn.Module):
    """
    一个轻量级的 CNN 特征提取器
    输入图像大小: 96x96
    """

    def __init__(self, out_dim=64):
        super(SimpleCNN, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 96 -> 48

        # Conv Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48 -> 24

        # Conv Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 24 -> 12

        # Conv Block 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(4, 4)  # 12 -> 3

        # Flatten 后的大小: 64通道 * 3 * 3 = 576
        self.fc = nn.Linear(64 * 3 * 3, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MultiModalPVNet(nn.Module):
    def __init__(self,
                 img_feat_dim=64,  # CNN 输出特征维度
                 numeric_input_dim=3,  # [Power, GHI, Zenith]
                 hidden_dim=128,  # GRU 隐藏层维度
                 num_layers=2,  # GRU 层数
                 output_seq_len=4):  # 预测未来4步
        super(MultiModalPVNet, self).__init__()

        # 1. 视觉支路 (Visual Encoder)
        self.cnn = SimpleCNN(out_dim=img_feat_dim)
        self.visual_encoder = TimeDistributed(self.cnn)

        # 2. 数值支路 (Numeric Encoder) - 可选
        # 也可以直接把数值拼接到图像特征上

        # 3. 时序模型 (GRU)
        # GRU 的输入维度 = 图像特征维度 + 数值特征维度
        self.fusion_dim = img_feat_dim + numeric_input_dim
        self.gru = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # 4. 输出层 (Decoder)
        # 将 GRU 的最后状态映射到 4 个时间步的预测值
        self.fc_out = nn.Linear(hidden_dim, output_seq_len)

    def forward(self, x_images, x_numeric):
        """
        x_images: (Batch, Seq=16, C=1, H=96, W=96)
        x_numeric: (Batch, Seq=16, Feat=3)
        """

        # A. 处理图像
        # Output: (Batch, Seq, img_feat_dim)
        img_features = self.visual_encoder(x_images)

        # B. 特征融合
        # 在特征维度进行拼接 (Dim=2)
        # Output: (Batch, Seq, img_feat_dim + numeric_dim)
        combined_features = torch.cat([img_features, x_numeric], dim=2)

        # C. GRU 时序编码
        # gru_out: (Batch, Seq, hidden_dim) - 每个时间步的输出
        # h_n: (Num_Layers, Batch, hidden_dim) - 最后一个时间步的隐状态
        gru_out, h_n = self.gru(combined_features)

        # D. 预测
        # 我们取 GRU 序列的最后一个时间步的输出，来预测未来
        last_step_out = gru_out[:, -1, :]  # (Batch, hidden_dim)

        # 映射到未来 4 个值
        predictions = self.fc_out(last_step_out)  # (Batch, 4)

        return predictions


# ==========================================
# 简单的测试代码，确保形状是对的
# ==========================================
if __name__ == "__main__":
    # 模拟一个 Batch 的数据
    batch_size = 8
    seq_len = 16

    # 随机生成假输入
    dummy_img = torch.randn(batch_size, seq_len, 1, 96, 96)  # (8, 16, 1, 96, 96)
    dummy_num = torch.randn(batch_size, seq_len, 3)  # (8, 16, 3)

    # 实例化模型
    model = MultiModalPVNet()
    print("模型架构:\n", model)

    # 前向传播
    output = model(dummy_img, dummy_num)

    print("\n形状检查:")
    print(f"输入图像: {dummy_img.shape}")
    print(f"输入数值: {dummy_num.shape}")
    print(f"模型输出: {output.shape}")  # 预期: (8, 4)

    if output.shape == (batch_size, 4):
        print("✅ 模型测试通过！形状匹配完美。")
    else:
        print("❌ 形状不对，请检查代码。")