import torch
import torch.nn as nn
import torch.nn.functional as F


class SatelliteEncoder(nn.Module):
    """
    Section II.B: CNN for Encoding Satellite Images
    采用 'Channel Stacking' 方式：将时间步(T)视为图像的通道(C)。
    输入: (Batch, Seq_Len, 1, H, W) -> view -> (Batch, Seq_Len, H, W)
    """

    def __init__(self, input_seq_len=16, out_dim=128):
        super(SatelliteEncoder, self).__init__()

        # 假设输入是 16帧，每帧1个通道(Band13) -> 总共 16个通道
        # 这样 CNN 可以同时捕获空间和短时序特征
        self.conv1 = nn.Conv2d(input_seq_len, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 96 -> 48

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48 -> 24

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 24 -> 12

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling -> 1x1

        # 最终映射到特征向量
        self.fc = nn.Linear(256, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (Batch, Seq, 1, H, W)
        # 1. 移除多余的 channel 维度 1，把 Seq 维度当作 Channel
        b, s, c, h, w = x.size()
        x = x.view(b, s * c, h, w)  # -> (Batch, 16, 96, 96)

        # 2. CNN 特征提取
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # 3. Flatten & Linear
        x = x.view(x.size(0), -1)  # (Batch, 256)
        x = self.dropout(x)
        x = self.fc(x)  # (Batch, out_dim)
        return x


class TimeSeriesEncoder(nn.Module):
    """
    Section II.C: GRU-Linear for Encoding Time Series
    专门处理数值序列的分支
    """

    def __init__(self, input_dim=3, hidden_dim=64, out_dim=64, num_layers=2):
        super(TimeSeriesEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        # "Linear" part of "GRU-Linear"
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Seq, 3)

        # GRU 输出: (Batch, Seq, Hidden)
        # h_n: (Num_Layers, Batch, Hidden)
        out, h_n = self.gru(x)

        # 取最后一个时间步的输出作为序列特征
        last_hidden = out[:, -1, :]  # (Batch, Hidden)

        # 通过 Linear 层
        features = self.act(self.linear(last_hidden))
        return features


class MultiModalPVNet(nn.Module):
    """
    论文对应的完整多模态模型 (Dual-Stream Architecture)
    """

    def __init__(self,
                 img_feat_dim=128,
                 ts_feat_dim=64,
                 output_seq_len=4):
        super(MultiModalPVNet, self).__init__()

        # 1. 两个独立分支
        self.visual_net = SatelliteEncoder(input_seq_len=16, out_dim=img_feat_dim)
        self.ts_net = TimeSeriesEncoder(input_dim=3, out_dim=ts_feat_dim)

        # 2. 融合层 (Fusion)
        # 将两个特征向量拼接
        fusion_dim = img_feat_dim + ts_feat_dim

        # 3. 预测层 (Decoder/Predictor)
        # 论文可能是直接 MLP 输出，也可能是另一个 GRU 解码
        # 这里使用 MLP 回归预测未来 4 个点
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_seq_len)
        )

    def forward(self, x_images, x_numeric):
        # A. 并行提取特征
        img_vec = self.visual_net(x_images)  # (Batch, 128)
        ts_vec = self.ts_net(x_numeric)  # (Batch, 64)

        # B. 融合 (Concatenate)
        # Section II.D: Feature Fusion
        combined = torch.cat([img_vec, ts_vec], dim=1)  # (Batch, 192)

        # C. 预测
        output = self.predictor(combined)  # (Batch, 4)

        return output


if __name__ == "__main__":
    # 测试形状
    model = MultiModalPVNet()
    dummy_img = torch.randn(8, 16, 1, 96, 96)
    dummy_ts = torch.randn(8, 16, 3)

    out = model(dummy_img, dummy_ts)
    print("模型输出形状:", out.shape)  # 应为 (8, 4)
    print("✅ 双流模型测试通过")