import torch
import torch.nn as nn


# ==========================================
# 1. 之前写好的 ConvLSTM 单元 (处理 96x96 的大视野)
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)

        i_conv, f_conv, o_conv, g_conv = conv_output.chunk(4, dim=1)
        i = torch.sigmoid(i_conv)
        f = torch.sigmoid(f_conv)
        o = torch.sigmoid(o_conv)
        g = torch.tanh(g_conv)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ==========================================
# 2. RICNN：感兴趣区域卷积网络 (聚焦中心电站)
# ==========================================
class RICNN(nn.Module):
    """
    Region of Interest CNN
    输入: ConvLSTM 最后一帧的特征图 (Batch, Channels, 96, 96)
    输出: 降维后的视觉特征向量 (Batch, Feature_Dim)
    """

    def __init__(self, in_channels, roi_size=16, out_dim=128):
        super(RICNN, self).__init__()
        self.roi_size = roi_size

        # 专门针对 RoI 区域的卷积层
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # roi_size -> roi_size // 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 汇聚成 1x1 像素
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, feature_map):
        # feature_map: (Batch, C, H=96, W=96)

        # 1. 微观裁剪 (RoI Cropping)
        # 找到图像中心 (光伏电站的确切位置)
        b, c, h, w = feature_map.size()
        center_y, center_x = h // 2, w // 2

        half_roi = self.roi_size // 2

        # 截取中心区域 (例如 16x16)
        # 注意边界检查，确保不会越界
        roi_feature = feature_map[:, :,
        center_y - half_roi: center_y + half_roi,
        center_x - half_roi: center_x + half_roi]

        # 2. 对中心区域进行专门的特征提取 (RICNN)
        x = self.roi_conv(roi_feature)  # (Batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten

        # 3. 输出视觉特征
        out = self.fc(x)
        return out


# ==========================================
# 3. 完整的视觉支路：ConvLSTM-RICNN
# ==========================================
class VisualBranch(nn.Module):
    def __init__(self, input_channels=1, lstm_hidden=16, roi_size=16, final_dim=128):
        super(VisualBranch, self).__init__()

        # 1. 负责捕捉云层运动的 ConvLSTM
        self.convlstm_cell = ConvLSTMCell(input_channels, lstm_hidden)
        self.lstm_hidden = lstm_hidden

        # 2. 负责聚焦电站的 RICNN
        self.ricnn = RICNN(in_channels=lstm_hidden, roi_size=roi_size, out_dim=final_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len=16, Channels=1, H=96, W=96)
        batch_size, seq_len, _, h, w = x.size()

        h_state = torch.zeros(batch_size, self.lstm_hidden, h, w).to(x.device)
        c_state = torch.zeros(batch_size, self.lstm_hidden, h, w).to(x.device)

        # 序列处理：ConvLSTM 看云移动
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            h_state, c_state = self.convlstm_cell(x_t, (h_state, c_state))

        # 提取最后一帧的特征图 (包含了前面所有的运动记忆)
        # h_state shape: (Batch, lstm_hidden, 96, 96)

        # 送入 RICNN 聚焦中心
        final_visual_feature = self.ricnn(h_state)  # (Batch, final_dim)

        return final_visual_feature


if __name__ == "__main__":
    print("🚀 开始测试 VisualBranch (ConvLSTM + RICNN)...")

    # 1. 设定模拟数据的参数
    batch_size = 2  # 模拟有 2 个样本
    seq_len = 16  # 过去 4 小时 (15min 分辨率就是 16 帧)
    channels = 1  # 卫星云图通常用单通道 (比如 Himawari-8 的 Band 13)
    height, width = 96, 96  # 裁剪后的大视野尺寸
    final_dim = 128  # 期望输出的特征向量维度

    # 2. 实例化视觉支路模型
    model = VisualBranch(
        input_channels=channels,
        lstm_hidden=16,  # ConvLSTM 中间隐层的通道数
        roi_size=16,  # RICNN 聚焦中心 16x16 的区域
        final_dim=final_dim
    )

    # 将模型调整为评估模式 (避免 BatchNorm/Dropout 在测试时带来的影响)
    model.eval()

    # 3. 生成假数据 (Dummy Data) 模拟实际输入
    # 维度: (Batch, Seq_Len, Channels, H, W)
    dummy_input = torch.randn(batch_size, seq_len, channels, height, width)
    print(f"\n📥 输入图像序列形状 : {dummy_input.shape}")
    print(f"   说明: [Batch大小={batch_size}, 序列长度={seq_len}, 通道数={channels}, 高度={height}, 宽度={width}]")

    # 4. 执行前向传播
    with torch.no_grad():  # 测试时不需要计算梯度
        output = model(dummy_input)

    # 5. 验证输出并打印结果
    print(f"\n📤 最终输出特征形状 : {output.shape}")
    print(f"   说明: [Batch大小={output.shape[0]}, 特征维度={output.shape[1]}]")

    if output.shape == (batch_size, final_dim):
        print("\n✅ 测试完美通过！")
        print("💡 模型已成功将庞大的时空视频流 (ConvLSTM)，")
        print("   精准聚焦并压缩成了中心电站的紧凑特征向量 (RICNN)。")
    else:
        print("\n❌ 测试失败，输出形状不符合预期。")
