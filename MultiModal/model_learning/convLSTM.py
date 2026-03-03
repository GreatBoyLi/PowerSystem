import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualConvLSTMCell(nn.Module):
    """
    手写 ConvLSTM 单元：把全连接换成卷积，适配时空数据
    输入不再是一维向量，而是二维特征图（带通道）
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ManualConvLSTMCell, self).__init__()
        self.input_channels = input_channels  # 输入通道数（比如光伏场景：功率+辐照=2通道）
        self.hidden_channels = hidden_channels  # 隐藏层通道数
        self.kernel_size = kernel_size  # 卷积核大小（常用3x3）
        self.padding = padding  # 填充，保证卷积后特征图尺寸不变

        # 核心改造：把 Linear 换成 Conv2d，输出维度是 4*hidden_channels（对应4个门）
        # x -> gates 的卷积变换：输入通道=input_channels，输出通道=4*hidden_channels
        self.x2h = nn.Conv2d(
            in_channels=input_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        # h -> gates 的卷积变换：输入通道=hidden_channels，输出通道=4*hidden_channels
        self.h2h = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, state):
        # x: (Batch, Input_Channels, H, W) → 当前时刻的空间特征图
        # state: ((Batch, Hidden_Channels, H, W), (Batch, Hidden_Channels, H, W)) → (h_prev, c_prev)
        h_prev, c_prev = state

        # 1. 卷积变换（替代原有的线性变换）：融合当前输入和历史隐状态
        # 输出形状：(Batch, 4*Hidden_Channels, H, W)
        gates = self.x2h(x) + self.h2h(h_prev)

        # 2. 切分成4个门：每个门形状 (Batch, Hidden_Channels, H, W)
        # 注意：卷积输出是 (Batch, C, H, W)，所以沿通道维度（dim=1）拆分
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

        # 3. 激活函数（和普通LSTM完全一致，逐元素激活）
        i = torch.sigmoid(i_gate)  # 输入门：空间维度上逐像素决定写入多少新信息
        f = torch.sigmoid(f_gate)  # 遗忘门：空间维度上逐像素决定保留多少旧记忆
        o = torch.sigmoid(o_gate)  # 输出门：空间维度上逐像素决定输出多少状态
        g = torch.tanh(g_gate)  # 候选记忆：空间维度上生成新的特征内容

        # 4. 状态更新（和普通LSTM一致，逐像素运算）
        c_next = f * c_prev + i * g  # 细胞状态：空间特征的长期记忆
        h_next = o * torch.tanh(c_next)  # 隐状态：空间特征的即时输出

        return h_next, c_next


class ManualConvLSTM(nn.Module):
    """
    将 ConvLSTM Cell 封装成处理时空序列的层
    输入形状：(Batch, Seq_Len, Input_Channels, H, W)
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ManualConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.cell = ManualConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Channels, H, W) → 时空序列输入
        batch_size, seq_len, input_channels, H, W = x.size()

        # 初始化 h0, c0：空间维度和输入一致，通道数=hidden_channels
        h = torch.zeros(batch_size, self.hidden_channels, H, W).to(x.device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W).to(x.device)

        outputs = []
        for t in range(seq_len):
            # 取出当前时刻的空间输入：(Batch, Input_Channels, H, W)
            x_t = x[:, t, :, :, :]
            # 放入 ConvLSTM Cell 计算
            h, c = self.cell(x_t, (h, c))
            outputs.append(h)

        # 堆叠输出：(Batch, Seq_Len, Hidden_Channels, H, W)
        return torch.stack(outputs, dim=1), (h, c)


if __name__ == "__main__":
    # 测试 ConvLSTM（贴合光伏场景：多电站空间分布+时序）
    batch_size = 4  # 批次大小
    seq_len = 16  # 时间步长（16个15分钟）
    input_channels = 2  # 输入通道：比如「光伏功率」+「GHI辐照」
    hidden_channels = 8  # 隐藏层通道数
    H, W = 10, 10  # 空间维度：10x10的光伏板阵列（模拟100个光伏板的空间分布）

    print("--- 测试 ConvLSTM ---")
    conv_lstm = ManualConvLSTM(input_channels=input_channels, hidden_channels=hidden_channels)

    # 构造模拟输入：(Batch, Seq_Len, Input_Channels, H, W)
    dummy_spatio_temporal_input = torch.randn(batch_size, seq_len, input_channels, H, W)

    # 前向传播
    conv_out, _ = conv_lstm(dummy_spatio_temporal_input)

    # 打印形状
    print(f"输入形状: {dummy_spatio_temporal_input.shape} (Batch, Seq, Channel, H, W)")
    print(f"输出形状: {conv_out.shape} (Batch, Seq, Hidden_Channel, H, W)")
    print("评价: 保留空间信息，同时处理时序依赖（时空双维度）。")