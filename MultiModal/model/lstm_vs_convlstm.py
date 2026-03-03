import torch
import torch.nn as nn


# ==========================================
# 1. 普通 LSTM (Standard LSTM)
# ==========================================
class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(StandardLSTM, self).__init__()
        # PyTorch 官方封装好的 LSTM
        # 输入必须是向量 (Vector)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Size)
        # 例如: (8, 16, 128) -> 意思是 batch=8, 时间长16, 每个时刻是一个长128的向量

        output, (hn, cn) = self.lstm(x)

        # output shape: (Batch, Seq_Len, Hidden_Size)
        # hn shape:     (Num_Layers, Batch, Hidden_Size)
        return output, hn


# ==========================================
# 2. ConvLSTM (Convolutional LSTM)
# ==========================================
class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 的核心单元 (类似于 nn.LSTMCell，但内部运算全是卷积)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # padding 保证输出图像大小不变 (H, W 保持不变)
        padding = kernel_size // 2

        # 核心变化点：使用 Conv2d 代替 Linear
        # 输入是 (当前时刻图像 x_t) + (上一时刻状态 h_t-1) 拼接
        # 输出是 4 个门的通道总和 (Input, Forget, Cell, Output) -> 4 * hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 1. 拼接输入和状态 (在通道维度 dim=1 拼接)
        # input_tensor: (Batch, Input_Dim, H, W)
        # h_cur:        (Batch, Hidden_Dim, H, W)
        # combined:     (Batch, Input+Hidden, H, W)
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # 2. 卷积运算 (代替全连接)
        combined_conv = self.conv(combined)

        # 3. 分割成 4 个部分 (cc_i, cc_f, cc_o, cc_g)
        # 每个部分的 shape: (Batch, Hidden_Dim, H, W)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 4. 经典的 LSTM 门控逻辑
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)  # 候选细胞状态

        # 5. 更新状态
        c_next = f * c_cur + i * g  # 新的 Cell State (逐元素相乘)
        h_next = o * torch.tanh(c_next)  # 新的 Hidden State

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    完整的 ConvLSTM 层，处理整个序列
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()

        # 可以是多层堆叠
        # 这里为了演示简单，只写一层
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Channels, Height, Width)
        # 典型的 5D 张量

        batch_size, seq_len, _, height, width = x.size()

        # 初始化状态 (h0, c0) 都是 0
        h, c = self.cell.init_hidden(batch_size, (height, width))

        outputs = []

        # 循环处理每一个时间步
        for t in range(seq_len):
            # 取出当前时刻的图像: (Batch, Channels, H, W)
            x_t = x[:, t, :, :, :]

            # 放入 Cell 运算
            h, c = self.cell(x_t, (h, c))

            # 记录输出
            outputs.append(h)

        # 堆叠输出 -> (Batch, Seq_Len, Hidden_Dim, H, W)
        outputs = torch.stack(outputs, dim=1)

        return outputs, (h, c)


# ==========================================
# 3. 运行测试与对比
# ==========================================
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10

    print("--- 1. 测试 Standard LSTM ---")
    # 假设输入是一个序列的向量 (Feature Vector)
    input_vector_size = 128
    hidden_vector_size = 64

    lstm = StandardLSTM(input_vector_size, hidden_vector_size)
    dummy_vec_input = torch.randn(batch_size, seq_len, input_vector_size)

    vec_out, _ = lstm(dummy_vec_input)
    print(f"输入形状: {dummy_vec_input.shape} (Batch, Seq, Vector_Dim)")
    print(f"输出形状: {vec_out.shape} (Batch, Seq, Hidden_Dim)")
    print("评价: 空间信息丢失，只处理一维数值。\n")

    print("--- 2. 测试 ConvLSTM ---")
    # 假设输入是一个序列的图像 (Image Frame)
    input_channels = 1  # 比如单通道卫星云图
    hidden_channels = 16  # 输出特征图的通道数
    height, width = 96, 96

    convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size=3, num_layers=1)
    dummy_img_input = torch.randn(batch_size, seq_len, input_channels, height, width)

    img_out, (h_n, c_n) = convlstm(dummy_img_input)
    print(f"输入形状: {dummy_img_input.shape} (Batch, Seq, Channel, H, W)")
    print(f"输出形状: {img_out.shape} (Batch, Seq, Hidden_Channel, H, W)")
    print("评价: 输出依然是图像(Feature Map)，完美保留了云层在 H,W 上的空间分布！")