import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualLSTMCell(nn.Module):
    """
    手写 LSTM 单元：不使用 nn.LSTM，只用 Linear 和数学公式
    """

    def __init__(self, input_size, hidden_size):
        super(ManualLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义权重参数
        # 我们可以把 4 个门 (Input, Forget, Cell, Output) 的权重合并写，提高效率
        # x -> gates 的变换矩阵 (W_xi, W_xf, W_xc, W_xo)
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True)

        # h -> gates 的变换矩阵 (W_hi, W_hf, W_hc, W_ho)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, state):
        # x: (Batch, Input_Size)
        # state: ((Batch, Hidden_Size), (Batch, Hidden_Size)) -> (h_prev, c_prev)
        h_prev, c_prev = state

        # 1. 执行线性变换 (对应公式中的 Wx*x + Wh*h + b)
        gates = self.x2h(x) + self.h2h(h_prev)

        # 2. 切分成 4 个部分
        # split 后每个形状: (Batch, Hidden_Size)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

        # 3. 激活函数 (手动实现门的开关)
        i = torch.sigmoid(i_gate)  # 输入门：决定写入多少新信息
        f = torch.sigmoid(f_gate)  # 遗忘门：决定保留多少旧记忆
        o = torch.sigmoid(o_gate)  # 输出门：决定输出多少状态
        g = torch.tanh(g_gate)  # 候选记忆：新生成的信息

        # 4. 状态更新公式
        c_next = f * c_prev + i * g  # 细胞状态 = 旧记忆保留 + 新记忆写入
        h_next = o * torch.tanh(c_next)  # 隐状态 = 输出门 * 激活后的细胞状态

        return h_next, c_next


class ManualLSTM(nn.Module):
    """
    将 Cell 封装成处理序列的层
    """

    def __init__(self, input_size, hidden_size):
        super(ManualLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Size)
        batch_size, seq_len, _ = x.size()

        # 初始化 h0, c0
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            # 取出当前时刻输入
            x_t = x[:, t, :]
            # 放入 Cell 计算
            h, c = self.cell(x_t, (h, c))
            outputs.append(h)

        # 堆叠输出: (Batch, Seq_Len, Hidden_Size)
        return torch.stack(outputs, dim=1), (h, c)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 16

    print("--- 1. 测试 Standard LSTM ---")
    # 假设输入是一个序列的向量 (Feature Vector)
    input_vector_size = 128
    hidden_vector_size = 64

    lstm = ManualLSTM(input_vector_size, hidden_vector_size)
    dummy_vec_input = torch.randn(batch_size, seq_len, input_vector_size)

    vec_out, _ = lstm(dummy_vec_input)
    print(f"输入形状: {dummy_vec_input.shape} (Batch, Seq, Vector_Dim)")
    print(f"输出形状: {vec_out.shape} (Batch, Seq, Hidden_Dim)")
    print("评价: 空间信息丢失，只处理一维数值。\n")
