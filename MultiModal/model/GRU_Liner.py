import torch
import torch.nn as nn


class TimeSeriesBranch(nn.Module):
    """
    GRU-Linear for Encoding Time Series
    负责处理历史数值数据 (如: Active Power, Clear-Sky GHI, Solar Zenith)
    """

    def __init__(self, input_dim=3, gru_hidden=64, num_layers=2, final_dim=64):
        """
        参数说明:
        input_dim: 输入特征的数量 (默认 3: 功率, 辐照度, 天顶角)
        gru_hidden: GRU 隐藏层的神经元数量
        num_layers: GRU 的堆叠层数 (通常 1 到 2 层就足够了)
        final_dim: 经过 Linear 层后的最终特征维度 (需要和图像特征拼接)
        """
        super(TimeSeriesBranch, self).__init__()

        # 1. GRU 模块 (捕捉时间序列的前后依赖关系)
        # batch_first=True 确保输入格式为 (Batch, Seq_Len, Feature_Dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0  # 多层时建议加点 dropout 防止过拟合
        )

        # 2. Linear 模块 (特征映射与浓缩)
        # 将 GRU 最后一个时间步的隐状态映射到目标维度
        self.linear = nn.Sequential(
            nn.Linear(gru_hidden, final_dim),
            # nn.ReLU()  # 添加非线性激活函数，增强表达能力
        )

    def forward(self, x):
        """
        x shape: (Batch, Seq_Len, Input_Dim)
        例如: (Batch, 16, 3)
        """
        # A. GRU 编码
        # gru_out 包含了所有时间步的隐状态输出: (Batch, Seq_Len, gru_hidden)
        # h_n 包含了最后时刻的隐状态: (num_layers, Batch, gru_hidden)
        gru_out, h_n = self.gru(x)

        # B. 提取序列的最终状态
        # 我们只关心历史序列走完之后的最终状态，即最后一个时间步的输出
        last_step_out = gru_out[:, -1, :]  # shape: (Batch, gru_hidden)

        # C. Linear 映射
        # 送入 Linear 层得到最终的数值特征向量
        ts_feature = self.linear(last_step_out)  # shape: (Batch, final_dim)

        return ts_feature


# ==========================================
# 测试代码 (模拟输入并验证形状)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始测试 TimeSeriesBranch (GRU-Linear)...")

    # 1. 设定模拟数据参数
    batch_size = 2  # 模拟 2 个样本
    seq_len = 16  # 过去 4 小时 (15min 间隔, 共 16 个点)
    input_dim = 3  # 输入特征: [Power_Norm, Clear_Sky_GHI, Solar_Zenith]
    final_dim = 64  # 期望输出的特征维度

    # 2. 实例化数值提取支路
    model = TimeSeriesBranch(
        input_dim=input_dim,
        gru_hidden=64,
        num_layers=2,
        final_dim=final_dim
    )

    # 3. 构造随机假数据 (Dummy Data)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"\n📥 输入序列形状 : {dummy_input.shape}")
    print(f"   说明: [Batch大小={batch_size}, 序列长度={seq_len}, 特征数量={input_dim}]")

    # 4. 前向传播测试
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # 5. 打印并检查输出
    print(f"\n📤 最终输出特征形状 : {output.shape}")
    print(f"   说明: [Batch大小={output.shape[0]}, 特征维度={output.shape[1]}]")

    if output.shape == (batch_size, final_dim):
        print("\n✅ 测试完美通过！")
        print("💡 模型已成功将 16 个时刻的数值变化规律 (GRU)，")
        print("   提取并压缩成了紧凑的数值特征向量 (Linear)。")
    else:
        print("\n❌ 测试失败，输出形状不符合预期。")