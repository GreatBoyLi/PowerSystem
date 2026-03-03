import torch.nn as nn
import torch

from MultiModal.model.convLSTM_RICNN import VisualBranch
from MultiModal.model.GRU_Liner import TimeSeriesBranch


class MultiModalPVNet(nn.Module):
    def __init__(self,
                 visual_dim=128,  # 视觉特征维度
                 ts_dim=64,  # 数值特征维度
                 output_seq_len=4  # 预测未来4个时间步
                 ):
        super(MultiModalPVNet, self).__init__()

        # 实例化两大支路
        self.visual_branch = VisualBranch(final_dim=visual_dim)
        self.ts_branch = TimeSeriesBranch(final_dim=ts_dim)

        # 融合后的总特征维度
        fusion_dim = visual_dim + ts_dim

        # 预测器 (Predictor)
        # 将融合后的特征映射到最终的预测结果
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_seq_len)  # 输出 (Batch, 4)
        )

    def forward(self, x_images, x_numeric):
        """
        x_images:  (Batch, 16, 1, 96, 96) 卫星云图序列
        x_numeric: (Batch, 16, 3)         历史功率等数值序列
        """
        # 1. 视觉提取
        v_feat = self.visual_branch(x_images)  # (Batch, 128)

        # 2. 数值提取
        t_feat = self.ts_branch(x_numeric)  # (Batch, 64)

        # 3. 特征融合 (沿着特征维度拼接)
        # 此时包含云层信息和电站状态的完整特征
        combined_feat = torch.cat([v_feat, t_feat], dim=1)  # (Batch, 192)

        # 4. 预测输出
        predictions = self.predictor(combined_feat)  # (Batch, 4)

        return predictions, v_feat, t_feat


# ==========================================
# 测试整个网络
# ==========================================
if __name__ == "__main__":
    print("🚀 开始测试完整多模态网络...")

    batch_size = 4
    seq_len = 16

    # 模拟 Dataset 传过来的数据
    dummy_imgs = torch.randn(batch_size, seq_len, 1, 96, 96)
    dummy_nums = torch.randn(batch_size, seq_len, 3)

    # 初始化主网络
    model = MultiModalPVNet(output_seq_len=4)

    # 前向传播
    output = model(dummy_imgs, dummy_nums)

    print(f"\n📥 输入图像 : {dummy_imgs.shape}")
    print(f"📥 输入数值 : {dummy_nums.shape}")
    print(f"📤 最终预测 : {output.shape} (预期为 Batch=4, 预测步数=4)")

    if output.shape == (batch_size, 4):
        print("\n✅ 完美！端到端 (End-to-End) 网络构建成功。")
    else:
        print("\n❌ 输出形状错误。")
