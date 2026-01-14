import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (max_len, d_model) -> (1, max_len, d_model) 方便广播
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, D_Model)
        return x + self.pe[:, :x.size(1), :]


class PVGPT(nn.Module):
    def __init__(self,
                 d_model=512,  # 论文配置: 512
                 nhead=8,  # 论文配置: 8 heads
                 num_encoder_layers=3,  # 论文配置: 3 layers
                 num_decoder_layers=3,  # 论文配置: 3 layers
                 dim_feedforward=2048,  # 论文配置: 2048
                 dropout=0.1):
        super(PVGPT, self).__init__()

        # --- 1. Embedding Layer  ---
        # Value Embedding: 卷积层提取局部特征 (Input: 1维功率值 -> d_model)
        self.value_embedding = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)

        # Temporal Embedding: 线性层映射时间特征 (Input: 4维时间特征 -> d_model)
        self.temporal_embedding = nn.Linear(4, d_model)

        # Positional Encoding: 这里的 dropout 用于防过拟合
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

        # --- 2. Transformer Backbone ---
        # 使用 PyTorch 原生 Transformer 模块，效率最高
        # batch_first=True 让输入变成 (Batch, Seq, Feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # --- 3. Output Projection ---
        # 将 Transformer 输出映射回 1 维功率值
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, tgt, src_time, tgt_time):
        """
        src: Encoder 输入功率序列 (Batch, 112, 1)
        tgt: Decoder 输入序列 (Batch, 16, 1) -> 训练时通常是 Teacher Forcing 或全0初始化
        src_time: Encoder 时间特征 (Batch, 112, 4)
        tgt_time: Decoder 时间特征 (Batch, 16, 4)
        """

        # --- Step A: Embedding ---
        # 1. Value Embedding (需调整维度适配 Conv1d: B, C, L)
        # (Batch, 112, 1) -> (Batch, 1, 112) -> Conv -> (Batch, 512, 112) -> (Batch, 112, 512)
        src_val = self.value_embedding(src.permute(0, 2, 1)).permute(0, 2, 1)
        tgt_val = self.value_embedding(tgt.permute(0, 2, 1)).permute(0, 2, 1)

        # 2. Temporal Embedding
        src_tmp = self.temporal_embedding(src_time)  # (Batch, 112, 512)
        tgt_tmp = self.temporal_embedding(tgt_time)  # (Batch, 16, 512)

        # 3. Sum & Positional Encoding
        # X = val(Xp) + tem(Xt) + pos(Xp)
        src_emb = self.pos_encoder(src_val + src_tmp)
        tgt_emb = self.pos_encoder(tgt_val + tgt_tmp)

        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        # --- Step B: Transformer ---
        # 生成 Mask: Decoder 必须防止看到未来 (Causal Mask)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask
        )

        # --- Step C: Prediction ---
        prediction = self.output_layer(output)

        return prediction


# 测试代码
if __name__ == "__main__":
    model = PVGPT()
    # 模拟一个 Batch 数据
    x = torch.randn(32, 112, 1)
    y = torch.randn(32, 16, 1)  # Decoder Input (可以是全0)
    x_t = torch.randn(32, 112, 4)
    y_t = torch.randn(32, 16, 4)

    out = model(x, y, x_t, y_t)
    print(f"Model Output Shape: {out.shape}")  # 应该输出 (32, 16, 1)