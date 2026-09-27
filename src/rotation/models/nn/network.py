"""行业打分网络：共享时间编码器（LSTM / GRU / Transformer）+ 行业嵌入 + 可选截面注意力。

可选 linear_skip：在输出上叠加一条"当日特征 → 线性打分"的残差通路（wide & deep 结构），
让网络同时保留线性截面信号，LSTM 只需学习线性部分之外的时序/非线性信息。

输入 x: [B, N, L, F]（B 个日期 × N 个行业 × L 天窗口 × F 个特征，已拼接市场状态特征）
输出   : [B, N]，每个行业一个预测值（标准化后的收益率量纲）
"""

from __future__ import annotations

import torch
from torch import nn


class IndustryScorer(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_industries: int,
        encoder: str = "lstm",
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        emb_dim: int = 8,
        cross_attention: bool = False,
        attn_heads: int = 4,
        tf_heads: int = 4,
        tf_ff_mult: int = 2,
        max_len: int = 256,
        linear_skip: bool = False,
    ):
        super().__init__()
        self.encoder_type = encoder
        self.input_proj = nn.Linear(n_inputs, hidden)
        if encoder in ("lstm", "gru"):
            rnn = nn.LSTM if encoder == "lstm" else nn.GRU
            self.encoder = rnn(
                hidden,
                hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
        elif encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=tf_heads,
                dim_feedforward=hidden * tf_ff_mult,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False
            )
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden))
            nn.init.normal_(self.pos_emb, std=0.02)
        else:
            raise ValueError(f"未知编码器：{encoder}")
        self.norm = nn.LayerNorm(hidden)
        self.embedding = nn.Embedding(n_industries, emb_dim) if emb_dim > 0 else None
        width = hidden + max(emb_dim, 0)
        self.cross = None
        if cross_attention:
            heads = attn_heads if width % attn_heads == 0 else 1
            self.cross = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.skip = nn.Linear(n_inputs, 1) if linear_skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, length, f = x.shape
        z = self.input_proj(x.reshape(b * n, length, f))
        if self.encoder_type == "transformer":
            z = z + self.pos_emb[:, -length:, :]
            h = self.encoder(z)[:, -1]
        else:
            out, _ = self.encoder(z)
            h = out[:, -1]
        h = self.norm(h).reshape(b, n, -1)
        if self.embedding is not None:
            ids = torch.arange(n, device=x.device)
            h = torch.cat([h, self.embedding(ids).unsqueeze(0).expand(b, n, -1)], dim=-1)
        if self.cross is not None:
            attended, _ = self.cross(h, h, h, need_weights=False)
            h = self.cross_norm(h + attended)
        out = self.head(h).squeeze(-1)
        if self.skip is not None:
            out = out + self.skip(x[:, :, -1, :]).squeeze(-1)
        return out
