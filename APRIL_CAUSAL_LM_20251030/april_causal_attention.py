import torch
import torch.nn as nn


class AprilCausalAttention(nn.Module):
    """
    Tek başlıklı kausal self-attention.
    Girdi-çıktı: [B, T, E] -> [B, T, D_out]
    """
    def __init__(self, embedding_dim: int, output_dim: int, context_length: int,
                 dropout_rate: float = 0.0, device: str = "cpu"):
        super().__init__()
        self.q = nn.Linear(embedding_dim, output_dim, bias=False, device=device)
        self.k = nn.Linear(embedding_dim, output_dim, bias=False, device=device)
        self.v = nn.Linear(embedding_dim, output_dim, bias=False, device=device)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length, device=device), diagonal=1).bool())
        self.scale = (output_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]
        B, T, _ = x.shape
        q = self.q(x)  # [B, T, D]
        k = self.k(x)  # [B, T, D]
        v = self.v(x)  # [B, T, D]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, T, T]
        m = self.mask[:T, :T]                                       # [T, T]
        attn = attn.masked_fill(m, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, T, D]
        return out
