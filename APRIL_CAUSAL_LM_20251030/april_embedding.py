import torch
import torch.nn as nn


def get_rotary_position_encoding(x: torch.Tensor, base: float = 10000.0, device: str = "cpu"):
    """
    x: [B, T, E]  (E çift sayı olmalı)
    RoPE uygulaması. Broadcast ile [T, E/2] sin/cos üretir.
    """
    B, T, E = x.shape
    assert E % 2 == 0, "Rotary için embedding boyutu çift olmalı."
    half = E // 2

    idx = torch.arange(0, half, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (idx / half))  # standardize edilmiş kullanım
    pos = torch.arange(0, T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
    angles = pos * freqs[None, :]  # [T, half]

    sin = torch.sin(angles)  # [T, half]
    cos = torch.cos(angles)  # [T, half]

    x_even = x[:, :, :half]     # [B, T, half]
    x_odd  = x[:, :, half:]     # [B, T, half]

    # broadcast: [B,T,half] * [T,half] -> [B,T,half]
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot  = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[:, :, :half] = x_even_rot
    out[:, :, half:] = x_odd_rot
    return out


class AprilEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int, device: str):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
        self.context_length = context_length
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        h = self.embedding(x)  # [B, T, E]
        h = get_rotary_position_encoding(h, device=self.device)  # [B, T, E]
        return h
