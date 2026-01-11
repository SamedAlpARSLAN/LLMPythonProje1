import torch
import torch.nn as nn


class AprilMultiHeadAttention(nn.Module):
    """
    PyTorch nn.MultiheadAttention (batch_first=True) ile kausal MHA sarmalayıcısı.
    Girdi/Çıktı: [B, T, E]
    """
    def __init__(self, embedding_dim: int, output_dim: int, context_length: int,
                 num_heads: int, dropout_rate: float = 0.0, device: str = "cpu"):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim,
                                         num_heads=num_heads,
                                         dropout=dropout_rate,
                                         batch_first=True,
                                         device=device)
        self.proj = nn.Linear(embedding_dim, output_dim, device=device)
        # geleceğe bakışı engelleyen kausal maske (True = maskelenecek)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length, device=device), diagonal=1).bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]
        B, T, E = x.shape
        attn_mask = self.mask[:T, :T]  # [T, T] bool (True -> -inf)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        out = self.proj(out)  # [B, T, output_dim]
        return out
