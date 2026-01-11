import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x.pow(3))))


class AprilMLP(nn.Module):
    """
    Gated-GELU tarzı MLP (proj -> gelu * proj -> down)
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, device: str = "cpu", dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, device=device)
        self.up_proj   = nn.Linear(embedding_dim, hidden_dim, device=device)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, device=device)
        self.act = GELU().to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(x))
        up   = self.up_proj(x)
        fuse = gate * up
        out  = self.down_proj(self.dropout(fuse))
        return out
