import torch
import torch.nn as nn


class AprilLayerNorm(nn.Module):
    """
    Basit LayerNorm: learnable gamma (weight) + beta (bias)
    """
    def __init__(self, embedding_dim: int, eps: float = 1e-5, device: str = "cpu"):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim, device=device))
        self.bias   = nn.Parameter(torch.zeros(embedding_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.weight + self.bias
