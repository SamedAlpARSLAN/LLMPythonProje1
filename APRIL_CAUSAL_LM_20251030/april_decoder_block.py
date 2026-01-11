import torch.nn as nn
from april_layer_norm import AprilLayerNorm
from april_mlp import AprilMLP
from april_multi_head_attention import AprilMultiHeadAttention


class AprilDecoderBlock(nn.Module):
    """
    Pre-LN (Norm -> SA -> Add -> Norm -> MLP -> Add)
    """
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, device: str):
        super().__init__()
        self.self_attention = AprilMultiHeadAttention(
            embedding_dim=embedding_dim,
            output_dim=embedding_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout_rate=0.0,
            device=device,
        )
        self.norm1 = AprilLayerNorm(embedding_dim, device=device)
        self.mlp   = AprilMLP(embedding_dim, embedding_dim, device=device, dropout=0.0)
        self.norm2 = AprilLayerNorm(embedding_dim, device=device)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.self_attention(y)
        y = self.norm2(x)
        x = x + self.mlp(y)
        return x
