from transformers import PretrainedConfig

class AprilConfig(PretrainedConfig):
    model_type = "april"

    def __init__(
        self,
        vocab_size: int = 32000,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        context_length: int = 128,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(**kwargs)

     
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context_length = context_length
        self.device = device  

  
        if not hasattr(self, "num_hidden_layers"):
            self.num_hidden_layers = num_layers
