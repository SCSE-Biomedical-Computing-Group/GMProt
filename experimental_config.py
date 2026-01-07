import json
from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    # Training
    batch_size: int = 256
    epochs: int = 300
    lr: float = 3e-4
    seed: int = 42
    patience: int = 100

    #physio features
    physio_feature_dim: int = 32
    blosum_feature_dim: int = 20

    #Transformer 
    transformer_heads: int = 8
    transformer_ff_dim : int = 512

    # Sequence branch
    seq_dense1: int = 512
    seq_dense2: int = 256
    seq_bottleneck_dim: int = 128
    seq_dropout: float = 0.3

    # GNN
    gnn_hidden: int = 64
    gnn_heads: int = 4
    gnn_layers: int = 3
    gnn_dense: int = 128
    gnn_bottleneck_dim : int = 128

    # Physio
    physio_proj: int = 32
    physio_bottleneck_dim: int = 128

    # Fusion
    fuse_dense: int = 256
    fuse_dropout: float = 0.3

    #shuffle buffer size
    shuffle_buffer_size: int = 5000

    #huber pearson loss delta
    huber_delta: float = 1.0
    alpha: float = 0.15
    use_rank_loss: bool = False
    beta: float = 0.05 #weight for soft rank loss
    epsilon: float = 1e-6 #numerical stability in pearson calculation(smoothing)

    #Attention Visualization graphs
    visualization_path : str = "./visualization/atten"

    def save_config(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)