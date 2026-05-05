from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    NONE = "none"


class AttentionType(Enum):
    FULL = "full"
    SLIDING = "sliding"
    FLASH = "flash"
    LINEAR = "linear"


class PoolingType(Enum):
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"


@dataclass
class Gene:
    vocab_dim: int = 4096
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    use_bias: bool = True
    attention_types: list[AttentionType] = field(default_factory=lambda: [AttentionType.FULL])
    hidden_act: ActivationType = ActivationType.GELU
    pooling_type: PoolingType = PoolingType.CLS
    layer_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    use_rms_norm: bool = True
    use_flash_attention: bool = False
    sliding_window: int = 4096
    dropout: float = 0.0
    use_rope: bool = True
    use_gated_activation: bool = False

    def to_dict(self) -> dict:
        return {
            "vocab_dim": self.vocab_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "use_bias": self.use_bias,
            "attention_types": [a.value for a in self.attention_types],
            "hidden_act": self.hidden_act.value,
            "pooling_type": self.pooling_type.value,
            "layer_norm_eps": self.layer_norm_eps,
            "rms_norm_eps": self.rms_norm_eps,
            "use_rms_norm": self.use_rms_norm,
            "use_flash_attention": self.use_flash_attention,
            "sliding_window": self.sliding_window,
            "dropout": self.dropout,
            "use_rope": self.use_rope,
            "use_gated_activation": self.use_gated_activation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Gene":
        attention_types = [AttentionType(a) for a in d.get("attention_types", ["full"])]
        hidden_act = ActivationType(d.get("hidden_act", "gelu"))
        pooling_type = PoolingType(d.get("pooling_type", "cls"))
        
        return cls(
            vocab_dim=d.get("vocab_dim", 4096),
            hidden_dim=d.get("hidden_dim", 512),
            num_layers=d.get("num_layers", 4),
            num_heads=d.get("num_heads", 8),
            head_dim=d.get("head_dim", 64),
            intermediate_size=d.get("intermediate_size", 2048),
            max_position_embeddings=d.get("max_position_embeddings", 2048),
            rope_theta=d.get("rope_theta", 10000.0),
            use_bias=d.get("use_bias", True),
            attention_types=attention_types,
            hidden_act=hidden_act,
            pooling_type=pooling_type,
            layer_norm_eps=d.get("layer_norm_eps", 1e-5),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            use_rms_norm=d.get("use_rms_norm", True),
            use_flash_attention=d.get("use_flash_attention", False),
            sliding_window=d.get("sliding_window", 4096),
            dropout=d.get("dropout", 0.0),
            use_rope=d.get("use_rope", True),
            use_gated_activation=d.get("use_gated_activation", False),
        )

    def validate(self) -> tuple[bool, list[str]]:
        errors = []
        
        if self.vocab_dim <= 0:
            errors.append("vocab_dim must be positive")
        if self.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")
        if self.num_layers <= 0:
            errors.append("num_layers must be positive")
        if self.num_heads <= 0:
            errors.append("num_heads must be positive")
        if self.head_dim <= 0:
            errors.append("head_dim must be positive")
        if self.intermediate_size <= 0:
            errors.append("intermediate_size must be positive")
        
        if self.hidden_dim % self.num_heads != 0:
            errors.append(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")
        if self.head_dim != self.hidden_dim // self.num_heads:
            errors.append(f"head_dim ({self.head_dim}) must equal hidden_dim / num_heads ({self.hidden_dim // self.num_heads})")
        
        if self.layer_norm_eps <= 0:
            errors.append("layer_norm_eps must be positive")
        if self.rms_norm_eps <= 0:
            errors.append("rms_norm_eps must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            errors.append("dropout must be in [0, 1)")
        
        return len(errors) == 0, errors

    def compute_params(self) -> int:
        embedding_params = self.vocab_dim * self.hidden_dim
        
        attention_params = 0
        for _ in range(self.num_layers):
            attention_params += 4 * self.hidden_dim * self.hidden_dim
            attention_params += 2 * self.hidden_dim * self.hidden_dim
        
        ffn_params = 0
        for _ in range(self.num_layers):
            ffn_params += self.hidden_dim * self.intermediate_size
            ffn_params += self.intermediate_size * self.hidden_dim
        
        output_params = self.hidden_dim * self.vocab_dim
        
        return embedding_params + attention_params + ffn_params + output_params

    def compute_memory(self) -> int:
        param_count = self.compute_params()
        bytes_per_param = 2 if self.use_bias else 1
        activation_memory = self.hidden_dim * self.max_position_embeddings * self.num_layers
        return param_count * bytes_per_param + activation_memory