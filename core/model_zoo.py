from core.gene_schema import Gene, ActivationType, AttentionType, PoolingType, ArchitectureFamily, QuantizationType


class ModelZoo:
    """Pre-trained architecture library."""

    ARCHITECTURES = {
        "gpt2": Gene(
            vocab_dim=50257,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            intermediate_size=3072,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            use_bias=True,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.GELU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
            arch_family=ArchitectureFamily.TRANSFORMER,
        ),
        "gpt2_medium": Gene(
            vocab_dim=50257,
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            head_dim=64,
            intermediate_size=4096,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            use_bias=True,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.GELU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
        ),
        "gpt2_large": Gene(
            vocab_dim=50257,
            hidden_dim=1280,
            num_layers=36,
            num_heads=20,
            head_dim=64,
            intermediate_size=5120,
            max_position_embeddings=1024,
            rope_theta=10000.0,
            use_bias=True,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.GELU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
        ),
        "llama2_7b": Gene(
            vocab_dim=32000,
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            intermediate_size=11008,
            max_position_embeddings=4096,
            rope_theta=100000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-5,
        ),
        "llama2_13b": Gene(
            vocab_dim=32000,
            hidden_dim=5120,
            num_layers=40,
            num_heads=40,
            head_dim=128,
            intermediate_size=13824,
            max_position_embeddings=4096,
            rope_theta=100000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-5,
        ),
        "mistral_7b": Gene(
            vocab_dim=32000,
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            intermediate_size=14336,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            use_bias=False,
            attention_types=[AttentionType.SLIDING],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            sliding_window=4096,
            rms_norm_eps=1e-5,
        ),
        "qwen2_0.5b": Gene(
            vocab_dim=151936,
            hidden_dim=896,
            num_layers=24,
            num_heads=14,
            head_dim=64,
            intermediate_size=4864,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-6,
        ),
        "qwen2_1.5b": Gene(
            vocab_dim=151936,
            hidden_dim=1536,
            num_layers=28,
            num_heads=12,
            head_dim=128,
            intermediate_size=6048,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-6,
        ),
        "tiny_llama": Gene(
            vocab_dim=32000,
            hidden_dim=2048,
            num_layers=22,
            num_heads=32,
            head_dim=64,
            intermediate_size=5632,
            max_position_embeddings=2048,
            rope_theta=100000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-5,
        ),
        "phi3_mini": Gene(
            vocab_dim=32064,
            hidden_dim=3072,
            num_layers=26,
            num_heads=32,
            head_dim=96,
            intermediate_size=8192,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-5,
            arch_family=ArchitectureFamily.TRANSFORMER,
        ),
        
        "bitnet_1.58b": Gene(
            vocab_dim=32000,
            hidden_dim=1600,
            num_layers=24,
            num_heads=25,
            head_dim=64,
            intermediate_size=3456,
            max_position_embeddings=4096,
            rope_theta=100000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.BITNET,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-5,
            arch_family=ArchitectureFamily.BITNET,
            quant_type=QuantizationType.BITNET_1BIT,
            quant_groupsize=128,
        ),
        "bitnet_700m": Gene(
            vocab_dim=32000,
            hidden_dim=960,
            num_layers=18,
            num_heads=15,
            head_dim=64,
            intermediate_size=2304,
            max_position_embeddings=2048,
            rope_theta=100000.0,
            use_bias=False,
            attention_types=[AttentionType.FULL],
            hidden_act=ActivationType.BITNET,
            pooling_type=PoolingType.CLS,
            use_rope=True,
            rms_norm_eps=1e-5,
            arch_family=ArchitectureFamily.BITNET,
            quant_type=QuantizationType.BITNET_1BIT,
            quant_groupsize=128,
        ),
        
        "mamba_130m": Gene(
            vocab_dim=32000,
            hidden_dim=768,
            num_layers=24,
            num_heads=0,
            head_dim=64,
            intermediate_size=1536,
            max_position_embeddings=8192,
            rope_theta=10000.0,
            use_bias=False,
            attention_types=[AttentionType.SSM],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
            arch_family=ArchitectureFamily.MAMBA,
            ssm_state_dim=256,
            ssm_conv_kernel=4,
            ssm_norms_before=True,
        ),
        "mamba_370m": Gene(
            vocab_dim=32000,
            hidden_dim=1024,
            num_layers=32,
            num_heads=0,
            head_dim=64,
            intermediate_size=2048,
            max_position_embeddings=8192,
            rope_theta=10000.0,
            use_bias=False,
            attention_types=[AttentionType.SSM],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
            arch_family=ArchitectureFamily.MAMBA,
            ssm_state_dim=256,
            ssm_conv_kernel=4,
            ssm_norms_before=True,
        ),
        
        "rwkv_430m": Gene(
            vocab_dim=32000,
            hidden_dim=1024,
            num_layers=20,
            num_heads=0,
            head_dim=64,
            intermediate_size=2048,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            use_bias=False,
            attention_types=[AttentionType.RWKV],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
            arch_family=ArchitectureFamily.RWKV,
            rwkv_time_mix=True,
            rwkv_layer_norm=True,
            rwkv_sigmoid_softcap=0.0,
        ),
        "rwkv_1b5": Gene(
            vocab_dim=32000,
            hidden_dim=1536,
            num_layers=24,
            num_heads=0,
            head_dim=64,
            intermediate_size=3072,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            use_bias=False,
            attention_types=[AttentionType.RWKV],
            hidden_act=ActivationType.SILU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
            arch_family=ArchitectureFamily.RWKV,
            rwkv_time_mix=True,
            rwkv_layer_norm=True,
            rwkv_sigmoid_softcap=0.0,
        ),
        
        "hyena_150m": Gene(
            vocab_dim=32000,
            hidden_dim=640,
            num_layers=18,
            num_heads=0,
            head_dim=64,
            intermediate_size=2560,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            use_bias=False,
            attention_types=[AttentionType.LINEAR],
            hidden_act=ActivationType.GELU,
            pooling_type=PoolingType.CLS,
            use_rope=False,
            arch_family=ArchitectureFamily.LINEAR,
        ),
    }

    @classmethod
    def get(cls, name: str) -> Gene:
        if name not in cls.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {name}. Available: {list(cls.ARCHITECTURES.keys())}")
        return cls.ARCHITECTURES[name]

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls.ARCHITECTURES.keys())

    @classmethod
    def info(cls, name: str) -> dict:
        gene = cls.get(name)
        return {
            "name": name,
            "gene": gene.to_dict(),
            "parameters": gene.compute_params(),
            "memory_mb": gene.compute_memory() / 1e6,
        }

    @classmethod
    def search(cls, min_params: int = None, max_params: int = None, has_rope: bool = None) -> list[str]:
        results = []
        for name, gene in cls.ARCHITECTURES.items():
            params = gene.compute_params()
            if min_params and params < min_params:
                continue
            if max_params and params > max_params:
                continue
            if has_rope is not None and gene.use_rope != has_rope:
                continue
            results.append(name)
        return results


if __name__ == "__main__":
    print("Available architectures:")
    for name in ModelZoo.list_all():
        info = ModelZoo.info(name)
        print(f"  {name}: {info['parameters']:,} params, {info['memory_mb']:.1f}MB")
    
    print("\nSearch: <1B params with RoPE:")
    print(ModelZoo.search(max_params=1_000_000_000, has_rope=True))