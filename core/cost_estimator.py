from dataclasses import dataclass
from typing import Optional
from core.gene_schema import Gene


@dataclass
class CostEstimates:
    """Cost estimates for an architecture."""
    inference_cost_per_1m_tokens: float
    training_cost_per_1k_tokens: float
    inference_latency_ms: float
    training_hours: float
    vram_gb: float
    gpu_hours: float


class CostEstimator:
    """Estimates real-world costs for architectures."""

    GPU_COST_PER_HOUR = {
        "A100-80GB": 3.50,
        "A100-40GB": 2.00,
        "A10": 0.80,
        "T4": 0.50,
        "H100": 5.00,
        "L40": 1.50,
    }

    TOKEN_Speeds = {
        "A100-80GB": 45000,
        "A100-40GB": 30000,
        "A10": 15000,
        "T4": 8000,
        "H100": 60000,
        "L40": 20000,
    }

    @classmethod
    def estimate_inference(
        cls,
        gene: Gene,
        gpu: str = "A100-40GB",
        batch_size: int = 1,
    ) -> CostEstimates:
        params = gene.compute_params()
        tokens_per_second = cls.TOKEN_Speeds.get(gpu, 10000) * batch_size
        latency = (1 / tokens_per_second) * 1000

        tokens_per_dollar = (tokens_per_second * 3600) / (cls.GPU_COST_PER_HOUR.get(gpu, 2.0) * 0.3)
        cost_per_m = (1 / tokens_per_dollar) * 1_000_000

        return CostEstimates(
            inference_cost_per_1m_tokens=round(cost_per_m, 2),
            training_cost_per_1k_tokens=0.0,
            inference_latency_ms=round(latency, 2),
            training_hours=0.0,
            vram_gb=round(gene.compute_memory() / 1e9, 2),
            gpu_hours=0.0,
        )

    @classmethod
    def estimate_training(
        cls,
        gene: Gene,
        gpu: str = "A100-40GB",
        training_tokens: int = 1_000_000_000,
    ) -> CostEstimates:
        params = gene.compute_params()
        flops_per_token = params * 6
        total_flops = flops_per_token * training_tokens

        gpu_tflops = {
            "A100-80GB": 312,
            "A100-40GB": 312,
            "A10": 250,
            "T4": 130,
            "H100": 513,
            "L40": 145,
        }
        tflops = gpu_tflops.get(gpu, 312)
        gpu_seconds = total_flops / (tflops * 1e12)
        gpu_hours = gpu_seconds / 3600

        cost_per_hour = cls.GPU_COST_PER_HOUR.get(gpu, 2.0)
        training_cost = gpu_hours * cost_per_hour

        return CostEstimates(
            inference_cost_per_1m_tokens=0.0,
            training_cost_per_1k_tokens=round(training_cost / (training_tokens / 1000), 2),
            inference_latency_ms=0.0,
            training_hours=round(gpu_hours, 1),
            vram_gb=round(gene.compute_memory() / 1e9, 2),
            gpu_hours=round(gpu_hours, 1),
        )

    @classmethod
    def full_estimate(
        cls,
        gene: Gene,
        gpu: str = "A100-40GB",
        batch_size: int = 1,
        training_tokens: int = 1_000_000_000,
    ) -> CostEstimates:
        inf = cls.estimate_inference(gene, gpu, batch_size)
        train = cls.estimate_training(gene, gpu, training_tokens)

        return CostEstimates(
            inference_cost_per_1m_tokens=inf.inference_cost_per_1m_tokens,
            training_cost_per_1k_tokens=train.training_cost_per_1k_tokens,
            inference_latency_ms=inf.inference_latency_ms,
            training_hours=train.training_hours,
            vram_gb=max(inf.vram_gb, train.vram_gb),
            gpu_hours=train.gpu_hours,
        )


if __name__ == "__main__":
    from core.model_zoo import ModelZoo

    print("Cost Estimates (GPU: A100-40GB)")
    print("=" * 60)

    for name in ["gpt2", "llama2_7b", "mistral_7b"]:
        gene = ModelZoo.get(name)
        est = CostEstimator.full_estimate(gene)

        print(f"\n{name}:")
        print(f"  VRAM: {est.vram_gb:.1f} GB")
        print(f"  Inference: ${est.inference_cost_per_1m_tokens:.2f}/M tokens")
        print(f"  Inference latency: {est.inference_latency_ms:.2f} ms/token")
        print(f"  Training (1T tokens): {est.training_hours:.0f} GPU hours")
        print(f"  Training cost: ${est.training_cost_per_1k_tokens:.2f}/1K tokens")