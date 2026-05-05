from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import time
from collections import defaultdict

from src.archgene.gene_schema import Gene
from src.archgene.model_zoo import ModelZoo
from src.archgene.cost_estimator import CostEstimator
from src.archgene.benchmark_integration import BenchmarkIntegration
from src.archgene.verifier import Verifier


app = FastAPI(
    title="ArchGene API",
    description="AI-powered architecture recommendations via API",
    version="1.0.0",
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

rate_limits = defaultdict(lambda: {"count": 0, "reset": 0})
rate_limit_window = 60
max_requests = 100


def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key from header."""
    valid_key = os.environ.get("ARCHGENE_API_KEY", "")
    if not valid_key:
        valid_key = os.environ.get("ARCHGENE_INTERNAL_KEY", "archgene-internal")
    
    if api_key != valid_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key


def check_rate_limit(api_key: str) -> None:
    """Check rate limit for API key."""
    now = time.time()
    
    if rate_limits[api_key]["reset"] < now:
        rate_limits[api_key] = {"count": 0, "reset": now + rate_limit_window}
    
    rate_limits[api_key]["count"] += 1
    
    if rate_limits[api_key]["count"] > max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_requests} requests per minute."
        )


class RecommendRequest(BaseModel):
    budget: float
    gpu: str = "A100-40GB"
    use_case: str = "fine-tuning"
    dataset_tokens: int = 1_000_000_000
    prefer_rope: bool = True
    prefer_flash: bool = True
    min_params_m: float = 0
    max_params_m: float = 70


class RecommendResponse(BaseModel):
    recommendations: List[dict]
    count: int


@app.get("/")
def root():
    return {
        "service": "ArchGene API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "recommend": "/recommend (POST, requires API key)",
            "zoo": "/zoo (GET, requires API key)",
            "cost": "/cost/{name} (GET, requires API key)",
            "benchmark": "/benchmark/{name} (GET, requires API key)",
        }
    }


@app.get("/public/zoo")
def list_zoo_public():
    """List all architectures in Model Zoo (public, no auth)."""
    return {"architectures": ModelZoo.list_all()}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_architecture(req: RecommendRequest, api_key: str = Depends(verify_api_key)):
    """Find architectures matching your requirements."""
    check_rate_limit(api_key)
    """Find architectures matching your requirements."""
    budget_hours = req.budget / CostEstimator.GPU_COST_PER_HOUR.get(req.gpu, 2.0)
    candidates = []
    
    for name in ModelZoo.list_all():
        gene = ModelZoo.get(name)
        est = CostEstimator.estimate_training(gene, gpu=req.gpu, training_tokens=req.dataset_tokens)
        
        if est.training_hours <= budget_hours * 1.5:
            score = 1.0
            if req.use_case == "research" and gene.num_layers >= 20:
                score += 0.2
            if req.use_case == "inference" and gene.hidden_dim <= 2048:
                score += 0.2
            if req.use_case == "fine-tuning" and 3000 <= gene.hidden_dim <= 8000:
                score += 0.3
                
            if req.prefer_rope and not gene.use_rope:
                score *= 0.8
            if req.prefer_flash and not gene.use_flash_attention:
                score *= 0.8
            
            params_m = gene.compute_params() / 1e6
            if req.min_params_m <= params_m <= req.max_params_m:
                candidates.append({
                    "name": name,
                    "score": score,
                    "parameters": gene.compute_params(),
                    "vram_gb": est.vram_gb,
                    "training_hours": est.training_hours,
                    "estimated_cost": est.training_hours * CostEstimator.GPU_COST_PER_HOUR.get(req.gpu, 2.0),
                    "hidden_dim": gene.hidden_dim,
                    "num_layers": gene.num_layers,
                    "num_heads": gene.num_heads,
                })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    return RecommendResponse(
        recommendations=candidates[:5],
        count=len(candidates)
    )


@app.get("/zoo")
def list_zoo(api_key: str = Depends(verify_api_key)):
    """List all architectures in Model Zoo."""
    check_rate_limit(api_key)
    return {"architectures": ModelZoo.list_all()}


@app.get("/zoo/{name}")
def get_architecture(name: str, api_key: str = Depends(verify_api_key)):
    """Get details for a specific architecture."""
    check_rate_limit(api_key)
    try:
        gene = ModelZoo.get(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Architecture '{name}' not found")
    
    est = CostEstimator.full_estimate(gene)
    bench = BenchmarkIntegration.estimate_mmlu(gene)
    
    return {
        "name": name,
        "gene": gene.to_dict(),
        "parameters": gene.compute_params(),
        "estimates": {
            "vram_gb": est.vram_gb,
            "training_hours": est.training_hours,
            "inference_cost_per_1m": est.inference_cost_per_1m_tokens,
            "mmlu": bench.mmlu,
        },
    }


@app.get("/cost/{name}")
def get_cost(name: str, gpu: str = "A100-40GB", api_key: str = Depends(verify_api_key)):
    """Get cost estimates for an architecture."""
    check_rate_limit(api_key)
    try:
        gene = ModelZoo.get(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Architecture '{name}' not found")
    
    est = CostEstimator.full_estimate(gene, gpu=gpu)
    
    return {
        "name": name,
        "gpu": gpu,
        "vram_gb": est.vram_gb,
        "inference_cost_per_1m_tokens": est.inference_cost_per_1m_tokens,
        "training_hours": est.training_hours,
        "training_cost_per_1k_tokens": est.training_cost_per_1k_tokens,
    }


@app.get("/benchmark/{name}")
def get_benchmark(name: str, api_key: str = Depends(verify_api_key)):
    """Get benchmark estimates for an architecture."""
    check_rate_limit(api_key)
    try:
        gene = ModelZoo.get(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Architecture '{name}' not found")
    
    bench = BenchmarkIntegration.full_benchmark(gene)
    
    return bench


@app.get("/verify")
def verify_architecture(
    vocab_dim: int = 32000,
    hidden_dim: int = 4096,
    num_layers: int = 32,
    num_heads: int = 32,
):
    """Verify an architecture meets constraints."""
    gene = Gene(
        vocab_dim=vocab_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    
    verifier = Verifier()
    result = verifier.verify_all(gene)
    
    return {
        "valid": result.is_valid,
        "violations": result.violations if hasattr(result, 'violations') else [],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)