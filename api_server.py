from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import sys
import os
import time
from collections import defaultdict
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gene_schema import Gene, ActivationType, AttentionType, PoolingType
from core.verifier import Verifier
from core.evaluation import Evaluator
from core.visualization import ArchitectureVisualizer
from core.exporter import Exporter

app = FastAPI(title="ArchGene API")

# Rate limiting storage
rate_limit_store = defaultdict(list)
RATE_LIMIT = 100  # requests per minute
rate_limit_lock = Lock()


def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit."""
    current_time = time.time()
    with rate_limit_lock:
        # Clean old requests
        rate_limit_store[client_id] = [
            ts for ts in rate_limit_store[client_id]
            if current_time - ts < 60
        ]
        if len(rate_limit_store[client_id]) >= RATE_LIMIT:
            return False
        rate_limit_store[client_id].append(current_time)
        return True


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Verify API key from header."""
    expected_key = os.getenv("ARCHGENE_API_KEY", "dev-key-change-in-production")
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class GeneInput(BaseModel):
    vocab_dim: int = 4096
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    use_bias: bool = True
    attention_types: List[str] = ["full"]
    hidden_act: str = "gelu"
    pooling_type: str = "cls"
    layer_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    use_rms_norm: bool = True
    use_flash_attention: bool = False
    sliding_window: int = 4096
    dropout: float = 0.0
    use_rope: bool = True
    use_gated_activation: bool = False


class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []


class StatsResponse(BaseModel):
    parameters: int
    memory_mb: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate", response_model=ValidationResponse)
def validate(gene_in: GeneInput, api_key: str = Depends(verify_api_key)):
    if not check_rate_limit("validate"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    gene = Gene(
        vocab_dim=gene_in.vocab_dim,
        hidden_dim=gene_in.hidden_dim,
        num_layers=gene_in.num_layers,
        num_heads=gene_in.num_heads,
        head_dim=gene_in.head_dim,
        intermediate_size=gene_in.intermediate_size,
        max_position_embeddings=gene_in.max_position_embeddings,
        rope_theta=gene_in.rope_theta,
        use_bias=gene_in.use_bias,
        attention_types=[AttentionType(a) for a in gene_in.attention_types],
        hidden_act=ActivationType(gene_in.hidden_act),
        pooling_type=PoolingType(gene_in.pooling_type),
        layer_norm_eps=gene_in.layer_norm_eps,
        rms_norm_eps=gene_in.rms_norm_eps,
        use_rms_norm=gene_in.use_rms_norm,
        use_flash_attention=gene_in.use_flash_attention,
        sliding_window=gene_in.sliding_window,
        dropout=gene_in.dropout,
        use_rope=gene_in.use_rope,
        use_gated_activation=gene_in.use_gated_activation,
    )
    is_valid, errors = gene.validate()
    return ValidationResponse(valid=is_valid, errors=errors)


@app.post("/verify")
def verify(gene_in: GeneInput, api_key: str = Depends(verify_api_key)):
    if not check_rate_limit("verify"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    gene = _gene_from_input(gene_in)
    verifier = Verifier()
    result = verifier.verify_all(gene)
    return {"verified": result.is_valid}


@app.post("/evaluate")
def evaluate(gene_in: GeneInput, api_key: str = Depends(verify_api_key)):
    if not check_rate_limit("evaluate"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    gene = _gene_from_input(gene_in)
    evaluator = Evaluator()
    result = evaluator.evaluate(gene)
    return result


@app.post("/visualize")
def visualize(gene_in: GeneInput, format: str = "ascii", api_key: str = Depends(verify_api_key)):
    if not check_rate_limit("visualize"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    gene = _gene_from_input(gene_in)
    viz = ArchitectureVisualizer()
    try:
        output = viz.visualize(gene, format)
        return {"visualization": output}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/export")
def export(gene_in: GeneInput, format: str = "json", api_key: str = Depends(verify_api_key)):
    if not check_rate_limit("export"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    gene = _gene_from_input(gene_in)
    exporter = Exporter()
    try:
        if format == "json":
            path = exporter.to_config(gene, "api_arch")
        elif format == "pytorch":
            path = exporter.to_pytorch(gene, "api_arch")
        elif format == "onnx":
            path = exporter.to_onnx(gene, "api_arch")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format: {format}")
        return {"path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/stats", response_model=StatsResponse)
def stats(gene_in: GeneInput, api_key: str = Depends(verify_api_key)):
    if not check_rate_limit("stats"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    gene = _gene_from_input(gene_in)
    is_valid, errors = gene.validate()
    if not is_valid:
        raise HTTPException(status_code=400, detail=errors)
    params = gene.compute_params()
    memory = gene.compute_memory()
    return StatsResponse(parameters=params, memory_mb=memory / 1e6)


def _gene_from_input(gene_in: GeneInput) -> Gene:
    return Gene(
        vocab_dim=gene_in.vocab_dim,
        hidden_dim=gene_in.hidden_dim,
        num_layers=gene_in.num_layers,
        num_heads=gene_in.num_heads,
        head_dim=gene_in.head_dim,
        intermediate_size=gene_in.intermediate_size,
        max_position_embeddings=gene_in.max_position_embeddings,
        rope_theta=gene_in.rope_theta,
        use_bias=gene_in.use_bias,
        attention_types=[AttentionType(a) for a in gene_in.attention_types],
        hidden_act=ActivationType(gene_in.hidden_act),
        pooling_type=PoolingType(gene_in.pooling_type),
        layer_norm_eps=gene_in.layer_norm_eps,
        rms_norm_eps=gene_in.rms_norm_eps,
        use_rms_norm=gene_in.use_rms_norm,
        use_flash_attention=gene_in.use_flash_attention,
        sliding_window=gene_in.sliding_window,
        dropout=gene_in.dropout,
        use_rope=gene_in.use_rope,
        use_gated_activation=gene_in.use_gated_activation,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)