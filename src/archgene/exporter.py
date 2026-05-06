from typing import Optional
import json
import torch
import torch.nn as nn
from pathlib import Path

from .gene_schema import Gene


class TransformerBlock(nn.Module):
    def __init__(self, gene: Gene, layer_idx: int = 0):
        super().__init__()
        self.gene = gene
        self.layer_idx = layer_idx
        
        self.self_attn = nn.MultiheadAttention(
            gene.hidden_dim,
            gene.num_heads,
            dropout=gene.dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(gene.hidden_dim, eps=gene.layer_norm_eps)
        
        self.mlp = nn.Sequential(
            nn.Linear(gene.hidden_dim, gene.intermediate_size),
            nn.GELU() if gene.hidden_act.value == "gelu" else nn.ReLU(),
            nn.Linear(gene.intermediate_size, gene.hidden_dim),
            nn.Dropout(gene.dropout)
        )
        
        self.mlp_norm = nn.LayerNorm(gene.hidden_dim, eps=gene.layer_norm_eps)
    
    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)[0]
        x = residual + x
        
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class LLMArchitecture(nn.Module):
    def __init__(self, gene: Gene):
        super().__init__()
        self.gene = gene
        
        self.embedding = nn.Embedding(gene.vocab_dim, gene.hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(gene, i) for i in range(gene.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(gene.hidden_dim, eps=gene.layer_norm_eps)
        self.lm_head = nn.Linear(gene.hidden_dim, gene.vocab_dim, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        for layer in self.layers:
            x = layer(x, key_padding_mask=None)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits


class Exporter:
    def __init__(self):
        self.output_dir = Path("exports")
        self.output_dir.mkdir(exist_ok=True)
    
    def to_pytorch(self, gene: Gene, path: Optional[str] = None) -> Path:
        model = LLMArchitecture(gene)
        
        if path is None:
            path = self.output_dir / "model.pt"
        else:
            path = Path(path)
        
        torch.save({
            "gene": gene.to_dict(),
            "model_state_dict": model.state_dict()
        }, path)
        
        return path
    
    def to_pytorch_scripted(self, gene: Gene, path: Optional[str] = None) -> Path:
        model = LLMArchitecture(gene)
        scripted = torch.jit.script(model)
        
        if path is None:
            path = self.output_dir / "model_scripted.pt"
        else:
            path = Path(path)
        
        torch.jit.save(scripted, path)
        
        return path
    
    def to_config(self, gene: Gene, path: Optional[str] = None) -> Path:
        config = {
            "architectural": {
                "vocab_size": gene.vocab_dim,
                "hidden_size": gene.hidden_dim,
                "intermediate_size": gene.intermediate_size,
                "num_hidden_layers": gene.num_layers,
                "num_attention_heads": gene.num_heads,
                "hidden_act": gene.hidden_act.value,
                "max_position_embeddings": gene.max_position_embeddings,
                "rope_theta": gene.rope_theta,
                "rms_norm_eps": gene.rms_norm_eps,
                "use_rope": gene.use_rope,
                "use_flash_attention": gene.use_flash_attention,
            },
            "training": {
                "max_learning_rate": 1e-4,
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "gradient_clip_norm": 1.0,
            }
        }
        
        if path is None:
            path = self.output_dir / "config.json"
        else:
            path = Path(path)
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        
        return path
    
    def to_huggingface(self, gene: Gene, model_dir: Optional[str] = None) -> Path:
        model_dir = Path(model_dir) if model_dir else self.output_dir / "huggingface_model"
        model_dir.mkdir(exist_ok=True)
        
        model = LLMArchitecture(gene)
        model.save_pretrained(model_dir)
        
        config = self.to_config(gene, model_dir / "config.json")
        
        return model_dir
    
    def to_onnx(self, gene: Gene, path: Optional[str] = None, sample_input: Optional[torch.Tensor] = None) -> Path:
        model = LLMArchitecture(gene)
        model.eval()
        
        if sample_input is None:
            batch_size = 1
            seq_len = gene.max_position_embeddings
            sample_input = torch.randint(0, gene.vocab_dim, (batch_size, seq_len))
        
        if path is None:
            path = self.output_dir / "model.onnx"
        else:
            path = Path(path)
        
        torch.onnx.export(
            model,
            sample_input,
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        )
        
        return path
    
    def export_all(self, gene: Gene, base_name: str = "architecture") -> dict[str, Path]:
        paths = {}
        
        paths["pytorch"] = self.to_pytorch(gene, f"{base_name}.pt")
        paths["config"] = self.to_config(gene, f"{base_name}_config.json")
        
        return paths