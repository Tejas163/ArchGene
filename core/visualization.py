from typing import Optional
import json
import torch
import torch.nn as nn
from pathlib import Path

from core.gene_schema import Gene


class ArchitectureVisualizer:
    def __init__(self):
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def to_ascii(self, gene: Gene) -> str:
        lines = []
        lines.append("-" * 50)
        lines.append("ARCHITECTURE VISUALIZATION")
        lines.append("-" * 50)
        lines.append(f"Vocab Size: {gene.vocab_dim:,}")
        lines.append(f"Hidden Dim: {gene.hidden_dim}")
        lines.append(f"Layers: {gene.num_layers}")
        lines.append(f"Heads: {gene.num_heads} x Head Dim: {gene.head_dim}")
        lines.append("")
        lines.append("Encoder Stack:")
        lines.append("-" * 30)
        
        for i in range(gene.num_layers):
            lines.append(f"Layer {i + 1}:")
            lines.append(f"  |-- Attention: {', '.join(a.value for a in gene.attention_types)}")
            lines.append(f"  +-- FFN: hidden_dim={gene.hidden_dim}, intermediate={gene.intermediate_size}")
            lines.append(f"       Activation: {gene.hidden_act.value}")
        
        lines.append("")
        lines.append("-" * 50)
        
        return "\n".join(lines)
    
    def to_mermaid(self, gene: Gene) -> str:
        lines = []
        lines.append("```mermaid")
        lines.append("graph TD")
        lines.append(f'  Input["Input: {gene.max_position_embeddings} tokens"] --> Embed')
        lines.append(f'  Embed["Embedding: {gene.vocab_dim} -> {gene.hidden_dim}"] --> PosEnc')
        lines.append('  PosEnc["RoPE Position Encoding"] --> Layer1')
        
        for i in range(1, gene.num_layers + 1):
            next_layer = f"Layer{i + 1}" if i < gene.num_layers else "Pool"
            attn_type = ", ".join(a.value for a in gene.attention_types)
            lines.append(f"  Layer{i} --> {next_layer}")
            lines.append(f'  Layer{i}["Attention: {attn_type}<br/>FFN: {gene.hidden_act.value}"]')
        
        lines.append(f"  Pool[\"{gene.pooling_type.value} Pooling\"] --> Output")
        lines.append(f'  Output["Output: {gene.vocab_dim} classes"]')
        lines.append("```")
        
        return "\n".join(lines)
    
    def to_json_schema(self, gene: Gene) -> str:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "LLMArchitecture",
            "type": "object",
            "properties": gene.to_dict(),
            "required": list(gene.to_dict().keys())
        }
        return json.dumps(schema, indent=2)
    
    def visualize(self, gene: Gene, format: str = "ascii") -> str:
        if format == "ascii":
            return self.to_ascii(gene)
        elif format == "mermaid":
            return self.to_mermaid(gene)
        elif format == "json":
            return self.to_json_schema(gene)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def save(self, gene: Gene, filename: str, format: str = "ascii"):
        content = self.visualize(gene, format)
        path = self.output_dir / f"{filename}.{format}"
        with open(path, "w") as f:
            f.write(content)
        return path
    
    def render_console(self, gene: Gene):
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        
        console = Console()
        content = self.to_ascii(gene)
        panel = Panel(content, title="Architecture Visualization", border_style="cyan")
        console.print(panel)