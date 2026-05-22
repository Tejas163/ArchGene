"""Kernel Generator — generates runnable PyTorch source code from a Gene."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import zipfile
import io

from .gene_schema import Gene, ActivationType, ArchitectureFamily


def _activation_import(gene: Gene) -> str:
    if gene.hidden_act == ActivationType.GELU:
        return "from torch.nn import GELU as Activation"
    elif gene.hidden_act == ActivationType.SILU:
        return "from torch.nn import SiLU as Activation"
    elif gene.hidden_act == ActivationType.RELU:
        return "from torch.nn import ReLU as Activation"
    return "from torch.nn import GELU as Activation"


def _norm_code(gene: Gene) -> str:
    if gene.use_rms_norm:
        return (
            "class RMSNorm(nn.Module):\n"
            "    def __init__(self, hidden_size: int, eps: float = 1e-6):\n"
            "        super().__init__()\n"
            "        self.weight = nn.Parameter(torch.ones(hidden_size))\n"
            f"        self.eps = eps\n"
            "\n"
            "    def forward(self, x):\n"
            "        norm = x.pow(2).mean(-1, keepdim=True)\n"
            f"        x = x * torch.rsqrt(norm + self.eps)\n"
            "        return x * self.weight\n"
        )
    else:
        return ""


def _norm_class(gene: Gene) -> str:
    return "RMSNorm" if gene.use_rms_norm else "nn.LayerNorm"


def _norm_eps(gene: Gene) -> str:
    return str(gene.rms_norm_eps if gene.use_rms_norm else gene.layer_norm_eps)


def _generate_model_py(gene: Gene) -> str:
    act_import = _activation_import(gene)
    norm_code = _norm_code(gene)
    norm_class = _norm_class(gene)
    norm_eps = _norm_eps(gene)

    kv_heads = gene.kv_heads
    is_gqa = kv_heads != gene.num_heads

    head_dim = gene.head_dim
    rope_enabled = gene.use_rope
    gated = gene.use_gated_activation
    flash = gene.use_flash_attention

    rope_code = ""
    if rope_enabled:
        rope_code = (
            "class RotaryEmbedding(nn.Module):\n"
            f"    def __init__(self, head_dim: int, max_seq_len: int = {gene.max_position_embeddings},"
            f" theta: float = {gene.rope_theta}):\n"
            "        super().__init__()\n"
            "        half = head_dim // 2\n"
            "        freqs = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)\n"
            "        t = torch.arange(max_seq_len)\n"
            "        freqs = torch.outer(t, freqs)\n"
            "        self.register_buffer('cos', freqs.cos())  # (max_seq_len, half)\n"
            "        self.register_buffer('sin', freqs.sin())\n"
            "\n"
            "    def forward(self, q, k):\n"
            "        seq_len = q.size(2)\n"
            "        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)\n"
            "        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)\n"
            "        cos = torch.cat([cos, cos], dim=-1)\n"
            "        sin = torch.cat([sin, sin], dim=-1)\n"
            "        q_embed = (q * cos) + (self._rotate_half(q) * sin)\n"
            "        k_embed = (k * cos) + (self._rotate_half(k) * sin)\n"
            "        return q_embed, k_embed\n"
            "\n"
            "    def _rotate_half(self, x):\n"
            "        x1, x2 = x.chunk(2, dim=-1)\n"
            "        return torch.cat((-x2, x1), dim=-1)\n"
            "\n"
        )

    gqa_comment = ""
    if is_gqa:
        gqa_comment = (
            f"        # Grouped Query Attention: {gene.num_heads} query heads, {kv_heads} KV heads\n"
        )

    mlp_code = (
        "class MLP(nn.Module):\n"
        "    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):\n"
        "        super().__init__()\n"
    )
    if gated:
        mlp_code += (
            "        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)\n"
            "        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)\n"
            "        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)\n"
            "        self.act = Activation()\n"
            "        self.dropout = nn.Dropout(dropout)\n"
            "\n"
            "    def forward(self, x):\n"
            "        x = self.act(self.gate_proj(x)) * self.up_proj(x)\n"
            "        x = self.down_proj(x)\n"
            "        x = self.dropout(x)\n"
            "        return x\n"
        )
    else:
        mlp_code += (
            "        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)\n"
            "        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)\n"
            "        self.act = Activation()\n"
            "        self.dropout = nn.Dropout(dropout)\n"
            "\n"
            "    def forward(self, x):\n"
            "        x = self.act(self.fc1(x))\n"
            "        x = self.fc2(x)\n"
            "        x = self.dropout(x)\n"
            "        return x\n"
        )

    if is_gqa:
        gqa_code = (
            "        kv_heads = k.size(1)\n"
            "        if kv_heads != self.num_heads:\n"
            "            n_repeat = self.num_heads // kv_heads\n"
            "            k = k[:, :, None, :, :].expand(-1, -1, n_repeat, -1, -1).reshape(\n"
            "                batch_size, self.num_heads, seq_len, self.head_dim\n"
            "            )\n"
            "            v = v[:, :, None, :, :].expand(-1, -1, n_repeat, -1, -1).reshape(\n"
            "                batch_size, self.num_heads, seq_len, self.head_dim\n"
            "            )\n"
        )
    else:
        gqa_code = ""

    rope_forward = ""
    rope_init = ""
    if rope_enabled:
        rope_init = "        self.rotary = RotaryEmbedding(head_dim)\n"
        rope_forward = (
            "        q, k = self.rotary(q, k)\n"
        )

    flash_code = ""
    if flash:
        flash_code = (
            "\n"
            "class FlashAttention(nn.Module):\n"
            "    def __init__(self, hidden_size: int, num_heads: int, head_dim: int = 64):\n"
            "        super().__init__()\n"
            "        self.num_heads = num_heads\n"
            "        self.head_dim = head_dim\n"
            "        self.scale = head_dim ** -0.5\n"
            "\n"
            "    def forward(self, q, k, v, causal_mask=None):\n"
            "        try:\n"
            "            from flash_attn import flash_attn_func\n"
            "            return flash_attn_func(q, k, v, causal=True)\n"
            "        except ImportError:\n"
            "            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale\n"
            "            if causal_mask is not None:\n"
            "                attn = attn.masked_fill(causal_mask, float('-inf'))\n"
            "            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)\n"
            "            return torch.matmul(attn, v)\n"
            "\n"
        )

    return f'''"""
Auto-generated model by ArchGene.
Architecture: {gene.arch_family.value} | hidden={gene.hidden_dim} layers={gene.num_layers} heads={gene.num_heads}
Params: {gene.compute_params():,} | Max context: {gene.max_position_embeddings}
RoPE: {gene.use_rope} | GQA: {is_gqa} | Gated activation: {gated} | RMSNorm: {gene.use_rms_norm}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
{act_import}

{norm_code}
{rope_code}
{flash_code}
class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 kv_heads: int = 0, dropout: float = {gene.dropout}):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.kv_heads = kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, {kv_heads} * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, {kv_heads} * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
{rope_init}
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Project K/V — may use fewer heads (GQA) than Q
        k = self.k_proj(x).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

{gqa_code}{rope_forward}
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


{mlp_code}
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 intermediate_size: int, dropout: float, layer_norm_eps: float,
                 kv_heads: int = 0):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads, head_dim,
                                   kv_heads=kv_heads, dropout=dropout)
        self.input_layernorm = {norm_class}(hidden_size, eps=layer_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, dropout)
        self.post_attention_layernorm = {norm_class}(hidden_size, eps=layer_norm_eps)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int = {gene.vocab_dim},
        hidden_size: int = {gene.hidden_dim},
        num_layers: int = {gene.num_layers},
        num_heads: int = {gene.num_heads},
        head_dim: int = {gene.head_dim},
        intermediate_size: int = {gene.intermediate_size},
        max_position_embeddings: int = {gene.max_position_embeddings},
        dropout: float = {gene.dropout},
        layer_norm_eps: float = {norm_eps},
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_heads, head_dim, intermediate_size, dropout, layer_norm_eps,
                kv_heads={gene.kv_heads}
            )
            for _ in range(num_layers)
        ])
        self.norm = {norm_class}(hidden_size, eps=layer_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
'''


def _generate_config_json(gene: Gene) -> str:
    model_type = gene.arch_family.value
    kv_heads = gene.kv_heads

    config = {
        "architectures": ["Model"],
        "model_type": model_type,
        "vocab_size": gene.vocab_dim,
        "hidden_size": gene.hidden_dim,
        "num_hidden_layers": gene.num_layers,
        "num_attention_heads": gene.num_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": gene.head_dim,
        "intermediate_size": gene.intermediate_size,
        "max_position_embeddings": gene.max_position_embeddings,
        "hidden_act": gene.hidden_act.value,
        "dropout": gene.dropout,
        "layer_norm_eps": gene.layer_norm_eps,
        "rms_norm_eps": gene.rms_norm_eps,
        "use_rms_norm": gene.use_rms_norm,
        "rope_theta": gene.rope_theta,
        "use_rope": gene.use_rope,
        "use_gated_activation": gene.use_gated_activation,
        "use_flash_attention": gene.use_flash_attention,
        "arch_family": gene.arch_family.value,
        "total_params": gene.compute_params(),
    }
    return json.dumps(config, indent=2)


def _generate_train_py(gene: Gene) -> str:
    return f'''"""
Auto-generated training script by ArchGene.
Architecture: {gene.arch_family.value} | {gene.compute_params():,} params

Usage:
    python train.py                       # Quick dummy data demo
    python train.py --steps 100           # Longer demo
    python train.py --real --data ./data  # Your own text data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from model import Model
import json
import time
import argparse
from pathlib import Path


class DummyDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(1, self.vocab_size, (self.seq_len,))
        return x, x.clone()


class TextDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int):
        self.seq_len = seq_len
        texts = []
        for f in Path(data_dir).glob("*.txt"):
            texts.append(f.read_text(encoding="utf-8"))
        text = "\\n".join(texts)
        self.tokens = torch.tensor([ord(c) for c in text[:100000]], dtype=torch.long)

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start:start + self.seq_len]
        y = self.tokens[start + 1:start + self.seq_len + 1]
        return x, y


def train(args):
    device = torch.device(args.device)

    model = Model()
    model.to(device)
    total = model.count_params()
    trainable = model.count_trainable_params()

    if args.real and args.data:
        dataset = TextDataset(args.data, args.seq_len)
    else:
        vocab_size = model.embed_tokens.num_embeddings
        dataset = DummyDataset(vocab_size, args.seq_len, args.steps * args.batch_size)

    val_size = max(1, int(len(dataset) * 0.05))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler() if args.amp else None

    print(f"Model: {{total:,}} params ({{trainable:,}} trainable)")
    print(f"Device: {{device}}  Steps: {{args.steps}}  Batch: {{args.batch_size}}")
    print(f"AMP: {{'on' if args.amp else 'off'}}")
    print()

    for step, (x, y) in enumerate(train_loader):
        if step >= args.steps:
            break

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device.type):
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()

        if step % args.log_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    if val_steps >= 5:
                        break
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits = model(vx)
                    vloss = loss_fn(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                    val_loss += vloss.item()
                    val_steps += 1
            model.train()

            lr = scheduler.get_last_lr()[0]
            print(
                f"step {{step:4d}} | "
                f"loss {{loss.item():.4f}} | "
                f"val {{val_loss / max(val_steps, 1):.4f}} | "
                f"lr {{lr:.2e}}"
            )

    torch.save(model.state_dict(), "model_state.pt")
    print(f"\\nSaved model_state.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--real", action="store_true", help="Use real text data")
    parser.add_argument("--data", help="Path to directory with .txt files")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
'''


def _generate_requirements_txt() -> str:
    return "torch>=2.0.0\n"


@dataclass
class GeneratedFiles:
    model_py: Path
    config_json: Path
    train_py: Path
    requirements_txt: Path
    output_dir: Path


class KernelGenerator:
    def generate(self, gene: Gene, output_dir: str = "generated") -> GeneratedFiles:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_py = out / "model.py"
        model_py.write_text(_generate_model_py(gene), encoding="utf-8")

        config_json = out / "config.json"
        config_json.write_text(_generate_config_json(gene), encoding="utf-8")

        train_py = out / "train.py"
        train_py.write_text(_generate_train_py(gene), encoding="utf-8")

        requirements_txt = out / "requirements.txt"
        requirements_txt.write_text(_generate_requirements_txt(), encoding="utf-8")

        return GeneratedFiles(model_py, config_json, train_py, requirements_txt, out)

    @staticmethod
    def generate_files_dict(gene: Gene) -> dict[str, str]:
        return {
            "model.py": _generate_model_py(gene),
            "config.json": _generate_config_json(gene),
            "train.py": _generate_train_py(gene),
            "requirements.txt": _generate_requirements_txt(),
        }

    @staticmethod
    def generate_zip(gene: Gene) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in KernelGenerator.generate_files_dict(gene).items():
                zf.writestr(f"archgene_generated/{name}", content)
        buf.seek(0)
        return buf.getvalue()
