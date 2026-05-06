from z3 import (
    Solver, Optimize, Bool, Int, Real, And, Or, Not, Implies,
    Sum, Product, If, sat, unsat
)
from typing import Optional
from dataclasses import dataclass

from .gene_schema import Gene


@dataclass
class VerificationResult:
    is_valid: bool
    model: Optional[dict]
    constraints_checked: list[str]
    errors: list[str]


class Verifier:
    def __init__(self):
        self.solver = Solver()
        self.optimizer = Optimize()
    
    def encode_gene(self, gene: Gene) -> dict:
        v_vocab_dim = Int("vocab_dim")
        v_hidden_dim = Int("hidden_dim")
        v_num_layers = Int("num_layers")
        v_num_heads = Int("num_heads")
        v_head_dim = Int("head_dim")
        v_intermediate_size = Int("intermediate_size")
        v_max_pos = Int("max_position_embeddings")
        v_rope_theta = Real("rope_theta")
        v_sliding_window = Int("sliding_window")
        v_dropout = Real("dropout")
        v_layer_norm_eps = Real("layer_norm_eps")
        v_rms_norm_eps = Real("rms_norm_eps")
        
        v_use_bias = Bool("use_bias")
        v_use_rms_norm = Bool("use_rms_norm")
        v_use_flash = Bool("use_flash_attention")
        v_use_rope = Bool("use_rope")
        v_use_gated = Bool("use_gated_activation")
        
        return {
            "vocab_dim": v_vocab_dim,
            "hidden_dim": v_hidden_dim,
            "num_layers": v_num_layers,
            "num_heads": v_num_heads,
            "head_dim": v_head_dim,
            "intermediate_size": v_intermediate_size,
            "max_position_embeddings": v_max_pos,
            "rope_theta": v_rope_theta,
            "sliding_window": v_sliding_window,
            "dropout": v_dropout,
            "use_bias": v_use_bias,
            "use_rms_norm": v_use_rms_norm,
            "use_flash_attention": v_use_flash,
            "use_rope": v_use_rope,
            "use_gated_activation": v_use_gated,
            "layer_norm_eps": v_layer_norm_eps,
            "rms_norm_eps": v_rms_norm_eps,
        }
    
    def check_wellformedness(self, gene: Gene) -> VerificationResult:
        solver = Solver()
        x = self.encode_gene(gene)
        
        solver.add(x["vocab_dim"] > 0)
        solver.add(x["hidden_dim"] > 0)
        solver.add(x["num_layers"] > 0)
        solver.add(x["num_heads"] > 0)
        solver.add(x["head_dim"] > 0)
        solver.add(x["intermediate_size"] > 0)
        
        solver.add(x["hidden_dim"] % x["num_heads"] == 0)
        solver.add(x["head_dim"] == x["hidden_dim"] / x["num_heads"])
        
        solver.add(x["layer_norm_eps"] > 0)
        solver.add(x["rms_norm_eps"] > 0)
        solver.add(x["dropout"] >= 0)
        solver.add(x["dropout"] < 1)
        
        check_result = solver.check()
        is_valid = check_result == sat
        
        model = None
        if is_valid:
            model_obj = solver.model()
            model = {d.name(): model_obj[d] for d in model_obj.decls()}
        
        return VerificationResult(
            is_valid=is_valid,
            model=model,
            constraints_checked=["positive_dims", "divisibility", "norm_eps", "dropout_range"],
            errors=[] if is_valid else ["Gene failed wellformedness check"]
        )
    
    def check_parameter_bounds(self, gene: Gene, max_params: int = 500_000_000) -> VerificationResult:
        solver = Optimize()
        x = self.encode_gene(gene)
        
        param_expr = (
            x["vocab_dim"] * x["hidden_dim"] +
            x["num_layers"] * (4 * x["hidden_dim"] * x["hidden_dim"] + 
                            2 * x["hidden_dim"] * x["intermediate_size"] +
                            x["intermediate_size"] * x["hidden_dim"]) +
            x["hidden_dim"] * x["vocab_dim"]
        )
        
        solver.add(param_expr <= max_params)
        
        check_result = solver.check()
        is_valid = check_result == sat
        
        model = None
        if is_valid:
            model_obj = solver.model()
            model = {d.name(): model_obj[d] for d in model_obj.decls()}
        
        return VerificationResult(
            is_valid=is_valid,
            model=model,
            constraints_checked=["parameter_count"],
            errors=[] if is_valid else [f"Gene exceeds {max_params} parameters"]
        )
    
    def check_memory_fit(self, gene: Gene, max_memory_bytes: int = 8_000_000_000) -> VerificationResult:
        solver = Optimize()
        x = self.encode_gene(gene)
        
        param_bytes = 2
        memory_expr = (
            (
                x["vocab_dim"] * x["hidden_dim"] +
                x["num_layers"] * (4 * x["hidden_dim"] * x["hidden_dim"] +
                                2 * x["hidden_dim"] * x["intermediate_size"] +
                                x["intermediate_size"] * x["hidden_dim"]) +
                x["hidden_dim"] * x["vocab_dim"]
            ) * param_bytes +
            x["hidden_dim"] * x["max_position_embeddings"] * x["num_layers"]
        )
        
        solver.add(memory_expr <= max_memory_bytes)
        
        check_result = solver.check()
        is_valid = check_result == sat
        
        model = None
        if is_valid:
            model_obj = solver.model()
            model = {d.name(): model_obj[d] for d in model_obj.decls()}
        
        return VerificationResult(
            is_valid=is_valid,
            model=model,
            constraints_checked=["memory_fit"],
            errors=[] if is_valid else [f"Gene exceeds {max_memory_bytes} bytes"]
        )
    
    def check_attention_compatibility(self, gene: Gene) -> VerificationResult:
        solver = Solver()
        x = self.encode_gene(gene)
        
        has_flash = x["use_flash_attention"] == True
        has_sliding = x["sliding_window"] > 0
        
        solver.add(Implies(has_flash, x["hidden_dim"] >= 128))
        solver.add(Implies(has_flash, x["head_dim"] >= 64))
        
        check_result = solver.check()
        is_valid = check_result == sat
        
        model = None
        if is_valid:
            model_obj = solver.model()
            model = {d.name(): model_obj[d] for d in model_obj.decls()}
        
        return VerificationResult(
            is_valid=is_valid,
            model=model,
            constraints_checked=["attention_compatibility"],
            errors=[] if is_valid else ["Attention configuration incompatible"]
        )
    
    def verify_all(self, gene: Gene, max_params: int = 500_000_000, max_memory: int = 8_000_000_000) -> VerificationResult:
        results = []
        
        results.append(self.check_wellformedness(gene))
        results.append(self.check_parameter_bounds(gene, max_params))
        results.append(self.check_memory_fit(gene, max_memory))
        results.append(self.check_attention_compatibility(gene))
        
        all_valid = all(r.is_valid for r in results)
        all_constraints = []
        all_errors = []
        for r in results:
            all_constraints.extend(r.constraints_checked)
            all_errors.extend(r.errors)
        
        return VerificationResult(
            is_valid=all_valid,
            model=None,
            constraints_checked=all_constraints,
            errors=all_errors
        )
    
    def score_gene(self, gene: Gene) -> float:
        is_valid, errors = gene.validate()
        if not is_valid:
            return 0.0
        
        verification = self.verify_all(gene)
        if not verification.is_valid:
            return 0.0
        
        param_count = gene.compute_params()
        score = min(param_count / 100_000_000, 1.0) * 0.5
        
        if gene.use_rope:
            score += 0.1
        if gene.use_flash_attention:
            score += 0.1
        if gene.use_rms_norm:
            score += 0.1
        
        if gene.hidden_act.value == "gelu":
            score += 0.1
        if "flash" in [a.value for a in gene.attention_types]:
            score += 0.1
        
        return score