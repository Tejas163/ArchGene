import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.archgene.gene_schema import Gene, ActivationType, AttentionType, PoolingType
from src.archgene.verifier import Verifier
from src.archgene.evaluation import Evaluator


class TestGeneSchema:
    def test_gene_defaults(self):
        gene = Gene()
        assert gene.vocab_dim == 4096
        assert gene.hidden_dim == 512
        assert gene.num_layers == 4
        assert gene.num_heads == 8

    def test_gene_validation_valid(self):
        gene = Gene(hidden_dim=512, num_heads=8)
        is_valid, errors = gene.validate()
        assert is_valid
        assert len(errors) == 0

    def test_gene_validation_invalid_divisibility(self):
        gene = Gene(hidden_dim=300, num_heads=8)
        is_valid, errors = gene.validate()
        assert not is_valid
        assert any("divisible" in e for e in errors)

    def test_gene_validation_invalid_intermediate_size(self):
        gene = Gene(intermediate_size=0)
        is_valid, errors = gene.validate()
        assert not is_valid
        assert any("intermediate" in e.lower() for e in errors)

    def test_gene_to_dict(self):
        gene = Gene()
        d = gene.to_dict()
        assert d["vocab_dim"] == 4096
        assert d["hidden_dim"] == 512

    def test_gene_from_dict(self):
        d = {"vocab_dim": 4096, "hidden_dim": 512, "num_layers": 4, "num_heads": 8}
        gene = Gene.from_dict(d)
        assert gene.vocab_dim == 4096
        assert gene.hidden_dim == 512

    def test_gene_compute_params(self):
        gene = Gene()
        params = gene.compute_params()
        assert params > 0
        assert params < 100_000_000

    def test_gene_compute_memory(self):
        gene = Gene()
        memory = gene.compute_memory()
        assert memory > 0

    def test_gene_kv_heads_default(self):
        gene = Gene()
        assert gene.kv_heads == gene.num_heads

    def test_gene_kv_heads_gqa(self):
        gene = Gene(num_kv_heads=4, num_heads=8)
        assert gene.kv_heads == 4


class TestVerifier:
    def test_verify_valid_gene(self):
        verifier = Verifier()
        gene = Gene()
        result = verifier.verify_all(gene)
        assert result.is_valid
    
    def test_verify_runs(self):
        verifier = Verifier()
        gene = Gene()
        result = verifier.verify_all(gene)
        assert len(result.constraints_checked) > 0


class TestEvaluator:
    def test_evaluate_valid_gene(self):
        evaluator = Evaluator()
        gene = Gene()
        score = evaluator.evaluate(gene)
        assert score.score > 0

    def test_evaluate_invalid_gene(self):
        evaluator = Evaluator()
        gene = Gene(hidden_dim=300, num_heads=8)
        score = evaluator.evaluate(gene)
        assert score.score == 0


class TestCLI:
    def test_version_command(self):
        from click.testing import CliRunner
        from cli.main import version
        
        runner = CliRunner()
        result = runner.invoke(version)
        assert result.exit_code == 0
        assert "0.4.0" in result.output

    def test_evaluate_command_help(self):
        from click.testing import CliRunner
        from cli.main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.output
    
    def test_verify_command(self):
        from click.testing import CliRunner
        from cli.main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["verify"])
        assert result.exit_code == 0


class TestDesignSession:
    def test_design_logic_produces_valid_gene(self):
        from src.archgene.design_session import DesignSession
        from src.archgene.gene_schema import Gene

        s = DesignSession()
        s.answers.use_case = "inference"
        s.answers.budget_tier = "mid"
        s.answers.constraints = ["low_latency"]
        s.answers.arch_family = "any"
        s.answers.target_params = "1b-7b"
        s.answers.context_length = 8192

        gene = s._design_architecture()
        assert isinstance(gene, Gene)
        assert gene.hidden_dim > 0
        assert gene.num_layers > 0
        assert gene.num_heads > 0

        v = s.verifier.verify_all(gene)
        assert v.is_valid

    def test_design_edge_case_produces_valid_gene(self):
        from src.archgene.design_session import DesignSession
        from src.archgene.gene_schema import Gene

        s = DesignSession()
        s.answers.use_case = "edge"
        s.answers.budget_tier = "free"
        s.answers.constraints = ["low_memory", "quantized"]
        s.answers.arch_family = "linear"
        s.answers.target_params = "under1b"
        s.answers.context_length = 2048

        gene = s._design_architecture()
        assert isinstance(gene, Gene)
        assert gene.compute_params() < 2_000_000_000

        v = s.verifier.verify_all(gene)
        assert v.is_valid

    def test_design_research_produces_valid_gene(self):
        from src.archgene.design_session import DesignSession

        s = DesignSession()
        s.answers.use_case = "research"
        s.answers.budget_tier = "high"
        s.answers.constraints = ["long_context"]
        s.answers.arch_family = "any"
        s.answers.target_params = "auto"
        s.answers.context_length = 131072

        gene = s._design_architecture()
        assert gene.max_position_embeddings >= 32768

        v = s.verifier.verify_all(gene)
        assert v.is_valid


class TestKernelGenerator:
    def test_generates_all_files(self):
        from src.archgene.kernel_generator import KernelGenerator
        from src.archgene.gene_schema import Gene
        import shutil

        gene = Gene(hidden_dim=512, num_layers=2, num_heads=8)
        gen = KernelGenerator()
        files = gen.generate(gene, "test_gen_files")

        assert files.model_py.exists()
        assert files.config_json.exists()
        assert files.train_py.exists()
        assert files.requirements_txt.exists()

        code = files.model_py.read_text()
        assert "class Model(nn.Module)" in code
        assert "class Attention(nn.Module)" in code
        assert "class MLP(nn.Module)" in code
        assert "class TransformerBlock(nn.Module)" in code

        shutil.rmtree("test_gen_files", ignore_errors=True)

    def test_generated_model_is_importable(self):
        from src.archgene.kernel_generator import KernelGenerator
        from src.archgene.gene_schema import Gene
        import sys, torch, shutil

        gene = Gene(hidden_dim=256, num_layers=2, num_heads=4, vocab_dim=2048)
        gen = KernelGenerator()
        files = gen.generate(gene, "test_gen_import")

        sys.path.insert(0, "test_gen_import")
        import model as _m
        m = _m.Model()
        assert m.count_params() > 0

        x = torch.randint(0, 100, (1, 16))
        logits = m(x)
        assert logits.shape == (1, 16, 2048)

        sys.path.pop(0)
        if "model" in sys.modules:
            del sys.modules["model"]
        shutil.rmtree("test_gen_import", ignore_errors=True)

    def test_generated_config_matches_gene(self):
        from src.archgene.kernel_generator import KernelGenerator
        from src.archgene.gene_schema import Gene
        import json, shutil

        gene = Gene(hidden_dim=768, num_layers=12, num_heads=12, vocab_dim=32000)
        gen = KernelGenerator()
        files = gen.generate(gene, "test_gen_config")

        config = json.loads(files.config_json.read_text())
        assert config["hidden_size"] == 768
        assert config["num_hidden_layers"] == 12
        assert config["num_attention_heads"] == 12
        assert config["vocab_size"] == 32000
        assert config["architectures"] == ["Model"]

        shutil.rmtree("test_gen_config", ignore_errors=True)

    def test_generated_model_with_gqa_rope_gated_rmsnorm(self):
        from src.archgene.kernel_generator import KernelGenerator
        from src.archgene.gene_schema import Gene
        import sys, torch, json, shutil

        gene = Gene(
            hidden_dim=512, num_layers=2, num_heads=8, head_dim=64,
            intermediate_size=2048, vocab_dim=4096,
            use_rope=True, use_gated_activation=True, use_rms_norm=True,
            num_kv_heads=4, dropout=0.0,
        )
        gen = KernelGenerator()
        files = gen.generate(gene, "test_gen_adv")
        sys.path.insert(0, "test_gen_adv")
        import model as _m2
        m = _m2.Model()
        x = torch.randint(0, 100, (1, 32))
        logits = m(x)
        assert logits.shape == (1, 32, 4096)

        config = json.loads(files.config_json.read_text())
        assert config["num_key_value_heads"] == 4

        src = files.model_py.read_text()
        assert "class RotaryEmbedding" in src
        assert "class RMSNorm" in src
        assert "self.gate_proj" in src

        sys.path.pop(0)
        if "model" in sys.modules:
            del sys.modules["model"]
        shutil.rmtree("test_gen_adv", ignore_errors=True)

    def test_generated_model_with_baseline_config(self):
        from src.archgene.kernel_generator import KernelGenerator
        from src.archgene.gene_schema import Gene
        import sys, torch, shutil

        gene = Gene(
            hidden_dim=256, num_layers=2, num_heads=4, vocab_dim=2048,
            use_rope=False, use_gated_activation=False, use_rms_norm=False,
        )
        gen = KernelGenerator()
        files = gen.generate(gene, "test_gen_base")
        sys.path.insert(0, "test_gen_base")
        import model as _m3
        m = _m3.Model()
        x = torch.randint(0, 100, (1, 16))
        logits = m(x)
        assert logits.shape == (1, 16, 2048)

        src = files.model_py.read_text()
        assert "class RotaryEmbedding" not in src
        assert "nn.LayerNorm" in src

        sys.path.pop(0)
        if "model" in sys.modules:
            del sys.modules["model"]
        shutil.rmtree("test_gen_base", ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])