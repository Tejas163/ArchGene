import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gene_schema import Gene, ActivationType, AttentionType, PoolingType
from core.verifier import Verifier
from core.evaluation import Evaluator


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
        from main import version
        
        runner = CliRunner()
        result = runner.invoke(version)
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_evaluate_command_help(self):
        from click.testing import CliRunner
        from main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Examples" in result.output
    
    def test_verify_command(self):
        from click.testing import CliRunner
        from main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["verify"])
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])