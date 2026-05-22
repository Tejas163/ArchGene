from typing import Optional
from dataclasses import dataclass
import json
from pathlib import Path

from .gene_schema import Gene
from .verifier import Verifier, VerificationResult


@dataclass
class EvaluationScore:
    gene: Gene
    score: float
    validation_errors: list[str]
    verification_errors: list[str]
    params: int
    memory_bytes: int


class Evaluator:
    def __init__(self, max_params: int = 500_000_000, max_memory: int = 8_000_000_000):
        self.verifier = Verifier()
        self.max_params = max_params
        self.max_memory = max_memory
    
    def evaluate(self, gene: Gene) -> EvaluationScore:
        is_valid, validation_errors = gene.validate()
        if not is_valid:
            return EvaluationScore(
                gene=gene,
                score=0.0,
                validation_errors=validation_errors,
                verification_errors=["Gene failed validation"],
                params=gene.compute_params(),
                memory_bytes=gene.compute_memory()
            )
        
        verification = self.verifier.verify_all(gene, self.max_params, self.max_memory)
        
        if not verification.is_valid:
            return EvaluationScore(
                gene=gene,
                score=0.0,
                validation_errors=[],
                verification_errors=verification.errors,
                params=gene.compute_params(),
                memory_bytes=gene.compute_memory()
            )
        
        score = self.verifier.score_gene(gene)
        
        return EvaluationScore(
            gene=gene,
            score=score,
            validation_errors=[],
            verification_errors=[],
            params=gene.compute_params(),
            memory_bytes=gene.compute_memory()
        )


@dataclass
class EvaluationRecord:
    gene: Gene
    score: float
    timestamp: str
    notes: str = ""


class EvaluationHistory:
    def __init__(self, path: Optional[str] = None):
        default_dir = Path.home() / ".archgene"
        default_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(path) if path else default_dir / "history.json"
        self.records: list[dict] = []
        self.load()
    
    def load(self):
        if self.path.exists():
            with open(self.path) as f:
                self.records = json.load(f)
    
    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)
    
    def add(self, record: EvaluationRecord):
        self.records.append({
            "gene": record.gene.to_dict(),
            "score": record.score,
            "timestamp": record.timestamp,
            "notes": record.notes
        })
        self.save()
    
    def get_all(self) -> list[EvaluationRecord]:
        return [
            EvaluationRecord(
                gene=Gene.from_dict(r["gene"]),
                score=r["score"],
                timestamp=r["timestamp"],
                notes=r.get("notes", "")
            )
            for r in self.records
        ]
    
    def get_top(self, n: int = 10) -> list[EvaluationRecord]:
        sorted_records = sorted(self.records, key=lambda r: r["score"], reverse=True)
        return [
            EvaluationRecord(
                gene=Gene.from_dict(r["gene"]),
                score=r["score"],
                timestamp=r["timestamp"],
                notes=r.get("notes", "")
            )
            for r in sorted_records[:n]
        ]


def evaluate_gene(gene: Gene) -> EvaluationScore:
    evaluator = Evaluator()
    return evaluator.evaluate(gene)