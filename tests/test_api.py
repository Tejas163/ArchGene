import pytest
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["ARCHGENE_API_KEY"] = "test-key"

from fastapi.testclient import TestClient
from api_server import app


client = TestClient(app)

TEST_API_KEY = "test-key"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class TestAPIValidate:
    def test_validate_valid_gene(self):
        response = client.post("/validate", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 64,
            "intermediate_size": 2048,
        }, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_invalid_divisibility(self):
        response = client.post("/validate", json={
            "vocab_dim": 4096,
            "hidden_dim": 300,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 37,
            "intermediate_size": 2048,
        }, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_validate_missing_fields_uses_defaults(self):
        response = client.post("/validate", json={}, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    def test_validate_no_api_key(self):
        response = client.post("/validate", json={"vocab_dim": 4096})
        assert response.status_code == 401


class TestAPIVerify:
    def test_verify_valid_gene(self):
        response = client.post("/verify", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 64,
            "intermediate_size": 2048,
        }, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "verified" in data

    def test_verify_invalid_gene(self):
        response = client.post("/verify", json={
            "hidden_dim": 300,
            "num_heads": 8,
        }, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "verified" in data


class TestAPIEvaluate:
    def test_evaluate_valid_gene(self):
        response = client.post("/evaluate", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 64,
            "intermediate_size": 2048,
        }, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert data["score"] > 0


class TestAPIVisualize:
    def test_visualize_ascii(self):
        response = client.post("/visualize", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 64,
            "intermediate_size": 2048,
        }, params={"format": "ascii"}, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "visualization" in data

    def test_visualize_json(self):
        response = client.post("/visualize", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
            "num_layers": 4,
        }, params={"format": "json"}, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "visualization" in data

    def test_visualize_invalid_format(self):
        response = client.post("/visualize", json={
            "vocab_dim": 4096,
        }, params={"format": "invalid"}, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 400


class TestAPIStats:
    def test_stats_valid_gene(self):
        response = client.post("/stats", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 64,
            "intermediate_size": 2048,
        }, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "parameters" in data
        assert "memory_mb" in data
        assert data["parameters"] > 0

    def test_stats_invalid_gene_returns_400(self):
        response = client.post("/stats", json={
            "hidden_dim": 300,
            "num_heads": 8,
            "head_dim": 37,
        }, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 400


class TestAPIExport:
    def test_export_json(self):
        response = client.post("/export", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
        }, params={"format": "json"}, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "path" in data

    def test_export_pytorch(self):
        response = client.post("/export", json={
            "vocab_dim": 4096,
            "hidden_dim": 512,
        }, params={"format": "pytorch"}, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "path" in data


class TestAPIAuthentication:
    def test_invalid_api_key(self):
        response = client.post("/validate", json={"vocab_dim": 4096}, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_missing_api_key(self):
        response = client.post("/validate", json={"vocab_dim": 4096})
        assert response.status_code == 401


class TestAPIRateLimit:
    def test_rate_limit_exceeded(self):
        # Make many requests to trigger rate limit
        for _ in range(100):
            response = client.post("/validate", json={"vocab_dim": 4096}, headers={"X-API-Key": TEST_API_KEY})
        # 101st request should be rate limited
        response = client.post("/validate", json={"vocab_dim": 4096}, headers={"X-API-Key": TEST_API_KEY})
        assert response.status_code == 429


if __name__ == "__main__":
    pytest.main([__file__, "-v"])