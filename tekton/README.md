# Tekton CI/CD for ArchGene

This directory contains Tekton pipelines for continuous integration.

## Files

```
tekton/
├── tasks/
│   ├── lint.yaml         # Syntax check and lint
│   ├── test.yaml        # pytest test suite
│   └── cli-test.yaml    # CLI smoke tests
├── pipelines/
│   └── ci.yaml         # Main CI pipeline
└── runs/
    └── ci-run.yaml     # Pipeline run definition
```

## Quick Start (Local)

### Install Tekton (if not installed)
```bash
kubectl apply -f https://storage.googleapis.com/tekton-releases/pipeline/latest/release.yaml
```

### Apply tasks and pipeline
```bash
kubectl apply -f tekton/tasks/
kubectl apply -f tekton/pipelines/
```

### Run CI pipeline
```bash
kubectl apply -f tekton/runs/ci-run.yaml
```

### Check status
```bash
tkn pipelinerun logs archgene-ci-run -f
```

## Pipeline Details

### Task: lint
- Python syntax check with py_compile
- Runs before tests

### Task: test  
- Runs pytest test suite
- 14 test cases

### Task: cli-test
- Smoke tests for CLI commands
- version, evaluate, verify, visualize

### Pipeline: archgene-ci
Runs in order: lint → test → cli-test

## Local Development (Without Tekton)

Run locally without Kubernetes:

```bash
# Lint
python -m py_compile main.py core/*.py

# Tests
pip install pytest
pytest tests/ -v

# CLI smoke test
python main.py version
python main.py evaluate
python main.py verify
```

## Environment Variables

| Variable | Description | Default |
|----------|------------|---------|
| PYTHONPATH | Project root | . |