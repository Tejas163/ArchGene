# Contributing to ArchGene

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/archgene.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest tests/ -v`

## Development Workflow

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_core.py -v

# With coverage
pytest tests/ -v --cov=core --cov-report=term-missing
```

### Running Locally
```bash
# Lint check
python -m py_compile main.py core/*.py

# CLI smoke test
python main.py version
python main.py evaluate
python main.py verify
```

### Building Documentation
```bash
# Documentation is in README.md and docs/
# Tutorial: docs/tutorial.md
```

## Code Style

- Use Python 3.12+
- Follow PEP 8
- Use descriptive variable names
- Add docstrings to new functions
- Keep functions focused (single responsibility)

## Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance

Example: `feat: add --save flag to evaluate command`

## Pull Request Process

1. Update tests if adding new features
2. Update README.md if changing CLI
3. Ensure all tests pass: `pytest tests/ -v`
4. Run CLI smoke test: `python main.py verify`
5. Submit PR with clear description

## Issue Reporting

Use the issue templates in `.github/ISSUE_TEMPLATE/`:
- `bug.md` for bugs
- `feature.md` for features

Include:
- ArchGene version: `python main.py version`
- Python version: `python --version`
- Steps to reproduce
- Expected vs actual behavior

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the work, not the person

---

## Quick Reference

| Command | Description |
|---------|------------|
| `pytest tests/ -v` | Run all tests |
| `python main.py evaluate` | Evaluate architecture |
| `python main.py verify` | Run Z3 verification |
| `python main.py visualize` | Visualize architecture |