# Changelog

All notable changes to ArchGene are documented here.

## [0.1.0] - 2026-05-05

### Added
- Initial release
- Gene schema for architecture encoding
- Z3/CVC5 verification layer
- Evaluation API with scoring
- CLI commands: evaluate, verify, visualize, export, bench, history
- Config file support (evaluate-file)
- Save to history (--save, --notes)
- Rich console output
- Tutorial documentation

### Known Issues
- PyTorch download is large (~67MB) — consider lighter alternatives for evaluation-only
- No interactive playground — planned for future

---

## Migration Guide

### Upgrading from 0.0.x to 0.1.0

**New CLI usage:**
```bash
# Old (0.0.x)
python main.py -d 512

# New (0.1.0)
python main.py evaluate -d 512
```

**Config format changed:**
```json
// Old
{"vocab": 4096, "hidden": 512}

// New
{"vocab_dim": 4096, "hidden_dim": 512}
```

**Migration steps:**
1. Update config files to use `_dim` suffix
2. Use subcommands: `evaluate`, `verify`, etc.
3. Use `--save` to track evaluations

### Future Migrations

When upgrading to 0.2.0:
1. Read CHANGELOG.md breaking changes
2. Run migration scripts if provided
3. Check config compatibility