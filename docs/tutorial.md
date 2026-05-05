# ArchGene Tutorial

Step-by-step guides for common use cases.

## Tutorial 1: Evaluate Your First Architecture

### Goal
Learn the basics by evaluating the default architecture.

```bash
# Step 1: Run default evaluation
python main.py evaluate
```

You should see:
- Gene table with architecture details
- Score: 0.394
- Parameters: ~19M
- Memory: 0.04 GB

### Understanding the Score

The score 0.394 means:
- **0.0-0.4**: Basic — passes validation but uses few best practices
- **0.5-0.7**: Good — uses some optimizations
- **0.8-1.0**: Excellent — uses RoPE, Flash attention, RMS norm, GELU

---

## Tutorial 2: Evaluate Different Sizes

### Goal
Compare architectures of different sizes.

**Small model (fast training, less accurate)**
```bash
python main.py evaluate -d 256 -l 2 -n 4 --save -t "small model"
```

**Medium model (balanced)**
```bash
python main.py evaluate -d 512 -l 4 -n 8 --save -t "medium model"
```

**Large model (most accurate, slower)**
```bash
python main.py evaluate -d 1024 -l 8 -n 16 -i 4096 --save -t "large model"
```

### View Comparison
```bash
python main.py history
```

---

## Tutorial 3: Find the Best Configuration

### Goal
Iterate to find an architecture that passes verification with a good score.

**Step 1: Test without saving**
```bash
python main.py evaluate -d 768 -l 6 -n 12
```

**Step 2: If score is 0, check errors**
```bash
python main.py evaluate -d 768 -n 12
# Error: hidden_dim (768) must be divisible by num_heads (12)
# Fix: Change -n to divisor of 768 (e.g., 12, 8, 6)
```

**Step 3: Save successful configs**
```bash
python main.py evaluate -d 768 -l 6 -n 12 --save -t "768 layers"
```

---

## Tutorial 4: Use Config Files

### Goal
Save and reuse complex configurations.

**Step 1: Create config file**
```json
{
  "vocab_dim": 32000,
  "hidden_dim": 768,
  "num_layers": 6,
  "num_heads": 12,
  "intermediate_size": 3072,
  "max_position_embeddings": 4096,
  "hidden_act": "gelu"
}
```

Save as `my_model.json`

**Step 2: Evaluate from config**
```bash
python main.py evaluate-file my_model.json
```

**Step 3: Modify and re-run**
Edit `my_model.json`, then:
```bash
python main.py evaluate-file my_model.json
```

---

## Tutorial 5: Verify Architecture Properties

### Goal
Formally verify architecture constraints.

**Basic verification**
```bash
python main.py verify
```

**Detailed verification**
```bash
python main.py verify --verbose
```

The verifier checks:
- ✓ `positive_dims` — All dimensions > 0
- ✓ `divisibility` — hidden_dim % num_heads == 0
- ✓ `parameter_count` — < 500M parameters
- ✓ `memory_fit` — Fits in 8GB memory
- ✓ `attention_compatibility` — Flash attention valid

---

## Tutorial 6: Visualize Architecture

### Goal
See architecture as diagram.

**ASCII diagram**
```bash
python main.py visualize
```

**Mermaid diagram (for docs)**
```bash
python main.py visualize --format mermaid
```

**JSON schema (for programmatic use)**
```bash
python main.py visualize --format json
```

---

## Tutorial 7: Export and Use

### Goal
Export architecture for training.

**Export to PyTorch**
```bash
python main.py export -f pytorch
# Output: exports/model.pt
```

**Export config**
```bash
python main.py export -f config
# Output: exports/config.json
```

**Run benchmark**
```bash
python main.py bench
```

Expected output:
```
Time per iteration: ~40 ms
Tokens/sec: ~2500
```

---

## Tutorial 8: Debug Failed Validation

### Goal
Fix architectures that score 0.

**Common errors:**

1. **Divisibility error**
   ```
   hidden_dim (300) must be divisible by num_heads (8)
   ```
   Fix: Use `hidden_dim` divisible by `num_heads`

2. **Too many parameters**
   ```
   Gene exceeds 500000000 parameters
   ```
   Fix: Reduce hidden_dim, num_layers, or intermediate_size

3. **Memory exceeded**
   ```
   Gene exceeds 8000000000 bytes
   ```
   Fix: Reduce max_position_embeddings or hidden_dim

**Quick diagnostic**
```bash
# Test each parameter individually
python main.py evaluate -d 512  # test hidden only
python main.py evaluate -l 8      # test layers only
```

---

## Common Patterns

### Pattern 1: GPT-style
```bash
python main.py evaluate -v 50257 -d 768 -l 12 -n 12 -i 3072
```

### Pattern 2: Small efficient
```bash
python main.py evaluate -v 10000 -d 256 -l 4 -n 4 -i 1024
```

### Pattern 3: Research baseline
```bash
python main.py evaluate -v 4096 -d 512 -l 6 -n 8 -i 2048
```

---

## Next Steps

- Try different architectures
- Save promising ones to history
- Export for training experiments
- Report issues at: https://github.com/your-repo/archgene/issues