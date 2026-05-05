"""
Test LLM-powered architecture analysis with smolagents.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.smolagents_system import (
    ArchitectAgent, 
    generate_gene, 
    generate_architecture_idea,
    LLM_MODEL,
    call_llm,
)


def test_llm_reasoning():
    """Test LLM reasoning for architecture analysis."""
    print("Testing LLM-powered architecture analysis...")
    print("=" * 50)
    
    # Generate a gene
    print("Generating gene...")
    gene_dict = generate_gene(vocab_dim=50257, hidden_dim=512, num_layers=6)
    print(f"Gene: vocab={gene_dict['vocab_dim']}, hidden={gene_dict['hidden_dim']}, layers={gene_dict['num_layers']}")
    print()
    
    # Generate architecture idea
    print("Generating architecture idea...")
    idea = generate_architecture_idea(gene_dict)
    print(idea)
    print()
    
    # Use LLM for analysis
    print("=" * 50)
    print(f"Calling LLM ({LLM_MODEL}) for reasoning...")
    print()
    
    task = f"""Analyze this transformer architecture:
- vocab_dim: {gene_dict['vocab_dim']}
- hidden_dim: {gene_dict['hidden_dim']}
- num_layers: {gene_dict['num_layers']}
- num_heads: {gene_dict['num_heads']}
- intermediate_size: {gene_dict['intermediate_size']}

Tell me:
1. What are the strengths?
2. What are the potential weaknesses?
3. What improvements would you suggest?
"""
    
    # Call LLM directly
    try:
        result = call_llm(task)
        print("LLM Response:")
        print("-" * 50)
        print(result)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_llm_reasoning()