import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.archgene.gene_schema import Gene, ActivationType
from src.archgene.model_zoo import ModelZoo
from src.archgene.cost_estimator import CostEstimator
from src.archgene.benchmark_integration import BenchmarkIntegration
from src.archgene.verifier import Verifier


st.set_page_config(
    page_title="ArchGene - Find Your Architecture",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 ArchGene - Find Your Architecture")
st.markdown("**AI-powered architecture recommendations** based on your budget, GPU, and use case.")


def recommend_architecture(
    budget_hours: float,
    gpu: str,
    use_case: str,
    dataset_size: int = 1_000_000_000,
) -> list:
    """Recommend architectures based on constraints."""
    candidates = []
    
    for name in ModelZoo.list_all():
        gene = ModelZoo.get(name)
        est = CostEstimator.estimate_training(gene, gpu=gpu, training_tokens=dataset_size)
        
        if est.training_hours <= budget_hours * 1.5:
            score = 1.0
            if use_case == "research" and gene.num_layers >= 20:
                score += 0.2
            if use_case == "inference" and gene.hidden_dim <= 2048:
                score += 0.2
            if use_case == "fine-tuning" and 3000 <= gene.hidden_dim <= 8000:
                score += 0.3
                
            candidates.append({
                "name": name,
                "score": score,
                "training_hours": est.training_hours,
                "vram_gb": est.vram_gb,
                "cost": est.training_hours * CostEstimator.GPU_COST_PER_HOUR.get(gpu, 2.0),
                "params": gene.compute_params(),
            })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:5]


col1, col2 = st.columns([1, 2])

with col1:
    st.header("🎯 Your Requirements")
    
    use_case = st.selectbox(
        "What's your use case?",
        ["fine-tuning", "inference", "research", "experimentation"],
        index=0
    )
    
    budget = st.number_input(
        "Budget ($)",
        min_value=10,
        max_value=10000,
        value=100,
        step=10,
        help="How much can you spend on training?"
    )
    
    gpu = st.selectbox(
        "GPU Available",
        ["A100-40GB", "A100-80GB", "H100", "A10", "T4", "L40"],
        index=0
    )
    
    dataset_tokens = st.number_input(
        "Dataset size (tokens)",
        min_value=1_000_000,
        value=1_000_000_000,
        step=1_000_000_000,
        format="%d",
        help="How many training tokens?"
    )
    
    st.header("⚙️ Preferences")
    
    prefer_rope = st.checkbox("Prefer RoPE embeddings", value=True)
    prefer_flash = st.checkbox("Prefer Flash Attention", value=True)
    
    min_params = st.slider("Min parameters (M)", 0, 70, 0)
    max_params = st.slider("Max parameters (M)", 0, 70, 70)

with col2:
    st.header("📋 Recommendations")
    
    if st.button("🔍 Find My Architecture", type="primary"):
        budget_hours = budget / CostEstimator.GPU_COST_PER_HOUR.get(gpu, 2.0)
        
        candidates = recommend_architecture(budget_hours, gpu, use_case, dataset_tokens)
        
        if prefer_rope or prefer_flash:
            filtered = []
            for c in candidates:
                gene = ModelZoo.get(c["name"])
                if prefer_rope and not gene.use_rope:
                    continue
                if prefer_flash and not gene.use_flash_attention:
                    continue
                filtered.append(c)
            candidates = filtered
        
        if min_params > 0 or max_params < 70:
            filtered = []
            for c in candidates:
                params_m = c["params"] / 1e6
                if min_params <= params_m <= max_params:
                    filtered.append(c)
            candidates = filtered
        
        if candidates:
            for i, c in enumerate(candidates):
                gene = ModelZoo.get(c["name"])
                est = CostEstimator.full_estimate(gene, gpu=gpu)
                bench = BenchmarkIntegration.estimate_mmlu(gene)
                
                with st.expander(f"#{i+1} {c['name'].upper()} - Score: {c['score']:.1f}", expanded=i==0):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Parameters", f"{c['params']/1e6:.1f}M")
                        st.metric("VRAM", f"{c['vram_gb']:.1f} GB")
                        st.metric("Training Cost", f"${c['cost']:.0f}")
                    
                    with col_b:
                        st.metric("Training Time", f"{c['training_hours']:.0f} hours")
                        st.metric("MMLU Est.", f"{bench.mmlu:.1%}")
                        st.metric("/hour", f"${CostEstimator.GPU_COST_PER_HOUR.get(gpu, 2.0)}")
                    
                    if use_case == "fine-tuning":
                        st.success("💡 Best for fine-tuning! LowerVRAM means you can use larger batches.")
                    elif use_case == "inference":
                        st.success("💡 Best for inference! Good throughput on your GPU.")
                    else:
                        st.success("💡 Best for research! Good balance of capability and cost.")
                        
                    st.caption(f"Hidden: {gene.hidden_dim}, Layers: {gene.num_layers}, Heads: {gene.num_heads}")
        else:
            st.warning("No architectures match your criteria. Try adjusting budget or GPU.")


st.divider()

st.header("📊 Model Zoo Overview")

zoo_df = []
for name in ModelZoo.list_all():
    gene = ModelZoo.get(name)
    est = CostEstimator.full_estimate(gene, gpu="A100-40GB")
    zoo_df.append({
        "Name": name,
        "Params": f"{gene.compute_params()/1e6:.1f}M",
        "VRAM": f"{est.vram_gb:.1f}GB",
        "Training": f"{est.training_hours:.0f}h",
        "Cost": f"${est.training_hours * 2:.0f}",
    })

if zoo_df:
    st.dataframe(pd.DataFrame(zoo_df), use_container_width=True)


with st.sidebar:
    st.header("About")
    st.markdown("""
    **ArchGene** helps you find the right LLM architecture for your budget and use case.
    
    Just input:
    - Your budget
    - Available GPU
    - Use case
    
    Get architecture recommendations with real cost estimates.
    """)
    
    st.header("Quick Links")
    st.link("CLI Documentation", "/main.py")
    st.link("GitHub", "https://github.com/Tejas163/ArchGene")


if __name__ == "__main__":
    pass