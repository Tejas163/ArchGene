"""ArchGene Web UI — Streamlit app for architecture design session."""

import streamlit as st

try:
    from archgene.design_session import design_from_answers
    from archgene.kernel_generator import KernelGenerator
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from archgene.design_session import design_from_answers
    from archgene.kernel_generator import KernelGenerator

st.set_page_config(
    page_title="ArchGene — Design LLM Architectures",
    page_icon="",
    layout="centered",
)

st.title("ArchGene Architecture Designer")
st.markdown(
    "Answer a few questions and get a **verified, cost-estimated, runnable** "
    "LLM architecture — before you spend on training."
)

if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.result = None
    st.session_state.gene = None

USE_CASES = {
    "Research / experimentation": "research",
    "Production inference": "inference",
    "Fine-tuning existing models": "finetuning",
    "Edge / mobile deployment": "edge",
}
BUDGET_TIERS = {
    "Free / low (<$100) — hobby": "free",
    "Mid ($100–$1K) — serious fine-tuning": "mid",
    "High ($1K–$10K) — production training": "high",
    "Enterprise ($10K+) — large-scale": "enterprise",
}
ARCH_FAMILIES = {
    "Transformer (proven, general-purpose)": "transformer",
    "Mixture of Experts (efficient scaling)": "moe",
    "Linear / State Space (long context)": "linear",
    "Any — pick the best for my needs": "any",
}
PARAM_RANGES = {
    "<1B — lightweight, fast": "under1b",
    "1B–7B — balanced": "1b-7b",
    "7B–30B — capable, higher cost": "7b-30b",
    "30B+ — maximum capability": "30b+",
    "Not sure — let the system decide": "auto",
}
CONTEXT_LENGTHS = {
    "2K (standard)": 2048,
    "8K (recommended)": 8192,
    "32K (long documents)": 32768,
    "128K+ (very long context)": 131072,
}
CONSTRAINT_OPTIONS = {
    "Long context (>32K tokens)": "long_context",
    "Low latency (<50ms per token)": "low_latency",
    "Low memory (<8GB VRAM)": "low_memory",
    "Quantization support": "quantized",
}

questions = [
    {
        "title": "What's your primary use case?",
        "key": "use_case",
        "type": "radio",
        "options": list(USE_CASES.keys()),
        "help": "This determines priority (efficiency vs capability) and device targets.",
    },
    {
        "title": "What's your budget range?",
        "key": "budget_tier",
        "type": "radio",
        "options": list(BUDGET_TIERS.keys()),
        "help": "Budget sets limits on GPU hours, parameter count, and memory.",
    },
    {
        "title": "Any specific constraints?",
        "key": "constraints",
        "type": "multiselect",
        "options": list(CONSTRAINT_OPTIONS.keys()),
        "help": "These reshape the architecture (e.g., long context → higher RoPE base freq).",
    },
    {
        "title": "Preferred architecture family?",
        "key": "arch_family",
        "type": "radio",
        "options": list(ARCH_FAMILIES.keys()),
        "help": "Transformer is proven. MoE scales efficiently. Linear models handle very long context.",
    },
    {
        "title": "Target parameter count?",
        "key": "target_params",
        "type": "radio",
        "options": list(PARAM_RANGES.keys()),
        "help": "More parameters → more capability but higher cost.",
    },
    {
        "title": "Maximum context length?",
        "key": "context_length",
        "type": "radio",
        "options": list(CONTEXT_LENGTHS.keys()),
        "help": "Longer context needs more memory and influences architecture choices.",
    },
]

if st.session_state.step < len(questions):
    q = questions[st.session_state.step]
    with st.container():
        st.subheader(f"Question {st.session_state.step + 1} of {len(questions)}")
        st.caption(q.get("help", ""))

        if q["type"] == "radio":
            choice = st.radio(
                q["title"],
                q["options"],
                key=f"q_{st.session_state.step}",
                index=None,
            )
        elif q["type"] == "multiselect":
            choice = st.multiselect(
                q["title"],
                q["options"],
                key=f"q_{st.session_state.step}",
            )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Back", disabled=st.session_state.step == 0):
                st.session_state.step -= 1
                st.rerun()
        with col2:
            disabled = not choice if q["type"] != "multiselect" else False
            if st.button("Next", disabled=disabled, type="primary"):
                st.session_state.answers[q["key"]] = choice
                st.session_state.step += 1
                st.rerun()

elif st.session_state.step == len(questions):
    with st.spinner("Designing your architecture..."):
        answers = st.session_state.answers
        mapped = {
            "use_case": USE_CASES.get(answers.get("use_case", ""), "research"),
            "budget_tier": BUDGET_TIERS.get(answers.get("budget_tier", ""), "mid"),
            "arch_family": ARCH_FAMILIES.get(answers.get("arch_family", ""), "any"),
            "target_params": PARAM_RANGES.get(answers.get("target_params", ""), "1b-7b"),
            "context_length": CONTEXT_LENGTHS.get(answers.get("context_length", ""), 8192),
        }
        raw_constraints = answers.get("constraints", [])
        mapped["constraints"] = [CONSTRAINT_OPTIONS[c] for c in raw_constraints if c in CONSTRAINT_OPTIONS]

        result = design_from_answers(**mapped)
        st.session_state.result = result
        st.session_state.gene = result["gene"]
        st.session_state.step += 1
    st.rerun()

elif st.session_state.step == len(questions) + 1:
    r = st.session_state.result
    gene = st.session_state.gene

    st.success("**Architecture design complete!**")

    valid = r["verified"]
    status_icon = "" if valid else ""
    st.metric("Z3 Verification", f"{status_icon} {'PASS' if valid else 'FAIL'}")
    cols = st.columns(3)
    cols[0].metric("Hidden Dim", gene.hidden_dim)
    cols[0].metric("Layers", gene.num_layers)
    cols[1].metric("Attention Heads", gene.num_heads)
    cols[1].metric("KV Heads", gene.kv_heads)
    cols[2].metric("Parameters", f"{r['params']:,}")
    cols[2].metric("VRAM", f"{r['vram_gb']:.1f} GB")

    st.subheader("Cost Estimates")
    cost_cols = st.columns(3)
    cost_cols[0].metric("Training", f"{r['training_hours']:.0f} GPU hrs")
    cost_cols[1].metric("Train Cost", f"${r['training_cost_per_1k']:.2f}/1K tok")
    cost_cols[2].metric("Inference", f"${r['inference_cost_per_1m']:.2f}/M tok")

    st.subheader("Architecture Details")
    st.code(
        f"Family: {r['arch_family']}\n"
        f"Use case: {r['use_case']}\n"
        f"Budget: {r['budget_tier']}\n"
        f"Context length: {r['context_length']:,}\n"
        f"Latency: {r['latency_ms']:.2f} ms/token\n"
        f"Score: {r['score']:.2f}",
        language="text",
    )

    if "constraints" in r and r["constraints"]:
        st.caption(f"Applied constraints: {', '.join(r['constraints'])}")

    st.subheader("Generated Code")
    st.markdown(
        "Download a runnable PyTorch package with RoPE, GQA, SwiGLU, and RMSNorm:"
    )
    zip_bytes = KernelGenerator.generate_zip(gene)
    st.download_button(
        label="Download archgene_generated.zip",
        data=zip_bytes,
        file_name="archgene_generated.zip",
        mime="application/zip",
        type="primary",
    )

    with st.expander("Preview model.py"):
        files = KernelGenerator.generate_files_dict(gene)
        st.code(files.get("model.py", ""), language="python")
    with st.expander("Preview train.py"):
        st.code(files.get("train.py", ""), language="python")
    with st.expander("Preview config.json"):
        st.code(files.get("config.json", ""), language="json")

    if st.button("Start over — design another architecture"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
