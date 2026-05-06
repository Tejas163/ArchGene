import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.archgene.gene_schema import Gene, ActivationType, AttentionType, PoolingType
from src.archgene.visualization import ArchitectureVisualizer
from src.archgene.verifier import Verifier


st.set_page_config(page_title="ArchGene Visualizer", layout="wide")
st.title("ArchGene - Neural Architecture Visualizer")


def main():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Gene Parameters")
        
        vocab_dim = st.slider("vocab_dim", 256, 65536, 4096, step=256)
        hidden_dim = st.slider("hidden_dim", 64, 2048, 512, step=64)
        num_layers = st.slider("num_layers", 1, 32, 4)
        num_heads = st.slider("num_heads", 1, 32, 8)
        head_dim = st.slider("head_dim", 16, 256, 64, step=16)
        intermediate_size = st.slider("intermediate_size", 256, 8192, 2048, step=256)
        max_position_embeddings = st.slider("max_position_embeddings", 256, 8192, 2048, step=256)
        rope_theta = st.number_input("rope_theta", 1000.0, 100000.0, 10000.0, step=1000.0)
        use_bias = st.checkbox("use_bias", value=True)
        
        st.subheader("Encoder Settings")
        attention_types = st.multiselect(
            "attention_types",
            options=[a.value for a in AttentionType],
            default=[AttentionType.FULL.value]
        )
        hidden_act = st.selectbox(
            "hidden_act",
            options=[a.value for a in ActivationType],
            index=1
        )
        pooling_type = st.selectbox(
            "pooling_type",
            options=[a.value for a in PoolingType],
            index=0
        )
        
        st.subheader("Advanced Settings")
        layer_norm_eps = st.number_input("layer_norm_eps", 1e-6, 1e-3, 1e-5, format="%.0e")
        rms_norm_eps = st.number_input("rms_norm_eps", 1e-7, 1e-4, 1e-6, format="%.0e")
        use_rms_norm = st.checkbox("use_rms_norm", value=True)
        use_flash_attention = st.checkbox("use_flash_attention", value=False)
        sliding_window = st.slider("sliding_window", 512, 8192, 4096, step=512)
        dropout = st.slider("dropout", 0.0, 0.9, 0.0, step=0.1)
        use_rope = st.checkbox("use_rope", value=True)
        use_gated_activation = st.checkbox("use_gated_activation", value=False)
        
        gene = Gene(
            vocab_dim=vocab_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            use_bias=use_bias,
            attention_types=[AttentionType(a) for a in attention_types],
            hidden_act=ActivationType(hidden_act),
            pooling_type=PoolingType(pooling_type),
            layer_norm_eps=layer_norm_eps,
            rms_norm_eps=rms_norm_eps,
            use_rms_norm=use_rms_norm,
            use_flash_attention=use_flash_attention,
            sliding_window=sliding_window,
            dropout=dropout,
            use_rope=use_rope,
            use_gated_activation=use_gated_activation,
        )
    
    with col2:
        st.header("Visualization")
        
        viz_format = st.radio("Format", ["ascii", "mermaid", "json"], horizontal=True)
        
        viz = ArchitectureVisualizer()
        
        try:
            if viz_format == "ascii":
                st.text(viz.to_ascii(gene))
            elif viz_format == "mermaid":
                st.markdown(viz.to_mermaid(gene))
            elif viz_format == "json":
                st.json(gene.to_dict())
        except Exception as e:
            st.error(f"Visualization error: {e}")
        
        st.header("Statistics")
        
        is_valid, errors = gene.validate()
        if is_valid:
            st.success("Gene is valid")
        else:
            st.error("Gene is invalid")
            for err in errors:
                st.write(f"- {err}")
        
        params = gene.compute_params()
        memory = gene.compute_memory()
        
        stats = {
            "Parameters": f"{params:,}",
            "Memory (MB)": f"{memory / 1e6:.2f}",
            "hidden_dim / num_heads": f"{hidden_dim // num_heads}",
        }
        st.table(pd.DataFrame(stats.items(), columns=["Metric", "Value"]))
        
        st.header("Z3 Verification")
        
        if st.button("Run Z3 Verification"):
            with st.spinner("Running Z3..."):
                verifier = Verifier()
                result = verifier.verify(gene)
                if result:
                    st.success("Z3 Verified: Architecture is well-formed")
                else:
                    st.error("Z3 Verification Failed")
        
        st.header("Export")
        
        export_format = st.selectbox("Export Format", ["json", "pytorch", "onnx"])
        
        if st.button("Save Architecture"):
            from src.archgene.exporter import Exporter
            exporter = Exporter()
            try:
                if export_format == "json":
                    path = exporter.export_json(gene, "arch_gene")
                elif export_format == "pytorch":
                    path = exporter.export_pytorch(gene, "arch_gene")
                else:
                    path = exporter.export_onnx(gene, "arch_gene")
                st.success(f"Saved to {path}")
            except Exception as e:
                st.error(f"Export error: {e}")


if __name__ == "__main__":
    main()