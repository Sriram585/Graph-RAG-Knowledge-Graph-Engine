import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from query_engine import GraphRAGEngine

st.set_page_config(
    page_title="Quantum ML Graph-RAG Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color mapping by entity type for visual clarity
LABEL_COLORS = {
    "TASK": "#FF6B6B",     # Red
    "METHOD": "#4ECDC4",   # Teal
    "METRIC": "#FFE66D",   # Yellow
    "DATASET": "#A8E6CF",  # Mint
    "ENTITY": "#C7CEEA"    # Fallback Lavender
}


@st.cache_resource
def load_engine():
    return GraphRAGEngine()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_graph_query(_engine, question: str, mode: str) -> dict:
    return _engine.query(question, mode=mode)

def render_pyvis_subgraph(triples: list, seed_nodes: list) -> str:
    """Builds an interactive HTML physics graph from the retrieved 2-hop triples."""
    net = Network(
        height="550px",
        width="100%",
        bgcolor="#0E1117",
        font_color="white",
        directed=True
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=120, spring_strength=0.08)

    added_nodes = set()

    for t in triples:
        src = t["source"]
        tgt = t["target"]
        src_label = t.get("source_label", "ENTITY")
        tgt_label = t.get("target_label", "ENTITY")
        rel = t["relation"]
        papers = ", ".join(t.get("papers", []))

        # Add source node
        if src not in added_nodes:
            is_seed = src in seed_nodes
            net.add_node(
                src,
                label=src,
                title=f"Type: {src_label} | Seed: {is_seed}",
                color=LABEL_COLORS.get(src_label, "#C7CEEA"),
                size=28 if is_seed else 18,
                borderWidth=3 if is_seed else 1
            )
            added_nodes.add(src)

        # Add target node
        if tgt not in added_nodes:
            is_seed = tgt in seed_nodes
            net.add_node(
                tgt,
                label=tgt,
                title=f"Type: {tgt_label} | Seed: {is_seed}",
                color=LABEL_COLORS.get(tgt_label, "#C7CEEA"),
                size=28 if is_seed else 18,
                borderWidth=3 if is_seed else 1
            )
            added_nodes.add(tgt)

        # Add directed relationship edge
        net.add_edge(
            src,
            tgt,
            label=rel,
            title=f"Relation: {rel} | Papers: [{papers}]",
            color="#8892B0",
            arrows="to"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    os.remove(tmp_path)
    return html_content


def main():
    st.title("Quantum ML Graph-RAG Knowledge Engine")
    st.caption("Multi-Hop Relational Retrieval & Louvain Community Sensemaking across 200 arXiv Papers")

    engine = load_engine()
    meta = engine.raw_data.get("metadata", {})

    # Sidebar Metrics & Legend
    with st.sidebar:
        st.header("Graph Index Telemetry")
        col1, col2 = st.columns(2)
        col1.metric("Papers Indexed", meta.get("total_papers", 200))
        col2.metric("Louvain Clusters", len(engine.communities))
        col3, col4 = st.columns(2)
        col3.metric("Canonical Nodes", engine.G.number_of_nodes())
        col4.metric("Relational Triples", engine.G.number_of_edges())

        st.divider()
        st.subheader("Node Legend")
        st.markdown(
            "- 🔴 **TASK** (Problem / Objective)\n"
            "- 🟢 **METHOD** (Algorithm / Circuit)\n"
            "- 🟡 **METRIC** (Evaluation Measure)\n"
            "- 🔵 **DATASET** (Benchmark Corpus)"
        )

        st.divider()
        search_mode = st.radio(
            "Retrieval Routing Mode",
            options=["auto", "local", "global"],
            index=0,
            help="Auto routes between 2-hop subgraph traversal and Louvain community summaries."
        )

    # Query Input
    default_q = "Which methods are used for quantum machine learning and what metrics or datasets evaluate them?"
    user_query = st.text_input("Ask a research question across the 200-paper corpus:", value=default_q)

    if st.button("Execute Graph-RAG Retrieval", type="primary"):
        with st.spinner("Traversing 2-hop subgraph & synthesizing citations..."):
            result = cached_graph_query(engine, user_query, search_mode)

        left_col, right_col = st.columns([1.1, 1.2])

        with left_col:
            st.subheader("Synthesized Research Answer")
            st.info(
                f"**Mode:** `{result['mode_used']}` | "
                f"**Seed Entities:** `{', '.join(result['seed_nodes']) or 'Global'}` | "
                f"**Triples Traversed:** `{result['triples_retrieved']}`"
            )
            st.markdown(result["answer"])

            with st.expander("Inspect Raw Retrieved Triples & arXiv Provenance"):
                st.json(result["subgraph"])

        with right_col:
            st.subheader("Interactive 2-Hop Subgraph")
            if result["subgraph"]:
                html_graph = render_pyvis_subgraph(result["subgraph"], result["seed_nodes"])
                components.html(html_graph, height=570, scrolling=False)
            else:
                st.warning("Global community mode active — no specific local subgraph nodes matched.")


if __name__ == "__main__":
    main()