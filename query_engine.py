import os
import json
import re
from difflib import get_close_matches
import networkx as nx
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in .env or Streamlit Secrets.")

client = Groq(api_key=api_key, max_retries=3, timeout=30.0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_GRAPH_PATH = os.path.join(BASE_DIR, "data", "master_knowledge_graph.json")

STOPWORDS = {
    "what", "which", "how", "does", "are", "the", "used", "for", "in", "to",
    "of", "and", "or", "with", "on", "by", "is", "can", "from", "papers",
    "methods", "tasks", "metrics", "datasets", "solve", "evaluate", "improve"
}


class GraphRAGEngine:
    def __init__(self, graph_path: str = MASTER_GRAPH_PATH):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"Master graph not found at {graph_path}. Run build_graph.py first."
            )

        with open(graph_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.G = nx.MultiDiGraph()
        self._build_networkx_graph()
        self.communities = self._compute_communities()

    def _build_networkx_graph(self):
        for n in self.raw_data.get("nodes", []):
            self.G.add_node(
                n["id"],
                label=n.get("label", "METHOD"),
                degree=n.get("degree", 0),
                papers=n.get("papers", [])
            )

        for e in self.raw_data.get("edges", []):
            self.G.add_edge(
                e["source"],
                e["target"],
                relation_type=e["relation_type"],
                weight=e.get("weight", 1),
                papers=e.get("papers", [])
            )

    def _compute_communities(self) -> list:
        """Partitions the graph into research clusters using Louvain community detection."""
        undirected = self.G.to_undirected()
        if undirected.number_of_nodes() == 0:
            return []

        raw_comms = nx.community.louvain_communities(undirected, seed=42)
        structured_comms = []
        for idx, comm in enumerate(sorted(raw_comms, key=len, reverse=True)):
            if len(comm) < 3:
                continue
            # Rank nodes inside the community by degree centrality
            ranked_nodes = sorted(
                comm, key=lambda n: self.G.nodes[n].get("degree", 0), reverse=True
            )
            top_entities = [
                f"{n} ({self.G.nodes[n].get('label', 'ENTITY')})"
                for n in ranked_nodes[:8]
            ]
            structured_comms.append({
                "community_id": idx + 1,
                "size": len(comm),
                "core_entities": top_entities
            })
        return structured_comms[:10]

    def find_seed_nodes(self, query: str, top_k: int = 5) -> list:
        """Maps natural language query to canonical graph nodes without using LLM tokens."""
        q_clean = re.sub(r"[^a-zA-Z0-9\s\-]", "", query.lower()).strip()
        q_tokens = {t for t in q_clean.split() if t not in STOPWORDS and len(t) > 2}

        all_nodes = list(self.G.nodes())
        scored_nodes = []

        for node in all_nodes:
            score = 0.0
            # 1. Exact substring match in query
            if len(node) > 3 and node in q_clean:
                score += 10.0 + (len(node) * 0.2)

            # 2. Token overlap score
            node_tokens = set(node.split())
            overlap = q_tokens.intersection(node_tokens)
            if overlap:
                score += (len(overlap) / max(len(node_tokens), 1)) * 5.0

            # 3. Slight tie-breaker boost for high-degree hub nodes
            if score > 0:
                score += min(self.G.nodes[node].get("degree", 0) * 0.05, 1.5)
                scored_nodes.append((node, score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        matched = [n for n, _ in scored_nodes[:top_k]]

        # Fallback to fuzzy matching if token overlap found nothing
        if not matched:
            for token in q_tokens:
                close = get_close_matches(token, all_nodes, n=2, cutoff=0.7)
                matched.extend(close)

        return list(dict.fromkeys(matched))[:top_k]

    def retrieve_2hop_subgraph(self, seed_nodes: list, max_triples: int = 45) -> dict:
        """Traverses 2 hops (inbound and outbound) from seed nodes and collects arXiv citations."""
        triples = []
        seen_triples = set()
        cited_papers = set()

        def collect_edges(nodes_to_expand, hop_level):
            next_frontier = set()
            for node in nodes_to_expand:
                # Outbound edges: (node) -[rel]-> (target)
                for _, tgt, data in self.G.out_edges(node, data=True):
                    key = (node, data["relation_type"], tgt)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        papers = data.get("papers", [])
                        cited_papers.update(papers)
                        triples.append({
                            "hop": hop_level,
                            "source": node,
                            "source_label": self.G.nodes[node].get("label", ""),
                            "relation": data["relation_type"],
                            "target": tgt,
                            "target_label": self.G.nodes[tgt].get("label", ""),
                            "weight": data.get("weight", 1),
                            "papers": papers[:4]
                        })
                    next_frontier.add(tgt)

                # Inbound edges: (source) -[rel]-> (node)
                for src, _, data in self.G.in_edges(node, data=True):
                    key = (src, data["relation_type"], node)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        papers = data.get("papers", [])
                        cited_papers.update(papers)
                        triples.append({
                            "hop": hop_level,
                            "source": src,
                            "source_label": self.G.nodes[src].get("label", ""),
                            "relation": data["relation_type"],
                            "target": node,
                            "target_label": self.G.nodes[node].get("label", ""),
                            "weight": data.get("weight", 1),
                            "papers": papers[:4]
                        })
                    next_frontier.add(src)
            return next_frontier

        # Hop 1: Direct connections
        hop1_neighbors = collect_edges(seed_nodes, hop_level=1)

        # Filter Hop 1 neighbors to top 12 by degree so Hop 2 doesn't explode on massive hub nodes
        hop1_filtered = sorted(
            hop1_neighbors - set(seed_nodes),
            key=lambda n: self.G.nodes[n].get("degree", 0),
            reverse=True
        )[:12]

        # Hop 2: Secondary multi-paper connections
        collect_edges(hop1_filtered, hop_level=2)

        # Prioritize Hop 1 triples first, then sort by paper citation weight
        triples.sort(key=lambda x: (x["hop"], -x["weight"]))
        selected_triples = triples[:max_triples]

        return {
            "seed_nodes": seed_nodes,
            "triples": selected_triples,
            "total_papers_referenced": len(cited_papers)
        }

    def query(self, user_question: str, mode: str = "auto") -> dict:
        """Routes between Local (2-hop) and Global (Community) search and synthesizes answer."""
        is_global = mode == "global" or any(
            kw in user_question.lower()
            for kw in ["overall", "main themes", "clusters", "overview", "broad trends", "communities"]
        )

        seed_nodes = self.find_seed_nodes(user_question)
        subgraph = self.retrieve_2hop_subgraph(seed_nodes) if seed_nodes else {"seed_nodes": [], "triples": []}

        context_lines = []
        if is_global or not subgraph["triples"]:
            context_lines.append("=== GLOBAL GRAPH COMMUNITIES (LOUVAIN CLUSTERS) ===")
            for c in self.communities:
                context_lines.append(
                    f"Cluster {c['community_id']} ({c['size']} entities): {', '.join(c['core_entities'])}"
                )

        if subgraph["triples"]:
            context_lines.append(f"\n=== 2-HOP RELATIONAL SUBGRAPH (Seed Entities: {', '.join(seed_nodes)}) ===")
            for t in subgraph["triples"]:
                papers_str = ", ".join(t["papers"])
                context_lines.append(
                    f"[Hop {t['hop']}] ({t['source']} : {t['source_label']}) "
                    f"--[{t['relation']}]--> "
                    f"({t['target']} : {t['target_label']}) | Cited in arXiv: [{papers_str}]"
                )

        assembled_context = "\n".join(context_lines)

        system_prompt = """You are a Quantum Machine Learning Graph-RAG Research Assistant.
Answer the user's question strictly using the provided Knowledge Graph Triples and Community Clusters.
RULES:
1. Explicitly trace multi-hop relationships (e.g., Method A -> SOLVES -> Task B -> EVALUATED_ON -> Dataset C).
2. Cite the exact arXiv paper IDs in brackets (e.g., [arXiv: 2310.06973v1]) for every claim you make.
3. If the graph context does not contain a specific relationship, state clearly what is missing rather than guessing."""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"GRAPH CONTEXT:\n{assembled_context}\n\nUSER QUESTION: {user_question}"
                    }
                ],
                reasoning_effort="low",
                max_completion_tokens=1500,
                temperature=0.2
            )
            answer = response.choices[0].message.content
        except RateLimitError:
            answer = "[Rate Limit Exceeded] Wait 60 seconds and retry."
        except Exception as e:
            answer = f"[Synthesis Error] {e}"

        return {
            "question": user_question,
            "mode_used": "GLOBAL + LOCAL" if is_global else "LOCAL_2_HOP",
            "seed_nodes": seed_nodes,
            "triples_retrieved": len(subgraph["triples"]),
            "answer": answer,
            "subgraph": subgraph["triples"]
        }


if __name__ == "__main__":
    engine = GraphRAGEngine()
    print(f"Loaded Master Graph: {engine.G.number_of_nodes()} nodes, {engine.G.number_of_edges()} edges.")
    print(f"Detected {len(engine.communities)} Louvain research communities.\n")

    # Test with 2 automatic queries, then drop into interactive CLI
    test_queries = [
        "What are the main themes and research clusters across these papers?",
        "Which methods are used for quantum machine learning and what metrics or datasets evaluate them?"
    ]

    for q in test_queries:
        print("=" * 80)
        print(f"QUESTION: {q}")
        res = engine.query(q)
        print(f"MODE: {res['mode_used']} | SEED NODES: {res['seed_nodes']} | TRIPLES: {res['triples_retrieved']}")
        print("-" * 80)
        print(res["answer"])
        print("\n")

    while True:
        user_q = input("Ask your Graph-RAG Engine (or type 'exit'): ").strip()
        if user_q.lower() in {"exit", "quit", ""}:
            break
        res = engine.query(user_q)
        print(f"\n[Matched Nodes: {res['seed_nodes']} | Triples Traversed: {res['triples_retrieved']}]")
        print(res["answer"])
        print("-" * 80)