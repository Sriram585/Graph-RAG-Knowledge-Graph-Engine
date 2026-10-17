import os
import glob
import json
import re
from collections import Counter
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(BASE_DIR, "data", "json")
OUTPUT_GRAPH_PATH = os.path.join(BASE_DIR, "data", "master_knowledge_graph.json")


def normalize_entity_name(name: str) -> str:
    """Normalizes entity strings and strips trailing acronyms like ' (VQE)'."""
    name = name.strip()
    # Remove trailing acronym in parentheses: e.g., "Quantum Machine Learning (QML)" -> "quantum machine learning"
    name = re.sub(r'\s*\([A-Za-z0-9\-_]+\)\s*$', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.lower()


def build_acronym_map(json_files: list) -> dict:
    """Scans all files for patterns like 'Full Name (ACRONYM)' to map standalone acronyms to their full name."""
    acronym_to_full = {}
    pattern = re.compile(r'^(.+?)\s*\(([A-Za-z0-9\-]{2,10})\)$')

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                raw_name = node.get("name", "").strip()
                match = pattern.match(raw_name)
                if match:
                    full_clean = match.group(1).strip().lower()
                    acronym = match.group(2).strip().lower()
                    acronym_to_full[acronym] = full_clean
        except Exception:
            continue
    return acronym_to_full


def aggregate_graphs():
    json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*_graph.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {JSON_DIR}")

    print(f"Scanning {len(json_files)} JSON files for entity resolution...")
    acronym_map = build_acronym_map(json_files)
    print(f"Resolved {len(acronym_map)} acronyms (e.g., VQE, QAOA, QML, NISQ).")

    G = nx.MultiDiGraph()
    node_labels = {}  # Tracks label votes per canonical entity
    node_papers = {}  # Tracks which papers mention each entity

    raw_node_count = 0
    raw_edge_count = 0

    for path in json_files:
        paper_id = os.path.basename(path).replace("_graph.json", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping corrupted file {paper_id}: {e}")
            continue

        # 1. Process Nodes
        local_name_map = {}
        for n in data.get("nodes", []):
            raw_name = n.get("name", "").strip()
            label = n.get("label", "METHOD")
            if not raw_name:
                continue

            raw_node_count += 1
            norm_name = normalize_entity_name(raw_name)
            canonical_name = acronym_map.get(norm_name, norm_name)
            local_name_map[raw_name] = canonical_name

            if canonical_name not in node_labels:
                node_labels[canonical_name] = Counter()
                node_papers[canonical_name] = set()

            node_labels[canonical_name][label] += 1
            node_papers[canonical_name].add(paper_id)

        # 2. Process Edges
        for e in data.get("edges", []):
            raw_src = e.get("source", "").strip()
            raw_tgt = e.get("target", "").strip()
            rel = e.get("relation_type", "SOLVES").strip()
            if not raw_src or not raw_tgt:
                continue

            raw_edge_count += 1
            src = local_name_map.get(raw_src, acronym_map.get(normalize_entity_name(raw_src), normalize_entity_name(raw_src)))
            tgt = local_name_map.get(raw_tgt, acronym_map.get(normalize_entity_name(raw_tgt), normalize_entity_name(raw_tgt)))

            # Skip self-loops created by entity resolution
            if src == tgt:
                continue

            # Ensure endpoint nodes exist even if LLM omitted them from the 'nodes' array
            for endpoint in (src, tgt):
                if endpoint not in node_labels:
                    node_labels[endpoint] = Counter({"METHOD": 1})
                    node_papers[endpoint] = {paper_id}
                else:
                    node_papers[endpoint].add(paper_id)

            G.add_edge(src, tgt, relation_type=rel, paper_id=paper_id)

    # 3. Populate final node attributes using majority-vote label
    for node_id in node_labels:
        majority_label = node_labels[node_id].most_common(1)[0][0]
        papers = sorted(list(node_papers[node_id]))
        if node_id not in G:
            G.add_node(node_id)
        G.nodes[node_id]["label"] = majority_label
        G.nodes[node_id]["papers"] = papers
        G.nodes[node_id]["degree"] = G.degree(node_id)

    # Deduplicate parallel edges (merge paper_ids on identical source-rel-target triples)
    deduped_edges = {}
    for u, v, data in G.edges(data=True):
        rel = data["relation_type"]
        key = (u, rel, v)
        if key not in deduped_edges:
            deduped_edges[key] = set()
        deduped_edges[key].add(data["paper_id"])

    master_data = {
        "metadata": {
            "total_papers": len(json_files),
            "raw_nodes": raw_node_count,
            "deduplicated_nodes": G.number_of_nodes(),
            "raw_edges": raw_edge_count,
            "deduplicated_edges": len(deduped_edges)
        },
        "nodes": [
            {
                "id": n,
                "label": G.nodes[n]["label"],
                "degree": G.nodes[n]["degree"],
                "paper_count": len(G.nodes[n]["papers"]),
                "papers": G.nodes[n]["papers"]
            }
            for n in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        ],
        "edges": [
            {
                "source": u,
                "relation_type": rel,
                "target": v,
                "weight": len(papers),
                "papers": sorted(list(papers))
            }
            for (u, rel, v), papers in deduped_edges.items()
        ]
    }

    with open(OUTPUT_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)

    print("\n--- Aggregation Complete ---")
    print(f"Raw Nodes: {raw_node_count} -> Canonical Nodes: {G.number_of_nodes()}")
    print(f"Raw Edges: {raw_edge_count} -> Canonical Triples: {len(deduped_edges)}")
    print(f"Saved master graph to: {OUTPUT_GRAPH_PATH}")

    print("\nTop 10 Most Connected Entities Across Your 200 Papers:")
    for item in master_data["nodes"][:10]:
        print(f"  - {item['id']} ({item['label']}) | Degree: {item['degree']} | Papers: {item['paper_count']}")


if __name__ == "__main__":
    aggregate_graphs()