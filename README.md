# Quantum ML Graph-RAG Knowledge Engine

[Live Interactive Demo (Streamlit Cloud)]
https://graph-rag-knowledge-graph-engine.streamlit.app/

An end-to-end Graph-based Retrieval-Augmented Generation (Graph-RAG) engine. This system ingests 200 arXiv Quantum Machine Learning research papers, resolves extracted entities into a canonical knowledge graph, and performs 2-hop relational subgraph traversal and Louvain community detection to synthesize citation-backed answers.

## System Architecture & Features

* **Zero-Loss Structured Extraction (`extractor.py`):** Processes 200 PDFs via PyMuPDF (front-matter + conclusion high-signal slicing) and the Groq API. Implements strict JSON Schema enforcement.

* **Canonical Entity Resolution (`build_graph.py`):** Deduplicates raw extracted entities across papers using parenthetical acronym mapping (e.g., unifying *"Variational Quantum Eigensolver (VQE)"* and *"VQE"*), majority-vote label resolution, and multi-paper edge weighting via NetworkX.

* **Hybrid Local + Global Graph Retrieval (`query_engine.py`):**

  * **Zero-Token Seed Matching:** Maps user queries to canonical graph nodes in <5ms using n-gram substring scoring and token overlap without burning LLM API calls.

  * **Supernode-Safe 2-Hop Traversal (Local Search):** Walks inbound and outbound relational triples while capping Hop-1 expansion by degree centrality to prevent context-window explosion on hub nodes.

  * **Louvain Community Clustering (Global Search):** Partitions the graph into modular research clusters to answer broad thematic queries across the entire corpus.

* **Interactive Physics Subgraph UI (`app.py`):** Renders the exact traversed 2-hop subgraph in real time using PyVis ForceAtlas2 physics alongside `[arXiv: ID]` cited answers in Streamlit.

## Tech Stack

* **Language:** Python 3.11
* **LLM / API:** Groq API (`llama3` / `gpt-oss-20b`)
* **Graph Processing:** NetworkX
* **Frontend & Visualization:** Streamlit, PyVis (HTML Physics Graphs)
* **Data Processing:** PyMuPDF (fitz), regex, JSON Schema
* **Deployment:** Docker, Streamlit Community Cloud

## Graph Index Telemetry

| Metric | Value | 
| ----- | ----- | 
| **Papers Indexed** | 200 arXiv QML Papers | 
| **Canonical Nodes** | 2255 | 
| **Relational Triples** | 3389 | 
| **Research Communities** | 10 | 
| **Average Query Latency** | ~1.4 seconds | 

## Local Reproduction

### Option 1: Native Python

1. Clone the repository:
   ```bash
   git clone https://github.com/Sriram585/Graph-RAG-Knowledge-Graph-Engine.git
   cd Graph-RAG-Knowledge-Graph-Engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq API key to a `.env` file:
   ```text
   GROQ_API_KEY="gsk_your_api_key_here"
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Option 2: Docker

1. Build the container:
   ```bash
   docker build -t graphrag-engine:v1 .
   ```

2. Run the container (injecting your `.env` file):
   ```bash
   docker run -p 8501:8501 --env-file .env graphrag-engine:v1
   ```

3. Open `http://localhost:8501` in your browser.

## Project Structure

```text
Graph-RAG-Knowledge-Graph-Engine/
├── data/
│   └── master_knowledge_graph.json  # Pre-compiled deduplicated graph index
├── app.py                           # Streamlit UI & PyVis rendering
├── query_engine.py                  # Hybrid 2-hop / Louvain retrieval logic
├── build_graph.py                   # NetworkX entity resolution & aggregation
├── extractor.py                     # PDF parsing & LLM JSON schema extraction
├── requirements.txt                 # Dependencies
├── Dockerfile                       # Containerization config
├── .dockerignore                    # Excludes venv/ and secrets from container
└── .gitignore
```