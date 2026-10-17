import os
import json
import re
import glob
import time
import fitz
from dotenv import load_dotenv
from groq import Groq, RateLimitError, BadRequestError

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in your .env file.")

# Set explicit timeout so requests don't hang indefinitely
client = Groq(api_key=api_key, max_retries=3, timeout=45.0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
JSON_DIR = os.path.join(BASE_DIR, "data", "json")

ALLOWED_LABELS = {"TASK", "METHOD", "METRIC", "DATASET"}
ALLOWED_RELATIONS = {"EVALUATED_ON", "USES_METRIC", "IMPROVES_UPON", "SOLVES"}


def ensure_directories():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)


def extract_high_signal_text(pdf_path: str, max_chars: int = 7500) -> str:
    """Extracts Front Matter and Conclusion, capped at 7500 chars to stay under 8000 TPM."""
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        if total == 0:
            return ""

        pages = []
        for p in range(min(2, total)):
            pages.append(doc[p].get_text("text"))

        if total > 2:
            pages.append(doc[total - 2].get_text("text"))

        text = "\n".join(pages)
        ref = re.search(r'\n(?:\d+\.?\s*)?(?:References|REFERENCES|Bibliography)\b', text)
        if ref:
            text = text[:ref.start()]

        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception as e:
        print(f"  [PDF Error] {e}")
        return ""


def sanitize_graph(raw_data: dict) -> dict:
    """Filters out hallucinated labels (e.g., 'CONCEPT'), stray strings, or malformed edges."""
    if not isinstance(raw_data, dict):
        return None

    clean_nodes = []
    seen_nodes = set()
    for n in raw_data.get("nodes", []):
        if isinstance(n, dict) and n.get("name") and n.get("label") in ALLOWED_LABELS:
            name = str(n["name"]).strip()
            if name and name not in seen_nodes:
                seen_nodes.add(name)
                clean_nodes.append({"name": name, "label": n["label"]})

    clean_edges = []
    for e in raw_data.get("edges", []):
        if (
            isinstance(e, dict)
            and e.get("source")
            and e.get("target")
            and e.get("relation_type") in ALLOWED_RELATIONS
        ):
            clean_edges.append({
                "source": str(e["source"]).strip(),
                "target": str(e["target"]).strip(),
                "relation_type": e["relation_type"]
            })

    if not clean_nodes:
        return None

    return {"nodes": clean_nodes, "edges": clean_edges}


def salvage_failed_json(failed_str: str) -> dict:
    """Recovers valid nodes and edges from Groq's failed_generation string on 400 errors."""
    if not failed_str:
        return None

    # 1. Attempt direct JSON load after stripping stray empty string elements like ,"",
    cleaned = re.sub(r',\s*""\s*(?=[,\]])', '', failed_str)
    cleaned = re.sub(r'\[\s*""\s*,', '[', cleaned)
    try:
        data = json.loads(cleaned)
        sanitized = sanitize_graph(data)
        if sanitized:
            return sanitized
    except Exception:
        pass

    # 2. Regex fallback: if JSON was truncated mid-stream, extract all complete node/edge objects
    nodes = []
    seen_nodes = set()
    for m in re.finditer(r'\{\s*"label"\s*:\s*"([^"]+)"\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}|\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}', failed_str):
        label = m.group(1) or m.group(4)
        name = (m.group(2) or m.group(3)).strip()
        if label in ALLOWED_LABELS and name and name not in seen_nodes:
            seen_nodes.add(name)
            nodes.append({"name": name, "label": label})

    edges = []
    for m in re.finditer(r'\{[^{}]*"source"\s*:\s*"([^"]+)"[^{}]*"target"\s*:\s*"([^"]+)"[^{}]*"relation_type"\s*:\s*"([^"]+)"[^{}]*\}|\{[^{}]*"relation_type"\s*:\s*"([^"]+)"[^{}]*"source"\s*:\s*"([^"]+)"[^{}]*"target"\s*:\s*"([^"]+)"[^{}]*\}', failed_str):
        src = m.group(1) or m.group(5)
        tgt = m.group(2) or m.group(6)
        rel = m.group(3) or m.group(4)
        if rel in ALLOWED_RELATIONS and src and tgt:
            edges.append({"source": src.strip(), "target": tgt.strip(), "relation_type": rel})

    if nodes:
        return {"nodes": nodes, "edges": edges}

    return None


def extract_graph(text: str) -> dict:
    system_prompt = """You are an automated Knowledge Graph extraction engine.
Extract up to 12 primary entities (nodes) and up to 12 relationships (edges) from the provided paper excerpt and output valid JSON.
STRICT RULES:
- Node 'label' MUST ONLY be one of: TASK, METHOD, METRIC, DATASET. Never use 'CONCEPT' or any other label. Skip items that do not fit these 4 categories.
- Edge 'relation_type' MUST ONLY be one of: EVALUATED_ON, USES_METRIC, IMPROVES_UPON, SOLVES."""

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "knowledge_graph",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "label": {"type": "string", "enum": ["TASK", "METHOD", "METRIC", "DATASET"]}
                            },
                            "required": ["name", "label"],
                            "additionalProperties": False
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "relation_type": {"type": "string", "enum": ["EVALUATED_ON", "USES_METRIC", "IMPROVES_UPON", "SOLVES"]}
                            },
                            "required": ["source", "target", "relation_type"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["nodes", "edges"],
                "additionalProperties": False
            }
        }
    }

    max_retries = 3
    current_text = text

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract Knowledge Graph in JSON format:\n\n{current_text}"}
                ],
                response_format=schema,
                reasoning_effort="low",
                max_completion_tokens=3000,
                temperature=0.2
            )

            raw_json = json.loads(response.choices[0].message.content)
            return sanitize_graph(raw_json)

        except BadRequestError as e:
            # Intercept 400 json_validate_failed and salvage the payload locally
            failed_gen = ""
            if isinstance(getattr(e, "body", None), dict):
                failed_gen = e.body.get("error", {}).get("failed_generation", "")

            salvaged = salvage_failed_json(failed_gen)
            if salvaged:
                print("[Salvaged]", end=" ", flush=True)
                return salvaged

            print(f"\n  [400 Validation Error, attempt {attempt + 1}/{max_retries}] Retrying...", end=" ", flush=True)
            current_text = current_text[:int(len(current_text) * 0.85)]
            time.sleep(5)

        except RateLimitError:
            print(f"\n  [Rate Limit, attempt {attempt + 1}/{max_retries}] Sleeping 60s...", end=" ", flush=True)
            time.sleep(60)

        except Exception as e:
            err_str = str(e)
            if "413" in err_str or "Request too large" in err_str:
                current_text = current_text[:int(len(current_text) * 0.70)]
                print(f"\n  [413 Overflow] Trimmed to {len(current_text)} chars. Retrying...", end=" ", flush=True)
                time.sleep(5)
                continue
            print(f"\n  [API Error, attempt {attempt + 1}/{max_retries}] {e}", end=" ", flush=True)
            time.sleep(10)

    return None


def run():
    ensure_directories()
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    print(f"Found {len(pdf_files)} PDFs. Running Groq extraction pipeline...\n")

    for idx, path in enumerate(pdf_files, 1):
        filename = os.path.basename(path)
        out_name = os.path.splitext(filename)[0] + "_graph.json"
        out_path = os.path.join(JSON_DIR, out_name)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        start = time.time()
        print(f"[{idx}/{len(pdf_files)}] {filename}...", end=" ", flush=True)

        text = extract_high_signal_text(path)
        if not text or len(text) < 100:
            print("Skipped (insufficient text).")
            continue

        graph = extract_graph(text)
        if not graph:
            print("Failed.")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start
        nodes = len(graph.get("nodes", []))
        edges = len(graph.get("edges", []))
        print(f"Done in {elapsed:.2f}s ({nodes} nodes, {edges} edges)")

        # Pace requests to stay under 8000 TPM on Groq Free Tier
        time.sleep(25)


if __name__ == "__main__":
    run()