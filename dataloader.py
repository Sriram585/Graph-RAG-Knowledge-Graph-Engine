import os
import urllib.request
import xml.etree.ElementTree as ET

DATA_DIR = "./data/pdfs"
os.makedirs(DATA_DIR, exist_ok=True)

# Query computer science / AI papers
QUERY = "cat:quant-ph+AND+(ti:quantum+machine+learning+OR+abs:quantum+machine+learning)"
MAX_RESULTS = 200

url = f"http://export.arxiv.org/api/query?search_query={QUERY}&start=0&max_results={MAX_RESULTS}"

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as response:
    xml_data = response.read()

root = ET.fromstring(xml_data)
ns = {"atom": "http://www.w3.org/2005/Atom"}

entries = root.findall("atom:entry", ns)
print(f"Found {len(entries)} papers. Starting download...")

for idx, entry in enumerate(entries, 1):
    pdf_link = None
    for link in entry.findall("atom:link", ns):
        if link.attrib.get("title") == "pdf":
            pdf_link = link.attrib.get("href")
            break

    if pdf_link:
        # Normalize to direct PDF URL
        if not pdf_link.endswith(".pdf"):
            pdf_link += ".pdf"
        
        paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1].replace("/", "_")
        target_path = os.path.join(DATA_DIR, f"{paper_id}.pdf")
        
        if not os.path.exists(target_path):
            try:
                urllib.request.urlretrieve(pdf_link, target_path)
                print(f"[{idx}/{len(entries)}] Downloaded: {paper_id}.pdf")
            except Exception as e:
                print(f"[{idx}/{len(entries)}] Failed: {paper_id}.pdf ({e})")
        else:
            print(f"[{idx}/{len(entries)}] Already exists: {paper_id}.pdf")