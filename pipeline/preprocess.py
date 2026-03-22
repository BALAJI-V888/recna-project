import re
from collections import defaultdict

def get_abstract(paper):
    abstract = paper.get("abstract", "")
    # Sometimes abstract is a dict with a 'text' key
    if isinstance(abstract, dict):
        abstract = abstract.get("text", "")
    elif isinstance(abstract, list):
        abstract = " ".join([a.get("text", "") if isinstance(a, dict) else a for a in abstract])
    return clean_text(abstract or "")

def clean_text(text):
    # Remove formula placeholders
    text = re.sub(r'\{\{formula:[^}]+\}\}', '', text)
    # Remove cite placeholders
    text = re.sub(r'\{\{cite:[^}]+\}\}', '', text)
    # Remove ref placeholders
    text = re.sub(r'\{\{ref:[^}]+\}\}', '', text)
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#paper[0] structure : dict_keys(['paper_id', 'title', 'abstract', 'categories', 'whole_text', 'sections'])

def extract_paper(paper):
    meta = paper.get("metadata", {})
    
    sections = defaultdict(list)
    for entry in paper.get("body_text", []):
        section = entry.get("section", "Unknown")
        text = clean_text(entry.get("text", ""))
        if text:
            sections[section].append(text)

    sections = {sec: " ".join(paragraphs) for sec, paragraphs in sections.items()}

    abstract = get_abstract(paper)  # ← fixed
    whole_text = abstract + " " + " ".join(sections.values())

    return {
        "paper_id": paper.get("paper_id"),
        "title": meta.get("title", ""),
        "abstract": abstract,
        "categories": meta.get("categories", ""),
        "whole_text": whole_text.strip(),
        "sections": sections
    }

