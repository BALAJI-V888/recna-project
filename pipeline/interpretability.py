
import json
import os
import numpy as np

from groq import Groq

from pipeline.novelty import compute_novelty
from pipeline.retrieval import get_sections, section_similarity, retrieve_top_k, compute_topk_section_scores
from utils.db import global_collection, section_collection
from config import weights, client

CS_PAPERS_MAPPED = r"C:\Users\sargu\OneDrive\Desktop\Final_year_project\ReCNA\dataset\unarxive\cs_papers_mapped.json"
loaded_papers = ""
with open(CS_PAPERS_MAPPED, "r", encoding = "utf-8") as f:
    loaded_papers = json.load(f)
paper_lookup = {p['paper_id']: p for p in loaded_papers}


def query_llm(prompt, model="llama-3.3-70b-versatile"):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a research assistant analyzing scientific paper novelty."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def _interpret_global(overall_novelty, section_scores, target_summary, retrieved_papers):
    
    # Build retrieved papers context for the prompt
    papers_context = ""
    for i, p in enumerate(retrieved_papers, 1):
        papers_context += f"\n{i}. {p['title']}\n   Abstract: {p.get('abstract','N/A')[:300]}\n"

    # Build section scores context
    section_context = "\n".join(
        [f"  - {sec}: {score:.4f}" for sec, score in section_scores.items()]
    ) if section_scores else "  Not available"

    prompt = f"""You are analyzing the novelty of a research paper.

Overall Novelty Score: {overall_novelty:.4f} (0 = not novel, 1 = highly novel)

Section-wise Novelty Scores:
{section_context}

Target Paper Summary:
{target_summary}

Most Similar Retrieved Papers:
{papers_context}

Based on the above, provide a concise novelty analysis:
1. Overall assessment of novelty
2. Which sections are most/least novel and why
3. How it compares to the retrieved similar papers
4. Key contributions that make it novel (or not)
"""
    return query_llm(prompt)

def _interpret_section(sections, section_scores, retrieved_papers):

    # Build per-section analysis block
    section_blocks = ""
    for sec_name, text in sections.items():
        score = section_scores.get(sec_name, "N/A")
        section_blocks += f"\n[{sec_name.upper()}] (novelty score: {score})\n{text[:1500]}\n"

    # Build retrieved papers context
    papers_context = ""
    for i, p in enumerate(retrieved_papers, 1):
        sec_novelty = p.get("section_novelty", {})
        sec_str = ", ".join([f"{k}: {v}" for k, v in sec_novelty.items()])
        papers_context += f"\n{i}. {p['title']}\n   Section novelty vs this paper: {sec_str}\n"

    prompt = f"""You are analyzing a research paper section by section for novelty.

Each section has a novelty score (0 = not novel, 1 = highly novel).

Paper Sections:
{section_blocks}

Most Similar Retrieved Papers (with per-section novelty):
{papers_context}

Provide a section-by-section novelty analysis:
- For each section: what is novel, what overlaps with existing work
- Which retrieved paper is most similar to each section
- Overall recommendation on the paper's contribution
"""
    return query_llm(prompt)

# ─── Main Dispatcher ──────────────────────────────────────────────────────────

def interpret_novelty(
    mode: str,
    sections: dict = None,           # {"Section Name": "full text", ...}
    section_scores: dict = None,     # {"Section Name": float, ...}
    overall_novelty: float = None,   # float 0–1
    target_summary: str = None,      # short summary of full target paper
    retrieved_papers: list[dict] = None,  # [{"title": str, "abstract": str, "sections": dict}, ...]
) -> str:
    """
    Unified novelty interpretation dispatcher.

    Modes:
        "global"  — Whole-paper summary + comparison against all top-k similar papers.
        "section" — Section-by-section summary + comparison against similar papers per section.

    Args:
        mode:             "global" | "section"
        sections:         Dict of section name → full text (required for "section" mode)
        section_scores:   Dict of section name → novelty score (0–1)
        overall_novelty:  Float novelty score for the whole paper (used in "global")
        target_summary:   Short summary of the target paper (used in "global")
        retrieved_papers: List of similar paper dicts with keys:
                            - title (str)
                            - abstract (str, optional)
                            - sections (dict, optional) — same structure as `sections`

    Returns:
        LLM-generated interpretation string.
    """
    retrieved_papers = retrieved_papers or []

    if mode == "global":
        if overall_novelty is None:
            raise ValueError("`overall_novelty` is required for mode='global'")
        if not target_summary:
            raise ValueError("`target_summary` is required for mode='global'")
        return _interpret_global(
            overall_novelty=overall_novelty,
            section_scores=section_scores or {},
            target_summary=target_summary,
            retrieved_papers=retrieved_papers,
        )

    elif mode == "section":
        if not sections:
            raise ValueError("`sections` dict is required for mode='section'")
        if not section_scores:
            raise ValueError("`section_scores` dict is required for mode='section'")
        return _interpret_section(
            sections=sections,
            section_scores=section_scores,
            retrieved_papers=retrieved_papers,
        )

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'global' or 'section'.")
    
    #papers lookup for optimizations O(1)


def get_paper_content(paper_id: str, mode: str) -> dict:
    """
    Fetch paper content by paper_id in O(1).
    
    Args:
        paper_id: unique paper identifier
        mode: "global" → returns whole_content
              "section" → returns all individual sections

    Returns:
        dict with the relevant content fields
    """
    
    paper = paper_lookup.get(paper_id)
    
    if paper is None:
        raise ValueError(f"Paper {paper_id} contents are not found in {CS_PAPERS_MAPPED}")
    
    
    mode = mode.strip().lower()
    if(mode == "global"):
             return {
            "paper_id":     paper["paper_id"],
            "title":        paper["title"],
            "whole_content": paper["whole_content"],
        }

    elif(mode == "section"):
           return {
            "paper_id":    paper["paper_id"],
            "title":       paper["title"],
            "abstract":    paper["abstract"],
            "introduction": paper["introduction"],
            "methodology": paper["methodology"],
            "discussion":  paper["discussion"],
            "conclusion":  paper["conclusion"],
            "other":       paper["other"],
           }
    else:
        raise ValueError(f"Invalid mode {mode}, please choose 'global' or 'section' mode...")
    
def run_interpretability(
    paper_id: str,
    mode: str,
    k: int = 5,
):
    """
    Full novelty interpretability pipeline.
    Just pass paper_id and mode — handles everything internally.

    Args:
        paper_id : ID of the target paper
        mode     : "global" | "section"
        k        : number of top-k papers to retrieve (default 5)

    Returns:
        dict with keys:
            - overall_novelty     : float (0–1)
            - section_scores      : {section_name: novelty_score}
            - per_paper_breakdown : [{title, section_scores}]
            - interpretation      : LLM-generated report string
    """

    # ── Step 1: Fetch target embeddings ───────────────────────────────────────
    target_global = global_collection.get(ids=[paper_id], include=["embeddings"])
    target_global_emb = target_global["embeddings"][0]

    target_sec_data = section_collection.get(
        where={"paper_id": paper_id},
        include=["embeddings", "metadatas"]
    )
    target_sections = {
        meta["section"]: np.array(emb).flatten()
        for emb, meta in zip(target_sec_data["embeddings"], target_sec_data["metadatas"])
    }

    # ── Step 2: Retrieve top-k similar papers ─────────────────────────────────
    retrieved_ids, _ = retrieve_top_k(global_collection, target_global_emb, paper_id, k=k)

    # ── Step 3: Compute overall novelty score ─────────────────────────────────
    overall_novelty = compute_novelty(
        global_collection,
        section_collection,
        target_global_emb,
        target_sections,
        paper_id,
        weights,
        k
    )

    # ── Step 4: Fetch section embeddings for all retrieved papers ─────────────
    retrieved_sections_map = get_sections(section_collection, retrieved_ids)

    # ── Step 5: Build retrieved_papers list (embeddings + text) ───────────────
    retrieved_papers_list = []
    for pid in retrieved_ids:
        try:
            content = get_paper_content(pid, mode="section")
        except ValueError:
            content = {"title": pid, "abstract": ""}
        retrieved_papers_list.append({
            "paper_id": pid,
            "title":    content["title"],
            "abstract": content["abstract"],
            "sections": retrieved_sections_map.get(pid, {})
        })

    # ── Step 6: Aggregate section scores across top-k ─────────────────────────
    section_scores, per_paper_breakdown = compute_topk_section_scores(
        target_sections,
        retrieved_papers_list,
        weights
    )

    # ── Step 7: Fetch target text content ─────────────────────────────────────
    target_content = get_paper_content(paper_id, mode=mode)

    # ── Step 8: Build interpret_novelty args based on mode ────────────────────
    retrieved_for_llm = [
        {
            "title":          p["title"],
            "abstract":       p["abstract"],
            "section_novelty": next(
                (b["section_scores"] for b in per_paper_breakdown if b["title"] == p["title"]),
                {}
            )
        }
        for p in retrieved_papers_list
    ]

    if mode == "global":
        interpretation = interpret_novelty(
            mode="global",
            overall_novelty=overall_novelty,
            target_summary=target_content["whole_content"][:3000],  # truncate for LLM
            section_scores=section_scores,
            retrieved_papers=retrieved_for_llm,
        )

    elif mode == "section":
        interpretation = interpret_novelty(
            mode="section",
            sections={
                "abstract":     target_content.get("abstract",     "")[:1500],
                "introduction": target_content.get("introduction", "")[:1500],
                "methodology":  target_content.get("methodology",  "")[:1500],
                "discussion":   target_content.get("discussion",   "")[:1500],
                "conclusion":   target_content.get("conclusion",   "")[:1500],
            },
            section_scores=section_scores,
            retrieved_papers=retrieved_for_llm,
        )

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'global' or 'section'.")

    return {
        "overall_novelty":     overall_novelty,
        "section_scores":      section_scores,
        "per_paper_breakdown": per_paper_breakdown,
        "interpretation":      interpretation,
    }