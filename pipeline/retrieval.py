import numpy as np

from pipeline.embeddings import fetch_embeddings
from utils.db import global_collection, section_collection

#Cosine similarity calculation

def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a).flatten()  # ensure 1D array
    vec_b = np.array(vec_b).flatten()  # ensure 1D array
    return float(np.dot(vec_a, vec_b)) # convert to scalar

def retrieve_top_k(global_collection, target_embedding, target_id, k=5):
    # Fetch k+1 to account for the target paper itself
    results = global_collection.query(
        query_embeddings=[target_embedding],
        n_results=k + 1
    )
    retrieved_ids  = results["ids"][0]
    distances      = results["distances"][0]

    # Filter out the target paper
    filtered_ids        = []
    filtered_similarities = []
    for rid, dist in zip(retrieved_ids, distances):
        if rid == target_id:
            continue
        filtered_ids.append(rid)
        filtered_similarities.append(max(0.0, min(1.0, 1 - dist)))

    # Return only top k
    return filtered_ids, filtered_similarities

def get_sections(section_collection, paper_ids):
    results = section_collection.get(
        where={"paper_id": {"$in": paper_ids}},
        include=["embeddings", "metadatas"]
    )
    sections_by_paper = {}
    for emb, meta in zip(results["embeddings"], results["metadatas"]):
        pid = meta["paper_id"]
        if pid not in sections_by_paper:
            sections_by_paper[pid] = {}
        sections_by_paper[pid][meta["section"]] = np.array(emb).flatten()  # ✅ always 1D
    return sections_by_paper

def section_similarity(target_sections, retrieved_sections, weights):
    score        = 0.0
    total_weight = 0.0
    section_scores = {}
    for section_name, weight in weights.items():
        if section_name in target_sections and section_name in retrieved_sections:
            sim           = cosine_similarity(
                np.array(target_sections[section_name]).flatten(),
                np.array(retrieved_sections[section_name]).flatten()
            )
            section_scores[section_name] = round(1 - float(sim), 4)  # novelty = 1 - sim
            score        += weight * float(sim)
            total_weight += weight
    weighted_scores = float(score / total_weight) if total_weight > 0 else 0.0
    return weighted_scores, section_scores

def compute_topk_section_scores(target_sections, retrieved_papers_list, weights):
    """
    Aggregates section_similarity() across all top-k papers.
    Uses MAX similarity per section (if even one paper is similar → not novel).
    """
    sim_accumulator = {sec: [] for sec in weights}
    per_paper_breakdown = []

    for retrieved in retrieved_papers_list:
        retrieved_sections = retrieved.get("sections", {})
        _,paper_section_scores = section_similarity(target_sections, retrieved_sections, weights)
        per_paper_breakdown.append({
            "title": retrieved.get("title", retrieved.get("paper_id", "Unknown")),
            "section_scores": paper_section_scores
        })
        for sec_name, novelty in paper_section_scores.items():
            sim_accumulator[sec_name].append(1 - novelty)  # convert back to similarity

    section_scores = {}
    for sec_name, sims in sim_accumulator.items():
        section_scores[sec_name] = round(1 - max(sims), 4) if sims else 1.0

    return section_scores, per_paper_breakdown

def plot_helper(sample_id):
    target_section_data = section_collection.get(where={'paper_id':sample_id}, include=['embeddings', 'metadatas'])
    target_section_emb = {}
    target_top_k_emb = []
    
    target_global_emd,_ = fetch_embeddings(sample_id)

    for emb, meta in zip(target_section_data['embeddings'], target_section_data['metadatas']):
        target_section_emb[meta['section']] = emb
        
    sample_ids,_ = retrieve_top_k(global_collection, target_global_emd, sample_id, 5)
    target_top_k_data = get_sections(section_collection, sample_ids)

    for paper_id, content in target_top_k_data.items():
        target_top_k_emb.append({
            'paper_id': paper_id,
            'sections' : content
        }) 
    return target_section_emb, target_top_k_emb