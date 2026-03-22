
from pipeline.retrieval import get_sections, section_similarity, retrieve_top_k

def compute_novelty(
        global_collection,
        section_collection,
        target_global_embedding,
        target_sections,
        target_id,
        weights,
        k
):

    retrieved_ids, similarities = retrieve_top_k(
        global_collection,
        target_global_embedding,
        target_id,
        k
    )
     # Fetch all sections in one ChromaDB call
    all_sections  = get_sections(section_collection, retrieved_ids)

    section_scores = []
    weighted_scores = [] #weighted aggregation
    for paper_id, retrievel_sim in zip(retrieved_ids, similarities):
        retrieved_sections = all_sections.get(paper_id, {})
        score,_ = section_similarity(target_sections, retrieved_sections, weights)
        if score > 0:
            section_scores.append(score)
            weighted_scores.append(score * retrievel_sim)

    if not section_scores:
        return 1.0  # no similar papers = highly novel

#     top_k_similarity = np.mean(section_scores)
    top_k_similarity = max(section_scores)
    
    novelty = 1 - top_k_similarity

#     # weighted aggregation
#     weighted_similarity = sum(weighted_scores) / sum(similarities)
#     novelty = 1 - weighted_similarity
    return novelty