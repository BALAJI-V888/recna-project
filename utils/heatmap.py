import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def plot_section_heatmap(target_sections, retrieved_sections_list, paper_id="Target Paper"):
    """
    Creates a section-wise similarity heatmap between a target paper and retrieved papers.

    Parameters
    ----------
    target_sections : dict
        {
            "introduction": embedding,
            "method": embedding,
            "results": embedding,
            ...
        }

    retrieved_sections_list : list
        [
            {"paper_id": "paper1", "sections": {...}},
            {"paper_id": "paper2", "sections": {...}},
        ]

    paper_id : str
        ID of the target paper (for title)
    """

    section_names = list(target_sections.keys())
    retrieved_ids = [p["paper_id"] for p in retrieved_sections_list]

    similarity_matrix = []

    for retrieved in retrieved_sections_list:

        row = []

        for sec in section_names:

            if sec in retrieved["sections"]:

                sim = np.dot(target_sections[sec], retrieved["sections"][sec]) / (
                    np.linalg.norm(target_sections[sec]) *
                    np.linalg.norm(retrieved["sections"][sec])
                )

            else:
                sim = 0

            row.append(sim)

        similarity_matrix.append(row)

    similarity_matrix = np.array(similarity_matrix)

    plt.figure(figsize=(10,6))

    sns.heatmap(
        similarity_matrix,
        xticklabels=section_names,
        yticklabels=retrieved_ids,
        cmap="coolwarm",
        annot=True,
        fmt=".2f"
    )

    plt.title(f"Section-wise Similarity Heatmap\nTarget: {paper_id}")
    plt.xlabel("Sections")
    plt.ylabel("Retrieved Papers")

    plt.show()