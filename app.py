import random
import time

from utils.db import global_collection, section_collection
from utils.heatmap import plot_section_heatmap
from pipeline.embeddings import fetch_embeddings
from pipeline.novelty import compute_novelty
from pipeline.retrieval import plot_helper
from pipeline.interpretability import run_interpretability
from config import weights

# -------------------------------
# Global variables
# -------------------------------
sample_id = None
target_global_emd = None
target_sections_emd = None


# -------------------------------
# STEP 1: Get random paper
# -------------------------------
def test_paper_id():
    print("\n🎲 Selecting random paper...")
    all_ids = global_collection.get()
    all_data = all_ids['ids']

    sample = random.choice(all_data)
    print(f"✅ Selected Paper ID: {sample}")

    return sample


# -------------------------------
# STEP 2: Fetch embeddings
# -------------------------------
def test_emd(sample_id):
    print("\n🧠 Fetching embeddings...")
    start = time.time()

    global_emb, section_emb = fetch_embeddings(sample_id)

    print(f"✅ Embeddings fetched in {time.time() - start:.2f}s")
    print(f"📊 Sections found: {list(section_emb.keys())}")

    return global_emb, section_emb


# -------------------------------
# MAIN EXECUTION
# -------------------------------
print("🚀 Starting ReCNA Pipeline...\n")

# Ensure valid paper (fix your loop bug)
while True:
    sample_id = test_paper_id()
    target_global_emd, target_sections_emd = test_emd(sample_id)

    if target_sections_emd:  # ensure not empty
        break
    else:
        print("⚠️ Empty sections, retrying...")

# -------------------------------
# STEP 3: Novelty computation
# -------------------------------
print("\n📊 Computing novelty score...")
start = time.time()

novelty_score = compute_novelty(
    global_collection,
    section_collection,
    target_global_emd,
    target_sections_emd,
    sample_id,
    weights,
    5
)

print(f"✅ Novelty Score Computed in {time.time() - start:.2f}s")
print(f"📌 Novelty Score: {novelty_score}")

# -------------------------------
# STEP 4: Prepare heatmap data
# -------------------------------
print("\n📈 Preparing heatmap data...")
start = time.time()

target_section_emb, target_top_k_emb = plot_helper(sample_id)

print(f"✅ Heatmap data ready in {time.time() - start:.2f}s")

# -------------------------------
# STEP 5: Plot heatmap
# -------------------------------
print("\n🖼️ Plotting heatmap...")
plot_section_heatmap(
    target_sections=target_section_emb,
    retrieved_sections_list=target_top_k_emb,
    paper_id=sample_id
)
print("✅ Heatmap displayed")

# -------------------------------
# STEP 6: LLM Interpretation (GLOBAL)
# -------------------------------
print("\n🌍 Running GLOBAL interpretation...")
start = time.time()

result_global = run_interpretability(
    paper_id=sample_id,
    mode="global"
)

print(f"✅ Global interpretation done in {time.time() - start:.2f}s")

# -------------------------------
# STEP 7: LLM Interpretation (SECTION)
# -------------------------------
print("\n🧩 Running SECTION-WISE interpretation...")
start = time.time()

result_section = run_interpretability(
    paper_id=sample_id,
    mode="section"
)

print(f"✅ Section interpretation done in {time.time() - start:.2f}s")

# -------------------------------
# STEP 8: Final Outputs
# -------------------------------
print("\n📦 Final Results:")

# ===== GLOBAL OUTPUT =====
print("\n🌍 GLOBAL INTERPRETATION")

overall_novelty_global = int(result_global['overall_novelty'] * 10)
global_interpretation = result_global['interpretation']

print(f"⭐ Global Novelty (scaled): {overall_novelty_global}")
print(f"\n🧠 Global Interpretation:\n{global_interpretation}")


# ===== SECTION OUTPUT =====
print("\n🧩 SECTION-WISE INTERPRETATION")

overall_novelty_section = int(result_section['overall_novelty'] * 10)
section_scores = result_section['section_scores']
section_interpretation = result_section['interpretation']

print(f"⭐ Section-Based Novelty (scaled): {overall_novelty_section}")
print(f"📊 Section Scores: {section_scores}")
print(f"\n🧠 Section Interpretation:\n{section_interpretation}")