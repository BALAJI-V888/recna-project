"""
Embedding Generation for Papers (Compact Version)
Generates and stores embeddings in ChromaDB
Loads model from local cache
"""
 
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import chromadb
import os
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
 
MODEL_PATH = r"C:\Users\sargu\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
CHROMA_PATH = r"C:\Users\sargu\OneDrive\Desktop\Final_year_project\ReCNA\dataset\chroma_db"
EMBEDDINGS_BACKUP = r"C:\Users\sargu\OneDrive\Desktop\Final_year_project\ReCNA\dataset\embeddings_backup.npz"
METADATA_BACKUP = r"C:\Users\sargu\OneDrive\Desktop\Final_year_project\ReCNA\dataset\embeddings_metadata_backup.npz"
TEXTS_BACKUP = r"C:\Users\sargu\OneDrive\Desktop\Final_year_project\ReCNA\dataset\texts_backup.npz"
 
SECTION_TYPES = ["introduction", "methodology", "discussion", "conclusion"]
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
 
print("Loading model...", end=" ")
try:
    model = SentenceTransformer(MODEL_PATH, device="cpu")
    print("model loaded successfully")
except Exception as e:
    print(f"Unable to load model {e}")
    exit(1)
 
# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
global_collection = chroma_client.get_or_create_collection(name="paper_embeddings")
section_collection = chroma_client.get_or_create_collection(name="section_embeddings")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING AND EMBEDDING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    
    return chunks
 
 
def get_avg_embedding(text):
    """Generate average embedding for text"""
    if not text or len(text.strip()) == 0:
        return None
    
    chunks = chunk_text(text)
    if len(chunks) == 0:
        return None
    
    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=16
    )
    
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    avg_embedding = normalize(avg_embedding.reshape(1, -1))[0]
    
    return avg_embedding.tolist()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GENERATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def generate_global_embeddings(papers):
    """Generate whole-paper embeddings"""
    print("Generating global embeddings...")
    
    global_ids = []
    global_embeddings = []
    global_texts = []
    global_metadata = []
    unprocessed = 0
    
    for p in tqdm(papers, desc="Global Embeddings"):
        pid = p.get("paper_id", "")
        title = p.get("title", "")
        whole_text = p.get("whole_content", "").strip()
        
        if not whole_text:
            unprocessed += 1
            continue
        
        avg_emb = get_avg_embedding(whole_text)
        if avg_emb is None:
            unprocessed += 1
            continue
        
        global_ids.append(pid)
        global_embeddings.append(avg_emb)
        global_texts.append(whole_text[:500])
        global_metadata.append({
            'paper_id': pid,
            'title': title
        })
    
    print(f"✓ Global embeddings: {len(global_ids)}")
    print(f"  Unprocessed: {unprocessed}")
    
    return global_ids, global_embeddings, global_texts, global_metadata
 
 
def generate_section_embeddings(papers):
    """Generate section-wise embeddings"""
    print("Generating section embeddings...")
    
    section_ids = []
    section_embeddings = []
    section_texts = []
    section_metadata = []
    
    for paper in tqdm(papers, desc="Section Embeddings"):
        pid = paper.get("paper_id", "")
        title = paper.get("title", "")
        
        for section in SECTION_TYPES:
            text = paper.get(section, "").strip()
            if not text:
                continue
            
            avg_emb = get_avg_embedding(text)
            if avg_emb is None:
                continue
            
            section_ids.append(f"{pid}_{section}")
            section_embeddings.append(avg_emb)
            section_texts.append(text[:500])
            section_metadata.append({
                'paper_id': pid,
                'title': title,
                'section': section
            })
    
    print(f"✓ Section embeddings: {len(section_ids)}")
    
    return section_ids, section_embeddings, section_texts, section_metadata
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB STORAGE
# ─────────────────────────────────────────────────────────────────────────────
 
def store_to_chromadb(global_ids, global_embeddings, global_texts, global_metadata,
                      section_ids, section_embeddings, section_texts, section_metadata):
    """Store embeddings to ChromaDB"""
    print("Storing to ChromaDB...")
    
    try:
        if global_ids:
            global_collection.add(
                ids=global_ids,
                embeddings=global_embeddings,
                documents=global_texts,
                metadatas=global_metadata
            )
            print(f"✓ Stored {len(global_ids)} global embeddings")
    except Exception as e:
        print(f"✗ Error storing global: {e}")
    
    try:
        if section_ids:
            section_collection.add(
                ids=section_ids,
                embeddings=section_embeddings,
                documents=section_texts,
                metadatas=section_metadata
            )
            print(f"✓ Stored {len(section_ids)} section embeddings")
    except Exception as e:
        print(f"✗ Error storing sections: {e}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# BACKUP UPDATE
# ─────────────────────────────────────────────────────────────────────────────
 
def update_backups(global_ids, global_embeddings, global_texts, global_metadata,
                   section_ids, section_embeddings, section_texts, section_metadata):
    """Update NPZ backup files with new embeddings"""
    print("Updating backups...")
    
    try:
        # Load existing
        emb_data = np.load(EMBEDDINGS_BACKUP, allow_pickle=True)
        meta_data = np.load(METADATA_BACKUP, allow_pickle=True)
        txt_data = np.load(TEXTS_BACKUP, allow_pickle=True)
        
        # Combine
        global_ids_all = np.concatenate([emb_data['global_ids'], np.array(global_ids)])
        global_emb_all = np.concatenate([emb_data['global_embeddings'], np.array(global_embeddings)])
        global_meta_all = np.concatenate([meta_data['global_metadata'], np.array(global_metadata)])
        global_txt_all = np.concatenate([txt_data['global_texts'], np.array(global_texts)])
        
        section_ids_all = np.concatenate([emb_data['section_ids'], np.array(section_ids)])
        section_emb_all = np.concatenate([emb_data['section_embeddings'], np.array(section_embeddings)])
        section_meta_all = np.concatenate([meta_data['section_metadata'], np.array(section_metadata)])
        section_txt_all = np.concatenate([txt_data['section_texts'], np.array(section_texts)])
        
        # Save
        np.savez(EMBEDDINGS_BACKUP,
                global_ids=global_ids_all,
                global_embeddings=global_emb_all,
                section_ids=section_ids_all,
                section_embeddings=section_emb_all)
        
        np.savez(METADATA_BACKUP,
                global_metadata=global_meta_all,
                section_metadata=section_meta_all)
        
        np.savez(TEXTS_BACKUP,
                global_texts=global_txt_all,
                section_texts=section_txt_all)
        
        print("✓ Backups updated")
        return True
    except Exception as e:
        print(f"✗ Error updating backups: {e}")
        return False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
 
def process_papers(papers_json_path, update_backups_flag=True):
    """
    Complete pipeline: Load JSON → Generate embeddings → Store → Update backups
    
    Args:
        papers_json_path: Path to JSON file with papers
        update_backups_flag: Whether to update NPZ backups
    
    Example:
        >>> process_papers("papers.json", update_backups_flag=True)
    """
    
    print("\n" + "="*80)
    print("EMBEDDING GENERATION PIPELINE")
    print("="*80 + "\n")
    
    # Load papers
    print(f"Loading papers from {papers_json_path}...")
    with open(papers_json_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    print(f"✓ Loaded {len(papers)} papers\n")
    
    # Generate embeddings
    global_ids, global_embeddings, global_texts, global_metadata = generate_global_embeddings(papers)
    print()
    section_ids, section_embeddings, section_texts, section_metadata = generate_section_embeddings(papers)
    print()
    
    # Store to ChromaDB
    store_to_chromadb(global_ids, global_embeddings, global_texts, global_metadata,
                     section_ids, section_embeddings, section_texts, section_metadata)
    print()
    
    # Update backups
    if update_backups_flag:
        update_backups(global_ids, global_embeddings, global_texts, global_metadata,
                      section_ids, section_embeddings, section_texts, section_metadata)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")
    
    return {
        "global_embeddings": len(global_ids),
        "section_embeddings": len(section_ids),
        "backups_updated": update_backups_flag
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
 
def check_embedding_exists(paper_id):
    """Check if embedding exists in ChromaDB"""
    try:
        result = global_collection.get(ids=[paper_id])
        return len(result['ids']) > 0
    except:
        return False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TESTING MODE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_embeddings(sample_id):
    #Fetching target global embeddings and target section embeddings

    #Global embeddings
    target_global = global_collection.get(ids=[sample_id], include=["embeddings"])
    target_global_emd = target_global["embeddings"][0]

    section_data = section_collection.get(
        where={'paper_id': sample_id}, include=['embeddings','metadatas']
    )
    #Section embeddings
    target_sections_emd = {}

    for emb, meta in zip(section_data['embeddings'], section_data['metadatas']):
        target_sections_emd[meta['section']] = emb

    return target_global_emd, target_sections_emd