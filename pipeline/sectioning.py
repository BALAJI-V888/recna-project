"""
Groq Section Mapper - LLM-Based (PDF to JSON)
Step 1: Extract text from PDF
Step 2: Classify entire text with Groq LLM
Step 3: Build JSON object with standardized sections
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List
from groq import Groq
import PyPDF2


from config import client


SAVE_PATH = r"C:\Users\sargu\OneDrive\Desktop\Final_year_project\ReCNA\dataset\unarxive\cs_papers_mapped.json"

 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Extract text from PDF
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from PDF file
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text content (string)
    
    Example:
        >>> text = extract_text_from_pdf("paper.pdf")
        >>> print(len(text))  # Character count
    """
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {e}")
    
    return text
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Classify paper content with Groq LLM
# ─────────────────────────────────────────────────────────────────────────────
 
def classify_paper_with_groq(paper_text: str, 
                             paper_id: str = "paper_001",
                             title: str = "") -> Dict[str, str]:
    """
    Send entire paper text to Groq LLM for section classification
    
    Args:
        paper_text: Full paper text extracted from PDF
        paper_id: Unique paper identifier
        title: Paper title (optional)
    
    Returns:
        Dictionary with classified sections:
        {
            "introduction": "introduction text...",
            "methodology": "methodology text...",
            "discussion": "discussion text...",
            "conclusion": "conclusion text...",
            "other": "other content..."
        }
    """
    
    prompt = f"""You are a scientific paper content classifier. 
 
Your task: Analyze the following paper text and classify it into the specified sections.
 
The paper content is:
===== PAPER START =====
{paper_text}
===== PAPER END =====
 
Instructions:
1. Read the entire paper content carefully
2. Identify and classify content into these exact sections:
   - introduction: Introduction, background, related work, motivation, literature review
   - methodology: Methods, methodology, approach, algorithm, technique, framework, architecture
   - discussion: Discussion, analysis, interpretation, implications, results analysis
   - conclusion: Conclusion, conclusions, future work, limitations, recommendations
   - other: Abstract, references, appendix, acknowledgments, figures, tables, any other content
 
3. Extract the actual text content for each section from the paper
4. Keep the original text without modification
5. If a section is not present, leave it empty
 
Return ONLY valid JSON with this exact format (no explanation, no markdown):
{{
  "introduction": "exact text from introduction section...",
  "methodology": "exact text from methodology section...",
  "discussion": "exact text from discussion section...",
  "conclusion": "exact text from conclusion section...",
  "other": "text from references, appendix, or other sections..."
}}
 
Important:
- Preserve the original text exactly as it appears
- Do not summarize or paraphrase
- Ensure each section contains only text belonging to that section
- Return valid JSON only"""
 
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4000
    )
    
    # Parse response
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    
    try:
        classified_sections = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {raw[:500]}")
        # Return empty structure on error
        classified_sections = {
            "introduction": "",
            "methodology": "",
            "discussion": "",
            "conclusion": "",
            "other": ""
        }
    
    return classified_sections
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build final JSON object
# ─────────────────────────────────────────────────────────────────────────────
 
def build_paper_json(classified_sections: Dict[str, str],
                     paper_id: str = "paper_001",
                     title: str = "") -> Dict[str, Any]:
    """
    Build final paper JSON object from classified sections
    
    Args:
        classified_sections: Dictionary with classified sections from Groq
        paper_id: Paper identifier
        title: Paper title
    
    Returns:
        Final paper JSON object:
        {
            "paper_id": "xxx",
            "title": "Paper Title",
            "introduction": "text...",
            "methodology": "text...",
            "discussion": "text...",
            "conclusion": "text...",
            "other": "text..."
        }
    """
    
    result = {
        "paper_id": paper_id,
        "title": title,
        "introduction": classified_sections.get("introduction", ""),
        "methodology": classified_sections.get("methodology", ""),
        "discussion": classified_sections.get("discussion", ""),
        "conclusion": classified_sections.get("conclusion", ""),
        "other": classified_sections.get("other", "")
    }
    
    return result
 
 

  
# ─────────────────────────────────────────────────────────────────────────────
# LOAD EXISTING PAPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def load_existing_papers() -> List[Dict[str, Any]]:
    """
    Load existing papers from JSON file
    
    Returns:
        List of papers or empty list if file doesn't exist
    """
    try:
        with open(SAVE_PATH, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            print(f"✓ Loaded {len(papers)} existing papers")
            return papers
    except FileNotFoundError:
        print(f"⚠️  File not found, starting fresh")
        return []
    except Exception as e:
        print(f"⚠️  Error loading file: {e}")
        return []
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SAVE PAPERS
# ─────────────────────────────────────────────────────────────────────────────
 
def save_paper_json(paper_json: Dict[str, Any], append: bool = True) -> bool:
    """
    Save single paper to JSON file
    
    Args:
        paper_json: Paper dictionary from pdf_to_paper_json()
        append: Whether to append to existing papers (True) or overwrite (False)
    
    Returns:
        True if successful
    
    Example:
        >>> paper = pdf_to_paper_json("paper.pdf", "p001", "Title")
        >>> save_paper_json(paper, append=True)
    """
    
    try:
        if append:
            # Load existing
            papers = load_existing_papers()
            
            # Check if paper already exists
            existing_ids = [p.get("paper_id") for p in papers]
            if paper_json.get("paper_id") in existing_ids:
                print(f"⚠️  Paper {paper_json.get('paper_id')} already exists, skipping")
                return False
            
            # Append new paper
            papers.append(paper_json)
            print(f"→ Appending paper {paper_json.get('paper_id')}")
        else:
            # Create new list with just this paper
            papers = [paper_json]
            print(f"→ Overwriting with new paper")
        
        # Create directory if it doesn't exist
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(papers)} papers to {SAVE_PATH}")
        return True
    
    except Exception as e:
        print(f"✗ Error saving paper: {e}")
        return False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# UPDATE PAPER
# ─────────────────────────────────────────────────────────────────────────────
 
def update_paper_json(paper_json: Dict[str, Any]) -> bool:
    """
    Update existing paper in JSON file
    
    Args:
        paper_json: Updated paper dictionary
    
    Returns:
        True if successful
    """
    
    try:
        # Load existing
        papers = load_existing_papers()
        
        # Find and update
        paper_id = paper_json.get("paper_id")
        found = False
        
        for i, p in enumerate(papers):
            if p.get("paper_id") == paper_id:
                papers[i] = paper_json
                found = True
                print(f"→ Updating paper {paper_id}")
                break
        
        if not found:
            print(f"⚠️  Paper {paper_id} not found, adding as new")
            papers.append(paper_json)
        
        # Save
        Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Updated {len(papers)} papers")
        return True
    
    except Exception as e:
        print(f"✗ Error updating paper: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION: PDF to JSON (3 steps combined)
# ─────────────────────────────────────────────────────────────────────────────
 
def pdf_to_paper_json(pdf_path: str,
                      paper_id: str = None,
                      title: str = "") -> Dict[str, Any]:
    """
    Complete pipeline: PDF → Extract Text → Classify with Groq → Build JSON
    
    Args:
        pdf_path: Path to PDF file
        paper_id: Unique paper identifier (auto-generated if None)
        title: Paper title (optional)
    
    Returns:
        Final paper JSON object with classified sections
    
    Example:
        >>> paper_json = pdf_to_paper_json("research_paper.pdf", "p001", "My Paper")
        >>> print(paper_json["introduction"])
    """
    
    # Auto-generate paper ID if not provided
    if paper_id is None:
        from datetime import datetime
        paper_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # print(f"Step 1: Extracting text from PDF: {pdf_path}")
    paper_text = extract_text_from_pdf(pdf_path)
    # print(f"  ✓ Extracted {len(paper_text):,} characters")
    
    # print(f"Step 2: Classifying content with Groq LLM...")
    classified_sections = classify_paper_with_groq(paper_text, paper_id, title)
    # print(f"  ✓ Classification complete")
    
    # print(f"Step 3: Building JSON object...")
    paper_json = build_paper_json(classified_sections, paper_id, title)
    # print(f"  ✓ JSON object created")
    
    return paper_json
 
def pdf_to_paper_json_and_save(pdf_path: str,
                               paper_id: str = None,
                               title: str = "",
                               append: bool = True) -> Dict[str, Any]:
    """
    Complete pipeline: PDF → JSON → Save to file
    
    Args:
        pdf_path: Path to PDF file
        paper_id: Unique paper identifier (auto-generated if None)
        title: Paper title (optional)
        append: Whether to append to existing papers
    
    Returns:
        Paper JSON object
    
    Example:
        >>> paper = pdf_to_paper_json_and_save("paper.pdf", "p001", "My Paper")
    """
        
    print("\n" + "="*80)
    print("PDF → JSON → SAVE PIPELINE")
    print("="*80 + "\n")
    
    # Step 1-3: Convert PDF to JSON
    print("Steps 1-3: Converting PDF to JSON with sections...")
    paper_json = pdf_to_paper_json(pdf_path, paper_id, title)
    print()
    
    # Step 4: Save to file
    print("Step 4: Saving to file...")
    save_paper_json(paper_json, append=append)

    
    return paper_json