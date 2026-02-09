import streamlit as st
import pdfplumber
import re
import uuid
import json
import os
from pathlib import Path
from typing import List, Dict

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PRA Regulatory Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PATH SETUP ---
# Robust path handling for Streamlit Cloud
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "DataExtraction" / "Documents"
PROCESSED_DIR = BASE_DIR / "DataExtraction" / "Processed"
PDF_PATH = DOCS_DIR / "corep-own-funds-instructions (1).pdf"
JSON_OUTPUT_PATH = PROCESSED_DIR / "regulatory_knowledge_base.json"

# --- 3. EXTRACTION ENGINE (Internal Class) ---
class LayoutAwareParser:
    """
    Production-Grade Parser that uses Visual Layout analysis (Fonts, Indentation)
    to extract regulatory rules with high precision.
    """
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.chunks = []
        self.current_template = "General Instructions"
        self.current_row = "Intro"
        self.buffer = []
        self.content_started = False

    def is_bold(self, char_list: List[Dict]) -> bool:
        """Heuristic: If >80% of chars in a line are bold, it's a header."""
        if not char_list: return False
        bold_chars = [c for c in char_list if "Bold" in c.get("fontname", "")]
        return len(bold_chars) / len(char_list) > 0.8

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'Page \d+ of \d+', '', text)
        return text

    def flush_chunk(self):
        if not self.buffer: return
        raw_text = " ".join(self.buffer)
        clean_text = self.clean_text(raw_text)
        
        # Noise filter
        if len(clean_text) < 10: return

        # THE SECRET SAUCE: Enriched Embedding Content
        # We inject the "Metadata" directly into the text for the Vector DB.
        embedding_content = (
            f"Regulation: PRA Rulebook / EBA ITS. "
            f"Template: {self.current_template}. "
            f"Row ID: {self.current_row}. "
            f"Instruction: {clean_text}"
        )

        self.chunks.append({
            "id": str(uuid.uuid4()),
            "metadata": {
                "source": "Annex II",
                "template": self.current_template,
                "row_id": self.current_row,
                "type": "instruction"
            },
            "page_content": clean_text,
            "embedding_content": embedding_content
        })
        self.buffer = []

    def parse(self, progress_bar=None) -> List[Dict]:
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                # Update UI Progress if provided
                if progress_bar:
                    progress_bar.progress((i + 1) / total_pages, text=f"Scanning Page {i+1}/{total_pages}...")

                lines = page.extract_text_lines()
                
                for line in lines:
                    text = line['text'].strip()
                    chars = line['chars']
                    if not text: continue

                    # 1. VISUAL FILTER: Skip Table of Contents
                    if not self.content_started:
                        if "C 01.00" in text and "OWN FUNDS" in text:
                            self.content_started = True
                        else:
                            continue

                    is_line_bold = self.is_bold(chars)

                    # 2. TEMPLATE SWITCH (Bold + "C 0...")
                    if is_line_bold and re.match(r'^C\s+\d{2}', text):
                        self.flush_chunk()
                        self.current_template = text
                        self.current_row = "Header"
                        continue

                    # 3. ROW SWITCH (Bold + Digits)
                    if is_line_bold and re.match(r'^\d{3,4}', text):
                        self.flush_chunk()
                        self.current_row = text.split(' ')[0].replace(',', '')
                        self.buffer.append(text)
                        continue

                    self.buffer.append(text)
        
        self.flush_chunk()
        return self.chunks

# --- 4. UI LOGIC ---

# Header
st.title("🏦 PRA COREP Reporting Assistant")
st.markdown("""
**Status:** Prototype Phase 1 (Data Ingestion)  
**Objective:** Context-Aware Extraction of EBA/PRA Regulatory Instructions.
""")
st.divider()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Bank_of_England_Badge.svg/1200px-Bank_of_England_Badge.svg.png", width=100)
    st.header("System Status")
    
    # Check File Status
    if PDF_PATH.exists():
        st.success(f"✅ Document Found:\n{PDF_PATH.name}")
    else:
        st.error(f"❌ Missing Document:\n{PDF_PATH.name}")
        st.info("Please upload the PDF to `DataExtraction/Documents/`")

    # Check Database Status
    if JSON_OUTPUT_PATH.exists():
        st.success("✅ Knowledge Base: Ready")
    else:
        st.warning("⚠️ Knowledge Base: Empty")

# Main Layout
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("⚙️ Data Ingestion Engine")
    st.markdown("Run this module to parse the raw PDF into semantic chunks optimized for RAG.")
    
    if st.button("🚀 Run Extraction Pipeline", type="primary", use_container_width=True):
        if not PDF_PATH.exists():
            st.error("Cannot run: Source file missing.")
        else:
            try:
                # Progress Bar UI
                progress_bar = st.progress(0, text="Initializing Engine...")
                
                # Run Parser
                parser = LayoutAwareParser(PDF_PATH)
                chunks = parser.parse(progress_bar=progress_bar)
                
                progress_bar.empty()
                
                # Save Data
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                with open(JSON_OUTPUT_PATH, 'w') as f:
                    json.dump(chunks, f, indent=4)
                
                # Success State
                st.balloons()
                st.success(f"Extraction Complete! Generated {len(chunks)} contextual chunks.")
                st.session_state['chunks'] = chunks # Cache in memory
                
            except Exception as e:
                st.error(f"Critical Error: {str(e)}")

with col2:
    st.subheader("📖 Knowledge Base Viewer")
    st.markdown("Inspect the extracted rules to verify `Row ID` and `Context` accuracy.")

    # Load Data (Priority: Session State -> Disk -> Empty)
    data_source = []
    if 'chunks' in st.session_state:
        data_source = st.session_state['chunks']
    elif JSON_OUTPUT_PATH.exists():
        with open(JSON_OUTPUT_PATH, 'r') as f:
            data_source = json.load(f)
            
    if data_source:
        # Search Filter
        search_term = st.text_input("🔍 Search Rule ID (e.g., '010')", placeholder="Type row number...")
        
        # Stats
        st.caption(f"Total Rules Indexed: {len(data_source)}")
        
        # Display List
        count = 0
        for chunk in data_source:
            # Filter Logic
            meta = chunk['metadata']
            if search_term and search_term not in meta['row_id'] and search_term.lower() not in meta['template'].lower():
                continue
                
            # Card UI
            with st.expander(f"📌 Row {meta['row_id']} | {meta['template'][:40]}...", expanded=False):
                st.markdown(f"**Template:** `{meta['template']}`")
                st.markdown(f"**Row ID:** `{meta['row_id']}`")
                st.markdown("**Instruction Content:**")
                st.info(chunk['page_content'])
                st.markdown("**RAG Optimization (Hidden Context):**")
                st.code(chunk['embedding_content'], language="text")
            
            count += 1
            if count > 50 and not search_term:
                st.warning("⚠️ Showing first 50 rules only. Use search to find specific rows.")
                break
    else:
        st.info("Waiting for extraction... Click the button on the left.")