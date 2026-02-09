import streamlit as st
import pdfplumber
import re
import uuid
import json
import shutil
import gc
import os
import sys
from pathlib import Path
from typing import List, Dict, Generator

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from fastembed import TextEmbedding
except ImportError:
    st.error("FastEmbed not installed. Please check requirements.txt")

st.set_page_config(
    page_title="PRA Regulatory Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "DataExtraction" / "Documents"
PROCESSED_DIR = BASE_DIR / "DataExtraction" / "Processed"
PDF_PATH = DOCS_DIR / "corep-own-funds-instructions (1).pdf"
JSON_OUTPUT_PATH = PROCESSED_DIR / "regulatory_knowledge_base.json"
DB_PATH = BASE_DIR / "vector_store"

class CustomFastEmbed(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name, threads=None)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [e.tolist() for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(self.model.embed([text])).tolist()

class LayoutAwareParser:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.chunks = []
        self.current_template = "General Instructions"
        self.current_row = "Intro"
        self.buffer = []
        self.content_started = False

    def is_bold(self, char_list: List[Dict]) -> bool:
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
        
        if len(clean_text) < 10: return

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
                if progress_bar:
                    progress_bar.progress((i + 1) / total_pages, text=f"Scanning Page {i+1}/{total_pages}...")

                lines = page.extract_text_lines()
                
                for line in lines:
                    text = line['text'].strip()
                    chars = line['chars']
                    if not text: continue

                    if not self.content_started:
                        if "C 01.00" in text and "OWN FUNDS" in text:
                            self.content_started = True
                        else:
                            continue

                    is_line_bold = self.is_bold(chars)

                    if is_line_bold and re.match(r'^C\s+\d{2}', text):
                        self.flush_chunk()
                        self.current_template = text
                        self.current_row = "Header"
                        continue

                    if is_line_bold and re.match(r'^\d{3,4}', text):
                        self.flush_chunk()
                        self.current_row = text.split(' ')[0].replace(',', '')
                        self.buffer.append(text)
                        continue

                    self.buffer.append(text)
        
        self.flush_chunk()
        return self.chunks

def batch_generator(data: List[Dict], batch_size: int) -> Generator[List[Document], None, None]:
    batch = []
    for item in data:
        meta = item.get("metadata", {}).copy()
        meta["display_text"] = item.get("page_content", "")
        meta["chunk_id"] = item.get("id", "")

        text_content = item.get("embedding_content", item.get("page_content", ""))
        
        doc = Document(page_content=text_content, metadata=meta)
        batch.append(doc)
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch

def build_vector_db(data: List[Dict], status_container):
    status_container.info("Initializing Embedding Engine...")
    try:
        embedding_engine = CustomFastEmbed()
    except Exception as e:
        status_container.error(f"Failed to load model: {e}")
        return False

    if DB_PATH.exists():
        try:
            shutil.rmtree(DB_PATH)
        except Exception:
            pass

    db = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embedding_engine
    )

    BATCH_SIZE = 50
    total = len(data)
    count = 0

    status_container.info(f"Embedding {total} chunks...")
    progress_bar = status_container.progress(0)

    for batch in batch_generator(data, BATCH_SIZE):
        try:
            db.add_documents(batch)
            count += len(batch)
            del batch
            gc.collect()
            progress_bar.progress(min(count / total, 1.0))
        except Exception as e:
            status_container.error(f"Batch Error: {e}")
            return False

    progress_bar.empty()
    return True

st.title("🏦 PRA COREP Reporting Assistant")
st.markdown("""
**Status:** Prototype Phase 1  
**Objective:** Context-Aware Extraction & Vector Embedding of EBA/PRA Regulatory Instructions.
""")
st.divider()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Bank_of_England_Badge.svg/1200px-Bank_of_England_Badge.svg.png", width=100)
    st.header("System Status")
    
    if PDF_PATH.exists():
        st.success(f"✅ Document Found:\n{PDF_PATH.name}")
    else:
        st.error(f"❌ Missing Document:\n{PDF_PATH.name}")
        st.info("Please upload the PDF to `DataExtraction/Documents/`")

    if DB_PATH.exists() and any(DB_PATH.iterdir()):
        st.success("✅ Vector Database: Ready")
    else:
        st.warning("⚠️ Vector Database: Not Built")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("⚙️ Ingestion Pipeline")
    st.markdown("Run the full pipeline: Extract Text → Chunk → Embed into Vector DB.")
    
    if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        if not PDF_PATH.exists():
            st.error("Cannot run: Source file missing.")
        else:
            status_box = st.empty()
            try:
                status_box.info("Step 1/2: Extracting Rules from PDF...")
                parser = LayoutAwareParser(PDF_PATH)
                chunks = parser.parse()
                
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                with open(JSON_OUTPUT_PATH, 'w') as f:
                    json.dump(chunks, f, indent=4)
                
                st.session_state['chunks'] = chunks
                
                status_box.info(f"Step 2/2: Building Vector Database with {len(chunks)} chunks...")
                success = build_vector_db(chunks, status_box)
                
                if success:
                    status_box.success("Pipeline Complete! Database is ready for RAG.")
                    st.balloons()
                else:
                    status_box.error("Pipeline Failed during Embedding.")
                
            except Exception as e:
                st.error(f"Critical Error: {str(e)}")

with col2:
    st.subheader("📖 Knowledge Base Viewer")
    st.markdown("Inspect the extracted rules.")

    data_source = []
    if 'chunks' in st.session_state:
        data_source = st.session_state['chunks']
    elif JSON_OUTPUT_PATH.exists():
        with open(JSON_OUTPUT_PATH, 'r') as f:
            data_source = json.load(f)
            
    if data_source:
        search_term = st.text_input("🔍 Search Rule ID (e.g., '010')", placeholder="Type row number...")
        
        st.caption(f"Total Rules Indexed: {len(data_source)}")
        
        count = 0
        for chunk in data_source:
            meta = chunk['metadata']
            if search_term and search_term not in meta['row_id'] and search_term.lower() not in meta['template'].lower():
                continue
                
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
        st.info("Waiting for pipeline execution...")