import json
import shutil
import os
import gc
import sys
from pathlib import Path
from typing import List, Dict, Generator

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from fastembed import TextEmbedding
except ImportError:
    print("Install fastembed: pip install fastembed")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "DataExtraction" / "Processed" / "regulatory_knowledge_base.json"
DB_PATH = PROJECT_ROOT / "vector_store"

class CustomFastEmbed(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name, threads=None)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Fix: Convert numpy arrays to standard python lists
        return [e.tolist() for e in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        # Fix: Convert single numpy array to standard python list
        return next(self.model.embed([text])).tolist()

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

def build_database():
    print("Starting DB Build...")
    
    if not DATA_PATH.exists():
        print(f"File missing: {DATA_PATH}")
        return

    try:
        embedding_engine = CustomFastEmbed()
    except Exception as e:
        print(f"Model Init Error: {e}")
        return

    if DB_PATH.exists():
        try:
            shutil.rmtree(DB_PATH)
        except Exception:
            pass

    db = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embedding_engine
    )

    try:
        with open(DATA_PATH, 'r') as f:
            raw_data = json.load(f)
    except Exception:
        return

    BATCH_SIZE = 50
    total = len(raw_data)
    count = 0

    for batch in batch_generator(raw_data, BATCH_SIZE):
        try:
            db.add_documents(batch)
            count += len(batch)
            del batch
            gc.collect()
            print(f"Embedded {count}/{total}...", end='\r')
        except Exception as e:
            print(f"\nBatch Error: {e}")

    print(f"\nSuccess. DB saved at: {DB_PATH}")

if __name__ == "__main__":
    build_database()