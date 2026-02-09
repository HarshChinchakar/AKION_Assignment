import pdfplumber
import json
import re
import uuid
from pathlib import Path
from typing import List, Dict

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "Documents" / "corep-own-funds-instructions (1).pdf"
OUTPUT_FILE = BASE_DIR / "Processed" / "regulatory_knowledge_base.json"

class LayoutAwareParser:
    def __init__(self):
        self.chunks = []
        self.current_template = "General Instructions"
        self.current_row = "Intro"
        self.buffer = []
        # Flag to skip the Table of Contents
        self.content_started = False

    def is_bold(self, char_list: List[Dict]) -> bool:
        """Checks if the majority of characters in a line are Bold."""
        if not char_list: return False
        bold_chars = [c for c in char_list if "Bold" in c.get("fontname", "")]
        # If >80% of characters are bold, it's a header/row identifier
        return len(bold_chars) / len(char_list) > 0.8

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'Page \d+ of \d+', '', text) # Remove page numbers
        return text

    def flush_chunk(self):
        if not self.buffer: return

        raw_text = " ".join(self.buffer)
        clean_text = self.clean_text(raw_text)
        
        if len(clean_text) < 10: return # Ignore noise

        # STRUCTURED PROMPT ENGINEERING (Baked into the data)
        # This ensures the embedding model understands the specific banking context.
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
            "embedding_content": embedding_content # USE THIS FOR CHROMA DB
        })
        self.buffer = []

    def parse(self):
        print(f"👁️  Starting Visual-Semantic Extraction...")
        
        with pdfplumber.open(INPUT_FILE) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract words/lines with detailed font information
                lines = page.extract_text_lines()
                
                for line in lines:
                    text = line['text'].strip()
                    chars = line['chars']
                    
                    if not text: continue

                    # 1. VISUAL FILTER: Skip Table of Contents
                    # We look for the first visual occurrence of "C 01.00"
                    if not self.content_started:
                        if "C 01.00" in text and "OWN FUNDS" in text:
                            self.content_started = True
                            print(f"   📍 content locked on Page {i+1}")
                        else:
                            continue # Skip TOC lines

                    # 2. VISUAL TRIGGER: Is this line BOLD?
                    is_line_bold = self.is_bold(chars)

                    # 3. LOGIC: Template Switch (Bold + "C 0...")
                    if is_line_bold and re.match(r'^C\s+\d{2}', text):
                        self.flush_chunk()
                        self.current_template = text
                        self.current_row = "Header"
                        print(f"   📂 Template Switch: {text}")
                        continue

                    # 4. LOGIC: Row Switch (Bold + Digits)
                    # Matches "010" or "010, 020" or "010 -"
                    if is_line_bold and re.match(r'^\d{3,4}', text):
                        self.flush_chunk()
                        # Extract just the ID "010" from "010 - Common Equity"
                        self.current_row = text.split(' ')[0].replace(',', '')
                        self.buffer.append(text)
                        continue

                    # 5. CONTENT: Normal text appends to current bucket
                    self.buffer.append(text)

                print(f"   Processed Page {i+1}", end='\r')

        self.flush_chunk()
        
        # Save
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(self.chunks, f, indent=4)

        print(f"\n✅ Extraction Complete. {len(self.chunks)} perfectly contextualized chunks.")

if __name__ == "__main__":
    parser = LayoutAwareParser()
    parser.parse()