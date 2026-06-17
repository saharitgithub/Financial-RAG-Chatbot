import os
import fitz  # PyMuPDF

def load_pdfs(data_folder):
    texts = []

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)

                doc = fitz.open(path)
                full_text = ""

                for page in doc:
                    # Extract text with better layout preservation
                    blocks = page.get_text("blocks")

                    for block in blocks:
                        # block[4] is the text content
                        block_text = block[4].strip()

                        if block_text:
                            full_text += block_text + "\n"

                    full_text += "\n"  # Page separator

                if full_text.strip():
                    texts.append(full_text)
                    print(f"✅ Loaded: {file} ({len(full_text)} chars)")

    return texts