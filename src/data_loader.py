import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from config import DATA_DIR, MARKDOWN_FILES, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_data():
    """Load Markdown files and split into chunks."""
    print(f"Loading Markdown files from: {DATA_DIR}")
    print(f"Available Markdown files: {MARKDOWN_FILES}")
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    for file in MARKDOWN_FILES:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
                print(f"Loaded {len(docs)} documents from {file_path}")
                chunks.extend(splitter.split_documents(docs))
                print(f"Split into {len(chunks)} chunks for {file}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print(f"Total chunks loaded: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    chunks = load_and_split_data()
    print(f"Loaded {len(chunks)} chunks")