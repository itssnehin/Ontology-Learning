import os
import logging
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DATA_DIR, MARKDOWN_FILES, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def load_and_split_data() -> List[Document]:
    """Load Markdown files and split into chunks."""
    
    logger.info(f"Loading Markdown files from: {DATA_DIR}")
    logger.info(f"Available Markdown files: {MARKDOWN_FILES}")
    
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    for file in MARKDOWN_FILES:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
                
                split_chunks = splitter.split_documents(docs)
                chunks.extend(split_chunks)
                logger.info(f"Split into {len(split_chunks)} chunks for {file}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info(f"Total chunks loaded: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    loaded_chunks = load_and_split_data()
    logger.info(f"Successfully loaded {len(loaded_chunks)} chunks.")