import os
import logging
import re
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DATA_DIR, MARKDOWN_FILES, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def _preprocess_text(text: str) -> str:
    """
    Applies light-touch cleaning and normalization to the raw text content.
    - Converts to lowercase
    - Removes Markdown images and links, keeping the alt text
    - Removes common boilerplate and footer text
    - Normalizes whitespace
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove Markdown images, but keep the alt text (e.g., ![alt text](path) -> alt text)
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)

    # 3. Remove Markdown links, but keep the link text (e.g., [link text](url) -> link text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    # 4. Remove common boilerplate/footer text found in datasheets
    boilerplate = [
        r'johanson technology, inc\. reserves the right to make design changes without notice\.',
        r'all sales are subject to johanson technology, inc\. terms and conditions\.',
        r'copyright © \d{4} texas instruments incorporated',
        r'submit document feedback',
        r'important notice and disclaimer',
        r'www\.ti\.com',
        r'www\.johansontechnology\.com'
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 5. Remove standalone special characters and normalize whitespace
    text = re.sub(r'[`*#_]+', '', text) # Remove markdown formatting characters
    text = re.sub(r'\s+', ' ', text).strip() # Collapse multiple whitespace characters into one

    return text


def load_and_split_data() -> List[Document]:
    """Load Markdown files, preprocess their content, and split into chunks."""
    logger.info(f"Loading Markdown files from: {DATA_DIR}")
    logger.info(f"Available Markdown files: {MARKDOWN_FILES}")
    
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    for file in MARKDOWN_FILES:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} documents from {file_path}")

                # PREPROCESSING STEP
                cleaned_docs = []
                for doc in docs:
                    cleaned_content = _preprocess_text(doc.page_content)
                    cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

                #CHUNKING
                chunks = splitter.split_documents(cleaned_docs)
                all_chunks.extend(chunks)
                logger.info(f"Split '{file.name}' into {len(chunks)} cleaned chunks.")
            except Exception as e:
                logger.error(f"Error loading or processing {file_path}: {e}", exc_info=True)
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info(f"Total chunks loaded and preprocessed: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    loaded_chunks = load_and_split_data()
    logger.info(f"Successfully loaded {len(loaded_chunks)} chunks.")
    if loaded_chunks:
        logger.info("--- Example of a preprocessed chunk ---")
        logger.info(loaded_chunks[0].page_content)
        logger.info("---------------------------------------")