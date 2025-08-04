import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY
from typing import List, Tuple, Dict
import numpy as np
from tiktoken import get_encoding
from utils import setup_logging

def embed_data(chunks: List, relations: List[str], themes: List[str], model_name: str = "text-embedding-ada-002") -> Dict[str, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]]:
    """
    Embed relations and themes per document using OpenAI embeddings.
    
    Args:
        chunks: List of document chunks (LangChain Document objects).
        relations: List of relation triples.
        themes: List of themes/concepts.
        model_name: Name of the embedding model.
    
    Returns:
        Dictionary mapping document source to (embedded_relations, embedded_themes).
    """
    setup_logging("../logs", "embedder")
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    EMBEDDING_COST_PER_1K_TOKENS = 0.0001  # $0.0001/1,000 tokens for text-embedding-ada-002
    total_tokens = 0
    total_cost = 0.0
    
    embedded_data = {}
    source = chunks[0].metadata.get("source", "unknown")
    print(f"Input Relations: {relations}")
    print(f"Input Themes: {themes}")
    
    embedded_relations = []
    if relations:
        try:
            relation_embeddings = embeddings.embed_documents(relations)
            relation_tokens = sum(len(tokenizer.encode(r)) for r in relations)
            relation_cost = (relation_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
            total_tokens += relation_tokens
            total_cost += relation_cost
            embedded_relations = [(r, np.array(e)) for r, e in zip(relations, relation_embeddings)]
            print(f"Tokens for relations: {relation_tokens}, Cost: ${relation_cost:.6f}")
        except Exception as e:
            print(f"Error embedding relations: {e}")
    
    embedded_themes = []
    if themes:
        try:
            valid_themes = [t for t in themes if t.strip()]  # Only exclude empty strings
            if valid_themes:
                theme_embeddings = embeddings.embed_documents(valid_themes)
                theme_tokens = sum(len(tokenizer.encode(t)) for t in valid_themes)
                theme_cost = (theme_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
                total_tokens += theme_tokens
                total_cost += theme_cost
                embedded_themes = [(t, np.array(e)) for t, e in zip(valid_themes, theme_embeddings)]
                print(f"Tokens for themes: {theme_tokens}, Cost: ${theme_cost:.6f}")
            else:
                print("No valid themes to embed")
        except Exception as e:
            print(f"Error embedding themes: {e}")
    
    embedded_data[source] = (embedded_relations, embedded_themes)
    print(f"Embedded {len(embedded_relations)} unique relations and {len(embedded_themes)} unique themes for {source}")
    print(f"Total OpenAI API Usage for Embeddings: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    
    return embedded_data