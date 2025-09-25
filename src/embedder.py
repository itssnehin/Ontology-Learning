import logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tiktoken import get_encoding
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from langchain_openai import OpenAIEmbeddings

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

def visualize_embeddings(
    embedded_data: Dict[str, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]],
    output_path: str = "../plots/embeddings_plot.png"
):
    """Visualize embeddings using t-SNE in a 2D scatter plot."""
    all_labels = []
    all_embeddings = []
    colors = []
    
    for source, (embedded_relations, embedded_themes) in embedded_data.items():
        for rel, emb in embedded_relations:
            all_labels.append(f"Relation: {rel}")
            all_embeddings.append(emb)
            colors.append("#1f77b4")  # Blue for relations
        for theme, emb in embedded_themes:
            all_labels.append(f"Theme: {theme}")
            all_embeddings.append(emb)
            colors.append("#ff7f0e")  # Orange for themes

    if not all_embeddings:
        logger.warning("No embeddings to visualize.")
        return

    embeddings_array = np.array(all_embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
    reduced_embeddings = tsne.fit_transform(embeddings_array)

    plt.figure(figsize=(14, 10))
    for i, (x, y) in enumerate(reduced_embeddings):
        plt.scatter(x, y, c=colors[i], alpha=0.7)
        plt.text(x + 0.1, y + 0.1, all_labels[i], fontsize=9)

    plt.title("t-SNE Visualization of Relations and Themes")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"Scatter plot saved to {output_path}")
    plt.show()

def embed_data(
    chunks: List,
    relations: List[str],
    themes: List[str],
    model_name: str = EMBEDDING_MODEL,
    visualize: bool = True
) -> Dict[str, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]]:
    """Embed relations and themes per document using OpenAI embeddings."""
    
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    EMBEDDING_COST_PER_1K_TOKENS = 0.0001
    total_tokens = 0
    total_cost = 0.0
    
    embedded_data = {}
    source = chunks[0].metadata.get("source", "unknown")
    logger.info(f"Input Relations: {relations}")
    logger.info(f"Input Themes: {themes}")
    
    embedded_relations = []
    if relations:
        try:
            relation_embeddings = embeddings.embed_documents(relations)
            relation_tokens = sum(len(tokenizer.encode(r)) for r in relations)
            relation_cost = (relation_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
            total_tokens += relation_tokens
            total_cost += relation_cost
            embedded_relations = [(r, np.array(e)) for r, e in zip(relations, relation_embeddings)]
            logger.info(f"Tokens for relations: {relation_tokens}, Cost: ${relation_cost:.6f}")
        except Exception as e:
            logger.error(f"Error embedding relations: {e}", exc_info=True)
    
    embedded_themes = []
    if themes:
        try:
            valid_themes = [t for t in themes if t and t.strip()]
            if valid_themes:
                theme_embeddings = embeddings.embed_documents(valid_themes)
                theme_tokens = sum(len(tokenizer.encode(t)) for t in valid_themes)
                theme_cost = (theme_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
                total_tokens += theme_tokens
                total_cost += theme_cost
                embedded_themes = [(t, np.array(e)) for t, e in zip(valid_themes, theme_embeddings)]
                logger.info(f"Tokens for themes: {theme_tokens}, Cost: ${theme_cost:.6f}")
            else:
                logger.warning("No valid themes to embed")
        except Exception as e:
            logger.error(f"Error embedding themes: {e}", exc_info=True)
    
    embedded_data[source] = (embedded_relations, embedded_themes)
    logger.info(f"Embedded {len(embedded_relations)} relations and {len(embedded_themes)} themes for {source}")
    logger.info(f"Total OpenAI API Usage for Embeddings: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    
    if visualize:
        visualize_embeddings(embedded_data)
    
    return embedded_data

if __name__ == "__main__":
    from .data_loader import load_and_split_data
    from .relation_extractor import extract_relations
    from .idea_extractor import extract_ideas
    
    sample_chunks = load_and_split_data()
    if sample_chunks:
        sample_relations = extract_relations(sample_chunks[:2])
        sample_themes = list(set(extract_ideas(sample_chunks[:2])))
        embedded_results = embed_data(sample_chunks[:2], sample_relations, sample_themes, visualize=True)
    else:
        logger.warning("No chunks found to run the embedder example.")