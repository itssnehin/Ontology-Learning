import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from tiktoken import get_encoding
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from langchain_openai import OpenAIEmbeddings

# Import centralized configuration
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_COST_PER_1K_TOKENS

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
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_obj, bbox_inches="tight")
    logger.info(f"Scatter plot saved to {output_path_obj}")
    # plt.show() # Commented out to prevent blocking in an automated pipeline

def embed_data(
    chunks: List,
    relations: List[str],
    themes: List[str],
    model_name: str = EMBEDDING_MODEL,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Embed relations and themes per document, tracking cost using centralized config.
    """
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    total_tokens = 0
    total_cost = 0.0
    
    embedded_data = {}
    source = chunks[0].metadata.get("source", "unknown") if chunks else "unknown"
    logger.info(f"Embedding data for source: {source}")
    
    embedded_relations = []
    if relations:
        try:
            relation_embeddings = embeddings.embed_documents(relations)
            relation_tokens = sum(len(tokenizer.encode(r)) for r in relations)
            relation_cost = (relation_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
            total_tokens += relation_tokens
            total_cost += relation_cost
            embedded_relations = [(r, np.array(e)) for r, e in zip(relations, relation_embeddings)]
            logger.info(f"Embedded {len(relations)} relations. Tokens: {relation_tokens}, Cost: ${relation_cost:.6f}")
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
                logger.info(f"Embedded {len(valid_themes)} themes. Tokens: {theme_tokens}, Cost: ${theme_cost:.6f}")
            else:
                logger.warning("No valid themes provided to embed.")
        except Exception as e:
            logger.error(f"Error embedding themes: {e}", exc_info=True)
    
    embedded_data[source] = (embedded_relations, embedded_themes)
    logger.info(f"Total OpenAI API Usage for Embeddings: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    
    if visualize:
        visualize_embeddings(embedded_data)
    
    return embedded_data

if __name__ == "__main__":
    from src.data_loader import load_and_split_data
    from src.relation_extractor import extract_relations
    from src.idea_extractor import extract_ideas

    logger.info("Running embedder.py as a standalone script for demonstration.")
    sample_chunks = load_and_split_data()
    
    if sample_chunks:
        # Use a small subset for demonstration to manage cost and time
        demo_chunks = sample_chunks[:2]
        logger.info(f"Processing {len(demo_chunks)} chunks for the demo.")
        
        relations_list = extract_relations(demo_chunks)
        themes_list = list(set(extract_ideas(demo_chunks)))
        
        embedded_results = embed_data(demo_chunks, relations_list, themes_list, visualize=True)
        logger.info("Embedder demonstration finished.")
    else:
        logger.warning("No markdown files found in the data directory. Cannot run embedder demonstration.")