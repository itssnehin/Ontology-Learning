import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY
from typing import List, Tuple, Dict
import numpy as np
from tiktoken import get_encoding
from utils import setup_logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(embedded_data: Dict[str, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]], output_path: str = "../plots/embeddings_plot.png"):
    """
    Visualize embeddings using t-SNE in a 2D scatter plot.
    
    Args:
        embedded_data: Dictionary mapping document source to (embedded_relations, embedded_themes).
        output_path: Path to save the scatter plot image.
    """
    all_labels = []
    all_embeddings = []
    colors = []
    
    for source, (embedded_relations, embedded_themes) in embedded_data.items():
        # Relations
        for rel, emb in embedded_relations:
            all_labels.append(f"Relation: {rel}")
            all_embeddings.append(emb)
            colors.append("#1f77b4")  # Blue for relations
        # Themes
        for theme, emb in embedded_themes:
            all_labels.append(f"Theme: {theme}")
            all_embeddings.append(emb)
            colors.append("#ff7f0e")  # Orange for themes

    if not all_embeddings:
        print("No embeddings to visualize.")
        return

    # Convert to NumPy array
    embeddings_array = np.array(all_embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    reduced_embeddings = tsne.fit_transform(embeddings_array)

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    for i, (x, y) in enumerate(reduced_embeddings):
        plt.scatter(x, y, c=colors[i], alpha=0.6)
        plt.text(x + 0.1, y + 0.1, all_labels[i], fontsize=8)

    plt.title("t-SNE Visualization of Relations and Themes")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save and show plot
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Scatter plot saved to {output_path}")
    plt.show()

def embed_data(chunks: List, relations: List[str], themes: List[str], model_name: str = "text-embedding-ada-002", visualize: bool = True) -> Dict[str, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]]:
    """
    Embed relations and themes per document using OpenAI embeddings.
    
    Args:
        chunks: List of document chunks (LangChain Document objects).
        relations: List of relation triples.
        themes: List of themes/concepts.
        model_name: Name of the embedding model.
        visualize: Whether to generate a t-SNE scatter plot of embeddings.
    
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
    
    if visualize:
        visualize_embeddings(embedded_data)
    
    return embedded_data

if __name__ == "__main__":
    from data_loader import load_and_split_data
    from relation_extractor import extract_relations
    from idea_extractor import extract_ideas
    chunks = load_and_split_data()
    relations = extract_relations(chunks[:10])
    themes = list(set(extract_ideas(chunks[:10])))
    embedded_data = embed_data(chunks[:10], relations, themes, visualize=True)