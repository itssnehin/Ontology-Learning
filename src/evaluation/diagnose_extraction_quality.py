import logging
from typing import List, Set
from langchain_core.documents import Document
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Import the core components needed for this test
from src.config import OPENAI_API_KEY
from src.data_loader import load_and_split_data
from src.idea_extractor import _extract_concepts_from_chunk # We use the internal function

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def _save_results(concepts_a: Set[str], concepts_b: Set[str], model_a: str, model_b: str, output_dir: Path):
    """Saves the comparison results to a CSV file and a Venn diagram."""
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. Save Raw Data to CSV ---
    common_concepts = sorted(list(concepts_a.intersection(concepts_b)))
    unique_to_a = sorted(list(concepts_a - concepts_b))
    unique_to_b = sorted(list(concepts_b - concepts_a))

    # Pad the lists to the same length for DataFrame creation
    max_len = max(len(common_concepts), len(unique_to_a), len(unique_to_b))
    
    # Create a dictionary for the DataFrame
    data_dict = {
        'Common_Concepts': common_concepts + [''] * (max_len - len(common_concepts)),
        f'Unique_to_{model_a}': unique_to_a + [''] * (max_len - len(unique_to_a)),
        f'Unique_to_{model_b}': unique_to_b + [''] * (max_len - len(unique_to_b))
    }
    
    df = pd.DataFrame(data_dict)
    csv_path = output_dir / "model_extraction_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Comparison data saved to {csv_path}")

    # --- 2. Generate and Save Venn Diagram ---
    plt.figure(figsize=(10, 8))
    
    venn2(
        subsets=(len(unique_to_a), len(unique_to_b), len(common_concepts)),
        set_labels=(f"{model_a}\n({len(concepts_a)} total)", f"{model_b}\n({len(concepts_b)} total)"),
        set_colors=('skyblue', 'lightgreen'),
        alpha=0.7
    )
    
    plt.title('Concept Extraction Overlap Between LLM Models', fontsize=16)
    
    plot_path = output_dir / "model_extraction_venn_diagram.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✅ Venn diagram saved to {plot_path}")
    plt.close()


def compare_model_extraction(chunks: List[Document], model_a: str, model_b: str):
    """
    Processes the same list of chunks with two different models and compares the results.
    """
    logger.info(f"--- Comparing concept extraction quality: '{model_a}' vs. '{model_b}' ---")

    concepts_a = set()
    concepts_b = set()

    logger.info(f"Processing {len(chunks)} chunks with '{model_a}'...")
    for chunk in chunks:
        extracted, _, _ = _extract_concepts_from_chunk(chunk, model_name=model_a)
        if extracted:
            concepts_a.update(extracted)

    logger.info(f"Processing {len(chunks)} chunks with '{model_b}'...")
    for chunk in chunks:
        extracted, _, _ = _extract_concepts_from_chunk(chunk, model_name=model_b)
        if extracted:
            concepts_b.update(extracted)

    # --- Analysis & Saving Results ---
    output_dir = Path("visualizations") # Define an output directory
    _save_results(concepts_a, concepts_b, model_a, model_b, output_dir)
    
    # --- Print Summary to Console ---
    common_concepts_count = len(concepts_a.intersection(concepts_b))
    unique_to_a_count = len(concepts_a - concepts_b)
    unique_to_b_count = len(concepts_b - concepts_a)
    
    print("\n" + "="*50)
    print("           EXTRACTION QUALITY DIAGNOSIS")
    print("="*50)
    print(f"\nModel A ({model_a}) extracted {len(concepts_a)} unique concepts.")
    print(f"Model B ({model_b}) extracted {len(concepts_b)} unique concepts.")
    print(f"Overlap (Common Concepts): {common_concepts_count}")
    print(f"Unique to '{model_a}': {unique_to_a_count}")
    print(f"Unique to '{model_b}': {unique_to_b_count}")
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("Check the generated 'visualizations/model_extraction_comparison.csv' file.")
    print(f"1. Is the 'Unique_to_{model_a}' column filled with more noise (plurals, generic terms)?")
    print(f"2. Is the 'Unique_to_{model_b}' column filled with more precise, domain-specific concepts?")
    print("If so, the model choice is the primary cause of the high review count.")
    print("="*50)

if __name__ == "__main__":
    # Ensure you have this dependency installed:
    # pip install matplotlib-venn
    
    print("Loading a sample of 30 chunks for the experiment...")
    all_chunks = load_and_split_data()
    sample_chunks = all_chunks[:30] if all_chunks else []

    if not sample_chunks:
        logger.error("No chunks found. Cannot run diagnosis.")
    else:
        nano_model = "gpt-4.1-nano"
        powerful_model = "gpt-4.1"

        compare_model_extraction(sample_chunks, nano_model, powerful_model)