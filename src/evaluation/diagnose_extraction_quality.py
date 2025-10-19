import logging
from typing import List, Set, Dict
from langchain_core.documents import Document
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the core components needed for this test
from src.config import OPENAI_API_KEY, MAX_WORKERS
from src.data_loader import load_and_split_data
from src.idea_extractor import _extract_concepts_from_chunk # We use the internal function

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def _save_results(all_results: Dict[str, Set[str]], output_dir: Path):
    """Saves the multi-model comparison results to CSV files and a bar chart."""
    output_dir.mkdir(exist_ok=True)
    model_names = list(all_results.keys())

    # --- 1. Calculate Summary Statistics ---
    summary_data = []
    all_concepts_union = set.union(*all_results.values())

    for model in model_names:
        concepts = all_results[model]
        # Find concepts unique to this model compared to all others
        other_concepts_union = set.union(*(s for m, s in all_results.items() if m != model))
        unique_to_this_model = concepts - other_concepts_union
        
        summary_data.append({
            "Model": model,
            "Total_Unique_Concepts": len(concepts),
            "Concepts_Unique_to_Model": len(unique_to_this_model),
            "Coverage_of_All_Concepts": f"{len(concepts.intersection(all_concepts_union)) / len(all_concepts_union) * 100:.2f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / "model_comparison_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"✅ Summary comparison data saved to {summary_csv_path}")

    # --- 2. Save Detailed Unique Concepts for Each Model ---
    for model in model_names:
        other_concepts_union = set.union(*(s for m, s in all_results.items() if m != model))
        unique_to_this_model = sorted(list(concepts - other_concepts_union))
        if unique_to_this_model:
            df_unique = pd.DataFrame(unique_to_this_model, columns=['Unique_Concepts'])
            unique_csv_path = output_dir / f"concepts_unique_to_{model.replace('.', '_')}.csv"
            df_unique.to_csv(unique_csv_path, index=False)
            logger.info(f"✅ Detailed unique concepts for {model} saved to {unique_csv_path}")

    # --- 3. Generate and Save Bar Chart ---
    fig, ax = plt.subplots(figsize=(12, 8))
    summary_df.set_index('Model').plot(kind='bar', ax=ax, rot=45, colormap='viridis')
    
    plt.title('Comparison of Concept Extraction Across LLM Models', fontsize=16)
    plt.ylabel('Number of Concepts', fontsize=12)
    plt.xlabel('Model Name', fontsize=12)
    plt.tight_layout()

    plot_path = output_dir / "model_comparison_barchart.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✅ Comparison bar chart saved to {plot_path}")
    plt.close()


def compare_model_extraction(chunks: List[Document], models_to_compare: List[str]):
    """
    Processes the same list of chunks with multiple different models and saves the results.
    """
    logger.info(f"--- Comparing concept extraction quality for models: {models_to_compare} ---")

    all_results: Dict[str, Set[str]] = {model: set() for model in models_to_compare}

    for model in models_to_compare:
        logger.info(f"Processing {len(chunks)} chunks with '{model}'...")
        # Use tqdm for a progress bar for each model
        for chunk in tqdm(chunks, desc=f"Model: {model}"):
            extracted, _, _ = _extract_concepts_from_chunk(chunk, model_name=model)
            if extracted:
                all_results[model].update(extracted)

    # --- Analysis & Saving Results ---
    output_dir = Path("visualizations") # Define an output directory
    _save_results(all_results, output_dir)
    
    # --- Print Final Summary to Console ---
    print("\n" + "="*50)
    print("      MULTI-MODEL EXTRACTION QUALITY DIAGNOSIS")
    print("="*50)
    summary_df = pd.read_csv(output_dir / "model_comparison_summary.csv")
    print(summary_df.to_string(index=False))
    print("\nANALYSIS:")
    print("Check the generated CSV files in the 'visualizations/' directory.")
    print("Look for models with a high 'Total_Unique_Concepts' but low 'Concepts_Unique_to_Model'.")
    print("This often indicates a model that is noisy or inconsistent.")
    print("A good model will have a reasonable total count and its unique concepts will be high-quality, domain-specific terms.")
    print("="*50)

if __name__ == "__main__":
    # 1. Load a sample of chunks for the experiment
    print("Loading a sample of 30 chunks...")
    all_chunks = load_and_split_data()
    sample_chunks = all_chunks[:30] if all_chunks else []

    if not sample_chunks:
        logger.error("No chunks found. Cannot run diagnosis.")
    else:
        # 2. Define the list of models
        models_to_test = [
            "gpt-3.5-turbo",    # The fast, cheap baseline
            "gpt-4.1-nano",     # Your previous model
            "gpt-4.1-mini",      # A modern, small model
            "gpt-4o",
            "gpt-4.1"            # The powerful, recommended model
        ]
        # 3. Run the comparison
        compare_model_extraction(sample_chunks, models_to_test)