import logging
from typing import List, Set, Dict, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

# Import the core components needed for this test
from src.config import MAX_WORKERS
from src.data_loader import load_and_split_data
from src.idea_extractor import extract_ideas

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def _normalize_string(s: str) -> str:
    """Normalizes a string by making it lowercase and stripping whitespace."""
    return s.lower().strip() if isinstance(s, str) else str(s)

def _calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1_score, 4)
    }

def load_gold_standard(gold_path: Path) -> Set[str]:
    """Loads and normalizes concepts from a gold standard JSON file."""
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {_normalize_string(c) for c in data.get('concepts', [])}

def _plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Generates and saves a grouped bar chart for model comparison."""
    logger.info("📊 Generating visualization...")
    
    # Plotting Precision, Recall, and F1-Score
    plot_df = results_df.set_index('Model')[['Precision', 'Recall', 'F1-Score']]
    
    ax = plot_df.plot(kind='bar', figsize=(14, 8), rot=0, colormap='viridis')
    
    plt.title('Head-to-Head Model Performance on Concept Extraction', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylim(0, 1.05) # Scores are between 0 and 1
    plt.legend(title='Metric')
    
    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10, padding=3)

    plt.tight_layout()
    plot_path = output_dir / "model_performance_comparison.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✅ Comparison bar chart saved to {plot_path}")
    plt.close()

def compare_model_extraction(chunks: List, models_to_compare: List[str], gold_concepts: Set[str]):
    """
    Processes the same list of chunks with multiple models and evaluates
    each against a gold standard, calculating precision, recall, and F1.
    """
    logger.info(f"--- Comparing concept extraction for models: {models_to_compare} ---")

    all_results = []

    for model in models_to_compare:
        logger.info(f"Processing {len(chunks)} chunks with '{model}'...")
        
        # Extract concepts using the specified model
        # The extract_ideas function returns (concepts, in_tokens, out_tokens)
        extracted_concepts_list, _, _ = extract_ideas(chunks, model_name=model, max_workers=MAX_WORKERS)
        generated_concepts = {_normalize_string(c) for c in extracted_concepts_list}

        # --- Evaluation Logic ---
        tp = len(generated_concepts.intersection(gold_concepts))
        fp = len(generated_concepts.difference(gold_concepts))
        fn = len(gold_concepts.difference(generated_concepts))

        metrics = _calculate_metrics(tp, fp, fn)
        
        result_entry = {
            "Model": model,
            **metrics, # Unpack precision, recall, f1
            "True_Positives": tp,
            "False_Positives": fp,
            "False_Negatives": fn,
            "Total_Extracted": len(generated_concepts)
        }
        all_results.append(result_entry)

    # --- Analysis & Saving Results ---
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(all_results).sort_values(by="F1-Score", ascending=False)
    
    # Save CSV
    summary_csv_path = output_dir / "model_performance_summary.csv"
    results_df.to_csv(summary_csv_path, index=False)
    logger.info(f"✅ Performance summary saved to {summary_csv_path}")
    
    # Generate Plot
    _plot_results(results_df, output_dir)
    
    # --- Print Final Summary to Console ---
    print("\n" + "="*80)
    print("      HEAD-TO-HEAD MODEL PERFORMANCE DIAGNOSIS (vs. Gold Standard)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("\nANALYSIS:")
    print("The model with the highest F1-Score provides the best balance of Precision and Recall.")
    print("A high Precision means the model is accurate and generates less noise.")
    print("A high Recall means the model is comprehensive and misses fewer important concepts.")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Compare LLM extraction quality against a gold standard for a single document.")
    parser.add_argument("document_name", type=str, help="The filename of the markdown document to test (e.g., 'LM158.md').")
    parser.add_argument("--gold", type=Path, required=True, help="Path to the document-specific gold standard JSON file.")
    parser.add_argument("--models", nargs='+', default=["gpt-3.5-turbo", "gpt-4o", "gpt-4.1"], help="List of OpenAI models to compare.")
    
    args = parser.parse_args()

    # 1. Load the specific document for the experiment
    logger.info(f"Loading and chunking the specified document: {args.document_name}")
    # The load_and_split_data function can filter by filename
    chunks = load_and_split_data(files_to_process=[args.document_name])

    if not chunks:
        logger.error(f"No chunks were loaded for '{args.document_name}'. Make sure the file is in 'data/raw_markdown/'.")
        return

    # 2. Load the gold standard
    try:
        gold_concepts = load_gold_standard(args.gold)
        logger.info(f"Successfully loaded {len(gold_concepts)} concepts from '{args.gold.name}'.")
    except FileNotFoundError:
        logger.error(f"Gold standard file not found at: {args.gold}")
        return

    # 3. Run the comparison
    compare_model_extraction(chunks, args.models, gold_concepts)

if __name__ == "__main__":
    main()