# src/evaluation/confidence_k_evaluation.py

import logging
from pathlib import Path
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def _normalize_string(s: str) -> str:
    """Normalizes a string by making it lowercase and stripping whitespace."""
    return s.lower().strip() if isinstance(s, str) else str(s)

def load_gold_standard_concepts(gold_path: Path) -> set:
    """Loads and normalizes concepts from the gold standard JSON file."""
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {_normalize_string(c) for c in data.get('concepts', [])}

def load_and_rank_concepts_by_confidence(tasks_path: Path) -> list:
    """Loads tasks from a pipeline output file and ranks them by confidence."""
    with open(tasks_path, 'r', encoding='utf-8') as f:
        # The file is a JSON object with a 'decisions' key which is a list of dicts
        data = json.load(f)
        decisions = data.get('decisions', [])

    # Sort the decisions by the 'confidence' key, highest first
    # Default to 0 confidence if the key is missing
    sorted_decisions = sorted(decisions, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Return just the list of concept names, now ranked by confidence
    return [_normalize_string(d.get('concept_name', '')) for d in sorted_decisions]

def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generates and saves a line plot comparing Coverage at K."""
    logger.info("📊 Generating Top-K Confidence vs. Coverage plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.lineplot(data=df, x='K', y='Coverage_at_K', marker='o', ax=ax)
    
    ax.set_title('Coverage of Gold Standard Concepts in Top-K Most Confident Extractions', fontsize=16, fontweight='bold')
    ax.set_xlabel('K (Number of Top-Confidence Concepts Considered)', fontsize=12)
    ax.set_ylabel('Coverage Rate of Gold Standard (%)', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xticks(df['K'].unique())
    
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    
    plot_path = output_dir / "confidence_k_coverage_plot.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✅ Top-K confidence plot saved to: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Top-K concept prominence by confidence score.")
    parser.add_argument(
        "pipeline_output", 
        type=Path,
        help="Path to the final integration_results_...json file from a pipeline run."
    )
    parser.add_argument(
        "--gold", 
        type=Path, 
        required=True,
        help="Path to the document-specific gold standard JSON file (e.g., data/lm158_gold_standard.json)."
    )
    parser.add_argument("-k", type=int, nargs='+', default=[25, 50, 100, 200, 500, 1000], help="A list of K values to test.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"), help="Directory to save the reports.")
    
    args = parser.parse_args()

    try:
        gold_concepts = load_gold_standard_concepts(args.gold)
        if not gold_concepts:
            logger.error("No concepts found in the gold standard file. Aborting.")
            return

        ranked_concepts = load_and_rank_concepts_by_confidence(args.pipeline_output)
        if not ranked_concepts:
            logger.error("No concepts with confidence scores found in the pipeline output file. Ensure you are using an 'integration_results' file.")
            return
            
        total_gold_concepts = len(gold_concepts)
        all_results = []
        
        logger.info(f"--- Running Confidence vs. Coverage analysis for '{args.pipeline_output.name}' ---")
        for k in sorted(args.k):
            # Take the top K concepts from the confidence-ranked list
            top_k_concepts = set(ranked_concepts[:k])
            
            hits = gold_concepts.intersection(top_k_concepts)
            hit_count = len(hits)
            
            coverage_at_k = hit_count / total_gold_concepts if total_gold_concepts > 0 else 0.0
            
            all_results.append({
                "K": k,
                "Gold_Concepts_in_Top_K": hit_count,
                "Coverage_at_K": coverage_at_k
            })

        results_df = pd.DataFrame(all_results)
        
        # Save CSV
        csv_path = args.output_dir / "confidence_k_evaluation_summary.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"✅ Confidence-K evaluation summary saved to: {csv_path}")
        
        print("\n--- Top-K Confidence vs. Coverage Summary ---")
        # Format for printing
        display_df = results_df.copy()
        display_df['Coverage_at_K'] = display_df['Coverage_at_K'].apply(lambda x: f"{x:.2%}")
        print(display_df.to_string(index=False))
        print("------------------------------------------")

        # Generate plot
        plot_results(results_df, args.output_dir)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Please check your file paths.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()