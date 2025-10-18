# src/evaluation/conceptual.py

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from typing import Set, List
import random

from src.config import MARKDOWN_DIR, LLM_MODEL, MAX_WORKERS
from src.data_loader import load_and_split_data # We can use the main loader now
from src.idea_extractor import extract_ideas

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Reduce library noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _plot_results(results_df: pd.DataFrame, output_path: Path):
    """Generates and saves the conceptual saturation plot based on chunks."""
    logger.info("📊 Generating visualization...")
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot 1: New concepts per chunk batch
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Chunks Processed', fontsize=12)
    ax1.set_ylabel('Newly Discovered Concepts per Batch', color=color1, fontsize=12)
    ax1.plot(results_df['chunks_processed'], results_df['new_concepts'], color=color1, alpha=0.7, label='New Concepts per Batch')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Optional: Add a rolling average to smooth the 'new concepts' line
    results_df['new_concepts_rolling'] = results_df['new_concepts'].rolling(window=10, min_periods=1).mean()
    ax1.plot(results_df['chunks_processed'], results_df['new_concepts_rolling'], color='darkblue', linestyle='--', label='Rolling Average (New Concepts)')

    # Create a second y-axis for the total concept growth
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Total Unique Concepts in Ontology', color=color2, fontsize=12)
    ax2.plot(results_df['chunks_processed'], results_df['total_concepts'], color=color2, marker='.', linestyle=':', markersize=4, label='Total Unique Concepts')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Conceptual Saturation Analysis (Per Chunk)', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')

    plt.savefig(output_path, dpi=300)
    logger.info(f"✅ Visualization saved to: {output_path}")
    plt.close()

def analyze_conceptual_saturation(chunk_step: int = 10):
    """
    Processes all documents, shuffles the resulting chunks, and analyzes
    the growth of new concepts incrementally to measure conceptual saturation.

    Args:
        chunk_step: The number of chunks to process in each iteration. A smaller
                    number gives a more granular plot but is slower.
    """
    logger.info("--- Starting Conceptual Saturation Analysis (Chunk-Based) ---")
    
    # 1. Load ALL chunks from ALL files into a single list
    logger.info("Loading and chunking all documents...")
    all_chunks = load_and_split_data()
    if not all_chunks:
        logger.error("No chunks were loaded. Aborting analysis.")
        return

    logger.info(f"Loaded a total of {len(all_chunks)} chunks from all files.")

    # 2. Shuffle the chunks to ensure random order processing
    random.shuffle(all_chunks)
    logger.info("Randomly shuffled all chunks for unbiased analysis.")

    seen_concepts = set()
    results = []
    
    # 3. Process the shuffled chunks in incremental steps
    for i in tqdm(range(0, len(all_chunks), chunk_step), desc="Analyzing Chunks"):
        chunk_batch = all_chunks[i : i + chunk_step]
        if not chunk_batch:
            continue

        # Extract ideas from the current batch of chunks
        # Note: We're calling the LLM here for each batch.
        batch_concepts_list, _, _ = extract_ideas(chunk_batch, model_name=LLM_MODEL, max_workers=MAX_WORKERS)
        batch_concepts = set(batch_concepts_list)
        
        # Calculate metrics for this batch
        newly_discovered_concepts = batch_concepts - seen_concepts
        num_new = len(newly_discovered_concepts)
        
        # Update the master set of seen concepts
        seen_concepts.update(batch_concepts)
        num_total = len(seen_concepts)
        
        chunks_processed = i + len(chunk_batch)
        
        results.append({
            'chunks_processed': chunks_processed,
            'new_concepts': num_new,
            'total_concepts': num_total
        })

    logger.info(f"--- Analysis Complete. Found {len(seen_concepts)} total unique concepts. ---")

    # 4. Save results and plot
    results_df = pd.DataFrame(results)
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "conceptual_saturation_data_chunks.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Raw chunk-based analysis data saved to: {csv_path}")
    
    plot_path = output_dir / "conceptual_saturation_plot_chunks.png"
    _plot_results(results_df, plot_path)

if __name__ == "__main__":
    # You can adjust the chunk_step here if needed for a quicker/coarser run
    analyze_conceptual_saturation(chunk_step=10)