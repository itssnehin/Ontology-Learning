import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from typing import Set

# It's crucial that this script can import your existing modules
from src.config import MARKDOWN_DIR, LLM_MODEL
from src.data_loader import UnstructuredMarkdownLoader, RecursiveCharacterTextSplitter, Document
from src.idea_extractor import extract_ideas

# Configure logging for the analysis
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Reduce noise from libraries during the run
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _process_single_file(file_path: Path) -> Set[str]:
    """
    Loads a single markdown file, splits it into chunks, and extracts concepts.
    Returns a set of unique concepts found in that file.
    """
    try:
        loader = UnstructuredMarkdownLoader(str(file_path))
        docs = loader.load()
        
        # We only need a simple splitter for this task
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        all_chunks = []
        for doc in docs:
            # We don't need the full preprocessing for this analysis
            chunks = splitter.split_documents([Document(page_content=doc.page_content.lower())])
            all_chunks.extend(chunks)

        if not all_chunks:
            return set()

        # Extract_ideas returns (concepts, in_tokens, out_tokens). We only need the concepts.
        concepts, _, _ = extract_ideas(all_chunks, model_name=LLM_MODEL, max_workers=4)
        return set(concepts)
        
    except Exception as e:
        logger.error(f"Failed to process file {file_path.name}: {e}")
        return set()

def _plot_results(results_df: pd.DataFrame, output_path: Path):
    """Generates and saves the conceptual saturation plot."""
    logger.info("📊 Generating visualization...")
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot 1: New concepts per file (the decaying line)
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Files Processed', fontsize=12)
    ax1.set_ylabel('Newly Discovered Concepts', color=color1, fontsize=12)
    ax1.plot(results_df['file_index'], results_df['new_concepts'], color=color1, marker='o', linestyle='-', label='New Concepts per File')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create a second y-axis for the total concept growth
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Total Unique Concepts in Ontology', color=color2, fontsize=12)
    ax2.plot(results_df['file_index'], results_df['total_concepts'], color=color2, marker='x', linestyle='--', label='Total Unique Concepts')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Conceptual Saturation Analysis', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add a unified legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')

    plt.savefig(output_path, dpi=300)
    logger.info(f"✅ Visualization saved to: {output_path}")
    plt.close()

def analyze_conceptual_saturation():
    """
    Main function to run the analysis. It processes files sequentially,
    tracks the growth of new concepts, and generates a plot.
    """
    logger.info("--- Starting Conceptual Saturation Analysis ---")
    
    # Get a sorted list of files to ensure a deterministic run
    files_to_process = sorted(list(MARKDOWN_DIR.glob("*.md")))
    if not files_to_process:
        logger.error("No markdown files found to analyze. Aborting.")
        return

    logger.info(f"Found {len(files_to_process)} files to process.")

    seen_concepts = set()
    results = []

    # Process each file one by one
    for i, file_path in enumerate(tqdm(files_to_process, desc="Analyzing Files")):
        
        current_file_concepts = _process_single_file(file_path)
        
        # Calculate how many are truly new
        newly_discovered_concepts = current_file_concepts - seen_concepts
        num_new = len(newly_discovered_concepts)
        
        # Update our set of all concepts seen so far
        seen_concepts.update(current_file_concepts)
        num_total = len(seen_concepts)
        
        logger.info(f"File {i+1}/{len(files_to_process)} ({file_path.name}): Found {num_new} new concepts. Total unique: {num_total}.")
        
        results.append({
            'file_index': i + 1,
            'file_name': file_path.name,
            'new_concepts': num_new,
            'total_concepts': num_total
        })

    # Convert results to a DataFrame for easier handling
    results_df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Save the raw data to CSV for further analysis
    csv_path = output_dir / "conceptual_saturation_data.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Raw analysis data saved to: {csv_path}")
    
    # Generate the plot
    plot_path = output_dir / "conceptual_saturation_plot.png"
    _plot_results(results_df, plot_path)

    logger.info("--- Analysis Complete ---")


if __name__ == "__main__":
    # This check ensures the script can be run directly from the command line
    analyze_conceptual_saturation()