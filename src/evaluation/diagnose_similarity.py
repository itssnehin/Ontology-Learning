import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from typing import Set, List
import concurrent.futures

# We need to import the core components to run the analysis
from src.config import LLM_MODEL, MAX_WORKERS
from src.data_loader import load_and_split_data
from src.idea_extractor import extract_ideas
from src.ontology_extension_manager import OntologyExtensionManager, _normalize_concept_name

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# --- NEW: HELPER FUNCTION FOR PARALLEL EXECUTION ---
def _analyze_single_concept_for_similarity(concept_name: str, manager: OntologyExtensionManager) -> float:
    """
    Analyzes a single concept's similarity and returns the top score.
    Designed to be run in a separate thread.
    """
    try:
        concept_dict = {
            'name': concept_name,
            'normalized_name': _normalize_concept_name(concept_name),
            'description': f"A component named {concept_name}"
        }
        matches = manager._find_concept_matches(concept_dict)
        return matches[0].similarity_score if matches else 0.0
    except Exception as e:
        logger.error(f"Error analyzing concept '{concept_name}' in a thread: {e}")
        return 0.0


def _plot_results(results_df: pd.DataFrame, thresholds: dict, output_path: Path):
    """Generates and saves the conceptual saturation plot."""
    logger.info("📊 Generating visualization...")
    
    plt.figure(figsize=(12, 7))
    sns.histplot(results_df['top_score'], bins=50, kde=True)
    
    # Overlay the current thresholds on the plot
    plt.axvline(thresholds['medium_similarity'], color='orange', linestyle='--', label=f"Uncertainty Threshold ({thresholds['medium_similarity']})")
    plt.axvline(thresholds['high_similarity'], color='red', linestyle='--', label=f"LLM Validation Threshold ({thresholds['high_similarity']})")
    
    plt.title('Distribution of Top Similarity Scores for New Concepts', fontsize=16)
    plt.xlabel('Top Cosine Similarity Score', fontsize=12)
    plt.ylabel('Number of Concepts', fontsize=12)
    plt.legend()
    
    plt.savefig(output_path, dpi=300)
    logger.info(f"✅ Visualization saved to: {output_path}")
    plt.close()


def analyze_similarity_distribution():
    """
    Runs the similarity analysis part of the pipeline in parallel to diagnose the
    distribution of top similarity scores.
    """
    logger.info("--- Starting Similarity Score Diagnosis (Parallel Mode) ---")

    # 1. Load data and extract concepts (no change here)
    logger.info("Step 1: Loading chunks and extracting concepts...")
    chunks = load_and_split_data()
    concepts, _, _ = extract_ideas(chunks, model_name=LLM_MODEL)
    if not concepts:
        logger.error("No concepts were extracted. Aborting diagnosis.")
        return
    logger.info(f"Found {len(concepts)} unique concepts to analyze.")

    # 2. Initialize the manager once (this is key for performance)
    logger.info("Step 2: Initializing OntologyExtensionManager and creating embeddings...")
    manager = OntologyExtensionManager()
    manager.load_existing_ontology()
    if not manager._existing_concepts:
        logger.warning("No existing concepts found. All similarity scores will be 0.")
    else:
        manager.create_concept_embeddings(manager._existing_concepts)

    # --- UPDATED: PARALLEL EXECUTION OF SIMILARITY ANALYSIS ---
    logger.info(f"Step 3: Calculating top similarity scores for {len(concepts)} concepts in parallel...")
    top_scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each concept, passing the manager instance
        future_to_concept = {executor.submit(_analyze_single_concept_for_similarity, concept, manager): concept for concept in concepts}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_concept), total=len(concepts), desc="Analyzing Similarities"):
            try:
                score = future.result()
                top_scores.append(score)
            except Exception as e:
                concept_name = future_to_concept[future]
                logger.error(f"Failed to process concept '{concept_name}' in parallel: {e}")

    # 4. Generate and save the plot (no change here)
    logger.info("Step 4: Generating score distribution plot...")
    results_df = pd.DataFrame(top_scores, columns=['top_score'])
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "similarity_score_distribution.png"
    _plot_results(results_df, manager.similarity_thresholds, plot_path)
    
    # 5. Print statistics (no change here)
    logger.info("\n--- Score Statistics ---")
    logger.info(f"Mean:   {results_df['top_score'].mean():.4f}")
    logger.info(f"Median: {results_df['top_score'].median():.4f}")
    logger.info(f"Std Dev:{results_df['top_score'].std():.4f}")
    logger.info(f"\n{results_df['top_score'].describe()}")

if __name__ == "__main__":
    analyze_similarity_distribution()