# src/evaluation/core_concept_evaluation.py

import json
import logging
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def _normalize_string(s: str) -> str:
    """Normalizes a string by making it lowercase and stripping whitespace."""
    if not isinstance(s, str):
        s = str(s)
    return s.lower().strip()

def load_concepts_from_gold_standard(gold_path: Path) -> Set[str]:
    """Loads and normalizes concepts from the gold standard JSON file."""
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {_normalize_string(c) for c in data.get('concepts', [])}

def load_concepts_from_pipeline_output(pipeline_output_path: Path) -> Set[str]:
    """Loads and normalizes concepts from the final pipeline tasks JSON file."""
    with open(pipeline_output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # The concepts are the 'name' in each task dictionary in the list
    return {_normalize_string(task.get('name', '')) for task in data if task.get('name')}

def evaluate_core_concepts(gold_concepts: Set[str], model_concepts: Set[str]) -> Dict[str, Any]:
    """
    Calculates metrics based on coverage of core concepts and signal-to-noise.
    """
    true_positives = len(gold_concepts.intersection(model_concepts))
    false_positives = len(model_concepts.difference(gold_concepts))
    false_negatives = len(gold_concepts.difference(model_concepts))

    # Metric 1: Coverage (Recall) of the Gold Standard
    coverage = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # Metric 2: Signal-to-Noise Ratio (higher is better)
    # How many true concepts were found for each noisy concept?
    signal_to_noise = true_positives / false_positives if false_positives > 0 else float('inf')

    # Metric 3: A combined score to balance coverage and noise
    # We use a harmonic mean, similar to an F1-score
    if signal_to_noise == float('inf'): # Perfect signal, no noise
        quality_score = coverage # The score is simply its perfect coverage
    elif (coverage + signal_to_noise) > 0:
        quality_score = 2 * (coverage * signal_to_noise) / (coverage + signal_to_noise)
    else:
        quality_score = 0.0
    
    return {
        "Total_Concepts_Extracted": len(model_concepts),
        "Core_Concepts_Found (TP)": true_positives,
        "Noise_Concepts (FP)": false_positives,
        "Core_Concepts_Missed (FN)": false_negatives,
        "Gold_Standard_Coverage (Recall)": round(coverage, 4),
        "Signal_to_Noise_Ratio": 'Infinity (No Noise)' if signal_to_noise == float('inf') else round(signal_to_noise, 4),
        "Ontology_Quality_Score": round(quality_score, 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a final pipeline output against a core concept gold standard.")
    parser.add_argument("pipeline_output", type=Path, help="Path to the final learned_ontology_tasks_...json file.")
    parser.add_argument("--gold", type=Path, default=Path("data/gold_standard.json"), help="Path to the gold standard JSON file.")
    parser.add_argument("--output", type=Path, default=Path("visualizations/final_ontology_evaluation.csv"), help="Path to save the summary CSV.")
    
    args = parser.parse_args()

    try:
        gold_concepts = load_concepts_from_gold_standard(args.gold)
        logger.info(f"Loaded {len(gold_concepts)} core concepts from the gold standard.")
    except FileNotFoundError:
        logger.error(f"FATAL: Gold standard file not found at: {args.gold}")
        return

    try:
        model_concepts = load_concepts_from_pipeline_output(args.pipeline_output)
        logger.info(f"Loaded {len(model_concepts)} concepts from the final pipeline output.")
    except FileNotFoundError:
        logger.error(f"FATAL: Pipeline output file not found at: {args.pipeline_output}")
        return

    results = evaluate_core_concepts(gold_concepts, model_concepts)
    results["Model"] = f"Final Ontology ({args.pipeline_output.stem})"
    
    summary_df = pd.DataFrame([results])
    # Define the column order for the best readability
    column_order = [
        "Model", "Ontology_Quality_Score", "Gold_Standard_Coverage (Recall)", 
        "Signal_to_Noise_Ratio", "Core_Concepts_Found (TP)", 
        "Noise_Concepts (FP)", "Core_Concepts_Missed (FN)", "Total_Concepts_Extracted"
    ]
    summary_df = summary_df[column_order]
    
    # Ensure the output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    
    logger.info(f"✅ Final ontology evaluation saved to: {args.output}")
    print("\n--- Final Ontology Accuracy Report ---")
    print(summary_df.to_string(index=False))
    print("------------------------------------\n")

if __name__ == "__main__":
    main()