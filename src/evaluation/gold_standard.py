import json
import logging
from pathlib import Path
from typing import Dict, Any, Set, Tuple

# Import the logger from the central config to ensure it's configured
from ..config import logger

def _normalize_string(s: str) -> str:
    """Normalizes a string by making it lowercase and stripping whitespace."""
    if not isinstance(s, str):
        s = str(s)
    return s.lower().strip()

def _calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    """Calculates precision, recall, and F1-score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def _extract_generated_relations(data: list) -> Set[Tuple[str, str, str]]:
    """Extracts relations from the pipeline's final 'learned_tasks' JSON file."""
    relations: Set[Tuple[str, str, str]] = set()
    
    for task in data:
        source = _normalize_string(task.get('name', ''))
        if not source:
            continue
            
        parent = _normalize_string(task.get('parent_class'))
        if parent:
            relations.add((source, 'subclass_of', parent))

        non_tax_rels = task.get('non_taxonomic_relations', [])
        if non_tax_rels:
            for rel in non_tax_rels:
                target = _normalize_string(rel.get('target'))
                rel_type = _normalize_string(rel.get('relation'))
                if target and rel_type:
                    relations.add((source, rel_type, target))
    return relations

def _extract_gold_relations(data: Dict) -> Set[Tuple[str, str, str]]:
    """Extracts relations from the manually created gold standard JSON file."""
    relations: Set[Tuple[str, str, str]] = set()
    for rel in data.get('relations', []):
        source = _normalize_string(rel.get('source', ''))
        rel_type = _normalize_string(rel.get('type', ''))
        target = _normalize_string(rel.get('target', ''))
        if source and rel_type and target:
            relations.add((source, rel_type, target))
    return relations

def evaluate_ontology(generated_path: Path, gold_standard_path: Path) -> Dict[str, Any]:
    """
    Compares a generated ontology against a gold standard and calculates metrics.
    """
    logger.info(f"Starting evaluation of '{generated_path.name}' against '{gold_standard_path.name}'")

    try:
        with open(generated_path, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)
        with open(gold_standard_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse evaluation files: {e}", exc_info=True)
        return {}

    # --- 1. Evaluate Concepts ---
    generated_concepts_raw = [task.get('name', '') for task in generated_data]
    generated_concepts: Set[str] = {_normalize_string(c) for c in generated_concepts_raw if c}
    
    gold_concepts_raw = gold_data.get('concepts', [])
    gold_concepts: Set[str] = {_normalize_string(c) for c in gold_concepts_raw if c}

    concept_tp = len(generated_concepts.intersection(gold_concepts))
    concept_fp = len(generated_concepts.difference(gold_concepts))
    concept_fn = len(gold_concepts.difference(generated_concepts))
    
    concept_metrics = _calculate_metrics(concept_tp, concept_fp, concept_fn)
    logger.info(f"Concept Evaluation Results: {concept_metrics}")

    # --- 2. Evaluate Relations ---
    generated_relations = _extract_generated_relations(generated_data)
    gold_relations = _extract_gold_relations(gold_data)

    relation_tp = len(generated_relations.intersection(gold_relations))
    relation_fp = len(generated_relations.difference(gold_relations))
    
    # --- THIS IS THE FIX ---
    relation_fn = len(gold_relations.difference(generated_relations))
    # --- END OF FIX ---

    relation_metrics = _calculate_metrics(relation_tp, relation_fp, relation_fn)
    logger.info(f"Relation Evaluation Results: {relation_metrics}")

    return {
        "source_files": {
            "generated": str(generated_path.name),
            "gold_standard": str(gold_standard_path.name)
        },
        "concept_evaluation": concept_metrics,
        "relation_evaluation": relation_metrics
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a generated ontology against a gold standard.")
    parser.add_argument("generated_file", type=Path, help="Path to the generated new_schema_objects_TIMESTAMP.jsonld file.")
    parser.add_argument("--gold", type=Path, default=Path("data/gold_standard.json"), help="Path to the gold standard JSON file (defaults to data/gold_standard.json).")
    
    args = parser.parse_args()

    if not args.generated_file.exists():
        logger.error(f"Generated file not found: {args.generated_file}")
    elif not args.gold.exists():
        logger.error(f"Gold standard file not found: {args.gold}")
    else:
        results = evaluate_ontology(args.generated_file, args.gold)
        
        print("\n--- Ontology Evaluation Report ---")
        if results:
            print(json.dumps(results, indent=2))
        else:
            print("Evaluation could not be completed.")
        print("--------------------------------\n")