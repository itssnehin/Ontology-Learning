import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Set, Tuple

import networkx as nx
import numpy as np

from ..config import logger

# Helper functions to load your specific JSON formats into NetworkX graphs
def _load_gold_standard_graph(file_path: Path) -> nx.DiGraph:
    """Loads the gold standard JSON into a NetworkX DiGraph."""
    G = nx.DiGraph()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    concepts = [c.lower().strip() for c in data.get('concepts', [])]
    G.add_nodes_from(concepts)
    
    for rel in data.get('relations', []):
        source = rel.get('source', '').lower().strip()
        target = rel.get('target', '').lower().strip()
        if source in G and target in G:
            G.add_edge(source, target, type=rel.get('type', ''))
            
    return G

def _load_generated_graph(file_path: Path) -> nx.DiGraph:
    """Loads the pipeline's generated JSON-LD into a NetworkX DiGraph."""
    G = nx.DiGraph()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add nodes first
    for item in data.get('@graph', []):
        name = item.get('name', '').lower().strip()
        if name:
            G.add_node(name, category=item.get('category', 'Unknown'))
            
    # Add edges
    relation_keys = ["manufacturer", "brand", "isRelatedTo", "worksWith", "hasPart", "subclass_of"]
    for item in data.get('@graph', []):
        source = item.get('name', '').lower().strip()
        if not source:
            continue
        for rel_type in relation_keys:
            target_obj = item.get(rel_type)
            if not target_obj:
                continue
            targets = target_obj if isinstance(target_obj, list) else [target_obj]
            for t in targets:
                target = (t.get('name', '') if isinstance(t, dict) else str(t)).lower().strip()
                if source in G and target in G:
                    G.add_edge(source, target, type=rel_type)
    return G

def compare_graph_metrics(G_gold: nx.DiGraph, G_generated: nx.DiGraph) -> Dict[str, Any]:
    """Calculates and compares key network metrics for two graphs."""
    
    # Basic counts
    metrics = {
        "gold_standard": {
            "nodes": G_gold.number_of_nodes(),
            "edges": G_gold.number_of_edges(),
        },
        "generated_ontology": {
            "nodes": G_generated.number_of_nodes(),
            "edges": G_generated.number_of_edges(),
        }
    }
    
    # Density (only meaningful if graphs are not empty)
    if G_gold.number_of_nodes() > 0:
        metrics["gold_standard"]["density"] = round(nx.density(G_gold), 5)
    if G_generated.number_of_nodes() > 0:
        metrics["generated_ontology"]["density"] = round(nx.density(G_generated), 5)
        
    # Average Degree (undirected for simplicity of comparison)
    if G_gold.number_of_nodes() > 0:
        gold_degrees = [d for n, d in G_gold.degree()]
        metrics["gold_standard"]["avg_degree"] = round(np.mean(gold_degrees), 2)
    if G_generated.number_of_nodes() > 0:
        gen_degrees = [d for n, d in G_generated.degree()]
        metrics["generated_ontology"]["avg_degree"] = round(np.mean(gen_degrees), 2)

    # (Add other metrics like clustering coefficient or centrality if needed)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform graph-based evaluation of a generated ontology.")
    parser.add_argument("generated_file", type=Path, help="Path to the generated new_schema_objects_...jsonld file.")
    parser.add_argument("--gold", type=Path, default=Path("data/gold_standard.json"), help="Path to the gold standard JSON file.")
    
    args = parser.parse_args()

    if not args.generated_file.exists():
        logger.error(f"Generated file not found: {args.generated_file}")
    elif not args.gold.exists():
        logger.error(f"Gold standard file not found: {args.gold}")
    else:
        logger.info("Loading graphs for evaluation...")
        G_gold = _load_gold_standard_graph(args.gold)
        G_generated = _load_generated_graph(args.generated_file)
        
        logger.info("Comparing graph-level metrics...")
        metric_comparison = compare_graph_metrics(G_gold, G_generated)
        
        print("\n--- Graph Structure Evaluation Report ---")
        print(json.dumps(metric_comparison, indent=2))
        print("---------------------------------------\n")