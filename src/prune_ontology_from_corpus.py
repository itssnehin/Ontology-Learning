#!/usr/bin/env python3
"""
Fixed version of the ontology pruning script that addresses the core issues
identified in the diagnostic analysis.
"""

import os
import glob
import re
import logging
import gc
from rdflib import Graph, Namespace
import torch
from gensim.utils import simple_preprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MARKDOWN_DIR = DATA_DIR / "raw_markdown"
OWL_FILE = DATA_DIR / "eclass_514en.owl"
PRUNED_OWL_FILE = DATA_DIR / "eclass_514en_pruned_fixed.owl"
MODEL_FILE = DATA_DIR / "electrotechnical_model.pt"

# FIXED PARAMETERS
SIMILARITY_THRESHOLD = 0.01  # Much lower threshold
TOP_K_SIMILARITY = 10  # Consider top-10 most similar terms
MIN_TOKENS_FOR_CLASS = 1  # Minimum valid tokens to consider a class
NUM_WORKERS = 2

# Domain-specific keywords for pre-filtering ECLASS
ELECTROTECHNICAL_KEYWORDS = [
    'electric', 'electronic', 'electrical', 'antenna', 'connector', 'cable', 
    'component', 'device', 'circuit', 'wire', 'sensor', 'resistor', 'capacitor',
    'inductor', 'diode', 'transistor', 'fpc', 'pcb', 'board', 'module',
    'frequency', 'signal', 'power', 'voltage', 'current', 'impedance',
    'communication', 'wireless', 'radio', 'bluetooth', 'wifi', 'gsm', 'lte'
]

def prefilter_eclass_classes(classes):
    """Pre-filter ECLASS classes to electrotechnical domain."""
    print("Pre-filtering ECLASS to electrotechnical domain...")
    relevant_classes = set()
    
    for class_uri in tqdm(classes, desc="Filtering classes"):
        class_name = class_uri.split("#")[-1] if "#" in class_uri else class_uri.split("/")[-1]
        class_name_lower = class_name.lower()
        
        # Check if any electrotechnical keyword is in the class name
        if any(keyword in class_name_lower for keyword in ELECTROTECHNICAL_KEYWORDS):
            relevant_classes.add(class_uri)
    
    print(f"Pre-filtered from {len(classes)} to {len(relevant_classes)} classes")
    print(f"Reduction ratio: {len(relevant_classes)/len(classes)*100:.1f}%")
    
    # Show some examples
    example_names = []
    for class_uri in list(relevant_classes)[:10]:
        name = class_uri.split("#")[-1] if "#" in class_uri else class_uri.split("/")[-1]
        example_names.append(name)
    print(f"Example relevant classes: {example_names}")
    
    return relevant_classes

def calculate_class_similarity_fixed(class_name, embeddings, word_to_idx, vocab):
    """
    Fixed similarity calculation that addresses the averaging issue.
    """
    tokens = simple_preprocess(class_name.lower())
    # Also split on common separators used in technical terms
    additional_tokens = []
    for separator in ['-', '_', '.', ' ']:
        additional_tokens.extend(class_name.lower().split(separator))
    tokens.extend(additional_tokens)
    
    # Filter valid tokens
    valid_tokens = [t for t in tokens if t in vocab and t and len(t) > 1]
    
    if len(valid_tokens) < MIN_TOKENS_FOR_CLASS:
        return None, valid_tokens
    
    try:
        # Get embeddings for the class tokens
        token_ids = [word_to_idx[t] for t in valid_tokens]
        class_vec = torch.mean(embeddings[token_ids], dim=0)
        
        # Calculate cosine similarities with all vocabulary
        similarities = torch.matmul(embeddings, class_vec) / (
            torch.norm(embeddings, dim=1) * torch.norm(class_vec) + 1e-8  # Add epsilon for stability
        )
        
        # FIX 1: Instead of averaging, use top-k similarity
        top_k_similarities, top_k_indices = torch.topk(similarities, k=min(TOP_K_SIMILARITY, len(similarities)))
        score = torch.mean(top_k_similarities).item()
        
        # FIX 2: Alternative scoring - maximum similarity
        max_score = torch.max(similarities).item()
        
        # FIX 3: Alternative scoring - percentile-based
        percentile_95 = torch.quantile(similarities, 0.95).item()
        
        return {
            'top_k_mean': score,
            'max_similarity': max_score,
            'percentile_95': percentile_95,
            'valid_tokens': valid_tokens,
            'top_similar_indices': top_k_indices.tolist()
        }, valid_tokens
        
    except Exception as e:
        logging.warning(f"Error calculating similarity for {class_name}: {e}")
        return None, valid_tokens

def prune_ontology_fixed(owl_file, model, similarity_threshold):
    """
    Fixed version of ontology pruning with multiple improvements.
    """
    print("Starting improved ontology pruning...")
    g = Graph()
    print("Parsing OWL file...")
    g.parse(owl_file, format="xml")
    print(f"Loaded ontology with {len(g)} triples.")

    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")

    # Extract all classes
    print("Extracting classes...")
    classes = set()
    for subj, pred, obj in g:
        if pred == OWL.Class or pred == RDFS.subClassOf:
            classes.add(str(subj))
    
    print(f"Found {len(classes)} total classes.")
    
    # Pre-filter to electrotechnical domain
    relevant_classes = prefilter_eclass_classes(classes)
    
    if not relevant_classes:
        print("No relevant classes found after pre-filtering!")
        return
    
    # Load model
    embeddings = model['embeddings']
    word_to_idx = model['word_to_idx']
    vocab = set(word_to_idx.keys())
    
    print(f"Model vocab size: {len(vocab)}")
    
    # Analyze classes with fixed similarity calculation
    print("Calculating similarities with fixed approach...")
    pruned_classes = set()
    similarity_stats = {
        'top_k_mean': [],
        'max_similarity': [],
        'percentile_95': [],
        'no_tokens': 0,
        'error_count': 0
    }
    
    for class_uri in tqdm(relevant_classes, desc="Processing classes"):
        class_name = class_uri.split("#")[-1] if "#" in class_uri else class_uri.split("/")[-1]
        
        similarity_result, valid_tokens = calculate_class_similarity_fixed(
            class_name, embeddings, word_to_idx, vocab
        )
        
        if similarity_result is None:
            if not valid_tokens:
                similarity_stats['no_tokens'] += 1
            else:
                similarity_stats['error_count'] += 1
            continue
        
        # Collect statistics
        for metric in ['top_k_mean', 'max_similarity', 'percentile_95']:
            similarity_stats[metric].append(similarity_result[metric])
        
        # MULTIPLE CRITERIA FOR INCLUSION
        include_class = False
        
        # Criterion 1: Top-k mean similarity
        if similarity_result['top_k_mean'] > similarity_threshold:
            include_class = True
        
        # Criterion 2: High maximum similarity (more lenient)
        if similarity_result['max_similarity'] > similarity_threshold * 2:
            include_class = True
        
        # Criterion 3: High percentile similarity
        if similarity_result['percentile_95'] > similarity_threshold * 1.5:
            include_class = True
        
        if include_class:
            pruned_classes.add(class_uri)
            logging.debug(f"INCLUDED: {class_name} - TopK: {similarity_result['top_k_mean']:.4f}, "
                         f"Max: {similarity_result['max_similarity']:.4f}, "
                         f"P95: {similarity_result['percentile_95']:.4f}")
    
    # Print comprehensive statistics
    print(f"\n=== SIMILARITY STATISTICS ===")
    for metric in ['top_k_mean', 'max_similarity', 'percentile_95']:
        values = similarity_stats[metric]
        if values:
            print(f"{metric}:")
            print(f"  Mean: {np.mean(values):.6f}")
            print(f"  Std:  {np.std(values):.6f}")
            print(f"  Min:  {np.min(values):.6f}")
            print(f"  Max:  {np.max(values):.6f}")
            print(f"  95th percentile: {np.percentile(values, 95):.6f}")
    
    print(f"\nClasses with no valid tokens: {similarity_stats['no_tokens']}")
    print(f"Classes with errors: {similarity_stats['error_count']}")
    print(f"Classes successfully processed: {len(similarity_stats['top_k_mean'])}")
    
    print(f"\n=== PRUNING RESULTS ===")
    print(f"Original ECLASS classes: {len(classes)}")
    print(f"Pre-filtered classes: {len(relevant_classes)}")
    print(f"Final pruned classes: {len(pruned_classes)}")
    print(f"Pruning ratio: {len(pruned_classes)/len(classes)*100:.2f}%")
    print(f"Effectiveness ratio: {len(pruned_classes)/len(relevant_classes)*100:.2f}%")
    
    if not pruned_classes:
        print("\n⚠️ WARNING: No classes survived pruning!")
        print("Consider:")
        print(f"- Lowering similarity threshold (current: {similarity_threshold})")
        print("- Enriching the training corpus")
        print("- Using pre-trained embeddings")
        return
    
    # Build pruned ontology
    print("Building pruned ontology graph...")
    pruned_g = Graph()
    
    for subj, pred, obj in tqdm(g, desc="Building pruned graph"):
        # Include triple if subject OR object is in pruned classes
        if str(subj) in pruned_classes or str(obj) in pruned_classes:
            pruned_g.add((subj, pred, obj))
    
    # Save pruned ontology
    pruned_g.serialize(destination=PRUNED_OWL_FILE, format="xml")
    print(f"Pruned ontology saved to {PRUNED_OWL_FILE}")
    print(f"Pruned ontology contains {len(pruned_g)} triples")
    
    # Show some examples of pruned classes
    print(f"\n=== SAMPLE PRUNED CLASSES ===")
    sample_classes = list(pruned_classes)[:20]
    for class_uri in sample_classes:
        name = class_uri.split("#")[-1] if "#" in class_uri else class_uri.split("/")[-1]
        print(f"  {name}")

def load_existing_model():
    """Load existing model or suggest training."""
    if not MODEL_FILE.exists():
        print(f"❌ Model file not found: {MODEL_FILE}")
        print("Please run the original prune_ontology_from_corpus.py first to train the model.")
        return None
    
    print("Loading existing model...")
    try:
        checkpoint = torch.load(MODEL_FILE, map_location='cpu')
        embeddings = checkpoint['embeddings']
        word_to_idx = checkpoint['word_to_idx']
        print(f"Model loaded: {len(word_to_idx)} vocab, {embeddings.shape[1]}D embeddings")
        return {'embeddings': embeddings, 'word_to_idx': word_to_idx}
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    print("=== FIXED ONTOLOGY PRUNING PIPELINE ===")
    
    # Load existing model
    model = load_existing_model()
    if model is None:
        exit(1)
    
    # Run fixed pruning
    prune_ontology_fixed(OWL_FILE, model, SIMILARITY_THRESHOLD)
    
    print("\n=== PIPELINE COMPLETED ===")
    print("Check the results and adjust SIMILARITY_THRESHOLD if needed.")