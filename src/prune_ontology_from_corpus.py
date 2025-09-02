#!/usr/bin/env python3
"""
ECLASS pruning without pre-filtering - relies entirely on similarity matching
since ECLASS uses opaque alphanumeric codes like C_BAF272001-gen.
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
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MARKDOWN_DIR = DATA_DIR / "raw_markdown"
OWL_FILE = DATA_DIR / "eclass_514en.owl"
PRUNED_OWL_FILE = DATA_DIR / "eclass_514en_pruned_no_prefilter.owl"
MODEL_FILE = DATA_DIR / "electrotechnical_model.pt"

# Adjusted parameters for similarity-only approach
SIMILARITY_THRESHOLD = 0.005  # Much lower since we're not pre-filtering
TOP_K_SIMILARITY = 20  # Consider more similar terms
MIN_VALID_TOKENS = 1  # Require at least 1 token match
BATCH_SIZE = 1000  # Process classes in batches for efficiency

def extract_meaningful_tokens_from_eclass_code(class_name):
    """
    Extract potentially meaningful tokens from ECLASS codes.
    Example: C_BAF272001-gen -> ['BAF', '272', '001', 'gen']
    """
    # Remove the C_ prefix
    clean_name = class_name.replace('C_', '')
    
    # Split on common separators
    parts = re.split(r'[-_]', clean_name)
    
    tokens = []
    for part in parts:
        # Split alphanumeric sequences
        # BAF272001 -> ['BAF', '272', '001'] 
        alpha_numeric_parts = re.findall(r'[A-Za-z]+|\d+', part)
        tokens.extend(alpha_numeric_parts)
    
    # Filter tokens (remove very short ones, keep meaningful parts)
    meaningful_tokens = []
    for token in tokens:
        token = token.lower().strip()
        if len(token) >= 2:  # Keep tokens with 2+ characters
            meaningful_tokens.append(token)
    
    return meaningful_tokens

def calculate_class_similarity_batch(class_batch, embeddings, word_to_idx, vocab):
    """
    Calculate similarities for a batch of classes efficiently.
    """
    results = {}
    
    for class_uri, class_name in class_batch:
        # Extract tokens from the ECLASS code
        tokens = extract_meaningful_tokens_from_eclass_code(class_name)
        
        # Find tokens that exist in our vocabulary
        valid_tokens = [t for t in tokens if t in vocab]
        
        if len(valid_tokens) < MIN_VALID_TOKENS:
            results[class_uri] = None
            continue
        
        try:
            # Get embeddings for valid tokens
            token_ids = [word_to_idx[t] for t in valid_tokens]
            class_vec = torch.mean(embeddings[token_ids], dim=0)
            
            # Calculate cosine similarities
            similarities = torch.matmul(embeddings, class_vec) / (
                torch.norm(embeddings, dim=1) * torch.norm(class_vec) + 1e-8
            )
            
            # Multiple similarity metrics
            top_k_similarities, top_k_indices = torch.topk(
                similarities, k=min(TOP_K_SIMILARITY, len(similarities))
            )
            
            results[class_uri] = {
                'top_k_mean': torch.mean(top_k_similarities).item(),
                'max_similarity': torch.max(similarities).item(),
                'percentile_95': torch.quantile(similarities, 0.95).item(),
                'valid_tokens': valid_tokens,
                'original_tokens': tokens,
                'class_name': class_name
            }
            
        except Exception as e:
            logging.warning(f"Error processing {class_name}: {e}")
            results[class_uri] = None
    
    return results

def prune_ontology_similarity_only(owl_file, model, similarity_threshold):
    """
    Prune ECLASS ontology using only similarity matching (no pre-filtering).
    """
    print("Starting similarity-only ontology pruning...")
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
    
    # Convert to (uri, name) pairs for processing
    class_pairs = []
    for class_uri in classes:
        name = class_uri.split("#")[-1] if "#" in class_uri else class_uri.split("/")[-1]
        class_pairs.append((class_uri, name))
    
    # Load model
    embeddings = model['embeddings']
    word_to_idx = model['word_to_idx']
    vocab = set(word_to_idx.keys())
    
    print(f"Model vocabulary size: {len(vocab)}")
    print(f"Processing {len(class_pairs)} classes in batches of {BATCH_SIZE}...")
    
    # Process classes in batches
    all_similarities = {}
    pruned_classes = set()
    
    for i in tqdm(range(0, len(class_pairs), BATCH_SIZE), desc="Processing batches"):
        batch = class_pairs[i:i+BATCH_SIZE]
        batch_results = calculate_class_similarity_batch(batch, embeddings, word_to_idx, vocab)
        all_similarities.update(batch_results)
        
        # Apply thresholds to determine inclusion
        for class_uri, result in batch_results.items():
            if result is None:
                continue
                
            # Multiple inclusion criteria (more lenient than pre-filtered version)
            include_class = False
            
            if result['top_k_mean'] > similarity_threshold:
                include_class = True
            elif result['max_similarity'] > similarity_threshold * 3:
                include_class = True
            elif result['percentile_95'] > similarity_threshold * 2:
                include_class = True
            
            if include_class:
                pruned_classes.add(class_uri)
    
    # Print statistics
    valid_results = [r for r in all_similarities.values() if r is not None]
    invalid_count = len([r for r in all_similarities.values() if r is None])
    
    print(f"\n=== PROCESSING STATISTICS ===")
    print(f"Total classes: {len(classes)}")
    print(f"Successfully processed: {len(valid_results)}")
    print(f"Failed to process (no valid tokens): {invalid_count}")
    
    if valid_results:
        top_k_means = [r['top_k_mean'] for r in valid_results]
        max_sims = [r['max_similarity'] for r in valid_results]
        p95_sims = [r['percentile_95'] for r in valid_results]
        
        print(f"\n=== SIMILARITY STATISTICS ===")
        print(f"Top-K mean similarities:")
        print(f"  Mean: {np.mean(top_k_means):.6f}")
        print(f"  Std:  {np.std(top_k_means):.6f}")
        print(f"  Max:  {np.max(top_k_means):.6f}")
        print(f"  95th percentile: {np.percentile(top_k_means, 95):.6f}")
        
        print(f"Max similarities:")
        print(f"  Mean: {np.mean(max_sims):.6f}")
        print(f"  Max:  {np.max(max_sims):.6f}")
        
        print(f"95th percentile similarities:")
        print(f"  Mean: {np.mean(p95_sims):.6f}")
        print(f"  Max:  {np.max(p95_sims):.6f}")
    
    print(f"\n=== PRUNING RESULTS ===")
    print(f"Similarity threshold used: {similarity_threshold}")
    print(f"Classes surviving pruning: {len(pruned_classes)}")
    print(f"Pruning ratio: {len(pruned_classes)/len(classes)*100:.2f}%")
    
    if len(pruned_classes) == 0:
        print("\n❌ NO CLASSES SURVIVED PRUNING")
        print("This suggests the similarity threshold is too high or the embeddings don't capture relevant concepts.")
        print("\nDiagnostic suggestions:")
        print("1. Lower the SIMILARITY_THRESHOLD further (try 0.001)")
        print("2. Check if your vocabulary contains relevant terms")
        print("3. Consider using pre-trained embeddings instead")
        return
    
    # Show examples of pruned classes
    print(f"\n=== SAMPLE PRUNED CLASSES ===")
    sample_pruned = list(pruned_classes)[:20]
    for class_uri in sample_pruned:
        if class_uri in all_similarities and all_similarities[class_uri]:
            result = all_similarities[class_uri]
            print(f"  {result['class_name']} (tokens: {result['valid_tokens']}, "
                  f"top-k: {result['top_k_mean']:.4f}, max: {result['max_similarity']:.4f})")
    
    # Build pruned ontology
    print("\nBuilding pruned ontology graph...")
    pruned_g = Graph()
    
    for subj, pred, obj in tqdm(g, desc="Building pruned graph"):
        if str(subj) in pruned_classes or str(obj) in pruned_classes:
            pruned_g.add((subj, pred, obj))
    
    # Save pruned ontology
    pruned_g.serialize(destination=PRUNED_OWL_FILE, format="xml")
    print(f"Pruned ontology saved to {PRUNED_OWL_FILE}")
    print(f"Pruned ontology contains {len(pruned_g)} triples")

def load_existing_model():
    """Load the existing trained model."""
    if not MODEL_FILE.exists():
        print(f"❌ Model file not found: {MODEL_FILE}")
        print("Please run prune_ontology_from_corpus.py first to train the model.")
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
    print("=== ECLASS PRUNING - SIMILARITY ONLY ===")
    
    # Load existing model
    model = load_existing_model()
    if model is None:
        exit(1)
    
    # Run similarity-only pruning
    prune_ontology_similarity_only(OWL_FILE, model, SIMILARITY_THRESHOLD)
    
    print("\n=== PIPELINE COMPLETED ===")
    print("If no classes survived, try lowering SIMILARITY_THRESHOLD further.")