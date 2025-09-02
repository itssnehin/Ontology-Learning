#!/usr/bin/env python3
"""
Explore ECLASS class names to understand their structure and find appropriate filtering criteria.
"""

import re
from rdflib import Graph, Namespace
from pathlib import Path
from collections import Counter
import random

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OWL_FILE = DATA_DIR / "eclass_514en.owl"

def explore_eclass_structure():
    """Explore ECLASS class names to understand their structure."""
    print("Loading ECLASS ontology...")
    g = Graph()
    g.parse(OWL_FILE, format="xml")
    print(f"Loaded {len(g)} triples")

    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")

    # Extract all classes
    classes = set()
    for subj, pred, obj in g:
        if pred == OWL.Class or pred == RDFS.subClassOf:
            classes.add(str(subj))
    
    print(f"Found {len(classes)} classes")
    
    # Extract class names
    class_names = []
    for class_uri in classes:
        name = class_uri.split("#")[-1] if "#" in class_uri else class_uri.split("/")[-1]
        class_names.append(name)
    
    # Analyze naming patterns
    print("\n=== ECLASS CLASS NAME ANALYSIS ===")
    
    # Sample of class names
    sample_names = random.sample(class_names, min(50, len(class_names)))
    print(f"\nSample class names:")
    for i, name in enumerate(sample_names[:20]):
        print(f"  {i+1:2d}: {name}")
    
    # Character analysis
    total_chars = sum(len(name) for name in class_names)
    avg_length = total_chars / len(class_names)
    print(f"\nNaming statistics:")
    print(f"  Average length: {avg_length:.1f} characters")
    print(f"  Shortest: {min(len(name) for name in class_names)} chars")
    print(f"  Longest: {max(len(name) for name in class_names)} chars")
    
    # Pattern analysis
    has_numbers = [name for name in class_names if any(c.isdigit() for c in name)]
    has_underscores = [name for name in class_names if '_' in name]
    has_dashes = [name for name in class_names if '-' in name]
    all_caps = [name for name in class_names if name.isupper() and len(name) > 1]
    has_mixed_case = [name for name in class_names if any(c.islower() for c in name) and any(c.isupper() for c in name)]
    
    print(f"\nPattern analysis:")
    print(f"  Contains numbers: {len(has_numbers)}/{len(class_names)} ({100*len(has_numbers)/len(class_names):.1f}%)")
    print(f"  Contains underscores: {len(has_underscores)}/{len(class_names)} ({100*len(has_underscores)/len(class_names):.1f}%)")
    print(f"  Contains dashes: {len(has_dashes)}/{len(class_names)} ({100*len(has_dashes)/len(class_names):.1f}%)")
    print(f"  All caps: {len(all_caps)}/{len(class_names)} ({100*len(all_caps)/len(class_names):.1f}%)")
    print(f"  Mixed case: {len(has_mixed_case)}/{len(class_names)} ({100*len(has_mixed_case)/len(class_names):.1f}%)")
    
    # Look for electrotechnical patterns
    print(f"\n=== SEARCHING FOR ELECTROTECHNICAL CLASSES ===")
    
    # Extended search terms
    search_terms = [
        'electric', 'electrical', 'electronic', 'electronics',
        'antenna', 'aerial', 'connector', 'connect', 'cable', 'wire', 'wiring',
        'component', 'device', 'circuit', 'resistor', 'capacitor', 'inductor',
        'diode', 'transistor', 'sensor', 'actuator', 'motor', 'generator',
        'power', 'voltage', 'current', 'frequency', 'signal', 'communication',
        'radio', 'wireless', 'bluetooth', 'wifi', 'gsm', 'lte', 'rf',
        'pcb', 'board', 'module', 'chip', 'ic', 'semiconductor',
        'switch', 'relay', 'fuse', 'breaker', 'transformer', 'coil',
        '27', '24', '23'  # Common ECLASS electrical category codes
    ]
    
    matches_by_term = {}
    for term in search_terms:
        matches = [name for name in class_names if term.lower() in name.lower()]
        if matches:
            matches_by_term[term] = matches
            print(f"\nTerm '{term}': {len(matches)} matches")
            if len(matches) <= 10:
                for match in matches:
                    print(f"  {match}")
            else:
                for match in matches[:5]:
                    print(f"  {match}")
                print(f"  ... and {len(matches)-5} more")
    
    # Find the most promising categories
    if matches_by_term:
        print(f"\n=== MOST PROMISING CATEGORIES ===")
        sorted_terms = sorted(matches_by_term.items(), key=lambda x: len(x[1]), reverse=True)
        for term, matches in sorted_terms[:10]:
            print(f"{term}: {len(matches)} classes")
    else:
        print("\nNo matches found with standard terms!")
        print("Let's look at URI structure instead...")
        
        # Examine URIs for patterns
        sample_uris = random.sample(list(classes), 20)
        print(f"\nSample URIs:")
        for uri in sample_uris:
            print(f"  {uri}")
    
    # Look for hierarchical patterns in URIs
    print(f"\n=== URI STRUCTURE ANALYSIS ===")
    uri_patterns = Counter()
    for class_uri in list(classes)[:1000]:  # Sample for performance
        # Extract the part after the last / or #
        if '#' in class_uri:
            base, fragment = class_uri.rsplit('#', 1)
            uri_patterns[base] += 1
        elif '/' in class_uri:
            parts = class_uri.split('/')
            if len(parts) >= 2:
                uri_patterns['/'.join(parts[:-1])] += 1
    
    print("Most common URI bases:")
    for base, count in uri_patterns.most_common(10):
        print(f"  {count:4d}: {base}")
    
    return {
        'class_names': class_names,
        'classes': classes,
        'matches_by_term': matches_by_term,
        'sample_names': sample_names
    }

def suggest_better_filtering(analysis_results):
    """Suggest better filtering approaches based on the analysis."""
    print(f"\n=== FILTERING RECOMMENDATIONS ===")
    
    matches = analysis_results.get('matches_by_term', {})
    
    if matches:
        # Create a more targeted keyword list
        effective_terms = [term for term, matches_list in matches.items() if len(matches_list) > 0]
        print(f"Effective search terms found: {effective_terms}")
        
        # Suggest hierarchical filtering
        print(f"\nRecommended filtering approach:")
        print(f"1. Use broader terms: {effective_terms[:10]}")
        print(f"2. Consider URI-based filtering for specific ECLASS categories")
        print(f"3. Use regex patterns for class codes (e.g., classes starting with '27')")
    else:
        print("No keyword matches found. Alternative approaches:")
        print("1. Use structural/hierarchical filtering based on URI patterns")
        print("2. Use numeric ECLASS category codes")
        print("3. Manual selection of relevant URI bases")
        print("4. Skip pre-filtering and rely on similarity scores only")
    
    # Generate updated filtering code
    print(f"\n=== SUGGESTED CODE UPDATE ===")
    if matches:
        effective_keywords = [term for term, match_list in matches.items() if len(match_list) > 0 and len(match_list) < 1000]
        print(f"Replace ELECTROTECHNICAL_KEYWORDS with:")
        print(f"ELECTROTECHNICAL_KEYWORDS = {effective_keywords}")
    
    print(f"\nAlternative: Skip pre-filtering entirely and use similarity-only approach")

if __name__ == "__main__":
    print("=== ECLASS STRUCTURE EXPLORER ===")
    results = explore_eclass_structure()
    suggest_better_filtering(results)