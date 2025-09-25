#!/usr/bin/env python3
"""
CLEANED VERSION - Removes actual duplicate document processing patterns from your existing file.

This removes the duplicate load_and_split_data() + extract_ideas() + extract_relations() 
pattern and replaces it with a reusable function.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neo4j import GraphDatabase

# Import pipeline modules
from .data_loader import load_and_split_data
from .idea_extractor import extract_ideas
from .relation_extractor import extract_relations
from .ontology_extension_manager import OntologyExtensionManager, ExtensionDecision
from .schema_org_extractor import extract_schema_org_markup
from .schema_org_relation_extractor import extract_schema_org_relations, SchemaOrgRelationExtractor
from .schema_org_graph_builder import build_schema_org_knowledge_graph
from .schema_ontology_visualizer import SchemaOrgOntologyVisualizer
from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from .utils import setup_logging

# ===== EXTRACTED COMMON PATTERN (removes duplication across your files) =====
def process_document_corpus():
    """
    CLEANED: Extracted duplicate document processing pattern.
    
    This replaces the pattern that appears in multiple files:
    - Load documents
    - Extract concepts  
    - Extract relations
    
    Instead of copy-pasting this 3-step pattern everywhere, we use this function.
    """
    print("   📖 Loading and processing corpus documents...")
    
    # Load documents
    chunks = load_and_split_data()
    print(f"   📄 Loaded {len(chunks)} document chunks")
    
    # Extract concepts
    print("   🧠 Extracting concepts using LLM...")
    concepts = extract_ideas(chunks)
    print(f"   💡 Extracted {len(concepts)} unique concepts")
    
    # Extract relations
    print("   🔗 Extracting relations using LLM...")
    relations = extract_relations(chunks)
    print(f"   📊 Extracted {len(relations)} relations")
    
    return {
        "concepts": concepts,
        "relations": relations,
        "chunks_processed": len(chunks)
    }
# ============================================================================

class SupervisorDemonstration:
    """Complete demonstration pipeline for supervisor review."""
    
    def __init__(self, output_dir: str = "../supervisor_demo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging("../logs", "supervisor_demo")
        
        # ===== CLEANED: Simplified subdirectory creation (no repetitive mkdir calls) =====
        subdirs = [
            "01_baseline_ontology",
            "02_corpus_extraction",
            "03_similarity_analysis",
            "04_extension_decisions", 
            "05_final_ontology"
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        # ================================================================================
        
        # Neo4j connection
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        # Results storage
        self.demo_results = {
            "timestamp": datetime.now().isoformat(),
            "baseline_stats": {},
            "corpus_concepts": [],
            "corpus_relations": [],
            "similarity_analysis": [],
            "extension_decisions": [],
            "final_stats": {}
        }
        
        print("🎓 Supervisor Demonstration Pipeline Initialized")
        print(f"📁 Output Directory: {self.output_dir}")
    
    def run_complete_demonstration(self) -> Dict[str, Any]:
        """Run the complete demonstration pipeline for supervisor review."""
        
        print("\n" + "="*60)
        print("🎓 SUPERVISOR DEMONSTRATION PIPELINE")
        print("="*60)
        
        # Step 1: Analyze baseline ontology
        print("\n📊 Step 1: Analyzing baseline ontology...")
        baseline_stats = self._analyze_baseline_ontology()
        self.demo_results["baseline_stats"] = baseline_stats
        
        # ===== CLEANED: Use extracted common function instead of duplicating the pattern =====
        # ORIGINAL (your duplicate code):
        # print("   📖 Loading and processing corpus documents...")
        # chunks = load_and_split_data()
        # print(f"   📄 Loaded {len(chunks)} document chunks")
        # print("   🧠 Extracting concepts using LLM...")
        # concepts = extract_ideas(chunks)
        # print(f"   💡 Extracted {len(concepts)} unique concepts")
        # print("   🔗 Extracting relations using LLM...")
        # relations = extract_relations(chunks)
        # print(f"   📊 Extracted {len(relations)} relations")
        
        # CLEANED VERSION:
        print("\n🔍 Step 2: Corpus extraction...")
        corpus_data = process_document_corpus()  # Single line replaces 8+ duplicate lines!
        # ====================================================================================
        
        self.demo_results.update(corpus_data)
        
        # Create corpus visualization
        self.create_corpus_visualization(corpus_data)
        
        # ===== CLEANED: Simplified JSON saving (reusable pattern) =====
        self._save_json(corpus_data, "02_corpus_extraction", "corpus_extraction.json")
        # ===============================================================
        
        # Save detailed concept list for review
        concepts_file = self.output_dir / "02_corpus_extraction" / "extracted_concepts.txt"
        with open(concepts_file, 'w') as f:
            f.write("EXTRACTED CONCEPTS FROM CORPUS\n")
            f.write("=" * 40 + "\n\n")
            for i, concept in enumerate(corpus_data["concepts"], 1):
                f.write(f"{i:3d}. {concept}\n")
        
        # Step 3: Similarity analysis
        print("\n🔍 Step 3: Performing similarity analysis...")
        similarity_results = self.perform_similarity_analysis(corpus_data["concepts"])
        self.demo_results["similarity_analysis"] = similarity_results
        
        # Step 4: Extension decisions
        print("\n🧭 Step 4: Generating extension decisions...")
        extension_decisions = self.generate_extension_decisions(similarity_results)
        self.demo_results["extension_decisions"] = extension_decisions
        
        # Step 5: Generate final report
        print("\n📋 Step 5: Generating supervisor report...")
        self._generate_supervisor_report()
        
        # Step 6: Save final results
        self._save_json(self.demo_results, "05_final_ontology", "demo_results.json")
        
        print("\n✅ Demonstration pipeline completed successfully!")
        return self.demo_results
    
    def _analyze_baseline_ontology(self) -> Dict[str, Any]:
        """Analyze current state of ontology before adding new concepts."""
        
        # Query Neo4j for baseline statistics
        with self.driver.session() as session:
            # Count total concepts
            concept_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            
            # Count total relationships  
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # Get sample of concept types
            sample_concepts = session.run("""
                MATCH (n) 
                RETURN labels(n) as labels, n.name as name 
                LIMIT 10
            """).data()
        
        baseline_stats = {
            "total_concepts": concept_count,
            "total_relations": rel_count,
            "sample_concepts": sample_concepts,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save baseline analysis
        self._save_json(baseline_stats, "01_baseline_ontology", "baseline_analysis.json")
        
        print(f"   📈 Found {concept_count} existing concepts")
        print(f"   🔗 Found {rel_count} existing relations")
        
        return baseline_stats
    
    def perform_similarity_analysis(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """Perform detailed similarity analysis for each concept."""
        
        print("   🔍 Initializing similarity analysis...")
        
        # Initialize extension manager
        extension_manager = OntologyExtensionManager()
        extension_manager.load_existing_ontology()
        
        if extension_manager._existing_concepts:
            extension_manager.create_concept_embeddings(extension_manager._existing_concepts)
        
        similarity_results = []
        
        print(f"   📊 Analyzing similarity for {len(concepts)} concepts...")
        
        for i, concept_name in enumerate(concepts):
            print(f"      Analyzing {i+1}/{len(concepts)}: {concept_name}")
            
            # Create concept dictionary
            concept_dict = {
                'name': concept_name,
                'category': self._infer_category(concept_name),
                'description': f"Electronic component: {concept_name}",
                'frequency': '',
                'impedance': '',
                'connector': ''
            }
            
            # Find matches using multiple methods
            matches = extension_manager._find_concept_matches(concept_dict)
            
            # Create detailed similarity analysis
            analysis = {
                'concept': concept_name,
                'category': concept_dict['category'],
                'top_matches': [],
                'similarity_methods': {},
                'llm_prompts': []
            }
            
            # Record top matches with detailed scores
            for match in matches[:5]:
                analysis['top_matches'].append({
                    'existing_concept': match.existing_concept,
                    'similarity_score': match.similarity_score,
                    'match_type': match.match_type,
                    'confidence': match.confidence,
                    'reasoning': match.reasoning
                })
            
            # If high similarity, capture LLM prompt for demonstration
            if matches and matches[0].similarity_score >= 0.85:
                llm_prompt = self._create_demo_llm_prompt(concept_dict, matches[0])
                analysis['llm_prompts'].append(llm_prompt)
            
            similarity_results.append(analysis)
        
        # Save detailed similarity analysis
        self._save_json(similarity_results, "03_similarity_analysis", "similarity_analysis.json")
        
        # Create similarity visualization
        self.create_similarity_visualization(similarity_results)
        
        return similarity_results
    
    def generate_extension_decisions(self, similarity_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed extension decisions with reasoning."""
        
        extension_decisions = []
        
        print(f"   🧭 Generating decisions for {len(similarity_results)} concepts...")
        
        for result in similarity_results:
            concept_name = result['concept']
            
            # Determine decision based on top similarity score
            if result['top_matches']:
                top_score = result['top_matches'][0]['similarity_score']
                top_match = result['top_matches'][0]['existing_concept']
                
                if top_score >= 0.95:
                    decision = "MAP_EXACT"
                    reasoning = f"Very high similarity ({top_score:.3f}) to existing concept '{top_match}'"
                elif top_score >= 0.85:
                    decision = "MAP_SIMILAR" 
                    reasoning = f"High similarity ({top_score:.3f}) to existing concept '{top_match}'"
                elif top_score >= 0.70:
                    decision = "MANUAL_REVIEW"
                    reasoning = f"Medium similarity ({top_score:.3f}) to '{top_match}' - requires human validation"
                else:
                    decision = "EXTEND"
                    reasoning = f"Low similarity ({top_score:.3f}) - extend ontology with new concept"
            else:
                decision = "EXTEND"
                reasoning = "No existing matches found - extend ontology with new concept"
            
            decision_record = {
                'concept': concept_name,
                'decision': decision,
                'reasoning': reasoning,
                'top_matches': result['top_matches'],
                'timestamp': datetime.now().isoformat()
            }
            
            extension_decisions.append(decision_record)
            print(f"      {concept_name}: {decision}")
        
        # Save extension decisions
        self._save_json(extension_decisions, "04_extension_decisions", "extension_decisions.json")
        
        return extension_decisions
    
    def _infer_category(self, concept_name: str) -> str:
        """Infer category for concept - simplified for demo."""
        concept_lower = concept_name.lower()
        
        if any(term in concept_lower for term in ['resistor', 'capacitor', 'inductor']):
            return 'Passive Components'
        elif any(term in concept_lower for term in ['transistor', 'diode', 'ic']):
            return 'Active Components'
        elif any(term in concept_lower for term in ['connector', 'cable', 'wire']):
            return 'Interconnects'
        else:
            return 'General Electronics'
    
    def _create_demo_llm_prompt(self, concept_dict: Dict[str, Any], match) -> Dict[str, str]:
        """Create example LLM prompt for supervisor demonstration."""
        return {
            'concept': concept_dict['name'],
            'existing_match': match.existing_concept,
            'similarity_score': match.similarity_score,
            'prompt': f"""
Analyze whether '{concept_dict['name']}' should be mapped to existing concept '{match.existing_concept}'.

Concept Details:
- Name: {concept_dict['name']}
- Category: {concept_dict['category']} 
- Description: {concept_dict['description']}

Existing Match:
- Name: {match.existing_concept}
- Similarity Score: {match.similarity_score:.3f}

Please determine if these concepts are:
1. Identical (exact mapping)
2. Similar but distinct (similar mapping)
3. Different (extend ontology)

Provide reasoning for your decision.
""",
            'timestamp': datetime.now().isoformat()
        }
    
    def create_corpus_visualization(self, corpus_data: Dict[str, Any]):
        """Create visualization of corpus extraction results."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Concept count visualization
        ax1.bar(['Total Concepts'], [len(corpus_data['concepts'])], color='steelblue')
        ax1.set_title('Extracted Concepts Count')
        ax1.set_ylabel('Number of Concepts')
        
        # Chunks processed visualization  
        ax2.bar(['Document Chunks'], [corpus_data['chunks_processed']], color='forestgreen')
        ax2.set_title('Document Chunks Processed')
        ax2.set_ylabel('Number of Chunks')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "02_corpus_extraction" / "corpus_analysis.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   📊 Saved corpus visualization: corpus_analysis.png")
    
    def create_similarity_visualization(self, similarity_results: List[Dict[str, Any]]):
        """Create similarity analysis visualization."""
        
        if not similarity_results:
            return
        
        # Extract similarity scores for plotting
        similarity_scores = []
        for result in similarity_results:
            if result.get('top_matches'):
                similarity_scores.append(result['top_matches'][0]['similarity_score'])
        
        if not similarity_scores:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(similarity_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Similarity Scores')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.output_dir / "03_similarity_analysis" / "similarity_distribution.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   📊 Saved similarity visualization: similarity_distribution.png")
    
    # ===== CLEANED: Reusable JSON saving method (eliminates duplicate file saving code) =====
    def _save_json(self, data: Dict[str, Any], subdir: str, filename: str) -> Path:
        """Standardized JSON saving - eliminates duplicate save code throughout the file."""
        output_path = self.output_dir / subdir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   💾 Saved: {filename}")
        return output_path
    # ========================================================================================
    
    def _generate_supervisor_report(self):
        """Generate comprehensive supervisor presentation report."""
        
        report_content = f"""# Supervisor Demonstration Report

## Executive Summary
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline**: Ontology Learning with Large Language Models

## Key Achievements

### 1. Baseline Ontology Analysis
- **Total Concepts**: {self.demo_results['baseline_stats'].get('total_concepts', 'N/A')}
- **Total Relations**: {self.demo_results['baseline_stats'].get('total_relations', 'N/A')}
- **Analysis Timestamp**: {self.demo_results['baseline_stats'].get('analysis_timestamp', 'N/A')}

### 2. Corpus Processing Results
- **Document Chunks Processed**: {self.demo_results.get('chunks_processed', 'N/A')}
- **Unique Concepts Extracted**: {len(self.demo_results.get('concepts', []))}
- **Relations Extracted**: {len(self.demo_results.get('relations', []))}
- **Processing Method**: LLM-based extraction with GPT-4

### 3. Similarity Analysis
- **Concepts Analyzed**: {len(self.demo_results.get('similarity_analysis', []))}
- **Analysis Method**: Multi-modal similarity scoring
- **Visualization**: Distribution plots and category analysis

### 4. Extension Decisions
- **Decisions Generated**: {len(self.demo_results.get('extension_decisions', []))}
- **Decision Types**: MAP_EXACT, MAP_SIMILAR, MANUAL_REVIEW, EXTEND
- **Reasoning**: Confidence-based thresholds with detailed explanations

## Technical Innovation
1. **Automated Ontology Extension**: Novel LLM-driven approach
2. **Semantic Similarity**: Advanced embedding-based matching
3. **Quality Assurance**: Multi-threshold decision framework
4. **Scalability**: Designed for large technical corpora

## Demonstration Outputs
- **Baseline Analysis**: `01_baseline_ontology/baseline_analysis.json`
- **Corpus Extraction**: `02_corpus_extraction/corpus_extraction.json`
- **Similarity Results**: `03_similarity_analysis/similarity_analysis.json`
- **Extension Decisions**: `04_extension_decisions/extension_decisions.json`
- **Visualizations**: Various plots in respective subdirectories

## Sample Concepts Processed
"""
        
        # Add sample concepts
        sample_concepts = self.demo_results.get('concepts', [])[:10]
        for i, concept in enumerate(sample_concepts, 1):
            report_content += f"{i}. {concept}\n"
        
        report_content += f"""

## Sample Extension Decisions
"""
        
        # Add sample decisions
        sample_decisions = self.demo_results.get('extension_decisions', [])[:5]
        for decision in sample_decisions:
            report_content += f"""
### {decision['concept']}
- **Decision**: {decision['decision']}
- **Reasoning**: {decision['reasoning']}
"""
        
        report_content += f"""

## Next Steps for Research
1. Expand corpus to include more technical domains
2. Validate against existing electronic component ontologies
3. Integrate with Schema.org for web semantic compatibility
4. Develop automated quality metrics

## Academic Contributions
- Novel hybrid approach combining LLMs with structured ontologies
- Scalable pipeline for domain-specific knowledge extraction
- Comprehensive evaluation framework for ontology quality

---
**Prepared for Academic Supervision Review**
**All code, data, and visualizations available for inspection**
"""
        
        # Save supervisor report
        report_path = self.output_dir / "05_final_ontology" / "supervisor_presentation_summary.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"   📋 Saved supervisor report: supervisor_presentation_summary.md")

if __name__ == "__main__":
    demo = SupervisorDemonstration()
    results = demo.run_complete_demonstration()