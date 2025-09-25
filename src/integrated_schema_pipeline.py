"""
This keeps all your existing functionality but replaces the verbose if/elif blocks 
with clean dictionary mapping logic.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import pickle
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import re
from difflib import SequenceMatcher

# --- CORRECTED IMPORTS ---
from .data_loader import load_and_split_data
from .idea_extractor import extract_ideas
from .schema_org_extractor import extract_schema_org_markup
from .schema_org_relation_extractor import extract_schema_org_relations, SchemaOrgRelationExtractor
from .schema_org_graph_builder import build_schema_org_knowledge_graph
from .ontology_extension_manager import OntologyExtensionManager, ExtensionDecision, ExtensionResult
from .utils import setup_logging
from .config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

@dataclass
class PipelineConfig:
    """Configuration for the integrated pipeline."""
    max_chunks: Optional[int] = None
    similarity_thresholds: Dict[str, float] = None
    enable_llm_validation: bool = True
    enable_technical_matching: bool = True
    cache_embeddings: bool = True
    output_dir: str = "../data/integrated_output"
    
    def __post_init__(self):
        if self.similarity_thresholds is None:
            self.similarity_thresholds = {
                'exact_match': 0.95,
                'high_similarity': 0.85,
                'medium_similarity': 0.70,
                'low_similarity': 0.50
            }

@dataclass
class IntegrationResults:
    """Results from the integrated pipeline execution."""
    total_concepts_extracted: int
    concepts_mapped_to_existing: int
    concepts_extending_ontology: int
    concepts_requiring_review: int
    confidence_scores: List[float]
    processing_time: float
    decisions: List[ExtensionResult]
    
    @property
    def automation_rate(self) -> float:
        """Percentage of concepts that received automated decisions."""
        total_automated = self.concepts_mapped_to_existing + self.concepts_extending_ontology
        return (total_automated / self.total_concepts_extracted * 100) if self.total_concepts_extracted > 0 else 0.0
    
    @property
    def average_confidence(self) -> float:
        """Average confidence score across all decisions."""
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0

class IntegratedSchemaOrgPipeline:
    """
    Schema.org pipeline with intelligent ontology extension management.
    
    This class integrates the existing Schema.org extraction pipeline with the new
    ontology extension manager to provide intelligent decisions about concept mapping
    vs. ontology extension.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the integrated pipeline.
        
        Args:
            config: Configuration object with pipeline parameters
        """
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging("../logs", "integrated_schema_pipeline")
        
        # Initialize extension manager
        self.extension_manager = OntologyExtensionManager()
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "chunks_processed": 0,
            "concepts_extracted": [],
            "extension_decisions": [],
            "schema_objects_created": [],
            "schema_objects_mapped": [],
            "integration_stats": {}
        }
        
        print("🔧 Integrated Schema.org Pipeline initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def run_integrated_pipeline(self) -> IntegrationResults:
        """
        Execute the complete integrated pipeline with ontology extension management.
        
        Returns:
            IntegrationResults object with comprehensive statistics and decisions
        """
        start_time = datetime.now()
        
        print("🚀" + "="*70)
        print("INTEGRATED SCHEMA.ORG PIPELINE WITH ONTOLOGY EXTENSION MANAGEMENT")
        print("="*70)
        
        # Step 1: Load and process documents
        print("\n📄 Step 1: Loading and processing documents...")
        chunks = load_and_split_data()
        if self.config.max_chunks:
            chunks = chunks[:self.config.max_chunks]
        
        self.results["chunks_processed"] = len(chunks)
        print(f"   ✅ Processed {len(chunks)} document chunks")
        
        # Step 2: Extract concepts from documents
        print("\n🧠 Step 2: Extracting concepts from documents...")
        extracted_concepts = extract_ideas(chunks)
        self.results["concepts_extracted"] = extracted_concepts
        print(f"   ✅ Extracted {len(extracted_concepts)} unique concepts")
        
        # Step 3: Load existing ontology for comparison
        print("\n📚 Step 3: Loading existing ontology...")
        self.extension_manager.load_existing_ontology()
        existing_concepts = self.extension_manager._existing_concepts
        print(f"   ✅ Loaded {len(existing_concepts)} existing concepts from ontology")
        
        # Step 4: Create embeddings for similarity comparison
        print("\n🧮 Step 4: Creating concept embeddings...")
        if existing_concepts:
            self.extension_manager.create_concept_embeddings(existing_concepts)
            print(f"   ✅ Created embeddings for existing concepts")
        
        # Step 5: Analyze each concept for extension vs mapping
        print("\n🔍 Step 5: Analyzing concepts for ontology decisions...")
        extension_decisions = []
        concepts_for_schema_creation = []
        concepts_for_mapping = []
        
        for i, concept_name in enumerate(extracted_concepts):
            print(f"   📋 Analyzing concept {i+1}/{len(extracted_concepts)}: {concept_name}")
            
            # Create concept dict (we have limited info from idea_extractor)
            concept_dict = {
                'name': concept_name,
                'category': self._infer_category(concept_name),
                'description': f"Electronic component: {concept_name}",
                # Technical properties would be extracted in a real implementation
                'frequency': '',
                'impedance': '',
                'connector': ''
            }
            
            # Analyze for extension decision
            decision = self.extension_manager.analyze_new_concept(concept_dict)
            extension_decisions.append(decision)
            
            # Route concept based on decision
            if decision.decision == ExtensionDecision.EXTEND:
                concepts_for_schema_creation.append(concept_dict)
                print(f"      🆕 Decision: Extend ontology")
            elif decision.decision in [ExtensionDecision.MAP_EXACT, ExtensionDecision.MAP_SIMILAR]:
                concepts_for_mapping.append({
                    'concept': concept_dict,
                    'target': decision.target_concept,
                    'confidence': decision.confidence
                })
                print(f"      🔗 Decision: Map to existing concept '{decision.target_concept}'")
            else:  # UNCERTAIN or MERGE_CONCEPTS
                concepts_for_schema_creation.append(concept_dict)  # Default to creation for now
                print(f"      ❓ Decision: Uncertain, defaulting to ontology extension")
        
        self.results["extension_decisions"] = extension_decisions
        
        # Step 6: Create Schema.org objects for new concepts
        print(f"\n🌐 Step 6: Creating Schema.org objects for {len(concepts_for_schema_creation)} new concepts...")
        schema_objects = []
        for concept in concepts_for_schema_creation:
            # Extract Schema.org markup using existing pipeline
            try:
                # Create a simple text representation for the schema extractor
                concept_text = f"{concept['name']}: {concept['description']}"
                markup = extract_schema_org_markup([concept_text])
                if markup:
                    schema_objects.extend(markup)
            except Exception as e:
                print(f"      ⚠️ Failed to create Schema.org object for {concept['name']}: {e}")
        
        self.results["schema_objects_created"] = schema_objects
        print(f"   ✅ Created {len(schema_objects)} Schema.org objects")
        
        # Step 7: Process mappings to existing concepts
        print(f"\n🔗 Step 7: Processing {len(concepts_for_mapping)} concept mappings...")
        for mapping in concepts_for_mapping:
            mapping_info = {
                'source_concept': mapping['concept']['name'],
                'target_concept': mapping['target'],
                'confidence': mapping['confidence'],
                'mapping_timestamp': datetime.now().isoformat()
            }
            self.results["schema_objects_mapped"].append(mapping_info)
        
        print(f"   ✅ Processed {len(concepts_for_mapping)} concept mappings")
        
        # Step 8: Generate integration statistics (CLEANED VERSION - No more verbose if/elif!)
        print("\n📊 Step 8: Generating integration statistics...")
        # ===== CLEANED DECISION COUNTING (replaces the 15 lines of if/elif blocks) =====
        integration_stats = self._analyze_decisions_clean(extension_decisions)
        # ================================================================================
        
        self.results["integration_stats"] = integration_stats
        
        # Step 9: Save comprehensive results
        print("\n💾 Step 9: Saving integration results...")
        self._save_integration_results()
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create final results object
        results = IntegrationResults(
            total_concepts_extracted=len(extracted_concepts),
            concepts_mapped_to_existing=integration_stats['decision_counts'].get('map_exact', 0) + integration_stats['decision_counts'].get('map_similar', 0),
            concepts_extending_ontology=integration_stats['decision_counts'].get('extend', 0),
            concepts_requiring_review=integration_stats['decision_counts'].get('uncertain', 0),
            confidence_scores=[d.confidence for d in extension_decisions],
            processing_time=processing_time,
            decisions=extension_decisions
        )
        
        print(f"\n🎉 INTEGRATION COMPLETE!")
        print(f"📊 Total concepts: {results.total_concepts_extracted}")
        print(f"🤖 Automation rate: {results.automation_rate:.1f}%")
        print(f"📈 Average confidence: {results.average_confidence:.2f}")
        print(f"⏱️ Processing time: {results.processing_time:.1f}s")
        
        return results
    
    # ===== CLEANED DECISION ANALYSIS (replaces your verbose if/elif blocks) =====
    def _analyze_decisions_clean(self, decisions):
        """
        CLEAN VERSION: Replaces the original verbose decision counting logic.
        
        ORIGINAL (your existing code):
        decision_counts = {
            'extend': 0,
            'map_exact': 0,
            'map_similar': 0,
            'merge': 0,
            'uncertain': 0
        }
        
        confidence_by_decision = {decision.value: [] for decision in ExtensionDecision}
        
        for decision in decisions:
            confidence_by_decision[decision.decision.value].append(decision.confidence)
            
            if decision.decision == ExtensionDecision.EXTEND:
                decision_counts['extend'] += 1
            elif decision.decision == ExtensionDecision.MAP_EXACT:
                decision_counts['map_exact'] += 1
            elif decision.decision == ExtensionDecision.MAP_SIMILAR:
                decision_counts['map_similar'] += 1
            elif decision.decision == ExtensionDecision.MERGE_CONCEPTS:
                decision_counts['merge'] += 1
            else:
                decision_counts['uncertain'] += 1
        
        CLEANED VERSION (below):
        """
        # Clean decision mapping - no more verbose if/elif blocks!
        decision_mapping = {
            ExtensionDecision.EXTEND: 'extend',
            ExtensionDecision.MAP_EXACT: 'map_exact', 
            ExtensionDecision.MAP_SIMILAR: 'map_similar',
            ExtensionDecision.MERGE_CONCEPTS: 'merge'
        }
        
        decision_counts = {}
        confidence_by_decision = {}
        
        # Single loop with dictionary lookup instead of if/elif chain
        for decision in decisions:
            decision_type = decision_mapping.get(decision.decision, 'uncertain')
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
            
            if decision_type not in confidence_by_decision:
                confidence_by_decision[decision_type] = []
            confidence_by_decision[decision_type].append(decision.confidence)
        
        return {
            'decision_counts': decision_counts,
            'confidence_by_decision': confidence_by_decision,
            'total_decisions': len(decisions),
            'automated_decisions': decision_counts.get('extend', 0) + decision_counts.get('map_exact', 0) + decision_counts.get('map_similar', 0),
            'uncertain_count': decision_counts.get('uncertain', 0),
            'average_confidence': np.mean([d.confidence for d in decisions]) if decisions else 0.0
        }
    # ===============================================================================
    
    def _infer_category(self, concept_name: str) -> str:
        """Infer category for concept based on name patterns."""
        concept_lower = concept_name.lower()
        
        if any(term in concept_lower for term in ['resistor', 'capacitor', 'inductor']):
            return 'Passive Components'
        elif any(term in concept_lower for term in ['transistor', 'diode', 'ic', 'microcontroller']):
            return 'Active Components'
        elif any(term in concept_lower for term in ['connector', 'cable', 'wire']):
            return 'Interconnects'
        else:
            return 'General Electronics'
    
    def _save_integration_results(self):
        """Save comprehensive integration results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results JSON
        results_file = self.output_dir / f"integration_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved integration results: {results_file.name}")
        
        # Save decision mappings
        mappings_file = self.output_dir / f"concept_mappings_{timestamp}.json"
        mappings_data = {
            'timestamp': self.results['timestamp'],
            'mappings': self.results['schema_objects_mapped'],
            'total_mappings': len(self.results['schema_objects_mapped'])
        }
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(mappings_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved concept mappings: {mappings_file.name}")
        
        # Save Schema.org objects
        schema_file = self.output_dir / f"schema_objects_{timestamp}.jsonld"
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(self.results['schema_objects_created'], f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved Schema.org objects: {schema_file.name}")
        
        # Generate human-readable report
        stats = self.results["integration_stats"]
        decisions = self.results["extension_decisions"]
        
        report_content = f"""
# Integrated Schema.org Pipeline Report

## Execution Summary
- **Timestamp**: {self.results['timestamp']}
- **Document Chunks Processed**: {self.results['chunks_processed']}
- **Concepts Extracted**: {len(self.results['concepts_extracted'])}
- **New Schema.org Objects**: {len(self.results['schema_objects_created'])}
- **Concept Mappings**: {len(self.results['schema_objects_mapped'])}

## Ontology Extension Decisions
- **Extend Ontology**: {stats['decision_counts'].get('extend', 0)} concepts
- **Map to Existing (Exact)**: {stats['decision_counts'].get('map_exact', 0)} concepts  
- **Map to Existing (Similar)**: {stats['decision_counts'].get('map_similar', 0)} concepts
- **Merge Concepts**: {stats['decision_counts'].get('merge', 0)} concepts
- **Uncertain/Review**: {stats['decision_counts'].get('uncertain', 0)} concepts

## Quality Metrics
- **Automation Rate**: {(stats['automated_decisions'] / stats['total_decisions'] * 100):.1f}%
- **Average Confidence**: {stats['average_confidence']:.2f}
- **Manual Review Required**: {stats['uncertain_count']} concepts

## Sample Decisions
"""
        
        # Add sample decisions
        for i, decision in enumerate(decisions[:5]):
            concept_name = self.results['concepts_extracted'][i] if i < len(self.results['concepts_extracted']) else f"Concept_{i}"
            report_content += f"""
### {i+1}. {concept_name}
- **Decision**: {decision.decision.value}
- **Confidence**: {decision.confidence:.2f}
- **Reasoning**: {decision.reasoning}
"""
            if decision.target_concept:
                report_content += f"- **Target**: {decision.target_concept}\n"
        
        report_content += f"""

## Integration Benefits Achieved
1. **Ontology Quality**: Prevented {stats['decision_counts'].get('map_exact', 0) + stats['decision_counts'].get('map_similar', 0)} potential duplicates
2. **Automation**: {(stats['automated_decisions'] / stats['total_decisions'] * 100):.1f}% of decisions automated
3. **Consistency**: Technical property matching applied to all concepts
4. **Traceability**: Full audit trail with confidence scores maintained

## Next Steps
1. Review {stats['uncertain_count']} concepts flagged for manual validation
2. Validate mapping decisions for {stats['decision_counts'].get('map_similar', 0)} similar concepts
3. Monitor ontology growth rate and quality metrics
4. Refine similarity thresholds based on validation feedback

---
*Generated by Integrated Schema.org Pipeline*
*Report ID: {timestamp}*
        """
        
        report_file = self.output_dir / f"integration_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"   ✅ Saved integration report: {report_file.name}")

def run_integrated_pipeline(config: PipelineConfig = None) -> IntegrationResults:
    """
    Main function to run the integrated Schema.org pipeline with ontology extension management.
    
    Args:
        config: Configuration object for pipeline parameters
        
    Returns:
        IntegrationResults with comprehensive statistics and decisions
    """
    pipeline = IntegratedSchemaOrgPipeline(config)
    return pipeline.run_integrated_pipeline()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run integrated Schema.org pipeline with ontology extension management")
    parser.add_argument("--max-chunks", type=int, help="Maximum chunks to process (for testing)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="High similarity threshold")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM validation")
    parser.add_argument("--output-dir", type=str, default="../data/integrated_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        max_chunks=args.max_chunks,
        similarity_thresholds={
            'exact_match': 0.95,
            'high_similarity': args.similarity_threshold,
            'medium_similarity': 0.70,
            'low_similarity': 0.50
        },
        enable_llm_validation=not args.disable_llm,
        output_dir=args.output_dir
    )
    
    try:
        # Run the integrated pipeline
        results = run_integrated_pipeline(config)
        
        print(f"\n🎉 INTEGRATION COMPLETE!")
        print(f"📊 Automation Rate: {results.automation_rate:.1f}%")
        print(f"📈 Average Confidence: {results.average_confidence:.2f}")
        print(f"⏱️ Processing Time: {results.processing_time:.1f}s")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()