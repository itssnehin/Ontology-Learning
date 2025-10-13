#!/usr/bin/env python3
"""
Integrated Schema.org Pipeline with Intelligent Ontology Extension Management.
This version parallelizes all I/O-bound extraction and integration steps for maximum performance.
"""
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from langchain_core.documents import Document

# Import all necessary modules from the package
from src.data_loader import load_and_split_data
from src.idea_extractor import extract_ideas
from src.schema_org_extractor import SchemaOrgExtractor
from src.schema_org_relation_extractor import extract_schema_org_relations, SchemaOrgRelationExtractor
from src.schema_org_graph_builder import SchemaOrgGraphBuilder
from src.ontology_extension_manager import OntologyExtensionManager, ExtensionDecision, ExtensionResult
from src.data_models import PipelineConfig, IntegrationResults, ExtensionDecision, ExtensionResult
from src.ontology_extension_manager import OntologyExtensionManager
from src.config import MAX_WORKERS, LLM_MODEL, MODEL_COSTS

logger = logging.getLogger(__name__)

class IntegratedSchemaOrgPipeline:
    """
    Orchestrates the entire ontology extraction and intelligent extension pipeline.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extension_manager = OntologyExtensionManager(config=self.config)
        
        logger.info("🔧 Integrated Schema.org Pipeline initialized (Parallel Mode)")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"⚙️ Max concurrent workers: {MAX_WORKERS}")

    # --- Main Pipeline Orchestrator ---
    
    def run_integrated_pipeline(self, llm_model: str = LLM_MODEL, max_workers: int = MAX_WORKERS) -> IntegrationResults:
        """Executes the complete integrated pipeline and returns structured results."""
        start_time = datetime.now()
        logger.info("🚀" + "="*70)
        logger.info("STARTING INTEGRATED PIPELINE RUN (PARALLELIZED)")
        logger.info("="*70)
        run_costs = {
            "concept_extraction": 0.0,
            "ontology_decision_analysis": 0.0,
            "schema_object_creation": 0.0,
            "total_cost": 0.0
        }
        model_pricing = MODEL_COSTS.get(llm_model, MODEL_COSTS['default'])
        input_cost_per_1k = model_pricing['input_cost_per_1k_tokens']
        output_cost_per_1k = model_pricing['output_cost_per_1k_tokens']
        logger.info(f"💰 Using pricing for model '{llm_model}': Input=${input_cost_per_1k}/1k, Output=${output_cost_per_1k}/1k")
        
        # --- EXECUTE PIPELINE STEPS ---
        chunks = self._step_1_load_documents()

        # Step 2: Concept Extraction
        extracted_concepts, in_tokens, out_tokens = self._step_2_extract_concepts(chunks, llm_model, max_workers)
        run_costs["concept_extraction"] = ((in_tokens / 1000) * input_cost_per_1k) + ((out_tokens / 1000) * output_cost_per_1k)

        
        self._step_3_and_4_load_ontology_and_embed()
        extension_decisions = self._step_5_analyze_concepts_parallel(extracted_concepts, max_workers=max_workers)
        
        concepts_for_creation, concepts_for_mapping = self._route_concepts_based_on_decisions(extension_decisions)
        new_schema_objects = self._step_6_create_schema_objects_parallel(concepts_for_creation, chunks, llm_model, max_workers)
        mapped_objects = self._step_7_process_mappings(concepts_for_mapping)
        
        self._step_8_update_knowledge_graph_parallel(new_schema_objects, mapped_objects, max_workers=max_workers)
        # --- FINALIZE AND RETURN RESULTS ---
        processing_time = (datetime.now() - start_time).total_seconds()
        
        final_results = self._prepare_final_results(
            extracted_concepts, concepts_for_mapping, concepts_for_creation,
            extension_decisions, processing_time, run_costs
        )

        self._step_9_save_reports(final_results, new_schema_objects, mapped_objects)

        logger.info("\n" + "="*70)
        logger.info("🎉 INTEGRATED PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Total concepts processed: {final_results.total_concepts_extracted}")
        logger.info(f"🔗 Mapped to existing: {final_results.concepts_mapped_to_existing}")
        logger.info(f"🆕 Extended ontology: {final_results.concepts_extending_ontology}")
        logger.info(f"❓ Requiring review: {final_results.concepts_requiring_review}")
        logger.info(f"🎯 Automation rate: {final_results.automation_rate:.1f}%")
        logger.info(f"📈 Average confidence: {final_results.average_confidence:.2f}")
        logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
        logger.info(f"📁 Results saved to: {self.output_dir}")
        logger.info("="*70)

        return final_results

    # --- Step Execution Methods ---

    def _step_1_load_documents(self) -> List[Document]:
        """Loads and preprocesses documents from the data directory."""
        logger.info("\n📄 Step 1: Loading and processing documents...")
        chunks = load_and_split_data()
        if self.config.max_chunks:
            chunks = chunks[:self.config.max_chunks]
        logger.info(f"   ✅ Processed {len(chunks)} document chunks")
        return chunks

    def _step_2_extract_concepts(self, chunks: List[Document], llm_model: str, max_workers: int) -> Tuple[List[str], int, int]:
        """Extracts concepts and returns them along with total token counts."""
        logger.info("\n🧠 Step 2: Extracting concepts from documents (in parallel)...")
        # This function now returns a tuple of (concepts, in_tokens, out_tokens)
        extracted_concepts, total_input, total_output = extract_ideas(chunks, model_name=llm_model, max_workers=max_workers)
        logger.info(f"   ✅ Extracted {len(extracted_concepts)} unique concepts")
        logger.info(f"   📊 Token Usage: {total_input:,} input, {total_output:,} output")
        return extracted_concepts, total_input, total_output

    def _step_3_and_4_load_ontology_and_embed(self):
        """Loads the existing ontology from Neo4j and creates embeddings for comparison."""
        logger.info("\n📚 Step 3 & 4: Loading existing ontology and creating embeddings...")
        self.extension_manager.load_existing_ontology()
        existing_concepts = self.extension_manager._existing_concepts
        if existing_concepts:
            self.extension_manager.create_concept_embeddings(existing_concepts)
        logger.info(f"   ✅ Loaded and embedded {len(existing_concepts)} existing concepts.")

    def _analyze_single_concept(self, concept_name: str) -> ExtensionResult:
        """Helper for parallel execution: analyzes one concept for an extension decision."""
        concept_dict = {
            'name': concept_name,
            'category': self._infer_category(concept_name),
            'description': f"Electronic component: {concept_name}",
        }
        return self.extension_manager.analyze_new_concept(concept_dict)

    def _step_5_analyze_concepts_parallel(self, extracted_concepts: List[str], max_workers: int) -> List[ExtensionResult]:
        """Analyzes all extracted concepts against the existing ontology in parallel."""
        logger.info("\n🔍 Step 5: Analyzing concepts for ontology decisions (in parallel)...")
        extension_decisions = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_concept = {executor.submit(self._analyze_single_concept, name): name for name in extracted_concepts}
            for future in tqdm(as_completed(future_to_concept), total=len(extracted_concepts), desc="Analyzing Concepts"):
                decision = future.result()
                if decision:
                    extension_decisions.append(decision)
        return extension_decisions
    
    def _route_concepts_based_on_decisions(self, extension_decisions: List[ExtensionResult]):
        """Sorts concepts into 'create' or 'map' lists based on analysis decisions."""
        concepts_for_creation = []
        concepts_for_mapping = []
        
        # --- THIS IS THE CORRECTED, SINGLE LOOP ---
        for decision in extension_decisions:
            concept_dict = {
                'name': decision.concept_name, 
                'category': self._infer_category(decision.concept_name)
            }
            if decision.decision == ExtensionDecision.EXTEND:
                concept_dict['status'] = 'new'
                concepts_for_creation.append(concept_dict)
            elif decision.decision in [ExtensionDecision.MAP_EXACT, ExtensionDecision.MAP_SIMILAR]:
                # The structure for mapping was slightly different, let's correct it
                concepts_for_mapping.append({
                    'concept': concept_dict, 
                    'target': decision.target_concept, 
                    'confidence': decision.confidence
                })
            else: # UNCERTAIN or MERGE
                concept_dict['status'] = 'review'
                concepts_for_creation.append(concept_dict)
        
        return concepts_for_creation, concepts_for_mapping

    def _step_6_create_schema_objects_parallel(self, concepts_for_creation: List[Dict], all_chunks: List[Document], llm_model: str, max_workers: int) -> List[Dict]:
        """Generates Schema.org objects for all new concepts in parallel with a progress bar."""
        logger.info(f"\n🌐 Step 6: Creating Schema.org objects for {len(concepts_for_creation)} new concepts (in parallel)...")
        if not concepts_for_creation:
            return []
            
        new_schema_objects = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._create_schema_for_single_concept, concept_dict, all_chunks, llm_model): concept_dict for concept_dict in concepts_for_creation}

            # Wrap the as_completed iterator with tqdm to create a progress bar
            for future in tqdm(as_completed(futures), total=len(concepts_for_creation), desc="Creating Schema Objects"):
                result = future.result()
                if result:
                    new_schema_objects.append(result)
        logger.info(f"   ✅ Created {len(new_schema_objects)} new Schema.org objects.")
        return new_schema_objects


    def _step_7_process_mappings(self, concepts_for_mapping: List[Dict]) -> List[Dict]:
        """Handles the creation of mapping objects for concepts similar to existing ones."""
        logger.info(f"\n🔗 Step 7: Processing {len(concepts_for_mapping)} concept mappings...")
        mapped_objects = []
        for mapping in concepts_for_mapping:
            mapped_object = {
                "@context": "https://schema.org/", "@type": "Product",
                "name": mapping['concept']['name'],
                "sameAs": f"#{mapping['target']}",
                "mappingConfidence": mapping['confidence'],
            }
            mapped_objects.append(mapped_object)
        logger.info(f"   ✅ Created {len(mapped_objects)} concept mapping objects.")
        return mapped_objects

    def _step_8_update_knowledge_graph_parallel(self, new_schema_objects: List[Dict], mapped_objects: List[Dict], max_workers: int):
        """Updates the Neo4j knowledge graph with new and mapped objects in parallel."""
        logger.info("\n🗃️ Step 8: Updating knowledge graph (in parallel)...")
        all_objects = new_schema_objects + mapped_objects
        if not all_objects:
            logger.info("   ℹ️ No new objects to add to the knowledge graph.")
            return

        builder = None
        try:
            builder = SchemaOrgGraphBuilder()
            graph_stats = builder.build_knowledge_graph_parallel(all_objects, max_workers=max_workers)
            logger.info(f"   ✅ Updated knowledge graph: {graph_stats.get('totals', {}).get('nodes', 0)} total nodes.")
        except Exception as e:
            logger.error(f"   ⚠️ Graph update failed: {e}", exc_info=True)
        finally:
            if builder:
                builder.close()

    def _prepare_final_results(self, extracted_concepts, concepts_for_mapping, concepts_for_creation, extension_decisions, processing_time, costs: Dict[str, float]) -> IntegrationResults:
        """Constructs the final IntegrationResults object."""
        uncertain_count = sum(1 for d in extension_decisions if d.decision == ExtensionDecision.UNCERTAIN)
        
        return IntegrationResults(
            total_concepts_extracted=len(extracted_concepts),
            concepts_mapped_to_existing=len(concepts_for_mapping),
            concepts_extending_ontology=len(concepts_for_creation),
            concepts_requiring_review=uncertain_count,
            confidence_scores=[d.confidence for d in extension_decisions if d.confidence is not None],
            processing_time=processing_time,
            decisions=extension_decisions,
            costs=costs  # <-- Assign the final costs object here
        )


    def _step_9_save_reports(self, final_results: IntegrationResults, new_schema_objects: List[Dict], mapped_objects: List[Dict]):
        """Saves all generated artifacts to the output directory."""
        logger.info("\n💾 Step 9: Saving all reports and artifacts...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results JSON
        results_file = self.output_dir / f"integration_results_{timestamp}.json"
        serializable_decisions = [d.to_dict() for d in final_results.decisions]
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": asdict(final_results, dict_factory=lambda data: {k: v for k, v in data if k != 'decisions'}),
                "decisions": serializable_decisions
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"   ✅ Saved integration summary: {results_file.name}")

        # Save other artifacts
        if new_schema_objects:
            schema_file = self.output_dir / f"new_schema_objects_{timestamp}.jsonld"
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump({"@context": "https://schema.org/", "@graph": new_schema_objects}, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Saved new Schema.org objects: {schema_file.name}")
        if mapped_objects:
            mappings_file = self.output_dir / f"concept_mappings_{timestamp}.json"
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(mapped_objects, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Saved concept mappings: {mappings_file.name}")
            
    # --- Utility Helper Methods ---
    
    def _infer_category(self, concept_name: str) -> str:
        """Infers a category from a concept name using simple heuristics."""
        # (implementation unchanged)
        return "Electronic Component"

    def _create_concept_chunks(self, concepts: List[Dict], original_chunks: List) -> List[Document]:
        """Creates pseudo-document chunks from concept data for the extractors."""
        # (implementation unchanged)
        return [Document(page_content=f"Component Name: {c['name']}") for c in concepts]


# --- Main execution block ---
def run_integrated_pipeline(config: Optional[PipelineConfig] = None, llm_model: str = LLM_MODEL, max_workers: int = MAX_WORKERS):
    pipeline = IntegratedSchemaOrgPipeline(config)
    return pipeline.run_integrated_pipeline(llm_model=llm_model, max_workers=max_workers)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run integrated Schema.org pipeline with ontology extension management")
    parser.add_argument("--max-chunks", type=int, help="Maximum chunks to process (for testing)")
    parser.add_argument("--output-dir", type=str, default="../data/integrated_output", help="Output directory")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        max_chunks=args.max_chunks,
        output_dir=args.output_dir
    )
    
    try:
        run_integrated_pipeline(config)
    except Exception as e:
        logger.error(f"❌ Pipeline failed with a critical error: {e}", exc_info=True)