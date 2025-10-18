#!/usr/bin/env python3
"""
Integrated Schema.org Pipeline with Intelligent Ontology Extension Management.
This version parallelizes all I/O-bound extraction and integration steps for maximum performance.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
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

    def run_integrated_pipeline(self, llm_model: str = LLM_MODEL, max_workers: int = MAX_WORKERS, progress_callback: Optional[Callable] = None, selected_files: Optional[List[str]] = None) -> IntegrationResults:
        """Executes the complete ontology learning pipeline and returns structured results."""
        start_time = datetime.now()
        logger.info("🚀" + "="*70)
        logger.info("STARTING ONTOLOGY LEARNING PIPELINE (PARALLELIZED)")
        logger.info("="*70)

        # --- Cost Tracking Initialization (remains the same) ---
        run_costs = {
            "concept_extraction": 0.0,
            "ontology_decision_analysis": 0.0,
            "total_cost": 0.0 # Simplified for this example
        }
        model_pricing = MODEL_COSTS.get(llm_model, MODEL_COSTS['default'])
        input_cost_per_1k = model_pricing['input_cost_per_1k_tokens']
        output_cost_per_1k = model_pricing['output_cost_per_1k_tokens']
        logger.info(f"💰 Using pricing for model '{llm_model}': Input=${input_cost_per_1k}/1k, Output=${output_cost_per_1k}/1k")
        
        # --- EXECUTE PIPELINE STEPS ---
        if progress_callback: progress_callback("Loading Documents", 10)
        # Pass the selected_files to the loading step
        chunks = self._step_1_load_documents(selected_files=selected_files)

        if progress_callback: progress_callback("Extracting Concepts", 25)
        extracted_concepts, in_tokens_extract, out_tokens_extract = self._step_2_extract_concepts(chunks, llm_model, max_workers)
        run_costs["concept_extraction"] = ((in_tokens_extract / 1000) * input_cost_per_1k) + ((out_tokens_extract / 1000) * output_cost_per_1k)
        
        if progress_callback: progress_callback("Loading Ontology", 50)
        self._step_3_and_4_load_ontology_and_embed()
        
        if progress_callback: progress_callback("Analyzing Decisions", 60)
        extension_decisions, cost_decision = self._step_5_analyze_concepts_parallel(extracted_concepts, max_workers, llm_model)
        run_costs["ontology_decision_analysis"] = cost_decision
        
        # --- THIS IS THE KEY CHANGED SECTION ---
        
        # 1. The routing function now returns a list of tasks for the ontology.
        #    The second variable (for mapped objects) is now empty.
        if progress_callback: progress_callback("Routing Decisions", 85)
        ontology_extension_tasks, _ = self._route_concepts_based_on_decisions(extension_decisions)
        
        # 2. We no longer run the schema object creation step, as we are building the ontology directly.
        #    This entire block can be commented out or deleted.
        # new_schema_objects, run_costs["schema_object_creation"] = self._step_6_create_schema_objects_parallel(...)
        # mapped_objects = self._step_7_process_mappings(...)
        
        # 3. The graph builder is now called with the list of ontology tasks.
        #    The second argument is an empty list because there are no separate mapped objects.
        if progress_callback: progress_callback("Extending Ontology Graph", 90)
        self._step_8_update_knowledge_graph_parallel(ontology_extension_tasks, [], max_workers=max_workers)
        
        # --- END OF CHANGED SECTION ---
        
        # --- FINALIZE AND RETURN RESULTS ---
        run_costs["total_cost"] = sum(run_costs.values())
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # This part needs a slight update to reflect the new meaning of the results
        final_results = self._prepare_final_results(
            extracted_concepts,
            extension_decisions, # Pass the raw decisions
            processing_time,
            run_costs
        )

        # The saving step now saves tasks, not schema objects
        self._step_9_save_reports(final_results, ontology_extension_tasks)

        logger.info("\n" + "="*70)
        logger.info("🎉 ONTOLOGY LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Total concepts processed: {final_results.total_concepts_extracted}")
        logger.info(f"🔗 Concepts mapped to existing classes: {final_results.concepts_mapped_to_existing}")
        logger.info(f"🆕 New classes learned: {final_results.concepts_extending_ontology}")

        return final_results

    # --- Step Execution Methods ---

    def _step_1_load_documents(self, selected_files: Optional[List[str]] = None) -> List[Document]:
        """Loads and preprocesses documents from the data directory."""
        logger.info("\n📄 Step 1: Loading and processing documents...")
        # Pass the file list to the actual data loader function
        chunks = load_and_split_data(files_to_process=selected_files)
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

    def _analyze_single_concept(self, concept_name: str) -> Tuple[ExtensionResult, float]: #<-- UPDATE RETURN SIGNATURE
        """Helper for parallel execution: analyzes one concept and returns its decision and cost."""
        concept_dict = {
            'name': concept_name,
            'category': self._infer_category(concept_name),
            'description': f"Electronic component: {concept_name}",
        }
        # This function now correctly returns a (result, cost) tuple
        return self.extension_manager.analyze_new_concept(concept_dict)

    def _step_5_analyze_concepts_parallel(self, extracted_concepts: List[str], max_workers: int, llm_model: str) -> Tuple[List[ExtensionResult], float]:
        """Analyzes concepts in parallel and returns the decisions and the total cost."""
        logger.info("\n🔍 Step 5: Analyzing concepts for ontology decisions (in parallel)...")
        extension_decisions = []
        total_cost = 0.0  # <-- Initialize cost for this stage

        # Ensure the manager is using the correct model for any validation calls
        self.extension_manager.llm.model_name = llm_model

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_concept = {executor.submit(self._analyze_single_concept, name): name for name in extracted_concepts}
            for future in tqdm(as_completed(future_to_concept), total=len(extracted_concepts), desc="Analyzing Concepts"):
                try:
                    # This is the line that was crashing silently inside the thread
                    decision, cost = future.result()
                    total_cost += cost
                    if decision:
                        extension_decisions.append(decision)
                except Exception as e:
                    # If one concept fails, log it and continue with the rest.
                    concept_name = future_to_concept[future]
                    logger.error(f"Failed to analyze concept '{concept_name}': {e}", exc_info=True)
        
        logger.info(f"   📊 Token Usage for Decision Analysis: Total Cost=${total_cost:.5f}")
        return extension_decisions, total_cost
    
    def _route_concepts_based_on_decisions(self, extension_decisions: List[ExtensionResult]):
        """
        Processes decisions to generate a list of ontology extension tasks,
        ensuring non-taxonomic relations are included.
        """
        ontology_extension_tasks = []
        
        for decision in extension_decisions:
            # We only create tasks for decisions that modify the ontology
            if decision.decision not in [ExtensionDecision.EXTEND, ExtensionDecision.UNCERTAIN]:
                continue

            task = {
                'name': decision.concept_name,
                'action': 'CREATE_CLASS'
            }
            
            if decision.decision == ExtensionDecision.UNCERTAIN:
                task['status'] = 'review'

            # Infer the parent class
            if decision.matches:
                task['parent_class'] = decision.target_concept or decision.matches[0].existing_concept
            else:
                task['parent_class'] = 'ElectronicComponent'
            
            # --- THIS IS THE CRITICAL FIX ---
            # Correctly retrieve and add the non-taxonomic relations to the task.
            if decision.non_taxonomic_relations:
                task['non_taxonomic_relations'] = decision.non_taxonomic_relations
            # --- END OF FIX ---
            
            ontology_extension_tasks.append(task)

        return ontology_extension_tasks, []



    def _create_schema_for_single_concept(self, concept_dict: Dict, all_chunks: List[Document], llm_model: str) -> Tuple[Optional[Dict], int, int]:
        """Helper for parallel execution: creates one Schema.org object and returns token counts."""
        in_tokens, out_tokens = 0, 0
        try:
            concept_name = concept_dict['name']
            pseudo_chunk = self._create_concept_chunks([concept_dict], all_chunks)
            schema_extractor = SchemaOrgExtractor(model_name=llm_model)
            
            # This function now returns (objects, in_tokens, out_tokens)
            base_objects, in_tokens, out_tokens = schema_extractor.extract_schema_org_data(pseudo_chunk, [concept_name])

            if not base_objects:
                return None, in_tokens, out_tokens
            
            # The relation extractor also needs to be updated to pass the model
            relations_data, rel_in_tokens, rel_out_tokens = extract_schema_org_relations(pseudo_chunk, [concept_name], model_name=llm_model)
            
            # Add the tokens from this step to the total
            in_tokens += rel_in_tokens
            out_tokens += rel_out_tokens
            
            
            relation_extractor = SchemaOrgRelationExtractor(model_name=llm_model)
            enhanced_objects = relation_extractor.generate_enhanced_schema_objects(base_objects, relations_data)
            
            obj = enhanced_objects[0] if enhanced_objects else None
            return obj, in_tokens, out_tokens
        except Exception as e:
            logger.warning(f"Failed to create Schema.org object for {concept_dict.get('name', 'N/A')}: {e}")
            return None, in_tokens, out_tokens


    def _step_6_create_schema_objects_parallel(self, concepts_for_creation: List[Dict], all_chunks: List[Document], llm_model: str, max_workers: int) -> Tuple[List[Dict], int, int]:
        """Generates Schema.org objects in parallel and returns objects and total token counts."""
        logger.info(f"\n🌐 Creating Schema.org objects for {len(concepts_for_creation)} new concepts...")
        if not concepts_for_creation:
            return [], 0, 0
            
        new_schema_objects = []
        total_input_tokens = 0
        total_output_tokens = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._create_schema_for_single_concept, c, all_chunks, llm_model): c for c in concepts_for_creation}
            
            for future in tqdm(as_completed(futures), total=len(concepts_for_creation), desc="Creating Schema Objects"):
                # Unpack the object and its token counts
                result_obj, in_tokens, out_tokens = future.result()
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                if result_obj:
                    new_schema_objects.append(result_obj)

        logger.info(f"   ✅ Created {len(new_schema_objects)} new Schema.org objects.")
        logger.info(f"   📊 Token Usage: {total_input_tokens:,} input, {total_output_tokens:,} output")
        return new_schema_objects, total_input_tokens, total_output_tokens


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

    def _prepare_final_results(self, extracted_concepts, extension_decisions, processing_time, costs) -> IntegrationResults:
        """Constructs the final IntegrationResults object for an ontology learning run."""
        
        # Recalculate counts based on the decisions made
        mapped_count = sum(1 for d in extension_decisions if d.decision in [ExtensionDecision.MAP_EXACT, ExtensionDecision.MAP_SIMILAR])
        extended_count = sum(1 for d in extension_decisions if d.decision == ExtensionDecision.EXTEND)
        uncertain_count = sum(1 for d in extension_decisions if d.decision == ExtensionDecision.UNCERTAIN)
        
        return IntegrationResults(
            total_concepts_extracted=len(extracted_concepts),
            concepts_mapped_to_existing=mapped_count,
            concepts_extending_ontology=extended_count,
            concepts_requiring_review=uncertain_count,
            confidence_scores=[d.confidence for d in extension_decisions if d.confidence is not None],
            processing_time=processing_time,
            decisions=extension_decisions,
            costs=costs
        )


    def _step_9_save_reports(self, final_results: IntegrationResults, ontology_extension_tasks: List[Dict]):
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
        if ontology_extension_tasks:
            tasks_file = self.output_dir / f"learned_ontology_tasks_{timestamp}.json"
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(ontology_extension_tasks, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Saved learned ontology tasks: {tasks_file.name}")
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
def run_integrated_pipeline(config: Optional[PipelineConfig] = None, llm_model: str = LLM_MODEL, max_workers: int = MAX_WORKERS, progress_callback: Optional[Callable] = None, selected_files: Optional[List[str]] = None):
    pipeline = IntegratedSchemaOrgPipeline(config)
    return pipeline.run_integrated_pipeline(llm_model=llm_model, max_workers=max_workers, progress_callback=progress_callback, selected_files=selected_files)
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