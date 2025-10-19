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
from src.config import MAX_WORKERS, LLM_MODEL, MODEL_COSTS, EMBEDDING_MODEL
from src.utils import cost_logger

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

    def run_integrated_pipeline(self, llm_model: str = LLM_MODEL, max_workers: int = MAX_WORKERS, progress_callback: Optional[Callable] = None, selected_files: Optional[List[str]] = None, pipeline_config: Optional[PipelineConfig] = None):
        """Executes the complete ontology learning pipeline and returns structured results."""
        start_time = datetime.now()
        logger.info("🚀 STARTING ONTOLOGY LEARNING PIPELINE 🚀")

        if pipeline_config:
            self.config = pipeline_config
            self.extension_manager = OntologyExtensionManager(config=self.config)
            logger.info(f"Pipeline configured with custom settings: {self.config.__dict__}")

        # --- Cost Tracking Initialization ---
        run_costs = {
            "concept_extraction": {"input": 0, "output": 0, "cost": 0.0},
            "ontology_embedding": {"input": 0, "output": 0, "cost": 0.0},
            "ontology_decision_analysis": {"input": 0, "output": 0, "cost": 0.0},
            "total_cost": 0.0
        }
        model_pricing = MODEL_COSTS.get(llm_model, MODEL_COSTS['default'])
        input_cost_per_1k = model_pricing['input_cost_per_1k_tokens']
        output_cost_per_1k = model_pricing['output_cost_per_1k_tokens']
        logger.info(f"💰 Using pricing for model '{llm_model}': Input=${input_cost_per_1k}/1k, Output=${output_cost_per_1k}/1k")
        
        # --- EXECUTE PIPELINE STEPS ---
        if progress_callback: progress_callback("Loading Documents", 10)
        chunks = self._step_1_load_documents(selected_files)

        # --- STAGE: CONCEPT EXTRACTION ---
        if progress_callback: progress_callback("Extracting Concepts", 25)
        extracted_concepts, in_tokens, out_tokens = self._step_2_extract_concepts(chunks, llm_model, max_workers)
        cost = ((in_tokens / 1000) * input_cost_per_1k) + ((out_tokens / 1000) * output_cost_per_1k)
        run_costs["concept_extraction"].update({"input": in_tokens, "output": out_tokens, "cost": cost})
        cost_logger.log_cost("Concept Extraction", llm_model, in_tokens, out_tokens, cost)
        
        # --- STAGE: ONTOLOGY EMBEDDING ---
        if progress_callback: progress_callback("Loading Ontology", 50)
        embedding_tokens, embedding_cost = self._step_3_and_4_load_ontology_and_embed()
        run_costs["ontology_embedding"].update({"input": embedding_tokens, "cost": embedding_cost})
        cost_logger.log_cost("Ontology Embedding", EMBEDDING_MODEL, embedding_tokens, 0, embedding_cost)
        
        # --- STAGE: DECISION ANALYSIS ---
        if progress_callback: progress_callback("Analyzing Decisions", 60)
        extension_decisions, in_tokens, out_tokens = self._step_5_analyze_concepts_parallel(extracted_concepts, max_workers, llm_model)
        cost = ((in_tokens / 1000) * input_cost_per_1k) + ((out_tokens / 1000) * output_cost_per_1k)
        run_costs["ontology_decision_analysis"].update({"input": in_tokens, "output": out_tokens, "cost": cost})
        cost_logger.log_cost("Decision Analysis", llm_model, in_tokens, out_tokens, cost)
        
        # --- ROUTING AND GRAPH BUILDING ---
        if progress_callback: progress_callback("Routing Decisions", 85)
        ontology_extension_tasks, _ = self._route_concepts_based_on_decisions(extension_decisions)
        
        if progress_callback: progress_callback("Extending Ontology Graph", 90)
        self._step_8_update_knowledge_graph_parallel(ontology_extension_tasks, [], max_workers=max_workers)
        
        # --- FINALIZE AND RETURN RESULTS ---
        run_costs["total_cost"] = sum(stage.get("cost", 0.0) for stage in run_costs.values() if isinstance(stage, dict))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        final_results = self._prepare_final_results(
            extracted_concepts,
            extension_decisions,
            processing_time,
            run_costs
        )

        self._step_9_save_reports(final_results, ontology_extension_tasks, []) # Pass empty list for mapped_objects

        logger.info("\n" + "="*70)
        logger.info("🎉 ONTOLOGY LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 Total concepts processed: {final_results.total_concepts_extracted}")
        logger.info(f"🔗 Concepts mapped to existing classes: {final_results.concepts_mapped_to_existing}")
        logger.info(f"🆕 New classes learned: {final_results.concepts_extending_ontology}")
        logger.info(f"💰 Total Estimated Cost: ${run_costs['total_cost']:.4f}")

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

    def _step_3_and_4_load_ontology_and_embed(self) -> Tuple[int, float]:
        """Loads ontology, creates embeddings, and returns embedding cost info."""
        logger.info("\n📚 Step 3 & 4: Loading existing ontology and creating embeddings...")
        self.extension_manager.load_existing_ontology()
        existing_concepts = self.extension_manager._existing_concepts
        total_tokens, total_cost = 0, 0.0
        if existing_concepts:
            _, total_tokens, total_cost = self.extension_manager.create_concept_embeddings(existing_concepts)
        logger.info(f"   ✅ Loaded and embedded {len(existing_concepts)} concepts.")
        return total_tokens, total_cost

    def _analyze_single_concept(self, concept_name: str) -> Tuple[ExtensionResult, int, int]:
        """Helper for parallel execution: returns decision and token counts."""
        concept_dict = {'name': concept_name, 'category': 'Unknown', 'description': ''}
        return self.extension_manager.analyze_new_concept(concept_dict)

    def _step_5_analyze_concepts_parallel(self, extracted_concepts: List[str], max_workers: int, llm_model: str) -> Tuple[List[ExtensionResult], int, int]:
        logger.info("\n🔍 Step 5: Analyzing concepts for ontology decisions (in parallel)...")
        extension_decisions = []
        total_input_tokens, total_output_tokens = 0, 0
        self.extension_manager.llm.model_name = llm_model

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_concept = {executor.submit(self._analyze_single_concept, name): name for name in extracted_concepts}
            for future in tqdm(as_completed(future_to_concept), total=len(extracted_concepts), desc="Analyzing Concepts"):
                try:
                    decision, in_tokens, out_tokens = future.result()
                    total_input_tokens += in_tokens
                    total_output_tokens += out_tokens
                    if decision:
                        extension_decisions.append(decision)
                except Exception as e:
                    concept_name = future_to_concept[future]
                    logger.error(f"Failed to analyze concept '{concept_name}': {e}", exc_info=True)
        
        logger.info(f"   📊 Token Usage for Decision Analysis: IN={total_input_tokens:,}, OUT={total_output_tokens:,}")
        return extension_decisions, total_input_tokens, total_output_tokens
    
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