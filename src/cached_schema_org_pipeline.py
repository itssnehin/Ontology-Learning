import logging
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

from src.data_loader import load_and_split_data
from src.idea_extractor import extract_ideas
from src.ontology_extension_manager import OntologyExtensionManager
from src.schema_org_extractor import SchemaOrgExtractor
from src.schema_org_relation_extractor import extract_schema_org_relations, SchemaOrgRelationExtractor
from src.schema_org_graph_builder import SchemaOrgGraphBuilder
from src.config import CACHE_DIR, MAX_WORKERS, LLM_MODEL
from src.data_models import PipelineConfig, ExtensionResult, ExtensionDecision

logger = logging.getLogger(__name__)

CACHE_PATHS = {
    "chunks": CACHE_DIR / "chunks.pkl",
    "concepts": CACHE_DIR / "concepts.pkl",
    "decisions": CACHE_DIR / "extension_decisions.pkl",
    "schema_objects": CACHE_DIR / "new_schema_objects.pkl",
    "mapped_objects": CACHE_DIR / "mapped_objects.pkl",
}

def clear_cache(start_step: str):
    """Clears cache files from the specified step onwards to force re-computation."""
    steps = list(CACHE_PATHS.keys())
    if start_step in steps:
        start_index = steps.index(start_step)
        for i in range(start_index, len(steps)):
            step_to_clear = steps[i]
            if CACHE_PATHS[step_to_clear].exists():
                logger.warning(f"Clearing cache for step: '{step_to_clear}' ({CACHE_PATHS[step_to_clear].name})")
                CACHE_PATHS[step_to_clear].unlink()

def run_cached_pipeline(resume_from: str = 'start', clear_downstream: bool = True, llm_model: str = LLM_MODEL, max_workers: int = MAX_WORKERS, progress_callback: Optional[Callable] = None):
    """
    Runs the ontology pipeline with caching, allowing resumption from intermediate steps.
    """
    logger.info("🚀" + "="*70)
    logger.info("STARTING CACHED PIPELINE RUN")
    logger.info(f"Resuming from step: '{resume_from}'")
    logger.info("="*70)

    if clear_downstream and resume_from != 'graph':
        clear_cache(resume_from)

    # --- Step 1: Load and Split Chunks ---
    if progress_callback: progress_callback("Loading Chunks", 10)
    if CACHE_PATHS["chunks"].exists() and resume_from != 'chunks':
        logger.info("📄 Loading chunks from cache...")
        with open(CACHE_PATHS["chunks"], "rb") as f:
            chunks = pickle.load(f)
    else:
        logger.info("📄 Step 1: Loading and processing documents...")
        chunks = load_and_split_data()
        with open(CACHE_PATHS["chunks"], "wb") as f:
            pickle.dump(chunks, f)
    logger.info(f"   ✅ Loaded {len(chunks)} chunks.")

    # --- Step 2: Extract Concepts ---
    if progress_callback: progress_callback("Extracting Concepts", 25)
    if CACHE_PATHS["concepts"].exists() and resume_from != 'concepts':
        logger.info("🧠 Loading concepts from cache...")
        with open(CACHE_PATHS["concepts"], "rb") as f:
            # The cached file only contains the concepts list
            extracted_concepts, _, _ = pickle.load(f)
    else:
        logger.info("\n🧠 Step 2: Extracting concepts (in parallel)...")
        # The function returns (concepts, in_tokens, out_tokens)
        extracted_concepts, in_tokens, out_tokens = extract_ideas(chunks, model_name=llm_model, max_workers=max_workers)
        with open(CACHE_PATHS["concepts"], "wb") as f:
            # Save all three for potential future cost analysis
            pickle.dump((extracted_concepts, in_tokens, out_tokens), f)
    logger.info(f"   ✅ Have {len(extracted_concepts)} unique concepts.")

    # --- Step 3 & 4: Analyze Concepts & Make Decisions ---
    if progress_callback: progress_callback("Analyzing Decisions", 50)
    if CACHE_PATHS["decisions"].exists() and resume_from != 'decisions':
        logger.info("🔍 Loading extension decisions from cache...")
        with open(CACHE_PATHS["decisions"], "rb") as f:
            extension_decisions = pickle.load(f)
    else:
        logger.info("\n🔍 Step 5: Analyzing concepts for ontology decisions (in parallel)...")
        config = PipelineConfig()
        # NOTE: Pass llm_model to manager if it's used for validation
        extension_manager = OntologyExtensionManager(config=config)
        extension_manager.load_existing_ontology()
        if extension_manager._existing_concepts:
            extension_manager.create_concept_embeddings(extension_manager._existing_concepts)
        
        extension_decisions = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_concept = {
                executor.submit(extension_manager.analyze_new_concept, {'name': name, 'category': 'Unknown'}): name
                for name in extracted_concepts
            }
            for future in tqdm(as_completed(future_to_concept), total=len(extracted_concepts), desc="Analyzing Concepts"):
                # analyze_new_concept returns (decision_result, cost)
                decision, cost = future.result() 
                if decision:
                    extension_decisions.append(decision)
                    
        with open(CACHE_PATHS["decisions"], "wb") as f:
            pickle.dump(extension_decisions, f)
    logger.info(f"   ✅ Processed {len(extension_decisions)} decisions.")
    
    # --- Route concepts based on decisions ---
    concepts_for_creation_dicts = []
    concepts_for_mapping = []
    for decision in extension_decisions:
        concept_dict = {'name': decision.concept_name, 'category': "Electronic Component"}
        if decision.decision == ExtensionDecision.EXTEND or decision.decision == ExtensionDecision.UNCERTAIN:
            concepts_for_creation_dicts.append(concept_dict)
        elif decision.decision in [ExtensionDecision.MAP_EXACT, ExtensionDecision.MAP_SIMILAR]:
            concepts_for_mapping.append(decision)

    # --- Step 5: Create Schema.org Objects ---
    if progress_callback: progress_callback("Creating Schema Objects", 75)
    if CACHE_PATHS["schema_objects"].exists() and resume_from != 'schema_objects':
        logger.info("🌐 Loading new Schema.org objects from cache...")
        with open(CACHE_PATHS["schema_objects"], "rb") as f:
            new_schema_objects = pickle.load(f)
    else:
        logger.info(f"\n🌐 Step 6: Creating Schema.org objects for {len(concepts_for_creation_dicts)} new concepts...")
        if concepts_for_creation_dicts:
            from src.integrated_schema_pipeline import IntegratedSchemaOrgPipeline # Import here to avoid circular dependency
            pipeline_instance = IntegratedSchemaOrgPipeline()
            # Pass all required arguments
            new_schema_objects = pipeline_instance._step_6_create_schema_objects_parallel(concepts_for_creation_dicts, chunks, llm_model, max_workers)
            with open(CACHE_PATHS["schema_objects"], "wb") as f:
                pickle.dump(new_schema_objects, f)
        else:
            new_schema_objects = []
            logger.info("   ✅ No new concepts require schema object creation.")
    logger.info(f"   ✅ Have {len(new_schema_objects)} new Schema.org objects.")

    # --- Step 6: Process Mappings ---
    if progress_callback: progress_callback("Processing Mappings", 85)
    logger.info(f"\n🔗 Step 7: Processing {len(concepts_for_mapping)} concept mappings...")
    mapped_objects = []
    for decision in concepts_for_mapping:
        mapped_objects.append({
            "@context": "https://schema.org/", "@type": "Product",
            "name": decision.concept_name, "sameAs": f"#{decision.target_concept}",
            "mappingConfidence": decision.confidence,
        })
    logger.info(f"   ✅ Created {len(mapped_objects)} mapping objects.")
    with open(CACHE_PATHS["mapped_objects"], "wb") as f:
        pickle.dump(mapped_objects, f)

    # --- Step 7: Update Knowledge Graph ---
    # CORRECTED LOGIC: Run this step unless the pipeline was specifically told to stop *before* it.
    if resume_from != 'graph':
        if progress_callback: progress_callback("Updating Knowledge Graph", 90)
        logger.info("\n🗃️ Step 8: Updating knowledge graph...")
        all_objects = new_schema_objects + mapped_objects
        if all_objects:
            builder = None
            try:
                builder = SchemaOrgGraphBuilder()
                builder.build_knowledge_graph_parallel(all_objects, max_workers=max_workers)
                logger.info("   ✅ Knowledge graph update complete.")
            finally:
                if builder:
                    builder.close()
        else:
            logger.info("   ℹ️ No objects to write to the graph.")
    else:
        logger.info("\n🗃️ Step 8: Skipping graph update as requested.")

    # --- Step 8: Finalize and Save Reports ---
    if progress_callback: progress_callback("Saving Reports", 98)
    if 'new_schema_objects' in locals() and 'mapped_objects' in locals():
        logger.info("\n💾 Generating final output files from cached run...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/integrated_output")
        output_dir.mkdir(exist_ok=True)

        if new_schema_objects:
            schema_file = output_dir / f"cached_new_schema_objects_{timestamp}.jsonld"
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump({"@context": "https://schema.org/", "@graph": new_schema_objects}, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Saved new Schema.org objects: {schema_file.name}")

        if mapped_objects:
            mappings_file = output_dir / f"cached_concept_mappings_{timestamp}.json"
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(mapped_objects, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Saved concept mappings: {mappings_file.name}")

    logger.info("\n🎉 CACHED PIPELINE RUN FINISHED! 🎉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the cached and resumable Schema.org pipeline.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="start",
        choices=['start', 'chunks', 'concepts', 'decisions', 'schema_objects', 'graph'],
        help="The step to resume the pipeline from. 'start' is an alias for 'chunks'."
    )
    args = parser.parse_args()

    start_step = args.resume_from
    if start_step == 'start':
        clear_cache('chunks')
        start_step = 'chunks'

    run_cached_pipeline(start_step)