import logging
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Import pipeline components and data models
from src.data_loader import load_and_split_data
from src.idea_extractor import extract_ideas
from src.ontology_extension_manager import OntologyExtensionManager
from src.schema_org_extractor import SchemaOrgExtractor
from src.schema_org_relation_extractor import extract_schema_org_relations, SchemaOrgRelationExtractor
from src.schema_org_graph_builder import SchemaOrgGraphBuilder
from src.config import CACHE_DIR, MAX_WORKERS
from src.data_models import PipelineConfig, ExtensionResult

logger = logging.getLogger(__name__)

# --- Cache File Definitions ---
CACHE_PATHS = {
    "chunks": CACHE_DIR / "chunks.pkl",
    "concepts": CACHE_DIR / "concepts.pkl",
    "decisions": CACHE_DIR / "extension_decisions.pkl",
    "schema_objects": CACHE_DIR / "new_schema_objects.pkl",
    "mapped_objects": CACHE_DIR / "mapped_objects.pkl",
}

def clear_cache(start_step: str):
    """Clears cache files from the specified step onwards."""
    steps = list(CACHE_PATHS.keys())
    if start_step in steps:
        start_index = steps.index(start_step)
        for i in range(start_index, len(steps)):
            step_to_clear = steps[i]
            if CACHE_PATHS[step_to_clear].exists():
                logger.warning(f"Clearing cache for step: '{step_to_clear}'")
                CACHE_PATHS[step_to_clear].unlink()

def run_cached_pipeline(resume_from: str = 'start', clear_downstream: bool = True):
    """
    Runs the ontology pipeline with caching at each major step.
    
    Args:
        resume_from: The step to start from. Can be 'start', 'concepts', 'decisions', 'schema', 'graph'.
        clear_downstream: If True, clears the cache for all steps after the resume point.
    """
    logger.info("🚀" + "="*70)
    logger.info("STARTING CACHED PIPELINE RUN")
    logger.info(f"Resuming from step: '{resume_from}'")
    logger.info("="*70)

    if clear_downstream:
        clear_cache(resume_from)

    # --- Step 1: Load and Split Chunks ---
    if CACHE_PATHS["chunks"].exists():
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
    if CACHE_PATHS["concepts"].exists():
        logger.info("🧠 Loading concepts from cache...")
        with open(CACHE_PATHS["concepts"], "rb") as f:
            extracted_concepts = pickle.load(f)
    else:
        logger.info("\n🧠 Step 2: Extracting concepts (in parallel)...")
        extracted_concepts = extract_ideas(chunks, max_workers=MAX_WORKERS)
        with open(CACHE_PATHS["concepts"], "wb") as f:
            pickle.dump(extracted_concepts, f)
    logger.info(f"   ✅ Have {len(extracted_concepts)} unique concepts.")

    # --- Step 3 & 4 are combined into Step 5 logic ---

    # --- Step 5: Analyze Concepts & Make Decisions ---
    if CACHE_PATHS["decisions"].exists():
        logger.info("🔍 Loading extension decisions from cache...")
        with open(CACHE_PATHS["decisions"], "rb") as f:
            extension_decisions = pickle.load(f)
    else:
        logger.info("\n🔍 Step 5: Analyzing concepts for ontology decisions (in parallel)...")
        config = PipelineConfig() # Using default config for this run
        extension_manager = OntologyExtensionManager(config=config)
        extension_manager.load_existing_ontology()
        extension_manager.create_concept_embeddings(extension_manager._existing_concepts)
        
        extension_decisions = []
        # (This part is a simplified version of the parallel analysis in the integrated pipeline)
        # For simplicity, we'll keep it sequential here, but it could be parallelized too.
        for name in tqdm(extracted_concepts, desc="Analyzing Concepts"):
            concept_dict = {'name': name, 'category': 'Unknown'} # Simplified for caching
            decision = extension_manager.analyze_new_concept(concept_dict)
            extension_decisions.append(decision)
            
        with open(CACHE_PATHS["decisions"], "wb") as f:
            pickle.dump(extension_decisions, f)
    logger.info(f"   ✅ Processed {len(extension_decisions)} decisions.")
    
    # --- Route concepts based on decisions ---
    concepts_for_creation = [d.concept_name for d in extension_decisions if d.decision == 'extend_ontology']
    concepts_for_mapping = [d for d in extension_decisions if d.decision in ['map_to_existing_exact', 'map_to_existing_similar']]
    
    # --- Step 6: Create Schema.org Objects ---
    if CACHE_PATHS["schema_objects"].exists():
        logger.info("🌐 Loading new Schema.org objects from cache...")
        with open(CACHE_PATHS["schema_objects"], "rb") as f:
            new_schema_objects = pickle.load(f)
    else:
        logger.info(f"\n🌐 Step 6: Creating Schema.org objects for {len(concepts_for_creation)} new concepts...")
        # (Simplified version of the parallel creation)
        schema_extractor = SchemaOrgExtractor()
        new_schema_objects = schema_extractor.extract_schema_org_data(chunks, concepts_for_creation)
        with open(CACHE_PATHS["schema_objects"], "wb") as f:
            pickle.dump(new_schema_objects, f)
    logger.info(f"   ✅ Created {len(new_schema_objects)} new Schema.org objects.")

    # --- Step 7: Process Mappings ---
    logger.info(f"\n🔗 Step 7: Processing {len(concepts_for_mapping)} concept mappings...")
    mapped_objects = []
    for decision in concepts_for_mapping:
        mapped_objects.append({
            "@context": "https://schema.org/", "@type": "Product",
            "name": decision.concept_name, "sameAs": f"#{decision.target_concept}",
            "mappingConfidence": decision.confidence,
        })
    logger.info(f"   ✅ Created {len(mapped_objects)} mapping objects.")


    # --- Step 8: Update Knowledge Graph ---
    if resume_from == 'graph':
        logger.info("\n🗃️ Step 8: Updating knowledge graph...")
        all_objects = new_schema_objects + mapped_objects
        if all_objects:
            builder = SchemaOrgGraphBuilder()
            builder.build_knowledge_graph_parallel(all_objects, max_workers=MAX_WORKERS)
            builder.close()
            logger.info("   ✅ Knowledge graph update complete.")
        else:
            logger.info("   ℹ️ No objects to write to the graph.")
    else:
        logger.info("\n🗃️ Step 8: Skipping graph update (not specified as resume point).")


    logger.info("\n🎉 CACHED PIPELINE RUN FINISHED! 🎉")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the cached and resumable Schema.org pipeline.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="start",
        choices=['start', 'chunks', 'concepts', 'decisions', 'schema', 'graph'],
        help="The step to resume the pipeline from. 'start' clears all caches."
    )
    args = parser.parse_args()

    # If resuming from start, clear all caches.
    if args.resume_from == 'start':
        clear_cache('chunks') # Clearing from the first step clears everything
        run_cached_pipeline('chunks')
    else:
        run_cached_pipeline(args.resume_from)