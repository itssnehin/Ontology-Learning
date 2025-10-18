import logging
import pickle
from pathlib import Path
from typing import Optional, Callable, List
from datetime import datetime, date, time

# We only need to import the ONE true pipeline and its config
from src.integrated_schema_pipeline import IntegratedSchemaOrgPipeline, PipelineConfig
from src.config import CACHE_DIR, MAX_WORKERS, LLM_MODEL
import json 

logger = logging.getLogger(__name__)

# CACHE_PATHS remains the same
CACHE_PATHS = {
    "chunks": CACHE_DIR / "chunks.pkl",
    "concepts": CACHE_DIR / "concepts.pkl",
    "decisions": CACHE_DIR / "extension_decisions.pkl",
}

def clear_cache(start_step: str):
    """Clears cache files from the specified step onwards."""
    steps = list(CACHE_PATHS.keys())
    if start_step in steps:
        start_index = steps.index(start_step) + 1
        for i in range(start_index, len(steps)):
            step_to_clear = steps[i]
            if CACHE_PATHS[step_to_clear].exists():
                logger.warning(f"Clearing downstream cache for step: '{step_to_clear}'")
                CACHE_PATHS[step_to_clear].unlink()

def run_cached_pipeline(resume_from: str = 'start', llm_model: str = LLM_MODEL, max_workers: int = MAX_WORKERS, progress_callback: Optional[Callable] = None, selected_files: Optional[List[str]] = None):
    """
    Orchestrates a resumable ontology learning pipeline by calling steps from the main
    IntegratedSchemaOrgPipeline and caching the results.
    """
    logger.info("🚀 STARTING CACHED ONTOLOGY LEARNING RUN")
    logger.info(f"Resuming from step: '{resume_from}'")
    
    if resume_from == 'start':
        clear_cache('chunks') # Special case to clear everything
        resume_from = 'chunks' # Start from the first actual step

    # Instantiate the one true pipeline
    config = PipelineConfig()
    pipeline = IntegratedSchemaOrgPipeline(config)

    # --- Step 1: Chunks ---
    if resume_from == 'chunks' or selected_files:
        if selected_files:
            logger.info("User selected specific files. Re-loading documents and clearing downstream cache.")
            clear_cache('chunks') # Clear concepts and decisions
        
        # Pass the selected files list to the loading step
        chunks = pipeline._step_1_load_documents(selected_files=selected_files)
        with open(CACHE_PATHS["chunks"], "wb") as f: pickle.dump(chunks, f)
    else:
        with open(CACHE_PATHS["chunks"], "rb") as f: chunks = pickle.load(f)
    logger.info(f"   ✅ Loaded {len(chunks)} chunks.")

    # --- Step 2: Concepts ---
    if progress_callback: progress_callback("Extracting Concepts", 25)
    if resume_from in ['chunks', 'concepts']:
        concepts, _, _ = pipeline._step_2_extract_concepts(chunks, llm_model, max_workers)
        with open(CACHE_PATHS["concepts"], "wb") as f: pickle.dump(concepts, f)
    else:
        with open(CACHE_PATHS["concepts"], "rb") as f: concepts = pickle.load(f)
    logger.info(f"   ✅ Have {len(concepts)} unique concepts.")

    # --- Step 3-5: Decisions ---
    if progress_callback: progress_callback("Analyzing Decisions", 50)
    if resume_from in ['chunks', 'concepts', 'decisions']:
        pipeline._step_3_and_4_load_ontology_and_embed()
        decisions, cost_of_decisions = pipeline._step_5_analyze_concepts_parallel(concepts, max_workers, llm_model)
        
        with open(CACHE_PATHS["decisions"], "wb") as f: pickle.dump(decisions, f)
    else:
        with open(CACHE_PATHS["decisions"], "rb") as f: decisions = pickle.load(f)
    logger.info(f"   ✅ Have {len(decisions)} extension decisions.")


    # --- Final Steps (Always run after loading/generating decisions) ---
    
    # Step 6: Routing
    if progress_callback: progress_callback("Routing Decisions", 85)
    ontology_extension_tasks, _ = pipeline._route_concepts_based_on_decisions(decisions)

    # Step 7: Graph Building (Ontology Learning)
    if progress_callback: progress_callback("Extending Ontology", 90)
    pipeline._step_8_update_knowledge_graph_parallel(ontology_extension_tasks, [], max_workers)

    # Step 8: Save Reports
    logger.info("\n💾 Saving learned ontology tasks from cached run...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/integrated_output")
    tasks_file = output_dir / f"cached_learned_tasks_{timestamp}.json"
    with open(tasks_file, 'w') as f: json.dump(ontology_extension_tasks, f, indent=2)
    logger.info(f"   ✅ Saved tasks to {tasks_file.name}")

    logger.info("\n🎉 CACHED PIPELINE RUN FINISHED! 🎉")