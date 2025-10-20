# src/evaluation/multi_db_evaluator.py

import logging
import json
import argparse
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase
from tqdm import tqdm  # --- ADDITION: Import tqdm for progress bars ---
from concurrent.futures import ThreadPoolExecutor, as_completed # --- ADDITION ---

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Import necessary components from your project
from ..config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    MAX_WORKERS,
    OPENAI_API_KEY,
    MODEL_COSTS
)
from ..initialize_baseline import initialize_database
from ..data_loader import load_and_split_data
from ..idea_extractor import extract_ideas
from ..ontology_extension_manager import OntologyExtensionManager
from ..schema_org_graph_builder import SchemaOrgGraphBuilder
from ..data_models import PipelineConfig, ExtensionDecision

class MultiDBEvaluator:
    # ... (the __init__ and close methods are unchanged) ...
    def __init__(self, models_and_dbs: dict):
        self.models_and_dbs = models_and_dbs
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connection verified for Multi-DB Evaluator.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def _build_ontology_for_model(self, model_name: str, db_name: str):
        # ... (cost tracking setup and steps 1-3 are unchanged) ...
        logger.info(f"\n{'='*20} BUILDING ONTOLOGY for Model: {model_name} in DB: {db_name} {'='*20}")
        run_costs = {"concept_extraction_cost": 0.0, "decision_analysis_cost": 0.0, "total_cost": 0.0}
        model_pricing = MODEL_COSTS.get(model_name, MODEL_COSTS['default'])
        input_price = model_pricing['input_cost_per_1k_tokens']
        output_price = model_pricing['output_cost_per_1k_tokens']
        
        logger.info(f"Step 1: Wiping and initializing baseline for '{db_name}'...")
        initialize_database(db_name)

        logger.info("Step 2: Loading and chunking all documents...")
        chunks = load_and_split_data()
        if not chunks:
            logger.error("No documents found. Aborting build.")
            return run_costs

        logger.info(f"Step 3: Extracting concepts with '{model_name}'...")
        concepts, in_tokens, out_tokens = extract_ideas(chunks, model_name=model_name)
        concept_cost = ((in_tokens / 1000) * input_price) + ((out_tokens / 1000) * output_price)
        run_costs["concept_extraction_cost"] = concept_cost
        run_costs["total_cost"] += concept_cost
        logger.info(f"   - Concept Extraction Cost: ${concept_cost:.4f}")
        
        # --- MODIFICATION: Parallelize Step 4 ---
        logger.info("Step 4: Analyzing concepts and generating extension tasks (in parallel)...")
        # Pass db_name_override to ensure OEM reads from the correct DB
        extension_manager = OntologyExtensionManager(db_name_override=db_name)
        extension_manager.llm.model_name = model_name
        
        decisions = []
        total_decision_in_tokens = 0
        total_decision_out_tokens = 0

        # This helper function is needed to pass arguments to the parallel executor
        def analyze_concept_worker(concept_name):
            concept_dict = {'name': concept_name, 'description': ''}
            return extension_manager.analyze_new_concept(concept_dict)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a future for each concept
            future_to_concept = {executor.submit(analyze_concept_worker, name): name for name in concepts}
            
            # --- ADDITION: TQDM Progress Bar ---
            for future in tqdm(as_completed(future_to_concept), total=len(concepts), desc=f"Analyzing Concepts for {model_name}"):
                try:
                    decision, in_tok, out_tok = future.result()
                    if decision:
                        decisions.append(decision)
                    total_decision_in_tokens += in_tok
                    total_decision_out_tokens += out_tok
                except Exception as e:
                    concept_name = future_to_concept[future]
                    logger.error(f"Error analyzing concept '{concept_name}' in parallel: {e}", exc_info=True)
        # --- END OF MODIFICATION ---

        decision_cost = ((total_decision_in_tokens / 1000) * input_price) + ((total_decision_out_tokens / 1000) * output_price)
        run_costs["decision_analysis_cost"] = decision_cost
        run_costs["total_cost"] += decision_cost
        logger.info(f"   - Decision Analysis Cost: ${decision_cost:.4f}")
        logger.info(f"   - Total Build Cost for {model_name}: ${run_costs['total_cost']:.4f}")
        
        ontology_tasks = [
            {
                'name': d.concept_name, 'action': 'CREATE_CLASS',
                'parent_class': d.target_concept or 'ElectronicComponent',
                'non_taxonomic_relations': d.non_taxonomic_relations or []
            }
            for d in decisions if d.decision in [ExtensionDecision.EXTEND, ExtensionDecision.MAP_SIMILAR]
        ]

        logger.info(f"Step 5: Building knowledge graph in '{db_name}'...")
        graph_builder = SchemaOrgGraphBuilder(database=db_name)
        try:
            graph_builder.build_knowledge_graph_parallel(ontology_tasks, max_workers=MAX_WORKERS)
        finally:
            graph_builder.close()

        logger.info(f"--- Finished building ontology for {model_name} ---")
        return run_costs

    # ... (the rest of the script, _evaluate_database and main, is unchanged) ...
    def _evaluate_database(self, db_name: str, queries: list) -> dict:
        logger.info(f"--- Evaluating query performance on DB: {db_name} ---")
        if not self.driver: return {}
        summary = {"total_queries": len(queries), "successful": 0, "empty_result": 0, "errors": 0}
        with self.driver.session(database=db_name) as session:
            for query in queries:
                try:
                    result = session.run(query['cypher'])
                    if result.peek(): summary["successful"] += 1
                    else: summary["empty_result"] += 1
                except Exception as e:
                    logger.warning(f"Query '{query['name']}' failed on DB '{db_name}': {e}")
                    summary["errors"] += 1
        summary["success_rate"] = ((summary["successful"] + summary["empty_result"]) / summary["total_queries"]) * 100
        summary["resultful_rate"] = (summary["successful"] / summary["total_queries"]) * 100
        return summary


    def run_full_comparison(self, queries_path: Path, build_ontologies: bool):
        if not self.driver: return
        build_costs = {}
        if build_ontologies:
            for model, db in self.models_and_dbs.items():
                cost_data = self._build_ontology_for_model(model, db)
                build_costs[model] = cost_data
        else:
            logger.info("Skipping ontology building phase as requested.")
            for model in self.models_and_dbs.keys():
                build_costs[model] = {"total_cost": 0, "concept_extraction_cost": 0, "decision_analysis_cost": 0}

        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        all_results = []
        for model, db in self.models_and_dbs.items():
            db_stats = self._evaluate_database(db, queries)
            db_stats["Model"] = model
            db_stats["Database"] = db
            final_stats = {**db_stats, **build_costs.get(model, {})}
            all_results.append(final_stats)
        
        return pd.DataFrame(all_results)

def main():
    parser = argparse.ArgumentParser(description="Build and evaluate ontologies from different models in separate Neo4j databases.")
    parser.add_argument("--queries", type=Path, default=Path("data/evaluation_queries.json"), help="Path to the JSON file containing evaluation queries.")
    parser.add_argument("--output", type=Path, default=Path("visualizations/multi_db_query_evaluation.csv"), help="Path to save the final comparison CSV.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the ontology building step and only run the evaluation.")
    args = parser.parse_args()
    models_to_evaluate = {"gpt-3.5-turbo": "gpt35turbo",
                          "gpt-4.1-nano": "gpt41nano",
                          "gpt-4.1": "gpt41"}
    evaluator = MultiDBEvaluator(models_to_evaluate)
    try:
        results_df = evaluator.run_full_comparison(args.queries, build_ontologies=not args.skip_build)
        if not results_df.empty:
            results_df['cost_effectiveness'] = results_df['resultful_rate'] / (results_df['total_cost'] + 0.0001)
            column_order = ["Model", "Database", "resultful_rate", "cost_effectiveness", "total_cost", "concept_extraction_cost", "decision_analysis_cost", "successful", "empty_result", "errors"]
            final_columns = [col for col in column_order if col in results_df.columns]
            results_df = results_df[final_columns].sort_values(by="cost_effectiveness", ascending=False)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(args.output, index=False)
            logger.info(f"✅ Multi-DB evaluation summary saved to: {args.output}")
            print("\n" + "="*80)
            print("      MULTI-DATABASE COST & FUNCTIONAL EVALUATION SUMMARY")
            print("="*80)
            print(results_df.to_string(index=False))
            print("="*80)
    finally:
        evaluator.close()

if __name__ == "__main__":
    main()