# src/evaluation/top_k_evaluation.py

import logging
from pathlib import Path
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Import Neo4j connection details
from ..config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class TopKEvaluator:
    # ... (__init__, close, _normalize_string, load_gold_standard_concepts are unchanged)
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connection verified for Top-K evaluation.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def _normalize_string(self, s: str) -> str:
        return s.lower().strip() if isinstance(s, str) else str(s)

    def load_gold_standard_concepts(self, gold_path: Path) -> set:
        with open(gold_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {self._normalize_string(c) for c in data.get('concepts', [])}

    def get_top_k_concepts_by_degree(self, db_name: str, k: int) -> set:
        # ... (this function is unchanged)
        if not self.driver: return set()
        query = "MATCH (n:OntologyClass) WITH n, COUNT{(n)--()} AS degree ORDER BY degree DESC LIMIT $k RETURN n.name AS conceptName"
        with self.driver.session(database=db_name) as session:
            results = session.run(query, k=k)
            return {self._normalize_string(record["conceptName"]) for record in results}

    def run_evaluation(self, db_names: list, gold_concepts: set, k_values: list):
        """Runs the Top-K evaluation for a list of databases across multiple K values."""
        all_results = []
        total_gold_concepts = len(gold_concepts)
        
        for k in sorted(k_values):
            logger.info(f"--- Running evaluation for K = {k} ---")
            for db_name in db_names:
                top_k_from_db = self.get_top_k_concepts_by_degree(db_name, k)
                
                if not top_k_from_db:
                    logger.warning(f"Could not retrieve any concepts from database '{db_name}'.")
                    hit_count = 0
                else:
                    hits = gold_concepts.intersection(top_k_from_db)
                    hit_count = len(hits)
                
                coverage_at_k = hit_count / total_gold_concepts if total_gold_concepts > 0 else 0.0
                
                all_results.append({
                    "Database": db_name,
                    "K": k,
                    "Gold_Concepts_in_Top_K": hit_count,
                    "Coverage_at_K": coverage_at_k
                })
        
        return pd.DataFrame(all_results)


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generates and saves a line plot comparing Coverage at K for each database."""
    logger.info("📊 Generating Top-K comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.lineplot(data=df, x='K', y='Coverage_at_K', hue='Database', marker='o', ax=ax)
    
    ax.set_title('Coverage of Gold Standard Concepts in Top-K Most Central Nodes', fontsize=16, fontweight='bold')
    ax.set_xlabel('K (Number of Top Concepts Considered)', fontsize=12)
    ax.set_ylabel('Coverage Rate (%)', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}')) # Format y-axis as percentage
    ax.set_xticks(df['K'].unique()) # Ensure all K values are shown as ticks
    
    plt.legend(title='Model-Generated Graph')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    
    plot_path = output_dir / "top_k_coverage_plot.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✅ Top-K coverage plot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Top-K concept prominence against a gold standard across multiple K values.")
    parser.add_argument("databases", nargs='+', help="List of Neo4j database names to evaluate.")
    parser.add_argument("--gold", type=Path, required=True, help="Path to the document-specific gold standard JSON file.")
    parser.add_argument("-k", type=int, nargs='+', default=[25, 50, 100, 200], help="A list of K values to test.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"), help="Directory to save the reports.")
    
    args = parser.parse_args()

    evaluator = TopKEvaluator()
    try:
        if evaluator.driver:
            gold_concepts = evaluator.load_gold_standard_concepts(args.gold)
            if not gold_concepts:
                logger.error("No concepts found in the gold standard file. Aborting.")
                return

            results_df = evaluator.run_evaluation(args.databases, gold_concepts, args.k)
            
            if not results_df.empty:
                # Pivot the table for a more readable summary
                pivot_df = results_df.pivot(index='K', columns='Database', values='Coverage_at_K')
                # Format as percentage strings for display
                for col in pivot_df.columns:
                    pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.2%}")

                # Save the raw data
                csv_path = args.output_dir / "top_k_evaluation_summary.csv"
                results_df.to_csv(csv_path, index=False)
                logger.info(f"✅ Top-K evaluation summary saved to: {csv_path}")
                
                print("\n--- Top-K Gold Standard Coverage Summary ---")
                print(pivot_df)
                print("------------------------------------------")

                # Generate the plot
                plot_results(results_df, args.output_dir)
    finally:
        evaluator.close()

if __name__ == "__main__":
    main()