# src/evaluation/final_ontology_analyzer.py

import logging
from pathlib import Path
import argparse
import pandas as pd
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Import Neo4j connection details from your project config
from ..config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class OntologyAnalyzer:
    """Analyzes and compares final, curated ontologies directly from multiple Neo4j databases."""

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connection verified for final ontology analysis.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def get_graph_statistics(self, db_name: str) -> dict:
        """Gets basic statistics about a graph."""
        with self.driver.session(database=db_name) as session:
            stats = session.run("""
                MATCH (c:OntologyClass)
                WITH count(c) AS nodes
                MATCH ()-[r]->()
                RETURN nodes, count(r) AS relationships
            """).single()
            
            if not stats or stats['nodes'] == 0:
                return {"nodes": 0, "relationships": 0, "graph_density": 0.0}

            # Calculate graph density
            nodes = stats['nodes']
            relationships = stats['relationships']
            # Density for a directed graph is E / (V * (V - 1))
            density = relationships / (nodes * (nodes - 1)) if nodes > 1 else 0.0
            
            return {
                "total_concepts": nodes, 
                "total_relationships": relationships,
                "graph_density": round(density, 6)
            }

    def analyze_hierarchy(self, db_name: str) -> dict:
        """Analyzes the depth of the :SUBCLASS_OF hierarchy."""
        with self.driver.session(database=db_name) as session:
            result = session.run("""
                MATCH path = (leaf:OntologyClass)-[:SUBCLASS_OF*]->(root:OntologyClass {name: 'Thing'})
                WHERE NOT (()-[:SUBCLASS_OF]->(leaf))
                RETURN length(path) AS depth
                ORDER BY depth DESC LIMIT 1
            """).single()
            leaf_result = session.run("""
                MATCH (leaf:OntologyClass)
                WHERE NOT (()-[:SUBCLASS_OF]->(leaf))
                RETURN count(leaf) as leaf_node_count
            """).single()

            return {
                "max_hierarchy_depth": result['depth'] if result else 0,
                "leaf_node_count": leaf_result['leaf_node_count'] if leaf_result else 0
            }

    def analyze_connectivity(self, db_name: str) -> dict:
        """Analyzes the degree distribution (connectivity) of concepts."""
        with self.driver.session(database=db_name) as session:
            result = session.run("""
                MATCH (c:OntologyClass)
                WITH COUNT{(c)--()} AS degree
                RETURN avg(degree) AS avg_degree, max(degree) AS max_degree
            """).single()
            return {
                "avg_connections_per_node": round(result['avg_degree'], 2) if result else 0,
                "max_connections_for_hub": result['max_degree'] if result else 0
            }

    def analyze_relationship_types(self, db_name: str) -> dict:
        """Counts the number of relationships for each type."""
        with self.driver.session(database=db_name) as session:
            results = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS count
            """)
            # Return as a dictionary for easy merging, focusing on key types
            rel_counts = {r['rel_type']: r['count'] for r in results}
            return {
                "taxonomic_relations": rel_counts.get("SUBCLASS_OF", 0),
                "non_taxonomic_relations": sum(v for k, v in rel_counts.items() if k != "SUBCLASS_OF")
            }

    def check_integrity(self, db_name: str) -> dict:
        """Performs integrity checks, such as finding orphan nodes."""
        with self.driver.session(database=db_name) as session:
            result = session.run("""
                MATCH (c:OntologyClass)
                WHERE c.source = 'learned_from_dataset'
                AND NOT (c)-[:SUBCLASS_OF*]->(:OntologyClass {name: 'Thing'})
                RETURN count(c) as orphan_count
            """).single()
            return {"orphan_nodes": result['orphan_count'] if result else 0}

    def run_all_analyses(self, db_name: str) -> dict:
        """Runs all analysis queries for a single database and aggregates the results."""
        if not self.driver: return {}
        logger.info(f"--- Analyzing Database: {db_name} ---")
        
        full_report = {"database": db_name}
        full_report.update(self.get_graph_statistics(db_name))
        full_report.update(self.analyze_hierarchy(db_name))
        full_report.update(self.analyze_connectivity(db_name))
        full_report.update(self.analyze_relationship_types(db_name))
        full_report.update(self.check_integrity(db_name))
        
        return full_report

def generate_reports(results_df: pd.DataFrame, output_dir: Path):
    """Generates and saves both CSV and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "final_graph_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Comparative CSV report saved to: {csv_path}")

    # Save Markdown
    md_path = output_dir / "final_graph_comparison_report.md"
    md_content = f"""
# Final Ontology Graph Comparison Report

This report provides a side-by-side structural analysis of the knowledge graphs
generated by different models, queried directly from their respective Neo4j databases.

{results_df.to_markdown(index=False)}
"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    logger.info(f"✅ Comparative Markdown report saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare final ontologies in multiple Neo4j databases.")
    parser.add_argument(
        "databases", 
        nargs='+', 
        help="A list of Neo4j database names to analyze (e.g., gpt41 gpt35turbo datasheetontology)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"), help="Directory to save the reports.")
    args = parser.parse_args()

    analyzer = OntologyAnalyzer()
    all_results = []
    try:
        if analyzer.driver:
            for db_name in args.databases:
                report_data = analyzer.run_all_analyses(db_name)
                all_results.append(report_data)
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                # Define a logical column order for the report
                column_order = [
                    "database", "total_concepts", "total_relationships", "graph_density",
                    "max_hierarchy_depth", "leaf_node_count", "taxonomic_relations", "non_taxonomic_relations",
                    "avg_connections_per_node", "max_connections_for_hub", "orphan_nodes"
                ]
                results_df = results_df[column_order]
                
                print("\n--- Final Graph Comparison Summary ---")
                print(results_df.to_string(index=False))
                print("------------------------------------")
                
                generate_reports(results_df, args.output_dir)

    finally:
        analyzer.close()

if __name__ == "__main__":
    main()