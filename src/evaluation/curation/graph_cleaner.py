# src/curation/graph_cleaner.py

import logging
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DB_NAME

# Configure basic logging if run as a script
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

class GraphCleaner:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connection verified for Graph Cleaner.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def prune_low_degree_nodes(self, degree_threshold: int = 1):
        """
        Removes 'island' or sparsely connected nodes from the graph.
        These are often extraction errors or highly irrelevant concepts.
        A threshold of 1 removes nodes with only one or zero connections.
        """
        if not self.driver: return
        
        query = f"""
        MATCH (c:OntologyClass)
        WHERE c.source = 'learned_from_dataset' // Only prune learned nodes
        AND size((c)--()) <= {degree_threshold}
        WITH c, c.name AS name
        DETACH DELETE c
        RETURN name
        """
        
        with self.driver.session(database=NEO4J_DB_NAME) as session:
            result = session.run(query)
            deleted_nodes = [record["name"] for record in result]
            
            if deleted_nodes:
                logger.info(f"Pruned {len(deleted_nodes)} low-degree nodes (<= {degree_threshold} connections).")
                logger.debug(f"Deleted nodes: {deleted_nodes}")
            else:
                logger.info("No low-degree nodes found to prune.")
    
    def run_all_cleaning_tasks(self):
        """Runs a sequence of cleaning and curation tasks on the graph."""
        logger.info("--- Starting Graph Curation Tasks ---")
        self.prune_low_degree_nodes(degree_threshold=1)
        # You can add more cleaning methods here in the future
        logger.info("--- Curation Complete ---")

if __name__ == "__main__":
    cleaner = GraphCleaner()
    try:
        cleaner.run_all_cleaning_tasks()
    finally:
        cleaner.close()