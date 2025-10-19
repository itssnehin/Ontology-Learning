# src/curation/graph_cleaner.py

import logging
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DB_NAME

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

    # --- NEW METHOD TO REMOVE ORPHANS ---
    def remove_orphan_concepts(self):
        """
        Finds and removes all learned concepts that have no hierarchical path
        back to the 'ElectronicComponent' root class. This removes out-of-domain noise.
        """
        if not self.driver: return
        
        query = """
        MATCH (c:OntologyClass)
        WHERE c.source = 'learned_from_dataset'
        AND NOT (c)-[:SUBCLASS_OF*0..]->(:OntologyClass {name: 'ElectronicComponent'})
        WITH c, c.name AS name
        DETACH DELETE c
        RETURN name AS removed_orphan_concept
        """
        
        with self.driver.session(database=NEO4J_DB_NAME) as session:
            result = session.run(query)
            deleted_nodes = [record["removed_orphan_concept"] for record in result]
            
            if deleted_nodes:
                logger.info(f"Removed {len(deleted_nodes)} orphan (out-of-domain) concepts.")
                # Log a sample of deleted nodes for debugging
                logger.debug(f"Sample of removed orphans: {deleted_nodes[:10]}")
            else:
                logger.info("No orphan concepts found to remove.")


    def prune_low_degree_nodes(self, degree_threshold: int = 1):
        """
        Removes sparsely connected nodes from the graph, which are often noise.
        Runs after orphan removal to clean up what's left.
        """
        if not self.driver: return
        
        query = """
        MATCH (c:OntologyClass)
        WHERE c.source = 'learned_from_dataset' // Only prune learned nodes
        AND COUNT{(c)--()} <= $degree_threshold
        WITH c, c.name AS name
        DETACH DELETE c
        RETURN name
        """
        
        with self.driver.session(database=NEO4J_DB_NAME) as session:
            result = session.run(query, degree_threshold=degree_threshold)
            deleted_nodes = [record["name"] for record in result]
            
            if deleted_nodes:
                logger.info(f"Pruned {len(deleted_nodes)} low-degree nodes (<= {degree_threshold} connections).")
                logger.debug(f"Sample of pruned nodes: {deleted_nodes[:10]}")
            else:
                logger.info("No low-degree nodes found to prune.")
    
    def run_all_cleaning_tasks(self):
        """Runs a sequence of cleaning and curation tasks on the graph."""
        logger.info("--- Starting Graph Curation Tasks ---")
        
        # Step 1: Remove concepts that are not part of the core electronics hierarchy.
        logger.info("Step 1: Removing out-of-domain orphan concepts...")
        self.remove_orphan_concepts()
        
        # Step 2: Remove any remaining nodes that are poorly connected.
        logger.info("Step 2: Pruning low-degree nodes...")
        self.prune_low_degree_nodes(degree_threshold=1)
        
        logger.info("--- Curation Complete ---")


if __name__ == "__main__":
    cleaner = GraphCleaner()
    try:
        cleaner.run_all_cleaning_tasks()
    finally:
        cleaner.close()