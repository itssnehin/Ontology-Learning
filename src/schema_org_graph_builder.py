import logging
import re
from typing import List, Dict, Any
from neo4j import GraphDatabase, Driver, Transaction
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized configuration
from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MAX_WORKERS, NEO4J_DB_NAME

logger = logging.getLogger(__name__)

class SchemaOrgGraphBuilder:
    """Builds a Neo4j knowledge graph from ontology learning tasks, supporting parallel writes."""
    
    def __init__(self, database: str = NEO4J_DB_NAME):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            logger.info(f"Neo4j connection verified for Graph Builder on DB: '{database}'.")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}", exc_info=True)
            self.driver = None
            raise
            
        self.database = database

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info(f"Neo4j connection for Graph Builder on DB '{self.database}' has been closed.")

    def _write_single_object_tx(self, tx: Transaction, task: Dict):
        """
        Executes a single ontology learning task within a transaction.
        1. Creates a new class and links it to its parent (taxonomic).
        2. Creates any discovered non-taxonomic relationships.
        3. Applies a :NeedsReview label if the task is uncertain.
        """
        if task.get('action') != 'CREATE_CLASS':
            return

        child_name = task.get('name')
        parent_name = task.get('parent_class')
        status = task.get('status')

        if not child_name:
            logger.warning(f"Skipping task with no name: {task}")
            return

        # 1. Create/Merge the new class node and set its source.
        class_query = """
            MERGE (c:OntologyClass {name: $name})
            ON CREATE SET
                c.source = 'learned_from_dataset',
                c.uri = 'https://example.org/ontology#' + apoc.text.slug($name, '_')
        """
        tx.run(class_query, name=child_name)

        # 2. Conditionally apply the :NeedsReview label.
        if status == 'review':
            review_query = "MATCH (c:OntologyClass {name: $name}) SET c:NeedsReview"
            tx.run(review_query, name=child_name)

        # 3. Create the hierarchical (taxonomic) :SUBCLASS_OF link.
        if parent_name:
            link_query = """
            MATCH (child:OntologyClass {name: $child_name})
            MATCH (parent:OntologyClass {name: $parent_name})
            MERGE (child)-[:SUBCLASS_OF]->(parent)
            """
            tx.run(link_query, child_name=child_name, parent_name=parent_name)

        # 4. Create non-taxonomic relationships.
        if relationships := task.get("non_taxonomic_relations"):
            for rel in relationships:
                target_name = rel.get("target")
                relation_type = rel.get("relation")
                
                if not target_name or not relation_type:
                    continue

                sanitized_relation_type = re.sub(r'[^a-zA-Z0-9_]', '_', relation_type).upper()
                if not sanitized_relation_type:
                    continue
                
                # Ensure target node exists
                tx.run("MERGE (c:OntologyClass {name: $name})", name=target_name)

                # Use a dynamic relationship type in the query
                rel_query = f"""
                MATCH (source:OntologyClass {{name: $source_name}})
                MATCH (target:OntologyClass {{name: $target_name}})
                MERGE (source)-[r:{sanitized_relation_type}]->(target)
                """
                tx.run(rel_query, source_name=child_name, target_name=target_name)

    def _write_object_session(self, task: Dict):
        """Manages a session from the connection pool to write a single task."""
        if not self.driver:
            raise ConnectionError("Neo4j driver is not available.")
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._write_single_object_tx, task)
        
    def build_knowledge_graph_parallel(self, ontology_tasks: List[Dict], max_workers: int = MAX_WORKERS) -> Dict[str, Any]:
        """Builds the knowledge graph by writing tasks to Neo4j in parallel."""
        if not self.driver:
            logger.error("Cannot build graph: Neo4j driver not initialized.")
            return {}
            
        logger.info(f"Building/updating graph in DB '{self.database}' with {len(ontology_tasks)} tasks using up to {max_workers} workers.")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._write_object_session, task): task for task in ontology_tasks}
            
            for future in tqdm(as_completed(futures), total=len(ontology_tasks), desc="Writing to Knowledge Graph"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to write a task to the graph: {e}", exc_info=True)
        
        logger.info("Graph update complete.")
        return self.get_graph_statistics()

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Gets basic statistics about the graph."""
        if not self.driver: return {}
        try:
            with self.driver.session(database=self.database) as session:
                stats = session.run("""
                    MATCH (n:OntologyClass)
                    WITH count(n) AS nodes
                    CALL { MATCH ()-[r]->() RETURN count(r) AS rels }
                    RETURN nodes, rels
                """).single()
                
                return {"totals": {"nodes": stats['nodes'], "relationships": stats['rels']}} if stats else {"totals": {"nodes": 0, "relationships": 0}}
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}", exc_info=True)
            return {}