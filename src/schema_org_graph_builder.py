import logging
import re
from typing import List, Dict, Any
from neo4j import GraphDatabase, Driver, Transaction
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
# Import centralized configuration
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MAX_WORKERS, NEO4J_DB_NAME
logger = logging.getLogger(__name__)

class SchemaOrgGraphBuilder:
    """Builds a Neo4j knowledge graph from Schema.org JSON-LD data, supporting parallel writes."""
    
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            logger.info("Neo4j connection verified successfully for Graph Builder.")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}", exc_info=True)
            self.driver = None # Ensure driver is None if connection fails
            raise
            
        self.database = NEO4J_DB_NAME
        self.type_to_label = {"Product": "Product", "Organization": "Organization", "Thing": "Thing"}
        self.relationship_properties = {
            "isAccessoryOrSparePartFor", "isRelatedTo", "hasPart", "worksWith",
            "manufacturer", "brand"
        }

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection for Graph Builder has been closed.")

    def _write_single_object_tx(self, tx: Transaction, task: Dict):
        """
        Executes a single ontology learning task.
        1. Creates a new class and links it to its parent (taxonomic).
        2. Creates any discovered non-taxonomic relationships.
        """
        if task.get('action') != 'CREATE_CLASS':
            return

        child_name = task.get('name')
        parent_name = task.get('parent_class')
        status = task.get('status')

        if not child_name:
            logger.warning(f"Skipping task with no name: {task}")
            return

        # 1. Create the new class node.
        class_query = """
            MERGE (c:OntologyClass {name: $name})
            ON CREATE SET
                c.source = 'learned_from_dataset',
                c.uri = 'https://example.org/ontology#' + $name
        """
        tx.run(class_query, name=child_name)

        # Now, apply the :NeedsReview label in a separate, conditional step
        # ONLY if the status is 'review'.
        if status == 'review':
            review_query = """
                MATCH (c:OntologyClass {name: $name})
                SET c:NeedsReview
            """
            tx.run(review_query, name=child_name)

        # 2. Create the hierarchical (taxonomic) link.
        if parent_name:
        # --- CORRECTED AND MORE ROBUST LINKING QUERY ---
            link_query = """
            // Anchor on the more specific child node that was just created/merged
            MATCH (child:OntologyClass {name: $child_name})
            
            // Anchor on the more general parent node
            MATCH (parent:OntologyClass {name: $parent_name})
            
            // A child is a SUBCLASS_OF a parent. The arrow points from specific to general.
            MERGE (child)-[:SUBCLASS_OF]->(parent)
            """
            tx.run(link_query, child_name=child_name, parent_name=parent_name)
            logger.info(f"LEARNED TAXONOMY: [{child_name}] -> IS_A -> [{parent_name}]")

        
        # 3. Create the non-taxonomic relationships.
        if relationships := task.get("non_taxonomic_relations"):
            logger.info(f"  Processing {len(relationships)} non-taxonomic relations for [{child_name}]...")
            for rel in relationships:
                target_name = rel.get("target")
                relation_type = rel.get("relation")
                
                if not target_name or not relation_type:
                    logger.warning(f"    - Skipping invalid relation for [{child_name}]: {rel}")
                    continue

                # Sanitize the relationship type to be a valid Cypher type
                sanitized_relation_type = re.sub(r'[^a-zA-Z0-9_]', '_', relation_type).upper()
                if not sanitized_relation_type:
                    logger.warning(f"    - Skipping relation with invalid type '{relation_type}'")
                    continue
                
                # Ensure the target node exists as a class
                tx.run("MERGE (c:OntologyClass {name: $name})", name=target_name)

                # Create the non-taxonomic relationship
                rel_query = f"""
                MATCH (source:OntologyClass {{name: $source_name}})
                MATCH (target:OntologyClass {{name: $target_name}})
                MERGE (source)-[r:{sanitized_relation_type}]->(target)
                """
                tx.run(rel_query, source_name=child_name, target_name=target_name)
                logger.info(f"    + LEARNED RELATION: [{child_name}] -[:{sanitized_relation_type}]-> [{target_name}]")


    def _write_object_session(self, schema_object: Dict):
        """Manages a session from the connection pool to write a single object."""
        if not self.driver:
            raise ConnectionError("Neo4j driver is not available.")
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._write_single_object_tx, schema_object)
        
    def build_knowledge_graph_parallel(self, schema_objects: List[Dict], max_workers: int = MAX_WORKERS) -> Dict[str, Any]:
        """Builds the knowledge graph by writing objects to Neo4j in parallel."""
        if not self.driver:
            logger.error("Cannot build graph: Neo4j driver not initialized.")
            return {}
            
        logger.info(f"Building/updating graph with {len(schema_objects)} objects using up to {max_workers} workers.")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._write_object_session, obj): obj for obj in schema_objects}
            
            for future in tqdm(as_completed(futures), total=len(schema_objects), desc="Writing to Knowledge Graph"):
                try:
                    future.result()  # Raise exceptions if any occurred during the write
                except Exception as e:
                    logger.error(f"Failed to write an object to the graph: {e}", exc_info=True)
        
        # After all writes are done, infer relationships in a final, single transaction
        logger.info("Inferring relationships (e.g., SAME_CATEGORY)...")
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._create_inferred_relationships)
        
        logger.info("Graph update and relationship inference complete.")
        return self.get_graph_statistics()

    def _create_inferred_relationships(self, tx: Transaction):
        """
        Creates relationships based on shared properties in batches to avoid memory issues.
        This now uses the APOC library, which is a standard requirement for production Neo4j.
        """
        logger.info("   (Batching) Inferring SAME_CATEGORY relationships...")

        # This query finds all products, then for each product (p1), it finds all other products (p2)
        # in the same category and creates the relationship. APOC handles running this in small batches.
        batching_query = """
        CALL apoc.periodic.iterate(
            'MATCH (p:Product) WHERE p.category IS NOT NULL RETURN p',
            'MATCH (p2:Product) WHERE p2.category = p.category AND elementId(p) < elementId(p2) MERGE (p)-[:SAME_CATEGORY]->(p2)',
            {batchSize: 1000, parallel: false}
        )
        """
        
        try:
            # We don't need to consume the results, just run it.
            # Use a longer timeout as this can be a long-running query on large datasets.
            tx.run(batching_query)
            logger.info("   ✅ Successfully submitted SAME_CATEGORY inference job.")
        except Exception as e:
            # This can happen if APOC is not installed.
            logger.error(
                "   ⚠️  Failed to run batched relationship inference. "
                "Please ensure the APOC plugin is installed in your Neo4j database. "
                f"Error: {e}"
            )

    def _sanitize_property_name(self, name: str) -> str:
        """Sanitizes property names for Neo4j compatibility."""
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Gets basic statistics about the graph using modern Cypher syntax."""
        if not self.driver: return {}
        try:
            with self.driver.session(database=self.database) as session:
                # Using COUNT{} to count relationships instead of the deprecated size()
                stats = session.run("""
                    MATCH (n)
                    WITH count(n) AS nodes
                    CALL {
                        MATCH ()-[r]->()
                        RETURN count(r) AS rels
                    }
                    RETURN nodes, rels
                """).single()
                
                return {"totals": {"nodes": stats['nodes'], "relationships": stats['rels']}} if stats else {"totals": {"nodes": 0, "relationships": 0}}
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}", exc_info=True)
            return {}