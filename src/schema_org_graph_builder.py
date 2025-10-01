import logging
import re
from typing import List, Dict, Any
from neo4j import GraphDatabase, Driver, Transaction
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
# Import centralized configuration
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MAX_WORKERS

logger = logging.getLogger(__name__)

class SchemaOrgGraphBuilder:
    """Builds a Neo4j knowledge graph from Schema.org JSON-LD data, supporting parallel writes."""
    
    def __init__(self, database: str = "neo4j"):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            logger.info("Neo4j connection verified successfully for Graph Builder.")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}", exc_info=True)
            self.driver = None # Ensure driver is None if connection fails
            raise
            
        self.database = database
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

    def _write_single_object_tx(self, tx: Transaction, schema_object: Dict):
        """Transaction function to write a single Schema.org object (node and its relations)."""
        if not isinstance(schema_object, dict) or 'name' not in schema_object:
            logger.warning(f"Skipping invalid schema object: {schema_object}")
            return
        
        # Create the main node
        self._create_schema_org_node(tx, schema_object)
        
        # Ensure any referenced organization nodes exist
        for org_prop in ["manufacturer", "brand"]:
            if org_name := schema_object.get(org_prop):
                tx.run("MERGE (o:Organization {name: $org_name})", org_name=org_name)
        
        # Create relationships from the object
        self._create_schema_org_relationships(tx, schema_object)

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

    def _create_schema_org_node(self, tx: Transaction, obj: Dict):
        """Creates a node in Neo4j, flattening any nested dictionary properties."""
        schema_type = obj.get("@type", "Thing")
        label = self.type_to_label.get(schema_type, "Product")
        
        node_props = {}
        for key, value in obj.items():
            # Skip metadata and relationship properties
            if key.startswith('@') or key in self.relationship_properties:
                continue

            prop_name = self._sanitize_property_name(key)
            
            # FLATTENING LOGIC: If the value is a dictionary, convert it to a string.
            if isinstance(value, dict):
                # Example: {"length": "10mm", "width": "5mm"} -> "length: 10mm, width: 5mm"
                node_props[prop_name] = json.dumps(value) 
            elif isinstance(value, list):
                # Ensure all items in a list are primitive types
                node_props[prop_name] = [str(item) for item in value]
            elif value is not None:
                # Handle primitive types
                node_props[prop_name] = value

        if 'name' in obj:
            query = f"MERGE (n:{label} {{name: $name}}) SET n += $props"
            tx.run(query, name=obj.get("name"), props=node_props)
        else:
            logger.warning(f"Skipping node creation for object with no name: {obj}")


    def _create_schema_org_relationships(self, tx: Transaction, obj: Dict):
        """Creates relationships for a given object."""
        source_name = obj.get("name")
        if not source_name: return

        if org_name := obj.get("manufacturer"):
            tx.run("MATCH (p:Product {name: $p_name}) MATCH (o:Organization {name: $o_name}) MERGE (p)-[:MANUFACTURED_BY]->(o)", p_name=source_name, o_name=org_name)
        if org_name := obj.get("brand"):
            tx.run("MATCH (p:Product {name: $p_name}) MATCH (o:Organization {name: $o_name}) MERGE (p)-[:BRANDED_BY]->(o)", p_name=source_name, o_name=org_name)
        
        # ... Add other relationship types here if needed ...

    def _create_inferred_relationships(self, tx: Transaction):
        """Creates relationships based on shared properties using the modern elementId()."""
        # Using elementId() instead of the deprecated id()
        tx.run("""
            MATCH (p1:Product), (p2:Product)
            WHERE p1.category = p2.category AND p1.category IS NOT NULL AND elementId(p1) < elementId(p2)
            MERGE (p1)-[:SAME_CATEGORY]->(p2)
        """)
        logger.info("Inferred SAME_CATEGORY relationships.")


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