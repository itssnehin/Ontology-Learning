import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from typing import List, Dict, Optional, Any
import json
from utils import setup_logging

class SchemaOrgGraphBuilder:
    """Build Neo4j knowledge graph from Schema.org JSON-LD data."""
    
    def __init__(self, database: str = "neo4j"):
        setup_logging("../logs", "schema_org_graph_builder")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.database = database
        
        # Schema.org to Neo4j label mappings
        self.type_to_label = {
            "Product": "Product",
            "Organization": "Organization", 
            "Person": "Person",
            "Place": "Place",
            "Thing": "Thing"
        }
        
        # Properties to exclude from node properties (handled as relationships)
        self.relationship_properties = {
            "isAccessoryOrSparePartFor", "isConsumableFor", "isRelatedTo", 
            "isSimilarTo", "hasPart", "isPartOf", "worksWith", "requires",
            "manufacturer", "brand"  # These create relationships to Organization nodes
        }
    
    def build_knowledge_graph(self, schema_objects: List[Dict]) -> None:
        """
        Build Neo4j knowledge graph from Schema.org objects.
        
        Args:
            schema_objects: List of Schema.org JSON-LD objects
        """
        print(f"Building Neo4j knowledge graph from {len(schema_objects)} Schema.org objects...")
        
        try:
            with self.driver.session(database=self.database) as session:
                # Clear existing Schema.org data (optional)
                # session.run("MATCH (n:Product)-[r]-() DELETE r, n")
                
                # Create nodes for each Schema.org object
                for obj in schema_objects:
                    self._create_schema_org_node(session, obj)
                
                # Create relationships between objects
                for obj in schema_objects:
                    self._create_schema_org_relationships(session, obj, schema_objects)
                
                # Create additional inferred relationships
                self._create_inferred_relationships(session)
                
                print("Knowledge graph construction completed successfully!")
                
        except Exception as e:
            print(f"Error building knowledge graph: {e}")
            raise
        finally:
            self.driver.close()
    
    def _create_schema_org_node(self, session, obj: Dict) -> None:
        """Create a Neo4j node from a Schema.org object."""
        # Determine node label
        schema_type = obj.get("@type", "Thing")
        label = self.type_to_label.get(schema_type, "Product")
        
        # Extract node properties (exclude relationship properties)
        node_props = {}
        for key, value in obj.items():
            if (not key.startswith("@") and 
                key not in self.relationship_properties and
                not key.startswith("is") and
                key != "additionalType"):
                
                # Handle different value types
                if isinstance(value, (str, int, float, bool)):
                    node_props[self._sanitize_property_name(key)] = value
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings for Neo4j
                    node_props[self._sanitize_property_name(key)] = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    # Skip complex objects for now
                    continue
        
        # Add metadata
        node_props["schemaType"] = schema_type
        node_props["sourceFormat"] = "Schema.org"
        
        if "additionalType" in obj:
            node_props["additionalType"] = obj["additionalType"]
        
        # Create node with unique constraint on name
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n += $props
        RETURN n
        """
        
        try:
            session.run(query, name=obj.get("name", "Unknown"), props=node_props)
            print(f"Created {label} node: {obj.get('name', 'Unknown')}")
        except Exception as e:
            print(f"Error creating node for {obj.get('name')}: {e}")
    
    def _create_schema_org_relationships(self, session, obj: Dict, all_objects: List[Dict]) -> None:
        """Create relationships based on Schema.org properties."""
        source_name = obj.get("name", "Unknown")
        
        # Handle manufacturer/brand relationships
        if "manufacturer" in obj:
            self._create_organization_relationship(session, source_name, obj["manufacturer"], "MANUFACTURED_BY")
        
        if "brand" in obj:
            self._create_organization_relationship(session, source_name, obj["brand"], "BRANDED_BY")
        
        # Handle product relationships
        relationships_map = {
            "isAccessoryOrSparePartFor": "ACCESSORY_FOR",
            "isConsumableFor": "CONSUMABLE_FOR", 
            "isRelatedTo": "RELATED_TO",
            "isSimilarTo": "SIMILAR_TO",
            "hasPart": "HAS_PART",
            "isPartOf": "PART_OF",
            "worksWith": "WORKS_WITH",
            "requires": "REQUIRES"
        }
        
        for prop, rel_type in relationships_map.items():
            if prop in obj:
                targets = obj[prop]
                if isinstance(targets, str):
                    targets = [targets]
                elif not isinstance(targets, list):
                    continue
                
                for target in targets:
                    self._create_product_relationship(session, source_name, target, rel_type)
    
    def _create_organization_relationship(self, session, product_name: str, org_name: str, rel_type: str) -> None:
        """Create relationship to an organization (manufacturer/brand)."""
        query = f"""
        MATCH (p:Product {{name: $product_name}})
        MERGE (o:Organization {{name: $org_name}})
        MERGE (p)-[r:{rel_type}]->(o)
        RETURN r
        """
        
        try:
            session.run(query, product_name=product_name, org_name=org_name)
            print(f"Created {rel_type} relationship: {product_name} -> {org_name}")
        except Exception as e:
            print(f"Error creating organization relationship: {e}")
    
    def _create_product_relationship(self, session, source_name: str, target_name: str, rel_type: str) -> None:
        """Create relationship between products."""
        query = f"""
        MATCH (s:Product {{name: $source_name}})
        MATCH (t:Product {{name: $target_name}})
        MERGE (s)-[r:{rel_type}]->(t)
        RETURN r
        """
        
        try:
            result = session.run(query, source_name=source_name, target_name=target_name)
            if result.single():
                print(f"Created {rel_type} relationship: {source_name} -> {target_name}")
        except Exception as e:
            print(f"Error creating product relationship: {e}")
    
    # Replace the _create_inferred_relationships method in schema_org_graph_builder.py
    # Around line 120-160
    def _create_inferred_relationships(self, session) -> None:
        """Create additional inferred relationships based on data patterns."""
        
        # Simplified category grouping - FIXED CYPHER SYNTAX
        query_category_groups = """
        MATCH (p1:Product), (p2:Product)
        WHERE p1.category = p2.category 
        AND p1.category IS NOT NULL
        AND p1 <> p2
        AND id(p1) < id(p2)
        MERGE (p1)-[:SAME_CATEGORY]->(p2)
        RETURN count(*) as relationships_created
        """
        
        # Simplified manufacturer grouping
        query_manufacturer_groups = """
        MATCH (p1:Product)-[:MANUFACTURED_BY]->(o:Organization)<-[:MANUFACTURED_BY]-(p2:Product)
        WHERE p1 <> p2 AND id(p1) < id(p2)
        MERGE (p1)-[:SAME_MANUFACTURER]->(p2)
        RETURN count(*) as relationships_created
        """
        
        try:
            # Create category relationships
            result1 = session.run(query_category_groups)
            record1 = result1.single()
            cat_count = record1["relationships_created"] if record1 else 0
            print(f"Created {cat_count} category-based relationships")
            
            # Create manufacturer relationships  
            result2 = session.run(query_manufacturer_groups)
            record2 = result2.single()
            mfg_count = record2["relationships_created"] if record2 else 0
            print(f"Created {mfg_count} manufacturer-based relationships")
            
        except Exception as e:
            print(f"Error creating inferred relationships: {e}")
            # Don't raise - continue pipeline execution


    
    def _sanitize_property_name(self, name: str) -> str:
        """Sanitize property names for Neo4j compatibility."""
        # Replace colons and special characters
        sanitized = name.replace(":", "_").replace("-", "_").replace("@", "")
        return sanitized
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the created knowledge graph."""
        stats = {}
        
        try:
            with self.driver.session(database=self.database) as session:
                # Node counts
                node_counts = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(n) as count
                    ORDER BY count DESC
                """)
                
                stats["nodes"] = {}
                for record in node_counts:
                    label = record["labels"][0] if record["labels"] else "Unknown"
                    stats["nodes"][label] = record["count"]
                
                # Relationship counts
                rel_counts = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as relationship_type, count(r) as count
                    ORDER BY count DESC
                """)
                
                stats["relationships"] = {}
                for record in rel_counts:
                    stats["relationships"][record["relationship_type"]] = record["count"]
                
                # Total counts
                total_nodes = session.run("MATCH (n) RETURN count(n) as total").single()["total"]
                total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as total").single()["total"]
                
                stats["totals"] = {
                    "nodes": total_nodes,
                    "relationships": total_rels
                }
                
        except Exception as e:
            print(f"Error getting graph statistics: {e}")
            return {}
        
        return stats

def build_schema_org_knowledge_graph(schema_objects: List[Dict]) -> Dict[str, Any]:
    """
    Main function to build Neo4j knowledge graph from Schema.org data.
    
    Args:
        schema_objects: List of Schema.org JSON-LD objects
        
    Returns:
        Dictionary containing graph statistics
    """
    builder = SchemaOrgGraphBuilder()
    builder.build_knowledge_graph(schema_objects)
    return builder.get_graph_statistics()

if __name__ == "__main__":
    # Test with sample Schema.org data
    sample_data = [
        {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": "WiFi 6E FPC Antenna",
            "description": "Flexible printed circuit antenna for WiFi 6E applications",
            "category": "Antenna",
            "manufacturer": "ACME Electronics",
            "additionalType": "http://www.productontology.org/id/Antenna_(radio)",
            "elec:frequency": "2.4-6 GHz",
            "elec:impedance": "50 ohms"
        },
        {
            "@context": "https://schema.org/",
            "@type": "Product", 
            "name": "PCB Connector",
            "description": "PCB mounting connector for antennas",
            "category": "Connector",
            "manufacturer": "ACME Electronics",
            "isAccessoryOrSparePartFor": "WiFi 6E FPC Antenna"
        }
    ]
    
    stats = build_schema_org_knowledge_graph(sample_data)
    print("Graph Statistics:")
    print(json.dumps(stats, indent=2))