from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from typing import List, Dict, Any

def build_graph_from_components(components: List[Dict[str, Any]], database: str = "neo4j"):
    """
    Build Neo4j graph from a list of component data.
    
    Args:
        components: List of component data dictionaries.
        database: Name of the Neo4j database to use.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    with driver.session(database=database) as session:
        for component_data in components:
            if not component_data:
                continue

            # Create component node
            session.run(
                """
                MERGE (c:Component {model_number: $model_number})
                SET c.component_type = $component_type, c.manufacturer = $manufacturer
                """,
                **component_data
            )

            # Create feature nodes and relationships
            for feature in component_data.get("features", []):
                session.run(
                    """
                    MATCH (c:Component {model_number: $model_number})
                    MERGE (f:Feature {name: $feature})
                    MERGE (c)-[:HAS_FEATURE]->(f)
                    """,
                    model_number=component_data["model_number"], feature=feature
                )

            # Create property nodes and relationships
            for prop in component_data.get("properties", []):
                session.run(
                    """
                    MATCH (c:Component {model_number: $model_number})
                    MERGE (p:Property {name: $name})
                    MERGE (c)-[:HAS_PROPERTY]->(p)
                    MERGE (p)-[:HAS_VALUE]->(v:Value {value: $value})
                    """,
                    model_number=component_data["model_number"], name=prop["name"], value=prop["value"]
                )

            # Create application nodes and relationships
            for app in component_data.get("applications", []):
                session.run(
                    """
                    MATCH (c:Component {model_number: $model_number})
                    MERGE (a:Application {name: $name})
                    MERGE (c)-[:USED_IN]->(a)
                    """,
                    model_number=component_data["model_number"], name=app["name"]
                )

            # Create relationship nodes and relationships
            for rel in component_data.get("relationships", []):
                session.run(
                    """
                    MATCH (c:Component {model_number: $model_number})
                    MERGE (t:Component {model_number: $target})
                    MERGE (c)-[:RELATES_TO {type: $type}]->(t)
                    """,
                    model_number=component_data["model_number"], target=rel["target"], type=rel["type"]
                )

    driver.close()
