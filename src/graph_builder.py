import logging
import re
from typing import List, Tuple, Dict
import numpy as np
from neo4j import GraphDatabase, Driver, Transaction

from .config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from .utils import setup_logging

logger = logging.getLogger(__name__)

def _build_subgraph_tx(tx: Transaction, source: str, embedded_relations: list, embedded_themes: list):
    """A single transaction to build a document's subgraph."""
    pattern = re.compile(r"(.+?) -> (.+?) -> (.+?)")

    # Create theme nodes
    for theme, _ in embedded_themes:
        tx.run(
            """
            MERGE (n:Theme {name: $name, source: $source})
            SET n.type = 'Class'
            """,
            name=theme.strip(), source=source
        )
    
    # Create nodes and relationships from relations
    for relation, _ in embedded_relations:
        try:
            match = pattern.match(relation)
            if not match:
                logger.warning(f"Skipping invalid relation format: {relation}")
                continue
            
            concept1, rel, concept2 = [g.strip() for g in match.groups()]
            
            # Ensure nodes exist
            tx.run("MERGE (n:Theme {name: $name, source: $source}) SET n.type = 'Class'", name=concept1, source=source)
            
            concept2_type = 'Property' if rel.lower() != "subclass_of" else 'Class'
            tx.run(
                "MERGE (n:Theme {name: $name, source: $source}) SET n.type = $type",
                name=concept2, source=source, type=concept2_type
            )
            
            # Create relationship
            if rel.lower() == "subclass_of":
                tx.run(
                    """
                    MATCH (c1:Theme {name: $concept1, source: $source})
                    MATCH (c2:Theme {name: $concept2, source: $source})
                    MERGE (c1)-[r:SUBCLASS_OF {source: $source}]->(c2)
                    """,
                    concept1=concept1, concept2=concept2, source=source
                )
            else:
                tx.run(
                    """
                    MATCH (c1:Theme {name: $concept1, source: $source})
                    MATCH (c2:Theme {name: $concept2, source: $source})
                    MERGE (c1)-[r:RELATION {type: $rel, source: $source}]->(c2)
                    """,
                    concept1=concept1, concept2=concept2, rel=rel, source=source
                )
        except Exception as e:
            logger.error(f"Error processing relation: {relation} (Error: {e})", exc_info=True)


def build_subgraphs(
    embedded_data: Dict[str, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]],
    database: str = "neo4j"
):
    """Build Neo4j subgraphs per document using embedded relations and themes."""
    setup_logging()
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        
        with driver.session(database=database) as session:
            for source, (embedded_relations, embedded_themes) in embedded_data.items():
                logger.info(f"Building subgraph for document: {source}")
                logger.debug(f"Themes: {[t for t, _ in embedded_themes]}")
                logger.debug(f"Relations: {[r for r, _ in embedded_relations]}")
                
                # Execute all operations for this document in a single transaction
                session.execute_write(
                    _build_subgraph_tx,
                    source,
                    embedded_relations,
                    embedded_themes
                )
    except Exception as e:
        logger.error(f"Failed to connect or build graph: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    from .data_loader import load_and_split_data
    from .relation_extractor import extract_relations
    from .idea_extractor import extract_ideas
    from .embedder import embed_data
    
    sample_chunks = load_and_split_data()
    if sample_chunks:
        limited_chunks = sample_chunks[:2]
        relations = extract_relations(limited_chunks)
        themes = list(set(extract_ideas(limited_chunks)))
        embedded_data = embed_data(limited_chunks, relations, themes, visualize=False)
        build_subgraphs(embedded_data, database="neo4j")
        logger.info("Subgraph building example completed.")
    else:
        logger.warning("No chunks found to run the graph builder example.")