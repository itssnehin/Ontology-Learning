import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from typing import List, Tuple, Dict
import numpy as np
import re
from utils import setup_logging
from data_loader import load_and_split_data
from relation_extractor import extract_relations
from idea_extractor import extract_ideas
from embedder import embed_data

def build_subgraphs(embedded_data: Dict[str, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]], database: str = "neo4j"):
    """
    Build Neo4j subgraphs per document using embedded relations and themes.
    
    Args:
        embedded_data: Dictionary mapping document source to (embedded_relations, embedded_themes) tuples.
        database: Name of the Neo4j database to use (default: 'neo4j').
    """
    setup_logging("../logs", "graph_builder")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    pattern = re.compile(r"(.+?) -> (.+?) -> (.+?)")
    
    try:
        with driver.session(database=database) as session:
            for source, (embedded_relations, embedded_themes) in embedded_data.items():
                print(f"Building subgraph for document: {source}")
                print(f"Themes: {list(t for t, _ in embedded_themes)}")
                print(f"Relations: {list(r for r, _ in embedded_relations)}")
                
                for theme, _ in embedded_themes:
                    session.run(
                        """
                        MERGE (n:Theme {name: $name, source: $source})
                        SET n.type = 'Class'
                        """,
                        name=theme.strip(), source=source
                    )
                
                for relation, _ in embedded_relations:
                    try:
                        match = pattern.match(relation)
                        if not match:
                            print(f"Skipping invalid relation: {relation}")
                            continue
                        concept1, rel, concept2 = match.groups()
                        session.run(
                            """
                            MERGE (n:Theme {name: $name, source: $source})
                            SET n.type = 'Class'
                            """,
                            name=concept1.strip(), source=source
                        )
                        session.run(
                            """
                            MERGE (n:Theme {name: $name, source: $source})
                            SET n.type = $type
                            """,
                            name=concept2.strip(), source=source, type='Property' if rel.lower() != "subclass_of" else 'Class'
                        )
                        edge_type = "SUBCLASS_OF" if rel.lower() == "subclass_of" else "RELATION"
                        if edge_type == "SUBCLASS_OF":
                            session.run(
                                """
                                MATCH (c1:Theme {name: $concept1, source: $source})
                                MATCH (c2:Theme {name: $concept2, source: $source})
                                MERGE (c1)-[r:SUBCLASS_OF {source: $source}]->(c2)
                                """,
                                concept1=concept1.strip(), concept2=concept2.strip(), source=source
                            )
                        else:
                            session.run(
                                """
                                MATCH (c1:Theme {name: $concept1, source: $source})
                                MATCH (c2:Theme {name: $concept2, source: $source})
                                MERGE (c1)-[r:RELATION {type: $rel, source: $source}]->(c2)
                                """,
                                concept1=concept1.strip(), concept2=concept2.strip(), rel=rel.strip(), source=source
                            )
                    except Exception as e:
                        print(f"Error processing relation: {relation} (Error: {e})")
    
    finally:
        driver.close()

if __name__ == "__main__":
    chunks = load_and_split_data()
    relations = extract_relations(chunks[:10])
    themes = list(set(extract_ideas(chunks[:10])))
    embedded_data = embed_data(chunks[:10], relations, themes)
    build_subgraphs(embedded_data, database="neo4j")