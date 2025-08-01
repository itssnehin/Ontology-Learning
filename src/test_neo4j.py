from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def test_connection():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j is running!' AS message")
            print(result.single()["message"])
    finally:
        driver.close()

if __name__ == "__main__":
    test_connection()