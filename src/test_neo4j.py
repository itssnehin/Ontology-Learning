# test_neo4j.py
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "ontology"))
try:
    with driver.session(database="neo4j") as session:
        result = session.run("RETURN 1")
        print("Connection successful:", result.single()[0])
finally:
    driver.close()