# prune_ontology.py
from neo4j import GraphDatabase
from rdflib import Graph, Namespace
import os
import config  # Import config file

# Neo4j driver
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))

def test_connection():
    """Test Neo4j connection."""
    try:
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 1")
            print("Connection successful:", result.single()[0])
    except Exception as e:
        print(f"Connection failed: {e}")

def import_owl():
    """Import OWL file into Neo4j."""
    try:
        if not os.path.exists(config.OWL_FILE):
            print(f"OWL file not found at: {config.OWL_FILE}")
            return

        g = Graph()
        g.parse(config.OWL_FILE, format="xml")  # Adjust format if RDF/XML or Turtle
        print(f"Loaded OWL file: {config.OWL_FILE}")
        print("Number of triples in OWL file:", len(g))

        # Print the first 10 triples for debug
        print("First 10 triples:")
        for subj, pred, obj in list(g)[:10]:
            print(subj, pred, obj)

        # Define namespaces
        RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        OWL = Namespace("http://www.w3.org/2002/07/owl#")

        # Identify classes based on subjects with rdfs:subClassOf
        classes = set()
        for subj, pred, obj in g:
            if pred == RDFS.subClassOf or pred == OWL.Class:
                classes.add(str(subj))
        print("Number of classes found:", len(classes))

        # Transaction for importing data
        with driver.session(database="neo4j") as session:
            tx = session.begin_transaction()
            for class_uri in classes:
                class_name = class_uri.split("#")[-1] if "#" in class_uri else class_uri.split("/")[-1]
                tx.run("MERGE (n:Class {uri: $uri, name: $name})",
                       uri=class_uri, name=class_name)
                print(f"Created node: {class_name}")
            tx.commit()

            tx = session.begin_transaction()
            subclass_rels = 0
            for subj, pred, obj in g:
                if pred == RDFS.subClassOf and str(subj) in classes and str(obj) in classes:
                    sub_name = str(subj).split("#")[-1] if "#" in str(subj) else str(subj).split("/")[-1]
                    super_name = str(obj).split("#")[-1] if "#" in str(obj) else str(obj).split("/")[-1]
                    tx.run("""
                        MATCH (sub:Class {name: $sub_name})
                        MATCH (super:Class {name: $super_name})
                        MERGE (sub)-[:SUBCLASS_OF]->(super)
                    """, sub_name=sub_name, super_name=super_name)
                    subclass_rels += 1
                    print(f"Created relationship: {sub_name} -> {super_name}")
            tx.commit()
            print("Number of subclass relationships found:", subclass_rels)

        print("OWL ontology imported successfully.")
    except Exception as e:
        print(f"OWL import failed: {e}")

def prune_ontology():
    """Prune the ontology to electrotechnical ideas only (segment 27)."""
    try:
        with driver.session(database="neo4j") as session:
            # Verify initial state
            result = session.run("MATCH (n:Class) RETURN count(n) AS node_count")
            initial_node_count = result.single()[0]
            result = session.run("MATCH ()-[r:SUBCLASS_OF]->() RETURN count(r) AS rel_count")
            initial_rel_count = result.single()[0]
            print(f"Initial state: {initial_node_count} nodes, {initial_rel_count} relationships")

            # Prune non-segment 27 nodes (based on hierarchyCode or URI pattern)
            session.run("""
                MATCH (n:Class)
                WHERE NOT n.uri CONTAINS '27' AND NOT n.name =~ 'C_27.*'
                DETACH DELETE n
            """)
            print("Non-electrotechnical nodes pruned.")

            # Verify pruned state
            result = session.run("MATCH (n:Class) RETURN count(n) AS node_count")
            node_count = result.single()[0]
            result = session.run("MATCH ()-[r:SUBCLASS_OF]->() RETURN count(r) AS rel_count")
            rel_count = result.single()[0]
            print(f"Verification: {node_count} nodes, {rel_count} relationships remain.")
    except Exception as e:
        print(f"Pruning failed: {e}")

def delete_data():
    """Delete all data from the Neo4j database."""
    try:
        with driver.session(database="neo4j") as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("All data deleted from Neo4j database.")
    except Exception as e:
        print(f"Delete failed: {e}")

def main():
    """Main function to run tests and imports."""
    test_connection()
    import_owl()  # Ensure the ontology is imported before pruning
    prune_ontology()
    # delete_data()  # Uncomment to delete all data (optional)

if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()