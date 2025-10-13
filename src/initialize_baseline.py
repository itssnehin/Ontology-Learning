import logging
import argparse
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

# Define the Ontology Hierarchy (Child -> Parent)
# Everything ultimately leads back to "Thing"
ONTOLOGY_HIERARCHY = [
    # --- Level 1: Core Schema.org ---
    ("Product", "Thing"),
    ("Organization", "Thing"),

    # --- Level 2: Domain High-Level Categories ---
    # We define ElectronicComponent as a subclass of Product
    ("ElectronicComponent", "Product"),

    # --- Level 3: Component Families ---
    ("PassiveComponent", "ElectronicComponent"),
    ("ActiveComponent", "ElectronicComponent"),
    ("RFComponent", "ElectronicComponent"),
    ("Interconnect", "ElectronicComponent"),
    ("Electromechanical", "ElectronicComponent"),
    ("Sensor", "ElectronicComponent"),

    # --- Level 4: Specific Classes (The benchmark concepts) ---
    # Passive
    ("Resistor", "PassiveComponent"),
    ("Capacitor", "PassiveComponent"),
    ("Inductor", "PassiveComponent"),
    # Active
    ("Diode", "ActiveComponent"),
    ("Transistor", "ActiveComponent"),
    ("IntegratedCircuit", "ActiveComponent"),
    # RF
    ("Antenna", "RFComponent"),
    ("FPCAntenna", "Antenna"), # Subclass of Antenna
    ("ChipAntenna", "Antenna"),
    # Interconnect
    ("Connector", "Interconnect"),
    ("Cable", "Interconnect"),
]

def initialize_database(db_name: str):
    """
    Wipes a database and builds a strict Schema.org class hierarchy rooted at 'Thing'.
    """
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info(f"✅ Connected to Neo4j. Targeting database: '{db_name}'")

        with driver.session(database=db_name) as session:
            # 1. Wipe Database
            logger.warning(f"🗑️ Wiping database '{db_name}'...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # 2. Create constraints for ontology classes (ensures uniqueness)
            try:
                session.run("CREATE CONSTRAINT FOR (c:OntologyClass) REQUIRE c.name IS UNIQUE")
            except Exception:
                pass # Constraint might already exist

            logger.info("🏗️ Building Ontology Hierarchy starting at 'Thing'...")

            # 3. Create the Absolute Root: Thing
            session.run("""
                MERGE (t:OntologyClass {name: 'Thing'})
                SET t.uri = 'https://schema.org/Thing',
                    t.description = 'The most generic type of item.'
            """)
            logger.info("   - Created Root: [Thing]")

            # 4. Build the tree based on the definition list
            for child_name, parent_name in ONTOLOGY_HIERARCHY:
                # We use MERGE to ensure we don't create duplicates if a parent appears twice
                query = """
                    MATCH (parent:OntologyClass {name: $parent_name})
                    MERGE (child:OntologyClass {name: $child_name})
                    ON CREATE SET 
                        child.uri = 'https://schema.org/' + $child_name,
                        child.source = 'Baseline'
                    MERGE (child)-[:SUBCLASS_OF]->(parent)
                """
                session.run(query, parent_name=parent_name, child_name=child_name)
                logger.info(f"   - Linked [{child_name}] -> [{parent_name}]")

            # Get stats
            count = session.run("MATCH (n:OntologyClass) RETURN count(n) as c").single()['c']
            logger.info(f"🎉 Hierarchy built successfully with {count} classes.")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a strict Schema.org hierarchy in Neo4j.")
    parser.add_argument(
        "--db-name",
        required=True,
        help="The name of the Neo4j database to target (e.g., 'schema_org_base')."
    )
    args = parser.parse_args()

    print(f"⚠️  WARNING: This will COMPLETELY WIPE data in '{args.db_name}'.")
    confirm = input("Type 'BUILD' to continue: ")
    if confirm == 'BUILD':
        initialize_database(args.db_name)
    else:
        print("Operation cancelled.")