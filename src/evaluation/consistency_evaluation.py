import logging
import json
from pathlib import Path
import argparse
from owlready2 import *

from ..config import logger

from io import BytesIO
import contextlib
import os

def evaluate_consistency(generated_file: Path, schema_file: Path):
    """
    Uses an OWL reasoner to check the logical consistency of the generated ontology
    against a formal schema.
    """
    logger.info("--- Starting Logical Consistency Evaluation ---")
    
    # 1. Load the formal schema (the rules) by reading the file manually
    try:
        with open(schema_file, "rb") as f:
            schema_content = BytesIO(f.read())
        
        # Load the ontology from the in-memory file object
        onto_schema = get_ontology("http://www.example.org/electronics_schema#").load(fileobj=schema_content)
        logger.info(f"✅ Successfully loaded schema from {schema_file.name}")

    except Exception as e:
        logger.error(f"Could not load schema file '{schema_file}'. Error: {e}", exc_info=True)
        return

    # 2. Load the generated data (the facts) into a new ontology
    logger.info("... Parsing generated data into a temporary ontology...")
    onto_data = get_ontology("http://www.example.org/generated_data#")

    try:
        with open(generated_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Could not read or parse the generated JSON file '{generated_file}': {e}", exc_info=True)
        return

    with onto_data:
        # Import the schema so the reasoner knows about our rules
        onto_data.imported_ontologies.append(onto_schema)

        # Dynamically create classes and individuals from the generated data
        for item in data.get('@graph', []):
            item_name_raw = item.get('name', '')
            item_type_str = item.get('@type', 'Thing')
            
            if not item_name_raw: continue

            # Sanitize name for OWL (replaces spaces and special chars, ensures it doesn't start with a digit)
            item_name = re.sub(r'[^a-zA-Z0-9_]', '_', item_name_raw)
            if item_name and item_name[0].isdigit():
                item_name = '_' + item_name

            # Find the class from the loaded schema
            SchemaClass = onto_schema[item_type_str]
            if not SchemaClass:
                logger.warning(f"Class '{item_type_str}' not found in schema. Using owl.Thing.")
                SchemaClass = Thing

            # Create an individual (an instance of the class)
            individual = SchemaClass(item_name)
            
            # Add properties
            if manuf_name := item.get('manufacturer'):
                manuf_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', manuf_name)
                if manuf_sanitized and manuf_sanitized[0].isdigit():
                    manuf_sanitized = '_' + manuf_sanitized
                
                ManufIndividual = onto_schema.Organization(manuf_sanitized)
                individual.manufacturer.append(ManufIndividual)
    
    # 3. Run the reasoner (HermiT is the default in OwlReady2)
    logger.info("🤖 Running the reasoner to check for inconsistencies...")
    try:
        # Suppress the verbose Java output from the reasoner
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                sync_reasoner([onto_data]) # Pass the ontology to the reasoner
        
        inconsistent_classes = list(onto_data.inconsistent_classes())
        
        print("\n--- Logical Consistency Report ---")
        if not inconsistent_classes:
            logger.info("✅ SUCCESS: The ontology is logically consistent.")
            print("✅ Status: CONSISTENT")
            print("   No logical contradictions were found against the schema.")
        else:
            logger.error(f"❌ FAILURE: Found {len(inconsistent_classes)} inconsistent classes.")
            print("❌ Status: INCONSISTENT")
            print("   The following concepts violate the rules defined in the schema:")
            for c in inconsistent_classes:
                reason = "Inconsistency found (e.g., disjoint class violation)."
                print(f"  - {c.name}: {reason}")
        print("------------------------------------\n")


    except OwlReadyInconsistentOntologyError as e:
        logger.error(f"❌ An unrecoverable inconsistency was found by the reasoner: {e}", exc_info=True)
        print(f"\n--- Logical Consistency Report ---")
        print(f"❌ CRITICAL ERROR: The ontology could not be processed due to a fundamental contradiction.")
        print(f"   Reasoner Message: {e}")
        print("------------------------------------\n")

# ... (the __main__ block is unchanged)

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Check ontology for logical consistency.")
    parser.add_argument("generated_file", type=Path, help="Path to the generated new_schema_objects_...jsonld file.")
    parser.add_argument("--schema", type=Path, default=Path("data/electronics_schema.owl"), help="Path to the formal OWL schema file.")

    args = parser.parse_args()

    # --- THIS IS THE IMPROVED CHECK ---
    if not args.generated_file.exists():
        print(f"❌ ERROR: Generated file not found at '{args.generated_file}'")
        logger.error(f"Generated file not found: {args.generated_file}")
        sys.exit(1) # Exit with an error code

    if not args.schema.exists():
        print(f"❌ ERROR: Schema file not found at '{args.schema}'")
        print("Please ensure you have created the data/electronics_schema.owl file.")
        logger.error(f"Schema file not found: {args.schema}")
        sys.exit(1) # Exit with an error code

    # If both files exist, proceed with the evaluation
    evaluate_consistency(args.generated_file, args.schema)