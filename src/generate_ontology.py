# src/generate_ontology.py

import os
import json
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Define project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# This is where we READ from
JSON_INPUT_DIR = os.path.join(BASE_DIR, "output", "structured_json_openai")
# This is where we WRITE to
ONTOLOGY_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "final_ontology")
ONTOLOGY_FILE_PATH = os.path.join(ONTOLOGY_OUTPUT_DIR, "master_ontology.ttl")

# Define our custom ontology namespace
EX = Namespace("http://eait.uq.edu.au/ontologies/electronics#")

# --- Helper Functions ---

def load_json_file(filepath):
    """Loads a single JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def sanitize_for_uri(text):
    """Removes spaces and special characters to make a valid URI component."""
    return text.replace(" ", "").replace("/", "_").replace("(", "").replace(")", "")

# --- Core Ontology Logic ---

def create_base_ontology():
    """Initializes the graph with foundational classes and binds prefixes."""
    g = Graph()
    g.bind("ex", EX)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    # Define a top-level class for all our components
    g.add((EX.ElectronicComponent, RDF.type, OWL.Class))
    g.add((EX.ElectronicComponent, RDFS.label, Literal("Electronic Component")))
    
    return g

def integrate_component_data(g, data):
    """
    Integrates the data from a single JSON file into the master graph.
    This function is the heart of the integration process.
    """
    # --- 1. Define the Component Class ---
    comp_type_str = data.get("component_type", "GenericComponent")
    class_uri = EX[sanitize_for_uri(comp_type_str)]
    
    # Check if the class already exists. If not, create it.
    if (class_uri, RDF.type, OWL.Class) not in g:
        g.add((class_uri, RDF.type, OWL.Class))
        # This is the corrected line:
        g.add((class_uri, RDFS.subClassOf, EX.ElectronicComponent)) # All components are subclasses of the main one
        g.add((class_uri, RDFS.label, Literal(comp_type_str)))

    # --- 2. Create the Component Individual ---
    model_num_str = data.get("model_number", "UnknownModel")
    individual_uri = EX[sanitize_for_uri(model_num_str)]

    # Add the individual and link it to its class
    g.add((individual_uri, RDF.type, OWL.NamedIndividual))
    g.add((individual_uri, RDF.type, class_uri))
    label = f"{data.get('manufacturer', '')} {model_num_str}"
    g.add((individual_uri, RDFS.label, Literal(label)))

    # --- 3. Add Manufacturer and other top-level fields ---
    g.add((individual_uri, EX.hasManufacturer, Literal(data.get('manufacturer'))))

    # --- 4. Process and Add Properties ---
    for prop in data.get("properties", []):
        prop_name_str = prop.get("name")
        prop_val_str = prop.get("value")
        if not prop_name_str or not prop_val_str:
            continue
            
        prop_uri = EX[f"has{sanitize_for_uri(prop_name_str)}"]

        # Check if the property definition exists. If not, create it.
        if (prop_uri, RDF.type, OWL.DatatypeProperty) not in g:
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            g.add((prop_uri, RDFS.domain, class_uri)) # Property applies to this class
            g.add((prop_uri, RDFS.range, XSD.string)) # Default to string for now
            g.add((prop_uri, RDFS.label, Literal(prop_name_str)))
        
        # Add the actual data triple for the individual
        g.add((individual_uri, prop_uri, Literal(prop_val_str)))

if __name__ == "__main__":
    # 1. Initialize our master knowledge graph
    master_graph = create_base_ontology()
    print("Initialized master ontology graph.")

    # 2. Find all the JSON files to process
    if not os.path.exists(JSON_INPUT_DIR):
        print(f"ERROR: Input directory not found: {JSON_INPUT_DIR}")
        print("Please run the extraction script first.")
    else:
        json_files = [f for f in os.listdir(JSON_INPUT_DIR) if f.endswith('.json')]
        print(f"Found {len(json_files)} extracted JSON files to integrate.")

        # 3. Loop through and integrate each file
        for filename in json_files:
            print(f"  - Integrating '{filename}'...")
            filepath = os.path.join(JSON_INPUT_DIR, filename)
            component_data = load_json_file(filepath)
            integrate_component_data(master_graph, component_data)
        
        # 4. Save the final, unified graph
        os.makedirs(ONTOLOGY_OUTPUT_DIR, exist_ok=True)
        master_graph.serialize(destination=ONTOLOGY_FILE_PATH, format="turtle")
        
        print(f"\n✅ Success! Unified ontology saved to: {ONTOLOGY_FILE_PATH}")