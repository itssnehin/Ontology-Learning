#!/usr/bin/env python3
# src/langchain_ontology.py

import os
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Load environment variables
load_dotenv()

# Define project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_markdown")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "structured_json_langchain")
ONTOLOGY_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "ontology_langchain")
ONTOLOGY_FILE_PATH = os.path.join(ONTOLOGY_OUTPUT_DIR, "langchain_ontology.ttl")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ONTOLOGY_OUTPUT_DIR, exist_ok=True)

# Define Pydantic models for structured extraction
class Property(BaseModel):
    name: str = Field(description="Name of the property")
    value: str = Field(description="Value of the property, including units if applicable")

class Relationship(BaseModel):
    type: str = Field(description="Type of relationship (e.g., 'compatibleWith', 'interfacesWith')")
    target: str = Field(description="Target component or concept name")
    description: Optional[str] = Field(default=None, description="Optional description of the relationship")

class Application(BaseModel):
    name: str = Field(description="Name of the application domain")
    description: Optional[str] = Field(default=None, description="Brief description of the application")

class Component(BaseModel):
    component_type: str = Field(description="General classification of the component")
    model_number: str = Field(description="Specific model number of the component")
    manufacturer: str = Field(description="Company that produces the component")
    properties: List[Property] = Field(description="List of technical properties")
    features: List[str] = Field(description="Key features of the component")
    applications: Optional[List[Application]] = Field(
        default=[],
        description="Application domains where this component is used"
    )
    relationships: Optional[List[Relationship]] = Field(
        default=[],
        description="Relationships to other components or concepts"
    )

# Initialize LangChain components
parser = PydanticOutputParser(pydantic_object=Component)

prompt_template = """
You are an expert in electronics engineering and knowledge graph creation.
Your task is to analyze a technical component datasheet, provided in Markdown format,
and extract key information into a structured format for ontology creation.

{format_instructions}

Extract the following information from the datasheet:
- Component type (general classification)
- Model number
- Manufacturer
- Technical properties (with name and value pairs, including units)
- Key features as a list of strings
- Applications where this component is used (name and optional description)
- Relationships to other components or concepts, including:
  * Type of relationship (e.g., "compatibleWith", "interfacesWith", "supportedBy")
  * Target component or concept name
  * Optional description of the relationship

Be precise and comprehensive in your extraction. For relationships, use standardized relationship types 
that would be appropriate for an ontology (e.g., "compatibleWith", "partOf", "requires").

Here is the datasheet content:
{datasheet_content}
"""

prompt = ChatPromptTemplate.from_template(
    template=prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Check if we have an API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key.strip():
    # Initialize the LLM with OpenAI
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0
    )
else:
    # Use a mock LLM for testing
    from langchain_core.language_models.fake import FakeListLLM
    
    # Create a mock response for our sample component
    mock_responses = [
        """
        {
          "component_type": "Field-Programmable Gate Array",
          "model_number": "XC4010E",
          "manufacturer": "Xilinx",
          "properties": [
            {
              "name": "Logic Cells",
              "value": "10,000"
            },
            {
              "name": "Configurable Logic Blocks",
              "value": "400"
            },
            {
              "name": "Flip-Flops",
              "value": "1,120"
            },
            {
              "name": "Maximum User I/O",
              "value": "192"
            },
            {
              "name": "System Clock Frequency",
              "value": "Up to 80 MHz"
            },
            {
              "name": "Internal Clock Frequency",
              "value": "Up to 100 MHz"
            },
            {
              "name": "Core Voltage",
              "value": "3.3V"
            },
            {
              "name": "I/O Voltage",
              "value": "5V tolerant"
            },
            {
              "name": "Power Consumption",
              "value": "500 mW (typical)"
            },
            {
              "name": "Technology",
              "value": "SRAM-based FPGA"
            },
            {
              "name": "Process Technology",
              "value": "0.35 μm"
            }
          ],
          "features": [
            "In-system programmability",
            "On-chip RAM",
            "Dedicated carry logic for high-speed arithmetic",
            "Four dedicated clock inputs",
            "Boundary scan (JTAG) support",
            "Multiple programmable I/O standards",
            "Hierarchical memory architecture",
            "Flexible routing resources"
          ],
          "applications": [
            {
              "name": "Digital signal processing",
              "description": "Used for implementing digital filters and signal processing algorithms"
            },
            {
              "name": "Custom microprocessor designs",
              "description": "Implementation of specialized processor architectures"
            },
            {
              "name": "Protocol converters",
              "description": "Converting between different communication protocols"
            },
            {
              "name": "Real-time systems",
              "description": "Systems requiring deterministic timing behavior"
            },
            {
              "name": "High-performance computing",
              "description": "Accelerating computationally intensive tasks"
            },
            {
              "name": "Telecommunications equipment",
              "description": "Used in network routing and switching equipment"
            }
          ],
          "relationships": [
            {
              "type": "compatibleWith",
              "target": "XC4000E series programming tools",
              "description": "Compatible with programming tools designed for the XC4000E series"
            },
            {
              "type": "interfacesWith",
              "target": "XC9500 CPLDs",
              "description": "Can be used alongside XC9500 CPLDs in system designs"
            },
            {
              "type": "supportedBy",
              "target": "Xilinx ISE Design Suite",
              "description": "Development and programming supported by Xilinx ISE Design Suite"
            }
          ]
        }
        """
    ]
    
    llm = FakeListLLM(responses=mock_responses)
    print("Using mock LLM for testing as no OpenAI API key was provided.")

# Chain for extraction
extraction_chain = prompt | llm | parser

def load_file(filepath):
    """Loads the entire content of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return None

def save_json(filepath, data):
    """Saves a Python dictionary to a JSON file with nice formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def extract_component_data(filepath):
    """Extract structured data from a datasheet using LangChain."""
    print(f"-> Loading raw markdown from: {os.path.basename(filepath)}")
    markdown_content = load_file(filepath)
    if not markdown_content:
        return None
    
    print(f"-> Processing with LangChain...")
    try:
        # Extract structured data using LangChain
        result = extraction_chain.invoke({"datasheet_content": markdown_content})
        print("-> Successfully extracted and parsed data.")
        return result.model_dump()
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return None

# Ontology generation functions
def sanitize_for_uri(text):
    """Removes spaces and special characters to make a valid URI component."""
    if not text:
        return "Unknown"
    return text.replace(" ", "").replace("/", "_").replace("(", "").replace(")", "")

def create_base_ontology():
    """Initializes the graph with foundational classes and binds prefixes."""
    g = Graph()
    
    # Define our custom ontology namespace
    EX = Namespace("http://example.org/ontology#")
    
    g.bind("ex", EX)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

    # Define a top-level class for all components
    g.add((EX.ElectronicComponent, RDF.type, OWL.Class))
    g.add((EX.ElectronicComponent, RDFS.label, Literal("Electronic Component")))
    
    return g, EX

def integrate_component_data(g, EX, data):
    """Integrates component data into the ontology graph."""
    # Define the Component Class
    comp_type_str = data.get("component_type", "GenericComponent")
    class_uri = EX[sanitize_for_uri(comp_type_str)]
    
    # Check if the class already exists. If not, create it.
    if (class_uri, RDF.type, OWL.Class) not in g:
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.subClassOf, EX.ElectronicComponent))
        g.add((class_uri, RDFS.label, Literal(comp_type_str)))

    # Create the Component Individual
    model_num_str = data.get("model_number", "UnknownModel")
    individual_uri = EX[sanitize_for_uri(model_num_str)]

    # Add the individual and link it to its class
    g.add((individual_uri, RDF.type, OWL.NamedIndividual))
    g.add((individual_uri, RDF.type, class_uri))
    label = f"{data.get('manufacturer', '')} {model_num_str}"
    g.add((individual_uri, RDFS.label, Literal(label)))

    # Add Manufacturer
    g.add((individual_uri, EX.hasManufacturer, Literal(data.get('manufacturer'))))

    # Add features
    for feature in data.get("features", []):
        g.add((individual_uri, EX.hasFeature, Literal(feature)))

    # Process and Add Properties
    for prop in data.get("properties", []):
        prop_name_str = prop.get("name")
        prop_val_str = prop.get("value")
        if not prop_name_str or not prop_val_str:
            continue
            
        prop_uri = EX[f"has{sanitize_for_uri(prop_name_str)}"]

        # Check if the property definition exists. If not, create it.
        if (prop_uri, RDF.type, OWL.DatatypeProperty) not in g:
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            g.add((prop_uri, RDFS.domain, class_uri))
            g.add((prop_uri, RDFS.range, XSD.string))
            g.add((prop_uri, RDFS.label, Literal(prop_name_str)))
        
        # Add the actual data triple for the individual
        g.add((individual_uri, prop_uri, Literal(prop_val_str)))
    
    # Process applications
    if "applications" in data and data["applications"]:
        # Create Application class if it doesn't exist
        if (EX.Application, RDF.type, OWL.Class) not in g:
            g.add((EX.Application, RDF.type, OWL.Class))
            g.add((EX.Application, RDFS.label, Literal("Application")))
        
        # Create usedIn object property if it doesn't exist
        if (EX.usedIn, RDF.type, OWL.ObjectProperty) not in g:
            g.add((EX.usedIn, RDF.type, OWL.ObjectProperty))
            g.add((EX.usedIn, RDFS.domain, EX.ElectronicComponent))
            g.add((EX.usedIn, RDFS.range, EX.Application))
            g.add((EX.usedIn, RDFS.label, Literal("used in")))
        
        for app in data.get("applications", []):
            app_name = app.get("name")
            if not app_name:
                continue
                
            app_uri = EX[f"Application_{sanitize_for_uri(app_name)}"]
            
            # Create application individual
            g.add((app_uri, RDF.type, OWL.NamedIndividual))
            g.add((app_uri, RDF.type, EX.Application))
            g.add((app_uri, RDFS.label, Literal(app_name)))
            
            # Add description if available
            if app.get("description"):
                g.add((app_uri, RDFS.comment, Literal(app.get("description"))))
            
            # Link component to application
            g.add((individual_uri, EX.usedIn, app_uri))
    
    # Process relationships
    for rel in data.get("relationships", []):
        if "type" in rel and "target" in rel:
            rel_type = rel["type"]
            rel_target = rel["target"]
            
            rel_uri = EX[f"{sanitize_for_uri(rel_type)}"]
            target_uri = EX[sanitize_for_uri(rel_target)]
            
            # Define the relationship property if it doesn't exist
            if (rel_uri, RDF.type, OWL.ObjectProperty) not in g:
                g.add((rel_uri, RDF.type, OWL.ObjectProperty))
                g.add((rel_uri, RDFS.domain, EX.ElectronicComponent))
                g.add((rel_uri, RDFS.label, Literal(rel_type)))
            
            # Add the relationship triple
            g.add((individual_uri, rel_uri, target_uri))
            
            # Add description if available
            if rel.get("description"):
                # Create a blank node for the relationship instance
                rel_instance = URIRef(f"{individual_uri}_{rel_uri}_{target_uri}")
                g.add((rel_instance, RDF.type, rel_uri))
                g.add((rel_instance, EX.source, individual_uri))
                g.add((rel_instance, EX.target, target_uri))
                g.add((rel_instance, RDFS.comment, Literal(rel.get("description"))))

def main():
    # Process all MD files in data/raw_markdown
    if not os.path.exists(RAW_DATA_DIR):
        print(f"ERROR: Raw data directory not found at '{RAW_DATA_DIR}'")
        print("Please create it and add your markdown datasheets.")
        return
    
    datasheet_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.md')]
    
    if not datasheet_files:
        print(f"No markdown files found in '{RAW_DATA_DIR}'.")
        return
    
    print(f"Found {len(datasheet_files)} datasheets to process.")
    
    # Initialize ontology graph
    master_graph, EX_namespace = create_base_ontology()
    print("Initialized master ontology graph.")
    
    # Process each datasheet
    for filename in datasheet_files:
        print(f"\n--- Processing: {filename} ---")
        datasheet_filepath = os.path.join(RAW_DATA_DIR, filename)
        
        # Extract structured data
        structured_data = extract_component_data(datasheet_filepath)
        
        if structured_data:
            # Save the structured data as JSON
            output_filename = os.path.splitext(filename)[0] + "_extracted.json"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            save_json(output_filepath, structured_data)
            print(f"Extracted data saved to: {output_filepath}")
            
            # Integrate into ontology
            integrate_component_data(master_graph, EX_namespace, structured_data)
            print(f"Data integrated into ontology.")
    
    # Save the final ontology
    master_graph.serialize(destination=ONTOLOGY_FILE_PATH, format="turtle")
    print(f"\n✅ Success! Unified ontology saved to: {ONTOLOGY_FILE_PATH}")

if __name__ == "__main__":
    main()