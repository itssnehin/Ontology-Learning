from src.data_loader import load_and_split_data
from src.structured_extractor import extract_component_from_chunk
from src.neo4j_graph_builder import build_graph_from_components

def main():
    """Main function to run the modular ontology learning pipeline."""
    # 1. Load and split data
    print("Loading and splitting documents...")
    chunks = load_and_split_data()
    print(f"Loaded {len(chunks)} chunks.")

    # 2. Extract structured data from each chunk
    all_components = []
    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing chunk {i+1}/{len(chunks)} ---")
        component_data = extract_component_from_chunk(chunk.page_content)
        if component_data:
            all_components.append(component_data)

    # 3. Build graph from extracted components
    if all_components:
        print("\nBuilding Neo4j graph from extracted components...")
        build_graph_from_components(all_components)
        print("Graph building complete.")
    else:
        print("No components were extracted, so no graph was built.")

if __name__ == "__main__":
    main()
