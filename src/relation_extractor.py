from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENAI_API_KEY
from tiktoken import get_encoding

def extract_relations(chunks, model_name=LLM_MODEL):
    """
    Extract relation triples for ontology construction from document chunks.
    
    Args:
        chunks: List of document chunks (LangChain Document objects).
        model_name: Name of the LLM model to use.
    
    Returns:
        List of relation triples in the format 'ClassA --relationship--> ClassB'.
    """
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    LLM_COST_PER_1K_TOKENS = 0.00336  # $0.00336/1,000 tokens for gpt-4o (July 2025)
    total_tokens = 0
    total_cost = 0.0
    
    relations = []
    
    for chunk in chunks:
        prompt = """
        You are an expert in ontology engineering and knowledge graph construction.
        Your task is to read the following technical document and extract an ontology subgraph that captures the main concepts and their relationships for IoT and antenna specifications.
        
        1. Identify the key domain concepts (classes), including:
           - Abstract categories (e.g., 'WiFi Compatibility', 'Frequency Range').
           - Specific entities that represent unique classes (e.g., '2.4 GHz Band', 'FPC Antenna').
           - Features and attributes (e.g., 'Adhesive Backing', 'Ground Plane Independence').
        2. Extract semantic relationships between these concepts (e.g., operatesIn, hasFeature, supports).
        3. Represent the ontology as a structured subgraph in the following format:
        
        OUTPUT FORMAT:
        NODES (Classes): [List of concepts]
        EDGES (Relationships): 
        - ClassA --relationship--> ClassB
        
        Ensure the output is concise, domain-relevant, and avoids trivial words (e.g., "the", "system", "data").
        Include frequency ranges (e.g., '2.4 GHz Band', '5 GHz Band') as subclasses of 'Frequency Range' and use them in relations (e.g., 'FPC Antenna --operatesIn--> 2.4 GHz Band').
        
        Document:
        {chunk}
        """
        input_tokens = len(tokenizer.encode(prompt.format(chunk=chunk.page_content)))
        response = llm.invoke(prompt.format(chunk=chunk.page_content))
        output_tokens = len(tokenizer.encode(response.content))
        chunk_tokens = input_tokens + output_tokens
        chunk_cost = (chunk_tokens / 1000) * LLM_COST_PER_1K_TOKENS
        total_tokens += chunk_tokens
        total_cost += chunk_cost
        print(f"Chunk Tokens: {chunk_tokens}, Cost: ${chunk_cost:.6f}")
        
        # Parse response
        lines = response.content.split("\n")
        edges_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("EDGES (Relationships):"):
                edges_section = True
                continue
            if edges_section and line.startswith("- Class"):
                relation = line.replace("Class", "").replace("--", " -> ").strip()
                relations.append(relation)
            if line == "":
                edges_section = False
    
    relations = list(set(relations))  # Deduplicate
    print(f"Total OpenAI API Usage for Relations: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    return relations

if __name__ == "__main__":
    from data_loader import load_and_split_data
    chunks = load_and_split_data()
    relations = extract_relations(chunks[:10])
    print("Extracted Relations:", relations)