from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENAI_API_KEY
from tiktoken import get_encoding

def extract_ideas(chunks, model_name=LLM_MODEL):
    """
    Extract a taxonomy of concepts from document chunks for ontology construction.
    
    Args:
        chunks: List of document chunks (LangChain Document objects).
        model_name: Name of the LLM model to use.
    
    Returns:
        List of concepts (flattened taxonomy for Neo4j nodes).
    """
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    LLM_COST_PER_1K_TOKENS = 0.00336  # $0.00336/1,000 tokens for gpt-4o (July 2025)
    total_tokens = 0
    total_cost = 0.0
    
    concepts = []
    
    for chunk in chunks:
        prompt = """
        You are an expert in ontology engineering and knowledge graph construction.
        Your task is to read the following technical document and extract an ontology subgraph that captures the main concepts and their relationships for IoT and antenna specifications.
        
        1. Identify the key domain concepts (classes), including:
           - Abstract categories (e.g., 'WiFi Compatibility', 'Frequency Range').
           - Specific entities that represent unique classes (e.g., '2.4 GHz Band', 'FPC Antenna').
           - Features and attributes (e.g., 'Adhesive Backing', 'Ground Plane Independence').
        2. Organize the concepts into a taxonomy (is_a / subclass relationships, e.g., '2.4 GHz Band' is_a 'Frequency Range').
        3. Represent the ontology as a structured subgraph in the following format:
        
        OUTPUT FORMAT:
        NODES (Classes): [List of concepts]
        EDGES (Relationships): 
        - ClassA --subclass_of--> ClassB
        
        Ensure the output is concise, domain-relevant, and avoids trivial words (e.g., "the", "system", "data").
        Include frequency ranges (e.g., '2.4 GHz Band', '5 GHz Band') as subclasses of 'Frequency Range'.
        
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
        nodes_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("NODES (Classes):"):
                nodes_section = True
                continue
            if nodes_section and line.startswith("- ") and not line.startswith("- Class"):
                concepts.append(line[2:].strip())
            if line.startswith("EDGES (Relationships):"):
                nodes_section = False
    
    concepts = list(set(concepts))  # Deduplicate
    print(f"Total OpenAI API Usage for Concepts: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    return concepts

if __name__ == "__main__":
    from data_loader import load_and_split_data
    chunks = load_and_split_data()
    concepts = extract_ideas(chunks[:10])
    print("Extracted Concepts:", concepts)