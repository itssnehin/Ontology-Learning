import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENAI_API_KEY
from tiktoken import get_encoding
from utils import setup_logging
import re

def extract_relations(chunks, model_name=LLM_MODEL):
    """
    Extract relation triples for ontology construction from document chunks.
    
    Args:
        chunks: List of document chunks (LangChain Document objects).
        model_name: Name of the LLM model to use.
    
    Returns:
        List of relation triples in the format 'ClassA -> relationship -> ClassB_or_Property'.
    """
    setup_logging("../logs", "relation_extractor")
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    LLM_COST_PER_1K_TOKENS = 0.00336  # $0.00336/1,000 tokens for gpt-4o (July 2025)
    total_tokens = 0
    total_cost = 0.0
    
    relations = []
    pattern = re.compile(r"(.+?) --(.+?)-->(.+?)(?:\n|$)")
    
    for chunk in chunks:
        prompt = """
        You are an expert in ontology engineering and knowledge graph construction.
        Your task is to read the following technical document and extract an ontology subgraph for IoT and antenna specifications, focusing on component-based concepts and their relations.
        
        1. Identify key domain concepts (classes) representing physical or abstract components (e.g., 'Antenna', 'WiFi Antenna', 'FPC Antenna', 'DAC', 'Resistor').
        2. Extract semantic relationships between these concepts and other concepts or properties (e.g., '2.4 GHz Band', 'Adhesive Backing'), using relations like 'operatesIn', 'hasFeature', 'supports'.
        3. Represent the ontology as a structured subgraph in the following format:
        
        OUTPUT FORMAT:
        NODES (Classes): [List of component-based concepts]
        EDGES (Relationships): 
        - ClassA --relationship--> ClassB_or_Property
        
        Ensure the output is concise, domain-relevant, and avoids trivial words (e.g., "the", "system", "data"). Properties (e.g., '2.4 GHz Band', 'Dimensions') can appear in relations but not in the taxonomy.
        
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
        print(f"Raw LLM Response for Relations:\n{response.content}")
        
        lines = response.content.split("\n")
        edges_section = False
        for line in lines:
            line = line.strip()
            if "EDGES (Relationships):" in line:
                edges_section = True
                continue
            if edges_section and line.startswith("- "):
                match = pattern.match(line[2:])
                if match:
                    class_a, rel, class_b = match.groups()
                    relation = f"{class_a.strip()} -> {rel.strip()} -> {class_b.strip()}"
                    relations.append(relation)
            if line == "" or "NODES" in line:
                edges_section = False
    
    relations = list(set(relations))  # Deduplicate
    print(f"Extracted Relations: {relations}")
    print(f"Total OpenAI API Usage for Relations: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    return relations

if __name__ == "__main__":
    from data_loader import load_and_split_data
    chunks = load_and_split_data()
    relations = extract_relations(chunks[:10])
    print("Final Extracted Relations:", relations)