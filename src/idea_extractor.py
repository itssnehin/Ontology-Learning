import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENAI_API_KEY
from tiktoken import get_encoding
from utils import setup_logging

def extract_ideas(chunks, model_name=LLM_MODEL):
    """
    Extract a taxonomy of component-based concepts from document chunks using a chain-of-thought approach.
    
    Args:
        chunks: List of document chunks (LangChain Document objects).
        model_name: Name of the LLM model to use.
    
    Returns:
        List of component-based concepts (flattened taxonomy for Neo4j nodes).
    """
    setup_logging("../logs", "idea_extractor")
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    LLM_COST_PER_1K_TOKENS = 0.00336  # $0.00336/1,000 tokens for gpt-4o (July 2025)
    total_tokens = 0
    total_cost = 0.0
    
    concepts = []
    
    for chunk in chunks:
        # Round 1: Break down topics
        cot_round_one_prompt = """
        The following is a technical document segment. Briefly break down the topics (both specific and general concepts) relevant to this document. Explain your reasoning step by step.
        
        Document:
        {chunk}
        
        ### OUTPUT FORMAT:
        - General concepts: [List]
        - Specific concepts: [List]
        - Reasoning: [Explanation]
        """
        input_tokens = len(tokenizer.encode(cot_round_one_prompt.format(chunk=chunk.page_content)))
        cot_round_one_response = llm.invoke(cot_round_one_prompt.format(chunk=chunk.page_content))
        output_tokens = len(tokenizer.encode(cot_round_one_response.content))
        chunk_tokens = input_tokens + output_tokens
        chunk_cost = (chunk_tokens / 1000) * LLM_COST_PER_1K_TOKENS
        total_tokens += chunk_tokens
        total_cost += chunk_cost
        print(f"Chunk Tokens (Round 1): {chunk_tokens}, Cost: ${chunk_cost:.6f}")
        print(f"Raw LLM Response (Round 1):\n{cot_round_one_response.content}")
        
        # Round 2: Extract taxonomy
        cot_round_two_prompt = """
        Based on the following topic breakdown, extract a taxonomy of component-based concepts for an ontology in IoT and antenna specifications. 
        Only include physical or abstract components (e.g., 'Antenna', 'FPC Antenna', 'DAC'), excluding properties (e.g., 'Frequency Range', 'Adhesive Backing').
        
        Topic Breakdown:
        {cot_round_one_response}
        
        OUTPUT FORMAT:
        NODES (Classes): [List of component-based concepts]
        EDGES (Relationships): 
        - ClassA --subclass_of--> ClassB
        """
        input_tokens = len(tokenizer.encode(cot_round_two_prompt.format(cot_round_one_response=cot_round_one_response.content)))
        response = llm.invoke(cot_round_two_prompt.format(cot_round_one_response=cot_round_one_response.content))
        output_tokens = len(tokenizer.encode(response.content))
        chunk_tokens = input_tokens + output_tokens
        chunk_cost = (chunk_tokens / 1000) * LLM_COST_PER_1K_TOKENS
        total_tokens += chunk_tokens
        total_cost += chunk_cost
        print(f"Chunk Tokens (Round 2): {chunk_tokens}, Cost: ${chunk_cost:.6f}")
        print(f"Raw LLM Response (Round 2):\n{response.content}")
        
        # Parse response
        lines = response.content.split("\n")
        nodes_section = False
        for line in lines:
            line = line.strip()
            if "NODES (Classes):" in line or "Nodes (Classes):" in line:
                nodes_section = True
                continue
            if nodes_section and line.startswith("- ") and not line.startswith("- Class"):
                concept = line[2:].strip()
                if concept and not any(p in concept.lower() for p in ["frequency", "adhesive", "dimensions", "ground plane", "vswr", "gain", "radiation", "return loss"]):
                    concepts.append(concept)
            if "EDGES (Relationships):" in line or line == "":
                nodes_section = False
    
    concepts = list(set(concepts))  # Deduplicate
    print(f"Extracted Concepts: {concepts}")
    print(f"Total OpenAI API Usage for Concepts: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    return concepts

if __name__ == "__main__":
    from data_loader import load_and_split_data
    concepts = extract_ideas(load_and_split_data()[:10])
    print("Final Extracted Concepts:", concepts)