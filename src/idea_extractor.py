import logging
import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from tiktoken import get_encoding

from src.config import LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

def extract_ideas(chunks: List[Document], model_name: str = LLM_MODEL) -> List[str]:
    """
    Extract a taxonomy of component-based concepts from document chunks.
    
    Args:
        chunks: List of document chunks.
        model_name: Name of the LLM model to use.
    
    Returns:
        List of component-based concepts.
    """
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    LLM_COST_PER_1K_TOKENS = 0.00336
    total_tokens = 0
    total_cost = 0.0
    
    concepts = []
    
    for chunk in chunks:
        # Round 1: Break down topics
        cot_round_one_prompt = f"""
        The following is a technical document segment. Briefly break down the topics (both specific and general concepts) relevant to this document. Explain your reasoning step by step.
        
        Document:
        {chunk.page_content}
        
        ### OUTPUT FORMAT:
        - General concepts: [List]
        - Specific concepts: [List]
        - Reasoning: [Explanation]
        """
        input_tokens_r1 = len(tokenizer.encode(cot_round_one_prompt))
        cot_round_one_response = llm.invoke(cot_round_one_prompt)
        output_tokens_r1 = len(tokenizer.encode(cot_round_one_response.content))
        
        chunk_tokens_r1 = input_tokens_r1 + output_tokens_r1
        chunk_cost_r1 = (chunk_tokens_r1 / 1000) * LLM_COST_PER_1K_TOKENS
        total_tokens += chunk_tokens_r1
        total_cost += chunk_cost_r1
        logger.debug(f"Chunk Tokens (Round 1): {chunk_tokens_r1}, Cost: ${chunk_cost_r1:.6f}")
        logger.debug(f"Raw LLM Response (Round 1):\n{cot_round_one_response.content}")
        
        # Round 2: Extract taxonomy in JSON format
        cot_round_two_prompt = f"""
        Based on the following topic breakdown, extract a taxonomy of component-based concepts for an ontology in IoT and antenna specifications. 
        Only include physical or abstract components (e.g., 'Antenna', 'FPC Antenna', 'DAC'), excluding properties (e.g., 'Frequency Range', 'Adhesive Backing').
        
        Topic Breakdown:
        {cot_round_one_response.content}
        
        OUTPUT FORMAT:
        Respond with a single JSON object containing one key: "nodes". The value of "nodes" should be a list of component-based concept strings.
        Example: {{"nodes": ["Antenna", "FPC Antenna", "DAC"]}}
        """
        input_tokens_r2 = len(tokenizer.encode(cot_round_two_prompt))
        response = llm.invoke(cot_round_two_prompt)
        output_tokens_r2 = len(tokenizer.encode(response.content))
        
        chunk_tokens_r2 = input_tokens_r2 + output_tokens_r2
        chunk_cost_r2 = (chunk_tokens_r2 / 1000) * LLM_COST_PER_1K_TOKENS
        total_tokens += chunk_tokens_r2
        total_cost += chunk_cost_r2
        logger.debug(f"Chunk Tokens (Round 2): {chunk_tokens_r2}, Cost: ${chunk_cost_r2:.6f}")
        logger.debug(f"Raw LLM Response (Round 2):\n{response.content}")
        
        # Parse JSON response
        try:
            # The response might be wrapped in ```json ... ```
            json_str = response.content.split('```json')[-1].split('```')[0].strip()
            data = json.loads(json_str)
            if "nodes" in data and isinstance(data["nodes"], list):
                # Filter out property-like concepts
                for concept in data["nodes"]:
                    if concept and not any(p in concept.lower() for p in ["frequency", "adhesive", "dimensions", "ground plane", "vswr", "gain", "radiation", "return loss"]):
                        concepts.append(concept)
            else:
                logger.warning(f"Invalid JSON format from LLM: 'nodes' key missing or not a list. Response: {response.content}")

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response.content}")

    deduplicated_concepts = sorted(list(set(concepts)))
    logger.info(f"Extracted Concepts: {deduplicated_concepts}")
    logger.info(f"Total OpenAI API Usage for Concepts: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    return deduplicated_concepts

if __name__ == "__main__":
    from .data_loader import load_and_split_data
    
    sample_chunks = load_and_split_data()
    if sample_chunks:
        final_concepts = extract_ideas(sample_chunks[:2])
        logger.info(f"Final Extracted Concepts: {final_concepts}")
    else:
        logger.warning("No chunks found to run idea extractor example.")