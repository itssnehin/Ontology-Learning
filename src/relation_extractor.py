import logging
import json
import re
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from tiktoken import get_encoding

from .config import LLM_MODEL, OPENAI_API_KEY
from .utils import setup_logging

logger = logging.getLogger(__name__)

def extract_relations(chunks: List[Document], model_name: str = LLM_MODEL) -> List[Dict[str, str]]:
    """
    Extract relation triples for ontology construction from document chunks.
    
    Args:
        chunks: List of document chunks.
        model_name: Name of the LLM model to use.
    
    Returns:
        List of relation dictionaries, e.g., [{"source": "ClassA", "type": "relationship", "target": "ClassB"}].
    """
    setup_logging()
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
    tokenizer = get_encoding("cl100k_base")
    LLM_COST_PER_1K_TOKENS = 0.00336
    total_tokens = 0
    total_cost = 0.0
    
    all_relations = []
    
    for chunk in chunks:
        prompt = f"""
        You are an expert in ontology engineering. Your task is to extract an ontology subgraph from the following technical document, focusing on component-based concepts and their relations.
        
        1. Identify key domain concepts (classes) and their properties.
        2. Extract semantic relationships between them using relations like 'operatesIn', 'hasFeature', 'subclass_of'.
        
        OUTPUT FORMAT:
        Respond with a single JSON object with one key: "relations". The value should be a list of objects, where each object represents a single relation with "source", "type", and "target" keys.
        
        Example:
        {{
            "relations": [
                {{"source": "FPC Antenna", "type": "subclass_of", "target": "Antenna"}},
                {{"source": "Antenna", "type": "hasFeature", "target": "Adhesive Backing"}}
            ]
        }}
        
        Document:
        {chunk.page_content}
        """
        input_tokens = len(tokenizer.encode(prompt))
        response = llm.invoke(prompt)
        output_tokens = len(tokenizer.encode(response.content))
        
        chunk_tokens = input_tokens + output_tokens
        chunk_cost = (chunk_tokens / 1000) * LLM_COST_PER_1K_TOKENS
        total_tokens += chunk_tokens
        total_cost += chunk_cost
        logger.debug(f"Chunk Tokens: {chunk_tokens}, Cost: ${chunk_cost:.6f}")
        logger.debug(f"Raw LLM Response for Relations:\n{response.content}")
        
        try:
            json_str = response.content.split('```json')[-1].split('```')[0].strip()
            data = json.loads(json_str)
            if "relations" in data and isinstance(data["relations"], list):
                all_relations.extend(data["relations"])
            else:
                logger.warning(f"Invalid JSON format from LLM: 'relations' key missing or not a list. Response: {response.content}")

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response.content}")
    
    # Deduplicate relations
    unique_relations = [dict(t) for t in {tuple(d.items()) for d in all_relations}]
    
    logger.info(f"Extracted Relations: {unique_relations}")
    logger.info(f"Total OpenAI API Usage for Relations: Tokens={total_tokens}, Cost=${total_cost:.6f}")
    
    # Convert to string format for compatibility with old graph builder if needed
    # This is now handled by the new graph builder that accepts dicts.
    relation_strings = [f"{r['source']} -> {r['type']} -> {r['target']}" for r in unique_relations]
    return relation_strings


if __name__ == "__main__":
    from .data_loader import load_and_split_data
    
    sample_chunks = load_and_split_data()
    if sample_chunks:
        final_relations = extract_relations(sample_chunks[:2])
        logger.info(f"Final Extracted Relations: {final_relations}")
    else:
        logger.warning("No chunks found to run relation extractor example.")