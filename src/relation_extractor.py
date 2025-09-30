import logging
import json
from typing import List, Dict, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from tiktoken import get_encoding
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized configuration
from src.config import (
    LLM_MODEL, OPENAI_API_KEY, PROMPTS, MAX_WORKERS,
    LLM_COST_PER_1K_TOKENS_INPUT, LLM_COST_PER_1K_TOKENS_OUTPUT
)

logger = logging.getLogger(__name__)


def _extract_relations_from_chunk(chunk: Document, model_name: str) -> Tuple[Optional[List[Dict]], float]:
    """
    Processes a single document chunk to extract relations and calculate cost.
    This function is designed to be run in a separate thread.
    """
    cost = 0.0
    try:
        llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        tokenizer = get_encoding("cl100k_base")
        
        prompt_template = PROMPTS["relation_extractor"]["main_prompt"]
        prompt = prompt_template.format(chunk_content=chunk.page_content)
        
        input_tokens = len(tokenizer.encode(prompt))
        response = llm.invoke(prompt)
        output_tokens = len(tokenizer.encode(response.content))

        # Calculate cost for this specific API call
        cost = (input_tokens / 1000 * LLM_COST_PER_1K_TOKENS_INPUT) + \
               (output_tokens / 1000 * LLM_COST_PER_1K_TOKENS_OUTPUT)

        # Parse the JSON response
        json_str = response.content.split('```json')[-1].split('```')[0].strip()
        data = json.loads(json_str)
        
        if "relations" in data and isinstance(data["relations"], list):
            return data["relations"], cost
        else:
            logger.warning(f"Invalid JSON format for relations: 'relations' key missing or not a list. Response: {response.content}")
            return [], cost

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON for relations from LLM response: {response.content}")
    except Exception as e:
        source_info = chunk.metadata.get('source', 'unknown_source')
        logger.error(f"An error occurred while processing relations for a chunk from {source_info}: {e}", exc_info=True)

    return None, cost


def extract_relations(chunks: List[Document], model_name: str = LLM_MODEL, max_workers: int = MAX_WORKERS) -> List[str]:
    """
    Extracts relation triples from document chunks concurrently and tracks cost.
    
    Args:
        chunks: A list of document chunks to process.
        model_name: The name of the LLM to use.
        max_workers: The number of concurrent threads for API calls, defaulting to the config value.
        
    Returns:
        A list of relation triples in the format 'Source -> Type -> Target'.
    """
    total_cost = 0.0
    all_relations = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(_extract_relations_from_chunk, chunk, model_name): chunk for chunk in chunks}
        
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Extracting Relations (Parallel)"):
            relations, cost = future.result()
            total_cost += cost
            if relations is not None:
                all_relations.extend(relations)

    logger.info(f"Total cost for relation extraction: ${total_cost:.4f}")

    # Deduplicate the list of dictionaries. This handles cases where the same relation is found in multiple chunks.
    # It works by converting each dict to a tuple of sorted items, which is hashable and can be put in a set.
    unique_relations_set = {tuple(sorted(d.items())) for d in all_relations if d.get('source') and d.get('type') and d.get('target')}
    unique_relations = [dict(t) for t in unique_relations_set]
    
    logger.info(f"Extracted {len(unique_relations)} unique relations from {len(chunks)} chunks.")
    
    # Convert back to the string format expected by the graph builder for compatibility
    relation_strings = [f"{r['source']} -> {r['type']} -> {r['target']}" for r in unique_relations]
    
    return relation_strings


if __name__ == "__main__":
    from src.data_loader import load_and_split_data

    logger.info("Running relation_extractor.py as a standalone script for demonstration.")
    sample_chunks = load_and_split_data()
    
    if sample_chunks:
        # Use a small subset for the demo to manage cost and time
        demo_chunks = sample_chunks[:5]
        logger.info(f"Processing {len(demo_chunks)} chunks for the demo.")
        
        final_relations = extract_relations(demo_chunks)
        logger.info(f"Final Extracted Relations (first 5): {final_relations[:5]}")
    else:
        logger.warning("No markdown files found in the data directory. Cannot run relation extractor demonstration.")