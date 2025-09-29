import logging
import json
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import LLM_MODEL, OPENAI_API_KEY, PROMPTS, MAX_WORKERS

logger = logging.getLogger(__name__)


def _extract_relations_from_chunk(chunk: Document, model_name: str) -> Optional[List[Dict]]:
    """
    Processes a single document chunk to extract relations.
    Designed to be run in a separate thread.
    """
    try:
        llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        prompt_template = PROMPTS["relation_extractor"]["main_prompt"]
        prompt = prompt_template.format(chunk_content=chunk.page_content)
        
        response = llm.invoke(prompt)
        
        json_str = response.content.split('```json')[-1].split('```')[0].strip()
        data = json.loads(json_str)
        
        if "relations" in data and isinstance(data["relations"], list):
            return data["relations"]
        else:
            logger.warning(f"Invalid JSON format for relations: 'relations' key missing or not a list.")
            return []

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON for relations: {response.content}")
    except Exception as e:
        source_info = chunk.metadata.get('source', 'unknown_source')
        logger.error(f"An error occurred while processing relations for a chunk from {source_info}: {e}", exc_info=True)

    return None


def extract_relations(chunks: List[Document], model_name: str = LLM_MODEL, max_workers: int = MAX_WORKERS) -> List[str]:
    """
    Extracts relation triples from document chunks concurrently.
    
    Args:
        chunks: A list of document chunks to process.
        model_name: The name of the LLM to use.
        max_workers: The number of concurrent threads for API calls.
        
    Returns:
        A list of relation triples in string format.
    """
    all_relations = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(_extract_relations_from_chunk, chunk, model_name): chunk for chunk in chunks}
        
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Extracting Relations (Parallel)"):
            result = future.result()
            if result is not None:
                all_relations.extend(result)

    # Deduplicate the list of dictionaries
    unique_relations = [dict(t) for t in {tuple(sorted(d.items())) for d in all_relations}]
    logger.info(f"Extracted {len(unique_relations)} unique relations from {len(chunks)} chunks.")
    
    # Convert back to the string format expected by the graph builder
    relation_strings = [f"{r['source']} -> {r['type']} -> {r['target']}" for r in unique_relations]
    
    return relation_strings