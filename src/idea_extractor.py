import logging
import json
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from tiktoken import get_encoding
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import LLM_MODEL, OPENAI_API_KEY, PROMPTS, MAX_WORKERS

logger = logging.getLogger(__name__)


def _extract_concepts_from_chunk(chunk: Document, model_name: str) -> Optional[List[str]]:
    """
    Processes a single document chunk to extract concepts using a two-step prompt.
    This function is designed to be run in a separate thread.
    """
    try:
        llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        
        # Get prompt templates from the centralized config
        prompt_template_r1 = PROMPTS["idea_extractor"]["round_one"]
        prompt_template_r2 = PROMPTS["idea_extractor"]["round_two"]

        # Round 1: Break down topics to prime the model
        cot_round_one_prompt = prompt_template_r1.format(chunk_content=chunk.page_content)
        cot_round_one_response = llm.invoke(cot_round_one_prompt)
        
        # Round 2: Extract the structured taxonomy
        cot_round_two_prompt = prompt_template_r2.format(cot_round_one_response=cot_round_one_response.content)
        response = llm.invoke(cot_round_two_prompt)

        # Parse the JSON response
        json_str = response.content.split('```json')[-1].split('```')[0].strip()
        data = json.loads(json_str)
        
        chunk_concepts = []
        if "nodes" in data and isinstance(data["nodes"], list):
            for concept in data["nodes"]:
                # Filter out property-like concepts to keep the ontology clean
                if concept and not any(p in concept.lower() for p in ["frequency", "adhesive", "dimensions", "ground plane", "vswr", "gain", "radiation", "return loss"]):
                    chunk_concepts.append(concept)
        else:
            logger.warning(f"Invalid JSON format from LLM: 'nodes' key missing or not a list. Response: {response.content}")
            
        return chunk_concepts

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM response: {response.content}")
    except Exception as e:
        source_info = chunk.metadata.get('source', 'unknown_source')
        logger.error(f"An error occurred while processing a chunk from {source_info}: {e}", exc_info=True)
    
    return None


def extract_ideas(chunks: List[Document], model_name: str = LLM_MODEL, max_workers: int = MAX_WORKERS) -> List[str]:
    """
    Extracts a taxonomy of component-based concepts from document chunks concurrently.
    
    Args:
        chunks: A list of document chunks to process.
        model_name: The name of the LLM to use.
        max_workers: The number of concurrent threads to use for API calls.
    
    Returns:
        A deduplicated and sorted list of extracted concepts.
    """
    all_concepts = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all chunk processing tasks to the thread pool
        future_to_chunk = {executor.submit(_extract_concepts_from_chunk, chunk, model_name): chunk for chunk in chunks}
        
        # Use tqdm to create a progress bar that updates as tasks are completed
        for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Extracting Concepts (Parallel)"):
            result = future.result()
            if result is not None:  # Check for errors (which return None)
                all_concepts.extend(result)

    deduplicated_concepts = sorted(list(set(all_concepts)))
    logger.info(f"Extracted {len(deduplicated_concepts)} unique concepts from {len(chunks)} chunks.")
    return deduplicated_concepts