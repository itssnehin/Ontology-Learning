import logging
import json
import re
from typing import List, Optional, Tuple, Dict, Any  # <-- THIS IS THE FIX
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from tiktoken import get_encoding
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import (
    LLM_MODEL, OPENAI_API_KEY, PROMPTS, MAX_WORKERS,
    LLM_COST_PER_1K_TOKENS_INPUT, LLM_COST_PER_1K_TOKENS_OUTPUT
)

logger = logging.getLogger(__name__)


def _extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    More robustly extracts a JSON object from a string that might contain other text.
    Handles markdown code blocks and looks for the first valid JSON object.
    """
    # 1. Try to find JSON within markdown code blocks ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Found a JSON code block, but it was malformed.")
            pass # Fall through to the next method

    # 2. If no code block, look for the first '{' and last '}'
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            potential_json = text[start:end+1]
            return json.loads(potential_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}. Raw text was: {text}")
        return None
        
    return None


def _extract_concepts_from_chunk(chunk: Document, model_name: str) -> Tuple[Optional[List[str]], float]:
    """Processes a single document chunk and returns concepts and the associated cost."""
    cost = 0.0
    try:
        llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        tokenizer = get_encoding("cl100k_base")
        
        prompt_template = PROMPTS["idea_extractor"]["main_prompt"]
        prompt = prompt_template.format(chunk_content=chunk.page_content)
        
        input_tokens = len(tokenizer.encode(prompt))
        response = llm.invoke(prompt)
        output_tokens = len(tokenizer.encode(response.content))
        
        cost = (input_tokens / 1000 * LLM_COST_PER_1K_TOKENS_INPUT) + \
               (output_tokens / 1000 * LLM_COST_PER_1K_TOKENS_OUTPUT)

        data = _extract_json_from_response(response.content)
        
        if data is None:
            return None, cost

        chunk_concepts = []
        if "nodes" in data and isinstance(data["nodes"], list):
            for concept in data["nodes"]:
                if concept and isinstance(concept, str) and not any(p in concept.lower() for p in ["frequency", "adhesive", "dimensions"]):
                    chunk_concepts.append(concept)
        else:
            logger.warning(f"Valid JSON parsed, but 'nodes' key is missing or not a list. Data: {data}")
            
        return chunk_concepts, cost

    except Exception as e:
        logger.error(f"An unhandled error occurred while processing chunk: {e}", exc_info=True)
        return None, cost


def extract_ideas(chunks: List[Document], model_name: str = LLM_MODEL, max_workers: int = MAX_WORKERS) -> List[str]:
    """Extracts concepts concurrently and tracks total cost."""
    total_cost = 0
    all_concepts = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_concepts_from_chunk, chunk, model_name): chunk for chunk in chunks}
        for future in tqdm(as_completed(futures), total=len(chunks), desc="Extracting Concepts (Parallel)"):
            concepts, cost = future.result()
            total_cost += cost
            if concepts:
                all_concepts.extend(concepts)

    logger.info(f"Total cost for concept extraction: ${total_cost:.4f}")
    deduplicated_concepts = sorted(list(set(all_concepts)))
    logger.info(f"Extracted {len(deduplicated_concepts)} unique concepts.")
    return deduplicated_concepts