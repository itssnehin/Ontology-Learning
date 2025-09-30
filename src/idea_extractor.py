import logging
import json
from typing import List, Optional, Tuple
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

def _extract_json_from_response(text: str) -> Optional[Dict]:
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
        # The error from the original traceback would be caught here
        logger.error(f"JSON parsing failed: {e}. Raw text was: {text}")
        return None
        
    return None


def _extract_concepts_from_chunk(chunk: Document, model_name: str) -> Tuple[Optional[List[str]], float]:
    """Processes a single document chunk and returns concepts and the associated cost."""
    cost = 0.0
    try:
        llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        tokenizer = get_encoding("cl100k_base")
        
        prompt_template_r1 = PROMPTS["idea_extractor"]["round_one"]
        prompt_template_r2 = PROMPTS["idea_extractor"]["round_two"]

        # Round 1
        cot_round_one_prompt = prompt_template_r1.format(chunk_content=chunk.page_content)
        input_tokens_r1 = len(tokenizer.encode(cot_round_one_prompt))
        cot_round_one_response = llm.invoke(cot_round_one_prompt)
        output_tokens_r1 = len(tokenizer.encode(cot_round_one_response.content))
        cost += (input_tokens_r1 / 1000 * LLM_COST_PER_1K_TOKENS_INPUT) + (output_tokens_r1 / 1000 * LLM_COST_PER_1K_TOKENS_OUTPUT)
        
        # Round 2
        cot_round_two_prompt = prompt_template_r2.format(cot_round_one_response=cot_round_one_response.content)
        input_tokens_r2 = len(tokenizer.encode(cot_round_two_prompt))
        response = llm.invoke(cot_round_two_prompt)
        output_tokens_r2 = len(tokenizer.encode(response.content))
        cost += (input_tokens_r2 / 1000 * LLM_COST_PER_1K_TOKENS_INPUT) + (output_tokens_r2 / 1000 * LLM_COST_PER_1K_TOKENS_OUTPUT)

        #json_str = response.content.split('```json')[-1].split('```')[0].strip()
        #data = json.loads(json_str)
        
        data = _extract_json_from_response(response.content)

        if data is None:
            # The robust parser failed and already logged the error.
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
        logger.error(f"Error processing chunk: {e}", exc_info=True)
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