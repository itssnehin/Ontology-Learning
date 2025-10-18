import logging
import json
import re
from typing import List, Optional, Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from tiktoken import get_encoding
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import (
    LLM_MODEL, OPENAI_API_KEY, PROMPTS, MAX_WORKERS, MODEL_COSTS
)

from src.config import (
    LLM_MODEL, OPENAI_API_KEY, PROMPTS, MAX_WORKERS, MODEL_COSTS
)

logger = logging.getLogger(__name__)

CONCEPT_STOPWORDS = {
    'figure', 'table', 'application', 'system', 'section', 'part', 'example',
    'information', 'note', 'introduction', 'overview', 'description', 'feature',
    'copyright', 'inc', 'page', 'date', 'january', 'february', 'march', 'april',
    'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
    # Add common irrelevant places found in the dataset
    'california', 'texas', 'germany', 'japan', 'china', 'india', 'france'
}


def _extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """More robustly extracts a JSON object from a string that might contain other text."""
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.warning("Found a JSON code block, but it was malformed.")
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}. Raw text was: {text}")
    return None

def _is_valid_concept(concept: str) -> bool:
    """Applies a set of rules to filter out low-quality or noisy concepts."""
    if not concept or not isinstance(concept, str):
        return False
        
    concept_lower = concept.lower()
    
    if concept_lower in CONCEPT_STOPWORDS: return False
    if len(concept) < 3: return False
    if concept.isnumeric() or not re.search(r'[a-zA-Z]', concept): return False
    if '/' in concept or 'www.' in concept or '.com' in concept: return False
        
    return True

def _extract_concepts_from_chunk(chunk: Document, model_name: str) -> Tuple[Optional[List[str]], float, float]:
    """Processes a single document chunk using a system prompt and returns concepts and cost."""
    input_tokens, output_tokens = 0, 0
    try:
        llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        tokenizer = get_encoding("cl100k_base")
        
        system_prompt_template = PROMPTS["idea_extractor"]["system_prompt"]
        user_prompt_template = PROMPTS["idea_extractor"]["user_prompt"]
        user_prompt = user_prompt_template.format(chunk_content=chunk.page_content)
        
        messages = [
            SystemMessage(content=system_prompt_template),
            HumanMessage(content=user_prompt)
        ]

        # Combine prompts for token calculation
        full_prompt_text = system_prompt_template + user_prompt
        input_tokens = len(tokenizer.encode(full_prompt_text))
        
        response = llm.invoke(messages)
        output_tokens = len(tokenizer.encode(response.content))

        data = _extract_json_from_response(response.content)
        
        if data is None:
            return None, input_tokens, output_tokens

        chunk_concepts = []
        if "nodes" in data and isinstance(data["nodes"], list):
            # Apply the validation filter here
            filtered_concepts = [concept for concept in data["nodes"] if _is_valid_concept(concept)]
            chunk_concepts.extend(filtered_concepts)
        else:
            logger.warning(f"Valid JSON parsed, but 'nodes' key is missing or not a list. Data: {data}")
            
        return chunk_concepts, input_tokens, output_tokens

    except Exception as e:
        logger.error(f"An unhandled error occurred while processing chunk: {e}", exc_info=True)
        return None, input_tokens, output_tokens


def extract_ideas(chunks: List[Document], model_name: str = LLM_MODEL, max_workers: int = MAX_WORKERS) -> Tuple[List[str], int, int]:
    """Extracts concepts concurrently and returns concepts and token counts."""
    total_input_tokens = 0
    total_output_tokens = 0
    all_concepts = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_concepts_from_chunk, chunk, model_name): chunk for chunk in chunks}
        for future in tqdm(as_completed(futures), total=len(chunks), desc="Extracting Concepts (Parallel)"):
            concepts, in_tokens, out_tokens = future.result()
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            if concepts:
                all_concepts.extend(concepts)

    deduplicated_concepts = sorted(list(set(all_concepts)))
    logger.info(f"Extracted {len(deduplicated_concepts)} unique concepts after filtering.")
    
    return deduplicated_concepts, total_input_tokens, total_output_tokens
