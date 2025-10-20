"""
Centralized configuration management for the ontology learning system.
"""

import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# --- 1. LOGGING SETUP MOVED TO THE TOP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "pipeline.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- 2. THE REST OF THE CONFIGURATION ---

# Define project root
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables
dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path)

# --- Environment-loaded variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
NEO4J_DB_NAME = os.getenv("NEO4J_DB_NAME", "datasheetontology") 

# --- Validation ---
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
if not NEO4J_URI:
    raise ValueError("NEO4J_URI not found in .env file.")

# --- Base directories ---
BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MARKDOWN_DIR = DATA_DIR / "raw_markdown"
CACHE_DIR = BASE_DIR / "cache"

# --- Create directories ---
DATA_DIR.mkdir(exist_ok=True)
MARKDOWN_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# --- OpenAI API settings ---
LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-ada-002"

EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.0001,
    "default": 0.0001
}

MODEL_COSTS = {}
try:
    costs_path = PROJECT_ROOT / "model_costs.json"
    with open(costs_path, 'r', encoding='utf-8') as f:
        MODEL_COSTS = json.load(f)
    logger.info("✅ Successfully loaded model costs from model_costs.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"FATAL: Could not load or parse model_costs.json. Costs will be zero. Error: {e}")
    MODEL_COSTS = {"default": {"input_cost_per_1k_tokens": 0, "output_cost_per_1k_tokens": 0}}

# --- Pipeline settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_WORKERS = 10

# --- Similarity thresholds ---
SIMILARITY_THRESHOLDS = {
    'exact_match': 0.96,
    'high_similarity': 0.90,
    'medium_similarity': 0.75,
    'low_similarity': 0.60
}

# --- Dynamically find files ---
OWL_FILES = list(DATA_DIR.glob("*.owl"))
MARKDOWN_FILES = list(MARKDOWN_DIR.glob("*.md"))
OWL_FILE = OWL_FILES[0] if OWL_FILES else None

# --- Load Prompts from JSON ---
PROMPTS_PATH = BASE_DIR / "src" / "prompts.json"
PROMPTS = {}
try:
    with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
        PROMPTS = json.load(f)
    logger.info("✅ Successfully loaded configurable prompts from prompts.json")
except Exception as e:
    logger.error(f"FATAL: Could not load or parse prompts.json. Error: {e}")
    raise

NON_TAXONOMIC_RELATION_PROMPT = ""
try:
    non_tax_path = Path(__file__).parent / "prompts" / "non_taxonomic_relation_prompt.txt"
    NON_TAXONOMIC_RELATION_PROMPT = non_tax_path.read_text(encoding="utf-8")
    logger.info("Successfully loaded the non-taxonomic relation prompt")
except Exception as e:
    logger.error(f"Could not load non-taxonomic relation prompt: {e}")

logger.info(f"Config loaded: OWL_FILE = {OWL_FILE}, found {len(OWL_FILES)} OWL files")