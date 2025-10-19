"""
Centralized configuration management for the ontology learning system.
"""

import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# --- 1. LOGGING SETUP MOVED TO THE TOP ---
# This ensures the logger is available for the rest of the file to use.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
    handlers=[
        # Add encoding='utf-8' to both handlers
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "pipeline.log", encoding='utf-8'),
        logging.StreamHandler() # The stream handler will pick up the system's encoding, but we can be explicit if needed.
    ]
)
# Set the log level for noisy libraries to WARNING to clean up the console output.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING) # <-- ADD THIS LINE

logger = logging.getLogger(__name__)


# --- 2. THE REST OF THE CONFIGURATION ---

# Define project root (directory containing src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env
dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path)

# --- Environment-loaded variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ontology")
NEO4J_DB_NAME = os.getenv("NEO4J_DB_NAME", "datasheetontology") 

# --- Validation ---
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
if not NEO4J_URI:
    raise ValueError("NEO4J_URI not found in environment variables. Please set it in your .env file.")

# --- Base directories ---
BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MARKDOWN_DIR = DATA_DIR / "raw_markdown"
CACHE_DIR = BASE_DIR / "cache"

# --- Create directories if they don't exist ---
DATA_DIR.mkdir(exist_ok=True)
MARKDOWN_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# --- OpenAI API settings ---
LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Prices are per 1,000 tokens, converted from per 1M.
# --- NEW: ADD EMBEDDING COSTS ---
EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.0001,
    "default": 0.0001 # Default to ada-002 price
}

# --- ADD THIS BLOCK TO LOAD THE CHAT MODEL COSTS ---
MODEL_COSTS = {}
try:
    # Assuming model_costs.json is in the root, next to this config file's parent dir
    costs_path = PROJECT_ROOT / "model_costs.json"
    with open(costs_path, 'r', encoding='utf-8') as f:
        MODEL_COSTS = json.load(f)
    logger.info("✅ Successfully loaded model costs from model_costs.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"FATAL: Could not load or parse model_costs.json. Costs will be zero. Error: {e}")
    # Provide a default structure to prevent crashes
    MODEL_COSTS = {
        "default": {"input_cost_per_1k_tokens": 0, "output_cost_per_1k_tokens": 0}
    }

# --- Pipeline settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_WORKERS = 10 #Parallelisation

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

# Get primary OWL file
OWL_FILE = OWL_FILES[0] if OWL_FILES else None

# --- Load Prompts from JSON ---
PROMPTS_PATH = BASE_DIR / "src" / "prompts.json"
PROMPTS = {}
try:
    with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
        PROMPTS = json.load(f)
    logger.info("✅ Successfully loaded configurable prompts from prompts.json")
except FileNotFoundError:
    logger.error(f"FATAL: prompts.json not found at {PROMPTS_PATH}. The application cannot continue.")
    raise
except json.JSONDecodeError:
    logger.error(f"FATAL: Could not parse prompts.json. Please check for JSON syntax errors.")
    raise


NON_TAXONOMIC_RELATION_PROMPT = ""
try:
    non_tax_path = Path(__file__).parent / "prompts" / "non_taxonomic_relation_prompt.txt"
    NON_TAXONOMIC_RELATION_PROMPT = Path(non_tax_path).read_text(encoding="utf-8")
    logger.info("Successfully loaded the non-taxonomic relation prompt")
except Exception as e:
    logger.error(f"Could not load non taxonomic relation prompt: {e}")


# --- Initial configuration logging ---
if not OWL_FILES:
    logger.warning(f"No OWL files found in {DATA_DIR}")

if not MARKDOWN_FILES:
    logger.warning(f"No markdown files found in {MARKDOWN_DIR}")

logger.info(f"Config loaded: OWL_FILE = {OWL_FILE}, found {len(OWL_FILES)} OWL files")