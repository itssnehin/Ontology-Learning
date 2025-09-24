"""
Centralized configuration management for the ontology learning system.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Define project root (directory containing src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env
dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path)

# --- Environment-loaded variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

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

# --- Create directories if they don't exist ---
DATA_DIR.mkdir(exist_ok=True)
MARKDOWN_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# --- OpenAI API settings ---
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- Pipeline settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Similarity thresholds ---
SIMILARITY_THRESHOLDS = {
    'exact_match': 0.95,
    'high_similarity': 0.85,
    'medium_similarity': 0.70,
    'low_similarity': 0.50
}

# --- Dynamically find files ---
OWL_FILES = list(DATA_DIR.glob("*.owl"))
MARKDOWN_FILES = list(MARKDOWN_DIR.glob("*.md"))

# Get primary OWL file
OWL_FILE = OWL_FILES[0] if OWL_FILES else None

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Initial configuration logging ---
if not OWL_FILES:
    logger.warning(f"No OWL files found in {DATA_DIR}")

if not MARKDOWN_FILES:
    logger.warning(f"No markdown files found in {MARKDOWN_DIR}")

logger.info(f"Config loaded: OWL_FILE = {OWL_FILE}, found {len(OWL_FILES)} OWL files")