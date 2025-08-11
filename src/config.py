import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import sys
# Add src/ to sys.path
sys.path.append(str(Path(__file__).parent))

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define project root (directory containing src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OpenAI API settings
LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Neo4j settings
NEO4J_URI = "bolt://127.0.0.1:7687"  # Changed from neo4j:// to bolt://
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "ontology"
OWL_FILE = ""
# Base directory for the project
BASE_DIR = Path(__file__).parent.parent  # Assumes config.py is in src/, points to code/

# Data directory
DATA_DIR = BASE_DIR / "data"

# Dynamically find OWL files in data directory
OWL_FILES = list(DATA_DIR.glob("*.owl"))
if not OWL_FILES:
    raise FileNotFoundError(f"No OWL files found in {DATA_DIR}")

# Use the first OWL file (or specify a particular one if multiple)
OWL_FILE = OWL_FILES[0]

MARKDOWN_DIR = DATA_DIR / "raw_markdown"
# Other config settings (example, adjust as per your project)
MARKDOWN_FILES = list(MARKDOWN_DIR.glob("*.md"))
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "ontology"

print(f"Config loaded: OWL_FILE = {OWL_FILE}, found {len(OWL_FILES)} OWL files")
# Pipeline settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100