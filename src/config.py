import os
import glob
from dotenv import load_dotenv
from pathlib import Path

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

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_markdown")
os.makedirs(DATA_DIR, exist_ok=True)
MARKDOWN_FILES = [
    os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, "*.md"))
]

# Pipeline settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100