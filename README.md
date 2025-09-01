# Getting Started with ECLASS Ontology Pruning Project

This repository contains scripts and tools for pruning the ECLASS ontology (version 5.1.4) to focus on electrotechnical ideas (segment 27) as part of the "Ontology-Guided Information Extraction & Extension for Electrotechnical Datasheets" project. Follow the steps below to set up your environment, install dependencies, and run the scripts.

## Prerequisites
- **Python 3.9+**: Ensure Python is installed (download from [python.org](https://www.python.org/downloads/)).
- **Neo4j Desktop**: Install from [neo4j.com/download](https://neo4j.com/download/) for database operations.
- **Git** (optional): For version control (download from [git-scm.com](https://git-scm.com/)).

## Project Structure
- `src/`: Contains Python scripts (e.g., `prune_ontology.py`, `test_neo4j.py`, `prune_ontology_from_corpus.py`).
- `data/`: Holds OWL files (e.g., `eclass_514en.owl`) and markdown datasheets (e.g., in `raw_markdown/`).
- `config.py`: Configuration file for Neo4j and file paths.

## Setting Up a Virtual Environment
1. **Navigate to Project Directory**:
   - Open a terminal (e.g., PowerShell) and cd to the project root:
     ```
     cd G:\Other computers\My Laptop (1)\Google Drive\University of Queensland - 2024\Semester 4\DATA7902\code
     ```

2. **Create Virtual Environment**:
   - Run the following command to create a virtual environment named `capstone-venv`:
     ```
     python -m venv capstone-venv
     ```

3. **Activate Virtual Environment**:
   - On Windows (PowerShell):
     ```
     .\capstone-venv\Scripts\Activate.ps1
     ```
   - You should see `(capstone-venv)` in your prompt.

4. **Deactivate (when done)**:
   - Run `deactivate` to exit the virtual environment.

## Installing Dependencies
1. **Install Required Packages**:
   - With the virtual environment activated, install dependencies listed in `requirements.txt` (create this file if not present) by running:
     ```
     pip install -r requirements.txt
     ```
   - If `requirements.txt` is missing, use:
     ```
     pip install rdflib neo4j gensim scikit-learn numpy
     ```

2. **Verify Installation**:
   - Check Python packages with `pip list` to ensure all are installed.

## Configuration
1. **Set Up `config.py`**:
   - Create `src/config.py` with the following content, adjusting paths as needed:
     ```python
     NEO4J_URI = "bolt://127.0.0.1:7687"
     NEO4J_USERNAME = "neo4j"
     NEO4J_PASSWORD = "ontology"
     OWL_FILE = "G:\\Other computers\\My Laptop (1)\\Google Drive\\University of Queensland - 2024\\Semester 4\\DATA7902\\code\\data\\eclass_514en.owl"
     ```
   - Ensure the `OWL_FILE` path matches your local setup.

2. **Start Neo4j**:
   - Open Neo4j Desktop, select your database, and click "Start". Verify at http://localhost:7474 with `neo4j:ontology`.

## Running the Scripts
### 1. `test_neo4j.py`
- **Purpose**: Tests Neo4j connection and basic functionality.
- **How to Run**:
  - Activate the virtual environment (if not already):
    ```
    .\capstone-venv\Scripts\Activate.ps1
    ```
  - Navigate to `src`:
    ```
    cd src
    ```
  - Execute:
    ```
    python test_neo4j.py
    ```
- **Expected Output**: "Connection successful: 1" if Neo4j is running.

### 2. `prune_ontology.py`
- **Purpose**: Imports the ECLASS OWL file into Neo4j and prunes to segment 27.
- **How to Run**:
  - Ensure Neo4j is running.
  - From `src`:
    ```
    python prune_ontology.py
    ```
- **Expected Output**: Class counts and pruning results (e.g., "Non-electrotechnical nodes pruned.").

### 3. `prune_ontology_from_corpus.py`
- **Purpose**: Prunes the ECLASS ontology based on a corpus of markdown datasheets in `data/raw_markdown`.
- **How to Run**:
  - Place markdown files in `data/raw_markdown/`.
  - From `src`:
    ```
    python prune_ontology_from_corpus.py
    ```
- **Expected Output**: Corpus size, model training confirmation, and pruned ontology save location (e.g., `data/eclass_514en_pruned.owl`).

## Troubleshooting
- **Neo4j Connection Failed**: Ensure Neo4j is running and the port (7687) is open. Check `config.py` paths.
- **Module Not Found**: Verify `pip install` completed successfully.
- **File Not Found**: Confirm `OWL_FILE` and markdown file paths in `config.py` and directory structure.

## Additional Notes
- Update `requirements.txt` with any new packages as you develop.
- Backup `data/eclass_514en.owl` before running pruning scripts.
- For pipeline integration, use the pruned OWL in Neo4j or further processing.

Happy pruning!