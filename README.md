
# End-to-End Ontology Learning Pipeline for Technical Documents

**Author:** Snehin Kukreja
**Course:** DATA7902 Capstone Project

## 1. Abstract

This project implements a complete, end-to-end pipeline for automated ontology learning from unstructured technical documents. The system addresses the "knowledge bottleneck," where valuable domain expertise is locked away in human-readable formats like PDFs and datasheets. By leveraging Large Language Models (LLMs), a hybrid AI decision engine, and a human-in-the-loop web interface, this pipeline transforms raw text into a structured, queryable, and standards-compliant knowledge graph in Neo4j. The final output is a formal ontology that captures not only concepts but also their taxonomic (hierarchical) and non-taxonomic (functional, compositional) relationships.

---

## 2. Key Features

-   **Automated PDF Processing:** Uses the `marker` library to convert complex, multi-column PDFs into clean Markdown.
-   **LLM-Powered Extraction:** Employs state-of-the-art models (e.g., GPT-4o) via zero-shot prompting to extract concepts and relationships.
-   **Hybrid AI Decision Engine:** A sophisticated `OntologyExtensionManager` that uses a multi-factor similarity score (semantic, lexical) and tunable thresholds to decide whether to map, merge, or create new concepts.
-   **Human-in-the-Loop Dashboard:** A Flask and JavaScript-based web UI that allows users to configure pipeline runs, monitor progress, and, most importantly, review and validate concepts that the AI is uncertain about.
-   **Graph-Based Knowledge Storage:** Builds and stores the learned ontology in a Neo4j graph database, using the `:OntologyClass` schema.
-   **Comprehensive Evaluation Suite:** Includes scripts for quantitative and qualitative analysis:
    -   **Conceptual Saturation:** To measure domain coverage.
    -   **Gold Standard Comparison:** To calculate Precision, Recall, and F1-score.
    -   **Model Performance Diagnosis:** To compare the extraction quality of different LLMs.
-   **Automated Curation:** Includes a post-processing script to programmatically clean the final graph by pruning noisy and disconnected nodes.

---

## 3. Technology Stack

| Category      | Technology                                                                                                                                                                                            |
| :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend**   | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)                                         |
| **Database**  | ![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)                                                                                                                           |
| **AI/ML**     | ![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-008664?logo=langchain&logoColor=white)                               |
| **Frontend**  | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black) |
| **Tooling**   | ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) ![PowerShell](https://img.shields.io/badge/PowerShell-5391FE?logo=powershell&logoColor=white) ![Visual Studio Code](https://img.shields.io/badge/VSCode-007ACC?logo=visualstudiocode&logoColor=white) |

---

## 4. System Architecture

The pipeline follows a four-stage process, transforming unstructured documents into a structured knowledge graph.

![System Architecture](visualizations/system_architecture.png)

---

## 5. Getting Started

### Prerequisites

-   Python 3.10+
-   Neo4j Desktop or Server (v5.x recommended)
-   An OpenAI API Key

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Set up the Python virtual environment:**
    ```powershell
    # Create the virtual environment
    python -m venv capstone-venv

    # Activate the environment
    .\capstone-venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    *   Create a file named `.env` in the project root directory.
    *   Add your credentials and configuration. It should look like this:
    ```env
    OPENAI_API_KEY="sk-..."
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="your_neo4j_password"
    NEO4J_DB_NAME="datasheetontology"
    ```

---

## 6. How to Use the Pipeline

### Step 1: Add Your Data

-   Convert your source PDFs into Markdown files using a tool like [Marker](https://github.com/VikParuchuri/marker).
-   Place all `.md` files into the `data/raw_markdown/` directory.

### Step 2: Initialize the Database

Before your first run, you must set up the baseline ontology schema in Neo4j.

```bash
# This will wipe the specified database and create the root classes.
python -m src.initialize_baseline --db-name datasheetontology
```
*(You will be prompted to type `BUILD` to confirm.)*

### Step 3: Start the Backend Server

This will launch the Flask application that powers the dashboard.

```bash
python -m src.ontology_management_backend
```

### Step 4: Use the Dashboard

-   Open your web browser and navigate to `http://localhost:5000`.
-   **Configure a Run:** On the "Pipeline Configuration" tab, select your desired LLM model, execution mode ("Start from Scratch" for the first run), and adjust any other settings.
-   **Run Pipeline:** Click the "Run Pipeline" button.
-   **Monitor Progress:** Watch the logs and progress bar for real-time updates.
-   **Review Concepts:** Navigate to the "Concept Review" tab to validate concepts the AI was uncertain about.
-   **Query the Graph:** Use the "QA System" tab to ask natural language questions about your newly built ontology.

### Step 5: Curation and Evaluation

After a pipeline run, you can use the provided scripts for analysis and cleaning.

```bash
# To run the conceptual saturation analysis
python -m src.evaluation.conceptual

# To run the multi-model quality comparison
python -m src.evaluation.diagnose_extraction_quality

# To prune and clean the final graph in Neo4j
python -m src.curation.graph_cleaner
```

---

## 7. Project Structure

```
.
├── data/
│   ├── raw_markdown/         # Input .md files go here
│   ├── integrated_output/    # Stores JSON results and cost logs
│   └── electronics_schema.owl # Formal axioms for consistency checks
├── src/
│   ├── curation/             # Scripts for cleaning the graph
│   │   └── graph_cleaner.py
│   ├── evaluation/           # Scripts for all evaluation tasks
│   │   ├── conceptual.py
│   │   ├── diagnose_extraction_quality.py
│   │   └── ...
│   ├── __init__.py
│   ├── cached_schema_org_pipeline.py # Orchestrates cached runs
│   ├── config.py             # Central configuration and API keys
│   ├── data_loader.py        # Loads and chunks documents
│   ├── idea_extractor.py     # Extracts concepts using an LLM
│   ├── integrated_schema_pipeline.py # The core pipeline logic
│   ├── initialize_baseline.py # Wipes and sets up the Neo4j database
│   ├── ontology_extension_manager.py # The hybrid AI decision engine
│   ├── ontology_management_backend.py # The Flask server and API
│   └── ...                   # Other pipeline modules
├── frontend/
│   └── templates/
│       └── dashboard.html    # The single-page web interface
├── visualizations/           # Output for generated plots and diagrams
├── capstone-venv/            # Python virtual environment
├── model_costs.json          # Cost data for different LLM models
└── requirements.txt          # Project dependencies
```
