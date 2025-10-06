
---

# Schema.org Ontology Extraction & Intelligent Extension Pipeline

## 🎯 Project Overview

This project implements an intelligent pipeline for extracting Schema.org ontologies from technical datasheets and automatically deciding whether to extend the ontology with new concepts or map to existing ones. The system combines large language models (LLMs), embedding-based similarity matching, and domain-specific technical property analysis to maintain ontology quality while ensuring comprehensive coverage.

## 🏗️ System Architecture

```
Datasheets (Markdown) → Concept Extraction → Ontology Decision Engine → Schema.org Objects → Neo4j Knowledge Graph
                           ↓                        ↓                        ↓
                    LLM Processing        Multi-Method Similarity      JSON-LD Generation
                                         + LLM Validation
```

## 🔬 Academic Contributions

### Novel Methodology
- **Hybrid Similarity Matching**: Combines semantic embeddings with domain-specific technical property matching.
- **Adaptive Decision Thresholds**: Dynamic adjustment based on ontology maturity and category density.
- **LLM-Powered Validation**: GPT-4 reasoning for ambiguous ontological decisions with explainable AI.
- **Technical Domain Specialization**: Electronic component property matching (frequency, impedance, connectors).

### Research Impact
- **Ontology Engineering**: Systematic approach to large-scale ontology evolution and maintenance.
- **Knowledge Graph Construction**: Automated quality preservation during rapid ontology growth.
- **Human-AI Collaboration**: Effective integration of automated decisions with expert oversight.
- **Reproducible Framework**: Comprehensive evaluation metrics and configurable parameters.

## 📁 Project Structure

```
code/
├── src/                               # Core application package
│   ├── __init__.py                    # Makes 'src' a package
│   ├── evaluation/                    # Evaluation scripts sub-package
│   │   ├── __init__.py                # Makes 'evaluation' a package
│   │   ├── gold_standard.py           # Precision/Recall/F1 evaluation
│   │   ├── consistency.py             # Logical consistency (OWL reasoner)
│   │   └── graph_evaluation.py        # Graph metrics comparison
│   ├── data_loader.py                 # Document loading and preprocessing
│   ├── idea_extractor.py              # LLM-based concept extraction
│   ├── ontology_extension_manager.py  # ⭐ Smart extension decisions
│   ├── integrated_schema_pipeline.py  # ⭐ Main pipeline orchestrator
│   ├── cached_schema_org_pipeline.py  # Resumable pipeline for development
│   └── ... (and other modules)
│
├── data/                              # Data directory
│   ├── raw_markdown/                  # Input datasheet files
│   ├── gold_standard.json             # Ground truth for evaluation
│   ├── electronics_schema.owl         # Formal schema for consistency checks
│   └── integrated_output/             # Timestamped pipeline results
│
├── frontend/                          # Web dashboard files
│   └── templates/
│       └── dashboard.html
│
├── cache/                             # Cached intermediate results
├── logs/                              # Application and pipeline logs
└── visualizations/                    # Generated charts and graphs
```

## 🚀 Key Features

### 1. Intelligent Ontology Extension Management
- **Multi-Method Similarity**: Embedding, lexical, technical specification, and category-based matching.
- **Confidence-Weighted Decisions**: Automated high-confidence decisions with manual review for ambiguous cases.
- **Technical Property Matching**: Specialized matchers for frequency ranges, impedance values, connector types.
- **LLM Validation**: GPT-4 expert reasoning for complex ontological decisions.

### 2. Comprehensive & Parallelized Pipeline
- **Concurrent Processing**: Employs multithreading for all I/O-bound tasks (LLM calls, database writes) for maximum performance.
- **Resumable Workflow**: Caching system allows the pipeline to be resumed from any stage, saving time and API costs during development.
- **Configurable Prompts**: All LLM prompts are centralized in a JSON file for easy tuning and experimentation.

### 3. Advanced Evaluation & Visualization Suite
- **Quantitative Metrics**: Calculates precision, recall, and F1-score against a gold standard.
- **Structural Analysis**: Compares graph-level metrics (density, connectivity) between the generated and gold standard ontologies.
- **Logical Consistency**: Uses an OWL reasoner (HermiT) to validate the ontology against a formal schema, ensuring no logical contradictions.
- **Interactive Dashboards**: Explorable visualizations for academic presentation.

## 🚀 Streamlining the Pipeline for Efficiency

Ontology extraction, especially with large language models, can be time-consuming and expensive. This project includes several features and best practices to streamline the process.

### Subgraph Generation and Merging
The pipeline processes documents by generating "subgraphs" (in-memory sets of concepts and relations) and then intelligently merging them into the main knowledge graph using Neo4j's `MERGE` functionality to prevent duplication.
```
[Document Chunks]
       |
       v
[Extractors: idea_extractor, relation_extractor]
       |
       +--> "Subgraph A" (in memory)
       +--> "Subgraph B"
       |
       v
[Graph Builder: schema_org_graph_builder]
       |
       |  (Takes Subgraph A)
       v
[Neo4j Database] --- MERGE node "Concept 1" --> (Node is created)
       |
       |  (Takes Subgraph B, which also has "Concept 1")
       v
[Neo4j Database] --- MERGE node "Concept 1" --> (Node already exists, do nothing)
```

### 1. Caching LLM and Embedding Results
The `cached_schema_org_pipeline.py` script saves the output of each major stage to the `cache/` directory, allowing you to re-run the pipeline from any point without reprocessing everything. For most development work, **the cached pipeline should be your default**.

### 2. Parallel Execution
Using Python's `concurrent.futures.ThreadPoolExecutor`, all I/O-bound tasks (API calls, database writes) are run in parallel, drastically reducing total runtime. The level of concurrency can be tuned in `config.py`.

### 3. Efficient Database Transactions
All graph updates for a batch of concepts are wrapped in a single Neo4j transaction, providing a significant performance boost over single-query approaches.

### 4. Ontology Management Dashboard
The backend server (`ontology_management_backend.py`) and frontend (`dashboard.html`) provide a UI for managing the pipeline, reviewing uncertain concepts, and visualizing the ontology's state.

## 🔬 Evaluation Framework

This project includes a multi-faceted evaluation framework located in the `src/evaluation/` package to ensure the quality, accuracy, and consistency of the generated ontology.

### 1. Gold Standard Comparison (`gold_standard.py`)
- **What it does:** Calculates **Precision, Recall, and F1-Score** for both extracted concepts and relationships.
- **How it works:** It compares the pipeline's output against a manually created `data/gold_standard.json` file, which represents the "perfect" extraction for a subset of documents.

### 2. Graph-Based Structural Analysis (`graph_evaluation.py`)
- **What it does:** Compares the high-level structural properties of the generated graph and the gold standard graph.
- **How it works:** It uses the `networkx` library to calculate and compare metrics like **node/edge counts, density, and average degree**, providing insight into the overall shape and connectivity of the ontology.

### 3. Logical Consistency Validation (`consistency.py`)
- **What it does:** Checks the generated ontology for logical contradictions using a formal reasoner.
- **How it works:** It uses the `OwlReady2` library and the HermiT reasoner to validate the extracted facts against a set of rules defined in a formal schema (`data/electronics_schema.owl`), such as class disjointness and property constraints.

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- Java (for the consistency checker)
- Neo4j Desktop

### Installation
```bash
# Clone repository and navigate into the 'code' directory
git clone <repository-url>
cd code

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the project and its dependencies in editable mode
pip install -e .
pip install -r requirements.txt

# Configure environment by copying the example and editing it
cp .env.example .env
# Edit .env with your OpenAI API key and Neo4j credentials
```

### Neo4j Setup
1.  Install and run Neo4j Desktop.
2.  Create a new database with the credentials specified in your `.env` file (default is `neo4j:ontology`).
3.  Ensure the database is started.

## 🚀 Usage

### Recommended Development Workflow
Use the cached pipeline to save time and API costs.
```bash
# Run the pipeline, resuming from the concept extraction step
python -m src.cached_schema_org_pipeline --resume-from concepts
```
See the script's `--help` for all resume options.

### Running a Full, Clean Pipeline
```bash
# Run the complete integrated pipeline from scratch
python -m src.integrated_schema_pipeline```

### Running Evaluations
```bash
# Run gold standard evaluation (precision/recall)
python -m src.evaluation.gold_standard "data/integrated_output/YOUR_OUTPUT_FILE.jsonld"

# Run graph metrics evaluation
python -m src.evaluation.graph_evaluation "data/integrated_output/YOUR_OUTPUT_FILE.jsonld"

# Run logical consistency evaluation
python -m src.evaluation.consistency "data/integrated_output/YOUR_OUTPUT_FILE.jsonld"
```

## 📞 Contact

For questions about this research or collaboration opportunities:
- **Project Lead**: Snehin Kukreja
- **Institution**: University of Queensland, DATA7902 Capstone Project
- **Research Area**: Ontology Engineering, Knowledge Graph Construction, LLM Applications

---