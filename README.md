
---
# An Intelligent Pipeline for Automated Ontology Learning

## 🎯 Project Goal

This project implements an advanced data science pipeline for **ontology learning**. Its primary function is to automatically **generate and extend a formal class hierarchy** (an ontology) by processing a corpus of unstructured technical documents (e.g., component datasheets).

The system identifies candidate concepts, intelligently decides whether they represent new knowledge, and places them within a coherent, hierarchical structure rooted in Schema.org principles. The entire process is managed and monitored through a web-based dashboard.

---

## 🏗️ System Architecture & Learning Workflow

The pipeline operates on a "learning-first" paradigm, where the ontology schema itself is the primary output.

1.  **Input:** A collection of Markdown files in `data/raw_markdown/`.
2.  **Concept Identification:** A zero-shot LLM prompt (`idea_extractor`) reads document chunks and identifies a "bag of concepts" (potential classes).
3.  **Ontology Learning Engine (`OntologyExtensionManager`):**
    *   For each candidate concept, it uses a hybrid similarity model (semantic embeddings, lexical matching) to compare it against all known classes in the Neo4j database.
    *   **EXTEND Decision:** If the concept is novel, the engine infers the most likely parent class from the existing hierarchy.
    *   **MAP Decision:** If the concept is a synonym or exact match for an existing class, it is ignored to prevent duplicates.
    *   **Non-Taxonomic Relations:** A secondary LLM prompt extracts relationships like `hasProperty`, `connectedTo`, etc., enriching the learned structure.
4.  **Ontology Task Generation:** The engine's decisions are converted into a list of explicit tasks (e.g., `{'action': 'CREATE_CLASS', 'name': 'Varactor Diode', 'parent_class': 'Diode', 'non_taxonomic_relations': [...]}`).
5.  **Graph Building:** The `SchemaOrgGraphBuilder` connects to Neo4j and executes these tasks, creating new `:OntologyClass` nodes and linking them into the hierarchy with `:SUBCLASS_OF` and other non-taxonomic relationships.
6.  **Output:** A single, unified, and expanded class hierarchy stored in the target Neo4j database.

---

## 🔧 Installation & Setup

### Prerequisites
-   Python 3.9+
-   Neo4j Desktop (with the APOC plugin installed)
-   Java (for the OWL consistency checker)

### Installation
1.  **Clone the repository** and navigate into the `code/` directory.
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure environment:**
    *   Copy the example file: `cp .env.example .env`
    *   Edit the new `.env` file and add your `OPENAI_API_KEY`.

### Neo4j Setup
1.  Open Neo4j Desktop and create a new project.
2.  Create a new database instance. The default credentials in `.env` are `neo4j` / `ontology`.
3.  **Crucially, create the database your pipeline will write to.** The default is `datasheetontology`.
4.  Install the **APOC** plugin via the "Plugins" tab for your database instance.
5.  Start the database.

---

## 🚀 How to Use the System: The Golden Workflow

Follow these steps in order to run the pipeline and see the results.

### Step 1: Initialize the Ontology Database

Before the first run, you must create the baseline hierarchy. This command wipes the target database and sets up the root classes (`Thing`, `Product`, etc.).

```bash
# Run from the 'code/' directory
python -m src.initialize_baseline --db-name "datasheetontology"
```
Type `BUILD` to confirm.

### Step 2: Start the Backend Server & Dashboard

This command launches the Flask web server, which provides the API and the user interface.

```bash
# Run from the 'code/' directory
python -m src.ontology_management_backend
```
Open your web browser and navigate to **`http://localhost:5000`**.

### Step 3: Run the Pipeline via the Dashboard

1.  On the "Pipeline Configuration" tab, select your desired settings.
2.  For the very first run, **always** choose **"Start from Scratch (Full Run)"** from the "Execution Mode" dropdown. This ensures all caches are cleared and the data is processed against the fresh baseline.
3.  Click the **"🚀 Run Pipeline"** button.
4.  Monitor the progress in the "Pipeline Progress" section and the log viewer.

### Step 4: Review and Curate the Learned Ontology

Once the pipeline is complete:
1.  Navigate to the **"Concept Review"** tab.
2.  The table will be populated with all the new classes the system learned but was uncertain about (marked with `:NeedsReview`).
3.  Use the **"Accept"** button for valid, correctly placed concepts to formalize them in the ontology.
4.  Use the **"Reject"** button for irrelevant or incorrectly extracted terms (e.g., addresses, part numbers) to remove them.

### Step 5: Query and Explore the Result

1.  **Using the QA System:** Go to the "QA System" tab on the dashboard and ask natural language questions about your newly built ontology.
2.  **Using Neo4j Browser:** Connect to your `datasheetontology` database and run Cypher queries to visualize the learned hierarchy:
    ```cypher
    // See the entire learned hierarchy
    MATCH path = (c:OntologyClass)-[:SUBCLASS_OF*]->(root:Thing)
    RETURN path

    // See both taxonomic and non-taxonomic relations for a specific class
    MATCH (c:OntologyClass {name: 'Varactor Diode'})-[r]-()
    RETURN c, r
    ```

