# Knowledge Graph Ontology Extraction & Schema Alignment

## Project Overview
This project proposes an automated pipeline for extracting ontologies from electrotechnical datasheets using a combination of classical Natural Language Processing (NLP) techniques and Large Language Models (LLMs). The primary objective is to convert unstructured markdown-formatted documents into structured, machine-readable ontologies aligned with external standards like the International Electrotechnical Commission Common Data Dictionary (IEC CDD).

## Objective
- Design and develop an automated pipeline for ontology extraction from electrotechnical datasheets.
- Convert unstructured markdown documents into structured, machine-readable ontologies.
- Align extracted ontologies with external standards (e.g., IEC CDD).
- Reduce human intervention through automated evaluation, refinement, and alignment strategies.

## Scope
- Design and implement preprocessing workflows.
- Implement entity and ontology extraction using NLP and LLMs.
- Conduct performance evaluation using automated metrics and domain expert validation.
- Develop a reusable, standards-compliant extraction method applicable to technical domains.
- The dataset will be limited to electrotechnical documents written in English.

## Methodology
The project follows a hybrid methodology, integrating traditional NLP with LLM-based extraction and synthetic data generation. The workflow includes:
1.  **Data Preparation & Synthetic Data Generation:** Utilizes real-world technical documents (scientific texts, electrical component datasheets, business/technical reports) and synthetically generated content to ensure comprehensive coverage and variability.
2.  **Data Preprocessing:** Involves Markdown parsing and cleaning, sentence segmentation and tokenization, removal of irrelevant content, lemmatization, Part-of-Speech tagging, and stopword filtering.
3.  **Ontology Extraction Workflow:**
    *   **Token Extraction:** Identifies candidate tokens (nouns, noun phrases) as potential classes/entities and numerical expressions as attribute values.
    *   **Semantic Annotation:** Maps identified entities to semantic types and recognizes syntactic structures for relationships.
    *   **Extending Existing Ontologies:** Compares extracted concepts against the IEC CDD and proposes new classes/properties where no exact match exists.
    *   **LLM-Based Extractors:** Leverages OpenAI's ChatGPT API for enhancing extraction quality, especially for complex descriptive paragraphs, and generating ontology triples.
    *   **Schema Alignment:** Aligns extracted ontology elements with the IEC CDD using lexical matching, semantic type checking, and manual verification.
4.  **Evaluation and Validation:** Assesses the effectiveness of the extraction and alignment processes using metrics such as Precision, Recall, F1 Score, Ontology Coverage, and Alignment Accuracy.
5.  **Iterative Ontology Refinement:** Employs a feedback-driven process at both corpus-level and incremental document addition to refine the ontology.

## Technologies Used
-   **Programming Language:** Python
-   **NLP Libraries:** SpaCy
-   **LLM Integration:** OpenAI API (ChatGPT)
-   **Ontology Tools:** Protégé, OWL / RDF / Turtle, OWLReady2
-   **Graph Database (Exploration):** Neo4j
-   **Reasoners:** HermiT / Pellet
-   **Data Manipulation:** Pandas
-   **Visualization:** Matplotlib / Seaborn

## Expected Results
-   A machine-readable ontology accurately representing knowledge of electronic components, aligned with IEC 61360 standards.
-   Ontologies expressed in OWL and serialized in Turtle/RDF/XML formats.
-   Performance metrics (precision, recall, F1 score, alignment accuracy) demonstrating the quality of the extraction process.
-   The ontology will capture core electrotechnical concepts (e.g., Resistor, Capacitor), properties (e.g., Rated Resistance, Tolerance), and relational structures (e.g., hasProperty, isSubtypeOf, alignedTo).

## Ethical and Privacy Considerations
This project does not involve any private, confidential, or personally identifiable information. All documents used are publicly available or generated for research purposes. Synthetic data is generated in accordance with OpenAI's terms of service and does not include or infer personal data. The project poses no ethical risk and does not require human participant involvement or ethical review.
