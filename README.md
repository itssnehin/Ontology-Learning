# Knowledge Graph Ontology Extraction & Schema Alignment

## Project Overview
This project proposes an automated pipeline for extracting ontologies from electrotechnical datasheets using a combination of classical Natural Language Processing (NLP) techniques and Large Language Models (LLMs). The primary objective is to convert unstructured, markdown-formatted documents into structured, machine-readable ontologies aligned with external standards, such as the International Electrotechnical Commission Common Data Dictionary (IEC CDD).

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
to do

## Technologies Used
-   **Programming Language:** Python
-   **NLP Libraries:** SpaCy
-   **LLM Integration:** OpenAI API (ChatGPT)

## Expected Results
-   A machine-readable ontology accurately representing knowledge of electronic components, aligned with IEC 61360 standards.
-   Ontologies expressed in OWL and serialized in Turtle/RDF/XML formats.
-   Performance metrics (precision, recall, F1 score, alignment accuracy) demonstrating the quality of the extraction process.
-   The ontology will capture core electrotechnical concepts (e.g., Resistor, Capacitor), properties (e.g., Rated Resistance, Tolerance), and relational structures (e.g., hasProperty, isSubtypeOf, alignedTo).

## Ethical and Privacy Considerations
This project does not involve any private, confidential, or personally identifiable information. All documents used are publicly available or generated for research purposes. Synthetic data is generated in accordance with OpenAI's terms of service and does not include or infer personal data. The project poses no ethical risk and does not require human participant involvement or ethical review.
