# Ontology Extraction with LangChain and Pydantic

This project demonstrates how to extract structured information from technical datasheets and build an ontology using LangChain and Pydantic.

## Project Structure

```
.
├── data/
│   └── raw_markdown/       # Raw markdown datasheets
├── output/
│   ├── structured_json_langchain/    # Extracted JSON data
│   └── ontology_langchain/           # Generated ontology files
└── src/
    ├── extract_openai.py             # Original OpenAI extraction script
    ├── generate_ontology.py          # Original ontology generation script
    ├── langchain_ontology.py         # New LangChain + Pydantic script
    └── openai_demo.py                # Simple OpenAI demo
```

## Features

- **Structured Data Extraction**: Uses Pydantic models to define the structure of the data to be extracted
- **LangChain Integration**: Leverages LangChain for prompt templating and LLM interaction
- **Ontology Generation**: Converts extracted data into RDF triples using rdflib
- **Rich Relationship Modeling**: Captures component relationships, applications, and properties

## Pydantic Models

The project uses the following Pydantic models to structure the extracted data:

- `Property`: Represents a technical property with name and value
- `Relationship`: Defines relationships between components
- `Application`: Describes application domains for components
- `Component`: The main model representing an electronic component

## Ontology Structure

The generated ontology includes:

- Classes for component types and applications
- Properties for technical specifications
- Relationships between components
- Application domains
- Features as literal values

## Usage

1. Place markdown datasheets in the `data/raw_markdown/` directory
2. Set your OpenAI API key in the `.env` file
3. Run the extraction script:

```bash
python src/langchain_ontology.py
```

4. View the extracted JSON data in `output/structured_json_langchain/`
5. Explore the generated ontology in `output/ontology_langchain/langchain_ontology.ttl`

## Requirements

- Python 3.8+
- langchain
- langchain-openai
- pydantic
- rdflib
- python-dotenv
- openai

## Future Improvements

- Add support for more datasheet formats
- Implement ontology visualization
- Create a web interface for exploring the ontology
- Add validation rules for extracted data
- Support for batch processing of multiple datasheets