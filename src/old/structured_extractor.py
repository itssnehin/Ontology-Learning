from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.pydantic_models import Component
import os

# Initialize LangChain components
parser = PydanticOutputParser(pydantic_object=Component)

prompt_template = """
You are an expert in electronics engineering and knowledge graph creation.
Your task is to analyze a technical component datasheet, provided in Markdown format,
and extract key information into a structured format for ontology creation.

{format_instructions}

Extract the following information from the datasheet:
- Component type (general classification)
- Model number
- Manufacturer
- Technical properties (with name and value pairs, including units)
- Key features as a list of strings
- Applications where this component is used (name and optional description)
- Relationships to other components or concepts, including:
  * Type of relationship (e.g., "compatibleWith", "interfacesWith", "supportedBy")
  * Target component or concept name
  * Optional description of the relationship

Be precise and comprehensive in your extraction. For relationships, use standardized relationship types 
that would be appropriate for an ontology (e.g., "compatibleWith", "partOf", "requires").

Here is the datasheet content:
{datasheet_content}
"""

prompt = ChatPromptTemplate.from_template(
    template=prompt_template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Check if we have an API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key.strip():
    # Initialize the LLM with OpenAI
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0
    )
else:
    # Use a mock LLM for testing
    from langchain_core.language_models.fake import FakeListLLM
    
    # Create a mock response for our sample component
    mock_responses = [
        """
        {
          "component_type": "Field-Programmable Gate Array",
          "model_number": "XC4010E",
          "manufacturer": "Xilinx",
          "properties": [
            {
              "name": "Logic Cells",
              "value": "10,000"
            }
          ],
          "features": [
            "In-system programmability"
          ],
          "applications": [
            {
              "name": "Digital signal processing"
            }
          ],
          "relationships": [
            {
              "type": "compatibleWith",
              "target": "XC4000E series programming tools"
            }
          ]
        }
        """
    ]
    
    llm = FakeListLLM(responses=mock_responses)
    print("Using mock LLM for testing as no OpenAI API key was provided.")

# Chain for extraction
extraction_chain = prompt | llm | parser

def extract_component_from_chunk(chunk_content: str) -> Optional[Dict[str, Any]]:
    """Extract structured data from a single text chunk using LangChain."""
    if not chunk_content:
        return None
    
    print("-> Processing a chunk with LangChain...")
    try:
        # Extract structured data using LangChain
        result = extraction_chain.invoke({"datasheet_content": chunk_content})
        print("-> Successfully extracted and parsed data from chunk.")
        return result.model_dump()
    except Exception as e:
        print(f"An error occurred during chunk extraction: {e}")
        return None
