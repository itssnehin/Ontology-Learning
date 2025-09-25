import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import re
from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from tiktoken import get_encoding

from src.config import LLM_MODEL, OPENAI_API_KEY
from src.utils import setup_logging

class SchemaOrgExtractor:
    """Extract Schema.org JSON-LD markup from document chunks for electronic components."""
    
    def __init__(self, model_name: str = LLM_MODEL):
        setup_logging("../logs", "schema_org_extractor")
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        self.tokenizer = get_encoding("cl100k_base")
        self.cost_per_1k_tokens = 0.00336  # GPT-4o cost
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Product Types Ontology base URL
        self.product_ontology_base = "http://www.productontology.org/id/"
        
        # Common electronic component mappings
        self.component_mappings = {
            "antenna": "Antenna_(radio)",
            "wifi antenna": "Antenna_(radio)",
            "fpc antenna": "Flexible_printed_circuit",
            "pcb": "Printed_circuit_board",
            "connector": "Electrical_connector",
            "cable": "Coaxial_cable",
            "resistor": "Resistor",
            "capacitor": "Capacitor",
            "inductor": "Inductor",
            "diode": "Diode",
            "transistor": "Transistor",
            "sensor": "Sensor",
            "module": "Electronic_module",
            "circuit": "Electronic_circuit"
        }
    
    def extract_schema_org_data(self, chunks: List, concepts: List[str]) -> List[Dict]:
        """
        Extract Schema.org JSON-LD markup for concepts found in document chunks.
        
        Args:
            chunks: List of document chunks
            concepts: List of extracted concepts/ideas
            
        Returns:
            List of Schema.org JSON-LD objects
        """
        schema_objects = []
        
        for concept in concepts:
            # Find relevant context from chunks
            context = self._find_concept_context(concept, chunks)
            
            # Generate Schema.org markup
            markup = self._generate_schema_markup(concept, context)
            if markup:
                schema_objects.append(markup)
        
        print(f"Generated Schema.org markup for {len(schema_objects)} concepts")
        print(f"Total API cost: ${self.total_cost:.6f}")
        
        return schema_objects
    
    def _find_concept_context(self, concept: str, chunks: List) -> str:
        """Find chunks that contain context about the concept."""
        relevant_chunks = []
        concept_lower = concept.lower()
        
        for chunk in chunks:
            chunk_text = chunk.page_content.lower()
            # Check if concept appears in chunk
            if concept_lower in chunk_text:
                relevant_chunks.append(chunk.page_content)
        
        # Combine relevant chunks (limit to avoid token limits)
        context = " ".join(relevant_chunks[:3])  # Use first 3 relevant chunks
        return context[:2000] if context else f"Electronic component: {concept}"
    
    def _generate_schema_markup(self, concept: str, context: str) -> Optional[Dict]:
        """Generate Schema.org JSON-LD markup for a concept."""
        
        prompt = f"""
        Create Schema.org JSON-LD markup for this electronic component based on the context provided.
        
        Component: {concept}
        Context: {context}
        
        Requirements:
        1. Use "Product" as the base @type
        2. Include standard Schema.org properties: name, description, category
        3. Add "additionalType" from Product Types Ontology when applicable
        4. Include technical properties as custom properties with "elec:" prefix
        5. Extract specifications like frequency, impedance, dimensions where available
        6. Format as valid JSON-LD
        
        Example format:
        {{
          "@context": {{
            "@vocab": "https://schema.org/",
            "elec": "https://example.org/electrical/"
          }},
          "@type": "Product",
          "name": "{concept}",
          "description": "Technical description from context",
          "category": "Electronic Component",
          "additionalType": "http://www.productontology.org/id/Appropriate_Type",
          "elec:technicalProperty": "value if available"
        }}
        
        OUTPUT: Valid JSON-LD only, no other text.
        """
        
        try:
            input_tokens = len(self.tokenizer.encode(prompt))
            response = self.llm.invoke(prompt)
            output_tokens = len(self.tokenizer.encode(response.content))
            
            # Track costs
            total_tokens = input_tokens + output_tokens
            cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            # Parse JSON response
            json_content = self._extract_json_from_response(response.content)
            if json_content:
                # Enhance with Product Types Ontology URI if not present
                if "additionalType" not in json_content:
                    uri = self._get_product_ontology_uri(concept)
                    if uri:
                        json_content["additionalType"] = uri
                
                return json_content
            else:
                print(f"Failed to parse JSON for concept: {concept}")
                return self._create_fallback_markup(concept, context)
                
        except Exception as e:
            print(f"Error generating markup for {concept}: {e}")
            return self._create_fallback_markup(concept, context)
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response, handling various formats."""
        try:
            # Try direct parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Look for JSON within code blocks or other text
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Look for standalone JSON object
            json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def _get_product_ontology_uri(self, concept: str) -> Optional[str]:
        """Get Product Types Ontology URI for a concept."""
        concept_lower = concept.lower().strip()
        
        # Check direct mappings
        if concept_lower in self.component_mappings:
            return f"{self.product_ontology_base}{self.component_mappings[concept_lower]}"
        
        # Check partial matches
        for key, value in self.component_mappings.items():
            if key in concept_lower or concept_lower in key:
                return f"{self.product_ontology_base}{value}"
        
        # Generate URI from concept name
        wiki_name = concept.replace(" ", "_").replace("-", "_")
        return f"{self.product_ontology_base}{wiki_name}"
    
    def _create_fallback_markup(self, concept: str, context: str) -> Dict:
        """Create basic Schema.org markup when extraction fails."""
        description = context[:200] + "..." if len(context) > 200 else context
        
        markup = {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": concept,
            "description": description,
            "category": "Electronic Component"
        }
        
        # Add Product Types Ontology URI
        uri = self._get_product_ontology_uri(concept)
        if uri:
            markup["additionalType"] = uri
        
        return markup

def extract_schema_org_markup(chunks: List, concepts: List[str]) -> List[Dict]:
    """
    Main function to extract Schema.org markup from chunks and concepts.
    
    Args:
        chunks: Document chunks from data_loader
        concepts: Extracted concepts from idea_extractor
        
    Returns:
        List of Schema.org JSON-LD objects
    """
    extractor = SchemaOrgExtractor()
    return extractor.extract_schema_org_data(chunks, concepts)

if __name__ == "__main__":
    from .data_loader import load_and_split_data
    from .idea_extractor import extract_ideas
    
    # Test with sample data
    chunks = load_and_split_data()
    concepts = extract_ideas(chunks[:10])
    
    schema_markup = extract_schema_org_markup(chunks[:10], concepts)
    
    print(f"Generated {len(schema_markup)} Schema.org objects:")
    for i, obj in enumerate(schema_markup[:3]):  # Show first 3
        print(f"\n{i+1}. {json.dumps(obj, indent=2)}")
