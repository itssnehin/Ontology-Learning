import sys
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from tiktoken import get_encoding
import logging

from src.config import LLM_MODEL, OPENAI_API_KEY, MODEL_COSTS

logger = logging.getLogger(__name__)

class SchemaOrgRelationExtractor:
    """Extract Schema.org relationships and properties from document chunks."""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        self.tokenizer = get_encoding("cl100k_base")
        
        # Standard Schema.org properties for products
        self.standard_properties = {
            "manufacturer": "Organization that produces the product",
            "model": "Product model identifier", 
            "category": "Product category classification",
            "material": "Materials used in construction",
            "color": "Product color",
            "weight": "Product weight",
            "height": "Product height",
            "width": "Product width", 
            "depth": "Product depth",
            "isAccessoryOrSparePartFor": "Product this is an accessory for",
            "isConsumableFor": "Product this is consumable for",
            "isRelatedTo": "Related products",
            "isSimilarTo": "Similar products"
        }
        
        # Electrical/electronic specific properties (custom namespace)
        self.electrical_properties = {
            "elec:frequency": "Operating frequency or frequency range",
            "elec:impedance": "Electrical impedance",
            "elec:voltage": "Operating voltage",
            "elec:current": "Operating current",
            "elec:power": "Power rating or consumption",
            "elec:gain": "Signal gain (for antennas/amplifiers)",
            "elec:polarization": "Signal polarization",
            "elec:connector": "Type of electrical connector",
            "elec:cableType": "Cable type or specification",
            "elec:mounting": "Mounting type or method",
            "elec:temperature": "Operating temperature range",
            "elec:protocol": "Communication protocol supported",
            "elec:standard": "Technical standards compliance"
        }
    
    def extract_relations_and_properties(self, chunks: List, concepts: List[str], model_name: str) -> Tuple[Dict[str, Dict], int, int]:
        """
        Extracts Schema.org relations/properties and returns data and token counts.
        """
        concept_data = {}
        total_input_tokens = 0
        total_output_tokens = 0
        
        for concept in concepts:
            context = self._find_concept_context(concept, chunks)
            
            # Pass the model_name down and capture token counts from each helper
            properties, p_in, p_out = self._extract_properties(concept, context, model_name)
            relations, r_in, r_out = self._extract_relations(concept, context, concepts, model_name)
            
            total_input_tokens += (p_in + r_in)
            total_output_tokens += (p_out + r_out)
            
            concept_data[concept] = {
                "properties": properties,
                "relations": relations,
                "context_source": context[:100] + "..." if len(context) > 100 else context
            }
        
        return concept_data, total_input_tokens, total_output_tokens
    
    def _find_concept_context(self, concept: str, chunks: List) -> str:
        """Find chunks containing context about the concept."""
        relevant_chunks = []
        concept_variations = [
            concept.lower(),
            concept.lower().replace(" ", ""),
            concept.lower().replace("-", " "),
            concept.lower().replace("_", " ")
        ]
        
        for chunk in chunks:
            chunk_text = chunk.page_content.lower()
            if any(var in chunk_text for var in concept_variations):
                relevant_chunks.append(chunk.page_content)
        
        return " ".join(relevant_chunks[:2])  # Limit context size
    
    def _extract_properties(self, concept: str, context: str, model_name: str) -> Tuple[Dict[str, str], int, int]:
        """Extract Schema.org properties for a concept."""
        
        prompt = f"""
        Extract technical properties for this electronic component based on the context.
        
        Component: {concept}
        Context: {context}
        
        Extract values for these Schema.org and electrical properties:
        
        Standard Schema.org properties:
        - manufacturer: Company that makes it
        - model: Model number or identifier
        - category: Product category
        - material: Materials used
        - weight: Weight specification
        - dimensions: Height, width, depth measurements
        
        Electrical properties (use elec: prefix):
        - elec:frequency: Operating frequency/range (e.g., "2.4-5 GHz")
        - elec:impedance: Impedance value (e.g., "50 ohms")
        - elec:voltage: Voltage specification
        - elec:power: Power rating
        - elec:gain: Signal gain (for antennas)
        - elec:connector: Connector type
        - elec:mounting: Mounting method
        - elec:temperature: Operating temperature
        - elec:protocol: Supported protocols
        - elec:standard: Compliance standards
        
        OUTPUT FORMAT: JSON object with property names as keys and extracted values as strings.
        Only include properties where you can extract actual values from the context.
        
        Example:
        {{
          "manufacturer": "Acme Electronics",
          "category": "Antenna",
          "elec:frequency": "2.4-6 GHz",
          "elec:impedance": "50 ohms"
        }}
        """
        in_tokens, out_tokens = 0, 0
        try:
            in_tokens = len(self.tokenizer.encode(prompt))
            self.llm.model_name = model_name  # Ensure the LLM instance uses the selected model
            response = self.llm.invoke(prompt)
            out_tokens = len(self.tokenizer.encode(response.content))
            
            properties = self._parse_json_response(response.content) or {}
            return properties, in_tokens, out_tokens
            
        except Exception as e:
            logger.error(f"Error extracting properties for {concept}: {e}")
            return {}, in_tokens, out_tokens

    
    def _extract_relations(self, concept: str, context: str, all_concepts: List[str], model_name: str) -> Tuple[List[Dict[str, str]], int, int]:
        """Extracts relationships and returns relations and token counts."""
        related_concepts = [c for c in all_concepts if c != concept and c.lower() in context.lower()]
        if not related_concepts:
            return [], 0, 0
        
        prompt = f"""
        Identify relationships between this component and related components based on the technical context.
        
        Main Component: {concept}
        Related Components Found: {related_concepts}
        Context: {context}
        
        Determine Schema.org relationships using these property types:
        - isAccessoryOrSparePartFor: Component A is an accessory/part of Component B
        - isConsumableFor: Component A is consumable for Component B  
        - isRelatedTo: General relationship between components
        - isSimilarTo: Components that are similar/comparable
        - hasPart: Component A has Component B as a part
        - isPartOf: Component A is part of Component B
        - worksWith: Components that work together
        - requires: Component A requires Component B to function
        
        OUTPUT FORMAT: JSON array of relationship objects:
        [
          {{
            "property": "relationship_type",
            "target": "target_component_name",
            "description": "brief explanation"
          }}
        ]
        
        Only include relationships you can clearly infer from the context.
        """
        in_tokens, out_tokens = 0, 0
        try:
            in_tokens = len(self.tokenizer.encode(prompt))
            self.llm.model_name = model_name
            response = self.llm.invoke(prompt)
            out_tokens = len(self.tokenizer.encode(response.content))
            
            relations = self._parse_json_response(response.content)
            return (relations if isinstance(relations, list) else []), in_tokens, out_tokens
            
        except Exception as e:
            logger.error(f"Error extracting relations for {concept}: {e}")
            return [], in_tokens, out_tokens
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Look for standalone JSON
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def generate_enhanced_schema_objects(self, base_schema_objects: List[Dict], 
                                        relations_data: Dict[str, Dict]) -> List[Dict]:
        """
        Enhance base Schema.org objects with extracted properties and relations.
        
        Args:
            base_schema_objects: Basic Schema.org objects from schema_org_extractor
            relations_data: Properties and relations data
            
        Returns:
            Enhanced Schema.org objects with properties and relations
        """
        enhanced_objects = []
        
        for schema_obj in base_schema_objects:
            concept_name = schema_obj.get("name", "")
            
            if concept_name in relations_data:
                data = relations_data[concept_name]
                
                # Add properties to schema object
                properties = data.get("properties", {})
                for prop, value in properties.items():
                    if prop.startswith("elec:"):
                        # Add electrical namespace to context
                        if "@context" not in schema_obj:
                            schema_obj["@context"] = "https://schema.org/"
                        
                        if isinstance(schema_obj["@context"], str):
                            schema_obj["@context"] = {
                                "@vocab": schema_obj["@context"],
                                "elec": "https://example.org/electrical/"
                            }
                        elif isinstance(schema_obj["@context"], dict):
                            schema_obj["@context"]["elec"] = "https://example.org/electrical/"
                    
                    schema_obj[prop] = value
                
                # Add relations
                relations = data.get("relations", [])
                for relation in relations:
                    prop = relation.get("property")
                    target = relation.get("target")
                    if prop and target:
                        if prop in schema_obj:
                            # Convert to array if needed
                            if isinstance(schema_obj[prop], str):
                                schema_obj[prop] = [schema_obj[prop]]
                            schema_obj[prop].append(target)
                        else:
                            schema_obj[prop] = target
            
            enhanced_objects.append(schema_obj)
        
        return enhanced_objects

def extract_schema_org_relations(chunks: List, concepts: List[str], model_name: str = LLM_MODEL) -> Tuple[Dict[str, Dict], int, int]:
    """
    Main function to extract relations and properties, returning data and token counts.
    """
    extractor = SchemaOrgRelationExtractor(model_name=model_name)
    return extractor.extract_relations_and_properties(chunks, concepts, model_name)

if __name__ == "__main__":
    # The __main__ block is for standalone testing and doesn't need to return costs
    from src.data_loader import load_and_split_data
    from src.idea_extractor import extract_ideas
    
    chunks = load_and_split_data()[:5]
    concepts, _, _ = extract_ideas(chunks)
    
    relations_data, total_in, total_out = extract_schema_org_relations(chunks, concepts)
    
    print(f"\nExtracted data for {len(relations_data)} concepts:")
    print(f"Total Tokens Used: Input={total_in}, Output={total_out}")
    for concept, data in list(relations_data.items())[:2]:
        print(f"\n{concept}:")
        print(f"  Properties: {data['properties']}")
        print(f"  Relations: {data['relations']}")