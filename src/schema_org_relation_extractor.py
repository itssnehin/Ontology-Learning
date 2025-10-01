import sys
from pathlib import Path
#sys.path.append(str(Path(__file__).parent))

import json
import re
from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from tiktoken import get_encoding
import logging

from src.config import LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__) # <-- ADD logger

class SchemaOrgRelationExtractor:
    """Extract Schema.org relationships and properties from document chunks."""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        self.tokenizer = get_encoding("cl100k_base")
        self.cost_per_1k_tokens = 0.00336  # GPT-4o cost
        self.total_tokens = 0
        self.total_cost = 0.0
        
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
    
    def extract_relations_and_properties(self, chunks: List, concepts: List[str]) -> Dict[str, Dict]:
        """
        Extract Schema.org relations and properties for concepts.
        
        Args:
            chunks: Document chunks containing technical information
            concepts: List of extracted concepts
            
        Returns:
            Dictionary mapping concepts to their properties and relations
        """
        concept_data = {}
        
        for concept in concepts:
            # Find context for this concept
            context = self._find_concept_context(concept, chunks)
            
            # Extract properties and relations
            properties = self._extract_properties(concept, context)
            relations = self._extract_relations(concept, context, concepts)
            
            concept_data[concept] = {
                "properties": properties,
                "relations": relations,
                "context_source": context[:100] + "..." if len(context) > 100 else context
            }
        
        logger.info(f"Extracted properties and relations for {len(concept_data)} concepts")
        logger.info(f"Total API cost: ${self.total_cost:.6f}")
        
        return concept_data
    
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
    
    def _extract_properties(self, concept: str, context: str) -> Dict[str, str]:
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
        
        try:
            input_tokens = len(self.tokenizer.encode(prompt))
            response = self.llm.invoke(prompt)
            output_tokens = len(self.tokenizer.encode(response.content))
            
            # Track costs
            total_tokens = input_tokens + output_tokens
            cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            # Parse response
            properties = self._parse_json_response(response.content)
            return properties if properties else {}
            
        except Exception as e:
            logger.error(f"Error extracting properties for {concept}: {e}")
            return {}
    
    def _extract_relations(self, concept: str, context: str, all_concepts: List[str]) -> List[Dict[str, str]]:
        """Extract relationships between concepts."""
        
        # Find other concepts mentioned in the same context
        related_concepts = []
        concept_lower = concept.lower()
        context_lower = context.lower()
        
        for other_concept in all_concepts:
            if other_concept != concept and other_concept.lower() in context_lower:
                related_concepts.append(other_concept)
        
        if not related_concepts:
            return []
        
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
        
        try:
            input_tokens = len(self.tokenizer.encode(prompt))
            response = self.llm.invoke(prompt)
            output_tokens = len(self.tokenizer.encode(response.content))
            
            # Track costs
            total_tokens = input_tokens + output_tokens
            cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            # Parse response
            relations = self._parse_json_response(response.content)
            return relations if isinstance(relations, list) else []
            
        except Exception as e:
            logger.error(f"Error extracting relations for {concept}: {e}")
            return []
    
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

def extract_schema_org_relations(chunks: List, concepts: List[str]) -> Dict[str, Dict]:
    """
    Main function to extract Schema.org relations and properties.
    
    Args:
        chunks: Document chunks
        concepts: Extracted concepts
        
    Returns:
        Dictionary with properties and relations for each concept
    """
    extractor = SchemaOrgRelationExtractor()
    return extractor.extract_relations_and_properties(chunks, concepts)

if __name__ == "__main__":
    from data_loader import load_and_split_data
    from idea_extractor import extract_ideas
    
    # Test with sample data
    chunks = load_and_split_data()
    concepts = extract_ideas(chunks[:10])
    
    relations_data = extract_schema_org_relations(chunks[:10], concepts)
    
    print(f"Extracted data for {len(relations_data)} concepts:")
    for concept, data in list(relations_data.items())[:2]:
        print(f"\n{concept}:")
        print(f"  Properties: {data['properties']}")
        print(f"  Relations: {data['relations']}")
