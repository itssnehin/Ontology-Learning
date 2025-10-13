#!/usr/bin/env python3
"""
Smart Ontology Extension Manager

CONCEPTUAL OVERVIEW:
===================
This module implements an intelligent decision-making system for ontology evolution,
specifically designed for Schema.org-based technical ontologies. When new concepts
are extracted from datasheets or technical documentation, the system must decide:
1. Should this concept extend the ontology as a new entity?
2. Should this concept be mapped to an existing ontology element?
3. Should existing concepts be merged or relationships refined?

The core challenge is balancing ontology completeness (capturing all domain concepts)
with ontological consistency (avoiding redundancy and maintaining semantic coherence).

THEORETICAL FOUNDATION:
======================
The approach is grounded in several key areas of computer science and knowledge engineering:

1. ONTOLOGY ENGINEERING PRINCIPLES:
   - Concept Similarity: Measures semantic relatedness between domain entities
   - Ontological Coherence: Maintains logical consistency and hierarchical structure
   - Knowledge Base Evolution: Systematic approach to ontology growth and refinement

2. INFORMATION RETRIEVAL THEORY:
   - Vector Space Models: Embedding-based semantic similarity computation
   - Similarity Metrics: Cosine similarity, Jaccard coefficients, edit distances
   - Threshold Optimization: Precision-recall trade-offs in similarity matching

3. MACHINE LEARNING INTEGRATION:
   - Embedding Models: Dense vector representations of concepts (OpenAI embeddings)
   - Large Language Models: Context-aware reasoning for ambiguous classification
   - Active Learning: Human-in-the-loop validation for uncertain decisions

ALGORITHMIC APPROACH:
====================
The system employs a multi-stage decision algorithm combining complementary similarity methods:

STAGE 1: RAPID SIMILARITY SCREENING
- Lexical Similarity: String matching algorithms (SequenceMatcher, substring detection)
- Embedding Similarity: High-dimensional semantic comparison using pre-trained models
- Category Filtering: Domain-specific hierarchical classification

STAGE 2: TECHNICAL SPECIFICATION MATCHING
For electronic components (primary domain), specialized matchers evaluate:
- Frequency Range Overlap: Quantitative comparison of operating frequencies
- Impedance Matching: Standard electrical impedance values (50Ω, 75Ω, 300Ω)
- Connector Type Normalization: SMA, BNC, N-type connector standardization
- Mounting Classification: Surface mount, through-hole, panel mount categorization

STAGE 3: CONFIDENCE-WEIGHTED DECISION SYNTHESIS
- Multi-method Score Fusion: Weighted combination of similarity measures
- Adaptive Thresholds: Dynamic adjustment based on ontology maturity
- Uncertainty Quantification: Confidence intervals for decision quality

STAGE 4: LLM-POWERED SEMANTIC VALIDATION
For ambiguous cases (similarity scores 0.70-0.95):
- Structured Reasoning: GPT-4 evaluation with domain expert persona
- Context-Aware Analysis: Consideration of technical specifications and relationships
- Explainable Decisions: Natural language reasoning with confidence scoring

SIMILARITY COMPUTATION DETAILS:
==============================
1. EMBEDDING SIMILARITY:
   - Model: OpenAI text-embedding-ada-002 (1536 dimensions)
   - Metric: Cosine similarity in high-dimensional space
   - Text Representation: "Name. Category: X. Description: Y. Frequency: Z..."#!/usr/bin/env python3
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import re
from difflib import SequenceMatcher
import logging
from dataclasses import dataclass, asdict

from src.config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, PROMPTS, NEO4J_DB_NAME
import re

# We need to import the PipelineConfig to use it as a type hint
from src.data_models import PipelineConfig, ExtensionDecision, ConceptMatch, ExtensionResult

logger = logging.getLogger(__name__)

class OntologyExtensionManager:
    """Intelligent manager for deciding ontology extensions vs mappings."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        # Use the provided config, or create a default one if none is given
        self.config = config or PipelineConfig()
        
        # Now, use the thresholds from the config object
        self.similarity_thresholds = self.config.similarity_thresholds

        # Initialize other components
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        
        self.technical_matchers = {
            'frequency': self._match_frequency_specs,
            'impedance': self._match_impedance_specs,
            'voltage': self._match_voltage_specs,
            'connector': self._match_connector_types,
            'mounting': self._match_mounting_types   
        }
        
        # Cached ontology data
        self._existing_concepts = None
        self._concept_embeddings = {}
        
        logger.info("🔍 Ontology Extension Manager initialized")
    
    def load_existing_ontology(self) -> Dict[str, Any]:
        """
        Load and cache existing ontology concepts from Neo4j.
        This version is smarter: it loads both instance data (:Product) AND
        class definitions (:OntologyClass) to build a comprehensive view of existing knowledge.
        """
        logger.info("📚 Loading existing ontology (classes and instances)...")
        concepts = []
        concept_names = set() # Use a set to prevent duplicates

        with self.driver.session(database=NEO4J_DB_NAME) as session:
            # Query 1: Get instance data (nodes created by previous pipeline runs)
            # This query is designed to not fail if the properties don't exist yet.
            instance_query = """
            MATCH (n:Product)
            RETURN n.name as name, n.category as category, n.description as description
            """
            try:
                for record in session.run(instance_query):
                    name = record['name']
                    if name and name not in concept_names:
                        concepts.append({
                            'name': name,
                            'category': record['category'] or '',
                            'description': record['description'] or ''
                            # Add other properties here if needed for embedding
                        })
                        concept_names.add(name)
            except Exception as e:
                logger.warning(f"Could not load instance data (:Product nodes). This is normal on a first run. Error: {e}")

            # Query 2: Get class definitions from the baseline hierarchy
            class_query = """
            MATCH (c:OntologyClass)
            RETURN c.name as name, c.description as description
            """
            try:
                for record in session.run(class_query):
                    name = record['name']
                    if name and name not in concept_names:
                        concepts.append({
                            'name': name,
                            'category': 'Ontology Class', # Differentiate classes from instances
                            'description': record['description'] or f'The ontological class representing {name}.'
                        })
                        concept_names.add(name)
            except Exception as e:
                logger.warning(f"Could not load class definitions (:OntologyClass nodes). Check baseline. Error: {e}")

        self._existing_concepts = concepts
        logger.info(f"   ✅ Loaded {len(concepts)} unique existing concepts (instances + classes).")
        return {'concepts': concepts}

    
    def create_concept_embeddings(self, concepts: List[Dict]) -> Dict[str, np.ndarray]:
        """Create embeddings for existing concepts for similarity matching."""
        print("🧠 Creating concept embeddings...")
        
        concept_texts = []
        concept_names = []
        
        for concept in concepts:
            # Create rich text representation
            text_parts = [concept['name']]
            
            if concept['category']:
                text_parts.append(f"Category: {concept['category']}")
            if concept['description']:
                text_parts.append(f"Description: {concept['description'][:200]}")
            if concept['frequency']:
                text_parts.append(f"Frequency: {concept['frequency']}")
            if concept['impedance']:
                text_parts.append(f"Impedance: {concept['impedance']}")
            if concept['connector']:
                text_parts.append(f"Connector: {concept['connector']}")
            
            concept_text = ". ".join(text_parts)
            concept_texts.append(concept_text)
            concept_names.append(concept['name'])
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(concept_texts)
        
        # Cache embeddings
        for name, embedding in zip(concept_names, embeddings):
            self._concept_embeddings[name] = np.array(embedding)
        
        print(f"   ✅ Created embeddings for {len(concept_names)} concepts")
        return self._concept_embeddings
    
    def analyze_new_concept(self, new_concept: Dict[str, Any]) -> ExtensionResult:
        """Analyze whether a new concept should extend the ontology or map to existing."""
        concept_name = new_concept.get('name', 'Unknown')
        logger.debug(f"🔍 Analyzing new concept: {concept_name}")
        
        if self._existing_concepts is None:
            self.load_existing_ontology()
            if self._existing_concepts:
                self.create_concept_embeddings(self._existing_concepts)
        
        matches = self._find_concept_matches(new_concept)
        decision_result = self._make_extension_decision(new_concept, matches) # Pass the full new_concept dict
        
        logger.debug(f"   🎯 Decision for '{concept_name}': {decision_result.decision.value} (Confidence: {decision_result.confidence:.2f})")
        
        return decision_result
    
    def _find_concept_matches(self, new_concept: Dict[str, Any]) -> List[ConceptMatch]:
        """Find potential matches using multiple similarity methods."""
        matches = []
        
        # Method 1: Embedding similarity
        embedding_matches = self._find_embedding_matches(new_concept)
        matches.extend(embedding_matches)
        
        # Method 2: Lexical similarity
        lexical_matches = self._find_lexical_matches(new_concept)
        matches.extend(lexical_matches)
        
        # Method 3: Technical property similarity
        technical_matches = self._find_technical_matches(new_concept)
        matches.extend(technical_matches)
        
        # Method 4: Category-based similarity
        category_matches = self._find_category_matches(new_concept)
        matches.extend(category_matches)
        
        # Deduplicate and rank matches
        unique_matches = self._deduplicate_matches(matches)
        ranked_matches = sorted(unique_matches, key=lambda x: x.similarity_score, reverse=True)
        
        return ranked_matches[:10]  # Top 10 matches
    
    def _find_embedding_matches(self, new_concept: Dict[str, Any]) -> List[ConceptMatch]:
        """Find matches using embedding similarity."""
        if not self._concept_embeddings:
            return []
        
        # Create embedding for new concept
        new_text = self._create_concept_text(new_concept)
        new_embedding = np.array(self.embeddings.embed_query(new_text))
        
        matches = []
        for existing_name, existing_embedding in self._concept_embeddings.items():
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
            
            if similarity >= self.similarity_thresholds['low_similarity']:
                matches.append(ConceptMatch(
                    existing_concept=existing_name,
                    similarity_score=similarity,
                    match_type='embedding',
                    confidence=similarity,
                    reasoning=f"Semantic embedding similarity: {similarity:.3f}"
                ))
        
        return matches
    
    def _find_lexical_matches(self, new_concept: Dict[str, Any]) -> List[ConceptMatch]:
        """Find matches using lexical similarity."""
        new_name = new_concept.get('name', '').lower()
        matches = []
        
        for existing_concept in self._existing_concepts:
            existing_name = existing_concept.get('name', '').lower()
            
            # Exact match
            if new_name == existing_name:
                matches.append(ConceptMatch(
                    existing_concept=existing_concept['name'],
                    similarity_score=1.0,
                    match_type='lexical_exact',
                    confidence=1.0,
                    reasoning="Exact name match"
                ))
                continue
            
            # Sequence similarity
            seq_similarity = SequenceMatcher(None, new_name, existing_name).ratio()
            if seq_similarity >= 0.8:
                matches.append(ConceptMatch(
                    existing_concept=existing_concept['name'],
                    similarity_score=seq_similarity,
                    match_type='lexical_similar',
                    confidence=seq_similarity,
                    reasoning=f"High lexical similarity: {seq_similarity:.3f}"
                ))
            
            # Substring matching
            if (new_name in existing_name or existing_name in new_name) and len(new_name) > 3:
                substring_score = min(len(new_name), len(existing_name)) / max(len(new_name), len(existing_name))
                if substring_score >= 0.6:
                    matches.append(ConceptMatch(
                        existing_concept=existing_concept['name'],
                        similarity_score=substring_score,
                        match_type='lexical_substring',
                        confidence=substring_score * 0.8,  # Lower confidence for substring
                        reasoning=f"Substring match: {substring_score:.3f}"
                    ))
        
        return matches
    
    def _find_technical_matches(self, new_concept: Dict[str, Any]) -> List[ConceptMatch]:
        """Find matches based on technical specifications."""
        matches = []
        
        for existing_concept in self._existing_concepts:
            technical_similarity = 0.0
            match_details = []
            
            # Check each technical property
            for prop, matcher_func in self.technical_matchers.items():
                if new_concept.get(prop) and existing_concept.get(prop):
                    prop_similarity = matcher_func(new_concept[prop], existing_concept[prop])
                    if prop_similarity > 0.7:
                        technical_similarity += prop_similarity
                        match_details.append(f"{prop}: {prop_similarity:.2f}")
            
            # If multiple technical properties match well
            if technical_similarity >= 1.5:  # At least 1.5 properties with >0.7 similarity
                avg_similarity = technical_similarity / len(match_details)
                matches.append(ConceptMatch(
                    existing_concept=existing_concept['name'],
                    similarity_score=avg_similarity,
                    match_type='technical_specs',
                    confidence=avg_similarity,
                    reasoning=f"Technical specification match: {', '.join(match_details)}"
                ))
        
        return matches
    
    def _find_category_matches(self, new_concept: Dict[str, Any]) -> List[ConceptMatch]:
        """Find matches within the same category."""
        new_category = new_concept.get('category', '').lower()
        if not new_category:
            return []
        
        matches = []
        for existing_concept in self._existing_concepts:
            existing_category = existing_concept.get('category', '').lower()
            
            if new_category == existing_category:
                # Same category - check if they might be variants
                name_similarity = SequenceMatcher(None, 
                    new_concept.get('name', '').lower(),
                    existing_concept.get('name', '').lower()
                ).ratio()
                
                if name_similarity >= 0.4:  # Moderate name similarity within same category
                    matches.append(ConceptMatch(
                        existing_concept=existing_concept['name'],
                        similarity_score=name_similarity,
                        match_type='category_variant',
                        confidence=name_similarity * 0.7,  # Lower confidence
                        reasoning=f"Same category variant: {name_similarity:.3f}"
                    ))
        
        return matches
    
    def _make_extension_decision(self, new_concept: Dict[str, Any], 
                                matches: List[ConceptMatch]) -> ExtensionResult:
        """Make the final decision on whether to extend or map."""
        
        concept_name = new_concept.get('name', 'Unknown')

        if not matches:
            return ExtensionResult(
                concept_name=concept_name, # <-- ADD THIS
                decision=ExtensionDecision.EXTEND,
                target_concept=None,
                confidence=0.9,
                reasoning="No similar concepts found in existing ontology",
                matches=[]
            )
        
        best_match = matches[0]
        
        if best_match.similarity_score >= self.similarity_thresholds['exact_match']:
            return ExtensionResult(
                concept_name=concept_name, # <-- ADD THIS
                decision=ExtensionDecision.MAP_EXACT,
                target_concept=best_match.existing_concept,
                confidence=best_match.confidence,
                reasoning=f"High similarity match: {best_match.reasoning}",
                matches=matches[:3]
            )
        
        elif best_match.similarity_score >= self.similarity_thresholds['high_similarity'] and self.config.enable_llm_validation:
            return self._llm_validate_similarity(new_concept, best_match, matches[:3])
        
        elif best_match.similarity_score >= self.similarity_thresholds['medium_similarity']:
            return ExtensionResult(
                concept_name=concept_name, # <-- ADD THIS
                decision=ExtensionDecision.UNCERTAIN,
                target_concept=best_match.existing_concept,
                confidence=0.5,
                reasoning=f"Medium similarity requires review: {best_match.reasoning}",
                matches=matches[:5]
            )
        
        else: # Low similarity
            return ExtensionResult(
                concept_name=concept_name, # <-- ADD THIS
                decision=ExtensionDecision.EXTEND,
                target_concept=None,
                confidence=0.8,
                reasoning="Low similarity to existing concepts suggests new concept",
                matches=matches[:3]
            )
    def _llm_validate_similarity(self, new_concept: Dict[str, Any], 
                                best_match: ConceptMatch,
                                top_matches: List[ConceptMatch]) -> ExtensionResult:
        """Use LLM to make final decision on high-similarity matches."""
        concept_name = new_concept.get('name', 'Unknown')
        
        existing_concept_info = next((c for c in self._existing_concepts if c['name'] == best_match.existing_concept), {})
        
        # --- USE THE NEW, MORE ROBUST PROMPT ---
        prompt_template = PROMPTS["ontology_extension_manager"]["llm_validation"]
        prompt = prompt_template.format(
            new_concept_name=new_concept.get('name', 'N/A'),
            new_concept_category=new_concept.get('category', 'N/A'),
            new_concept_description=new_concept.get('description', 'N/A')[:200],
            existing_concept_name=existing_concept_info.get('name', 'N/A'),
            existing_concept_category=existing_concept_info.get('category', 'N/A'),
            existing_concept_description=existing_concept_info.get('description', 'N/A')[:200],
            match_reasoning=best_match.reasoning,
            match_score=f"{best_match.similarity_score:.3f}"
        )
        
        try:
            response = self.llm.invoke(prompt)
            
            # --- USE A ROBUST JSON PARSER ---
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if not match:
                raise json.JSONDecodeError("No JSON object found in LLM response", response.content, 0)
            
            result = json.loads(match.group(0))
            
            llm_decision = result.get('decision')
            decision = ExtensionDecision.MAP_SIMILAR if llm_decision == "SAME_ENTITY" else ExtensionDecision.EXTEND
            
            return ExtensionResult(
                concept_name=concept_name,
                decision=decision,
                target_concept=best_match.existing_concept if decision != ExtensionDecision.EXTEND else None,
                confidence=result.get('confidence', 0.85), # Default to high confidence if key is missing
                reasoning=f"LLM validation: {result.get('reasoning', 'No reasoning provided.')}",
                matches=top_matches
            )
            
        except (json.JSONDecodeError, AttributeError) as e:
            # This will now catch the "Expecting value" error
            logger.warning(f"   ⚠️ LLM validation for '{concept_name}' failed to produce valid JSON. Falling back to UNCERTAIN. Error: {e}")
            return ExtensionResult(
                concept_name=concept_name,
                decision=ExtensionDecision.UNCERTAIN,
                target_concept=best_match.existing_concept,
                confidence=0.5,
                reasoning="LLM validation failed, requires manual review.",
                matches=top_matches
            )

    
    def _deduplicate_matches(self, matches: List[ConceptMatch]) -> List[ConceptMatch]:
        """Remove duplicate matches, keeping the best score for each concept."""
        concept_best_match = {}
        
        for match in matches:
            existing_name = match.existing_concept
            if (existing_name not in concept_best_match or 
                match.similarity_score > concept_best_match[existing_name].similarity_score):
                concept_best_match[existing_name] = match
        
        return list(concept_best_match.values())
    
    def _create_concept_text(self, concept: Dict[str, Any]) -> str:
        """Create text representation of concept for embedding."""
        text_parts = [concept.get('name', '')]
        
        if concept.get('category'):
            text_parts.append(f"Category: {concept['category']}")
        if concept.get('description'):
            text_parts.append(f"Description: {concept['description'][:200]}")
        if concept.get('frequency'):
            text_parts.append(f"Frequency: {concept['frequency']}")
        if concept.get('impedance'):
            text_parts.append(f"Impedance: {concept['impedance']}")
        
        return ". ".join(text_parts)
    
    # Technical property matchers
    def _match_frequency_specs(self, freq1: str, freq2: str) -> float:
        """Match frequency specifications."""
        # Extract frequency ranges/values
        def extract_freq_range(freq_str):
            # Handle patterns like "2.4-5.8 GHz", "900 MHz", "1-6 GHz"
            freq_str = freq_str.lower().replace(' ', '')
            
            if 'ghz' in freq_str:
                multiplier = 1000
            elif 'mhz' in freq_str:
                multiplier = 1
            else:
                return None
            
            # Extract numbers
            numbers = re.findall(r'[\d.]+', freq_str)
            if not numbers:
                return None
            
            numbers = [float(n) * multiplier for n in numbers]
            
            if len(numbers) == 1:
                return (numbers[0], numbers[0])  # Single frequency
            elif len(numbers) >= 2:
                return (min(numbers), max(numbers))  # Range
            
            return None
        
        range1 = extract_freq_range(freq1)
        range2 = extract_freq_range(freq2)
        
        if not range1 or not range2:
            return 0.0
        
        # Calculate overlap
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])
        
        if overlap_start <= overlap_end:
            overlap = overlap_end - overlap_start
            total_range = max(range1[1], range2[1]) - min(range1[0], range2[0])
            return overlap / total_range if total_range > 0 else 1.0
        
        return 0.0
    
    def _match_impedance_specs(self, imp1: str, imp2: str) -> float:
        """Match impedance specifications."""
        # Extract impedance values
        def extract_impedance(imp_str):
            numbers = re.findall(r'[\d.]+', imp_str.lower())
            return float(numbers[0]) if numbers else None
        
        val1 = extract_impedance(imp1)
        val2 = extract_impedance(imp2)
        
        if val1 is None or val2 is None:
            return 0.0
        
        # Common impedance values: 50, 75, 300, 600 ohms
        if val1 == val2:
            return 1.0
        
        # Allow small tolerance
        tolerance = 0.1
        if abs(val1 - val2) / max(val1, val2) <= tolerance:
            return 0.9
        
        return 0.0
    
    def _match_voltage_specs(self, volt1: str, volt2: str) -> float:
        """Match voltage specifications."""
        def extract_voltage(volt_str):
            numbers = re.findall(r'[\d.]+', volt_str.lower())
            return float(numbers[0]) if numbers else None
        
        val1 = extract_voltage(volt1)
        val2 = extract_voltage(volt2)
        
        if val1 is None or val2 is None:
            return 0.0
        
        if val1 == val2:
            return 1.0
        
        # Allow 10% tolerance for voltage
        tolerance = 0.1
        if abs(val1 - val2) / max(val1, val2) <= tolerance:
            return 0.8
        
        return 0.0
    
    def _match_connector_types(self, conn1: str, conn2: str) -> float:
        """Match connector types."""
        conn1_clean = conn1.lower().strip()
        conn2_clean = conn2.lower().strip()
        
        if conn1_clean == conn2_clean:
            return 1.0
        
        # Common connector synonyms
        synonyms = {
            'sma': ['sma', 'sub miniature a'],
            'bnc': ['bnc', 'bayonet neill-concelman'],
            'n-type': ['n-type', 'n connector', 'type n'],
            'mmcx': ['mmcx', 'micro-miniature coaxial'],
        }
        
        for standard, variants in synonyms.items():
            if conn1_clean in variants and conn2_clean in variants:
                return 1.0
        
        # Partial string matching
        return SequenceMatcher(None, conn1_clean, conn2_clean).ratio()
    
    def _match_mounting_types(self, mount1: str, mount2: str) -> float:
        """Match mounting types."""
        mount1_clean = mount1.lower().strip()
        mount2_clean = mount2.lower().strip()
        
        if mount1_clean == mount2_clean:
            return 1.0
        
        # Common mounting synonyms
        synonyms = {
            'surface_mount': ['smd', 'smt', 'surface mount', 'surface-mount'],
            'through_hole': ['through hole', 'through-hole', 'tht', 'thru-hole'],
            'panel_mount': ['panel mount', 'panel-mount', 'chassis mount'],
        }
        
        for standard, variants in synonyms.items():
            if mount1_clean in variants and mount2_clean in variants:
                return 1.0
        
        return SequenceMatcher(None, mount1_clean, mount2_clean).ratio()

# Integration with existing pipeline
def analyze_datasheet_concepts(datasheet_concepts: List[Dict[str, Any]]) -> List[ExtensionResult]:
    """Analyze concepts from new datasheets for ontology extension decisions."""
    
    manager = OntologyExtensionManager()
    results = []
    
    print(f"🔍 Analyzing {len(datasheet_concepts)} concepts from new datasheets...")
    
    for concept in datasheet_concepts:
        result = manager.analyze_new_concept(concept)
        results.append(result)
        
        # Log decision
        print(f"   📋 {concept.get('name', 'Unknown')}: {result.decision.value} (confidence: {result.confidence:.2f})")
    
    # Summary statistics
    decisions = [r.decision for r in results]
    extend_count = decisions.count(ExtensionDecision.EXTEND)
    map_count = decisions.count(ExtensionDecision.MAP_EXACT) + decisions.count(ExtensionDecision.MAP_SIMILAR)
    uncertain_count = decisions.count(ExtensionDecision.UNCERTAIN)
    
    print(f"\n📊 Analysis Summary:")
    print(f"   🆕 Extend ontology: {extend_count}")
    print(f"   🔗 Map to existing: {map_count}")
    print(f"   ❓ Uncertain/Review: {uncertain_count}")
    
    return results

if __name__ == "__main__":
    # Test with sample concepts
    sample_concepts = [
        {
            'name': 'WiFi 6E Dual-Band Antenna',
            'category': 'Antenna', 
            'description': 'High-gain dual-band antenna for WiFi 6E applications',
            'frequency': '2.4-6 GHz',
            'impedance': '50 ohms',
            'connector': 'SMA'
        },
        {
            'name': 'Bluetooth Low Energy Module',
            'category': 'Module',
            'description': 'Ultra-low power Bluetooth 5.0 communication module',
            'frequency': '2.4 GHz',
            'voltage': '3.3V',
            'connector': 'Header pins'
        },
        {
            'name': 'PCB Coaxial Connector',
            'category': 'Connector',
            'description': 'Right-angle SMA connector for PCB mounting',
            'impedance': '50 ohms',
            'connector': 'SMA',
            'mounting': 'surface mount'
        }
    ]
    
    # Run analysis
    results = analyze_datasheet_concepts(sample_concepts)
    
    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED ANALYSIS RESULTS")
    print("="*60)
    
    for i, (concept, result) in enumerate(zip(sample_concepts, results)):
        print(f"\n{i+1}. {concept['name']}")
        print(f"   Decision: {result.decision.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Reasoning: {result.reasoning}")
        
        if result.matches:
            print(f"   Top matches:")
            for match in result.matches[:3]:
                print(f"     • {match.existing_concept}: {match.similarity_score:.3f} ({match.match_type})")
                print(f"       {match.reasoning}")
        print()
    
    print("="*60)