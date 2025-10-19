from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class ExtensionDecision(Enum):
    EXTEND = "extend_ontology"
    MAP_EXACT = "map_to_existing_exact"
    MAP_SIMILAR = "map_to_existing_similar"
    MERGE_CONCEPTS = "merge_concepts"
    UNCERTAIN = "requires_manual_review"

@dataclass
class ConceptMatch:
    existing_concept: str
    similarity_score: float
    match_type: str
    confidence: float
    reasoning: str

    def to_dict(self):
        return asdict(self)

@dataclass
class ExtensionResult:
    concept_name: str
    decision: ExtensionDecision
    target_concept: Optional[str]
    confidence: float
    reasoning: str
    matches: List[ConceptMatch]
    non_taxonomic_relations: List[Dict[str, str]] = None # <-- ADD THIS FIELD

    def to_dict(self):
        return {
            "concept_name": self.concept_name,
            "decision": self.decision.value,
            "target_concept": self.target_concept,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "matches": [m.to_dict() for m in self.matches],
            "non_taxonomic_relations": self.non_taxonomic_relations  # <-- INCLUDE IN DICT
        }

# --- Data Models from integrated_schema_pipeline.py ---

@dataclass
class PipelineConfig:
    """Configuration for the integrated pipeline."""
    max_chunks: Optional[int] = None
    similarity_thresholds: Optional[Dict[str, float]] = None
    enable_llm_validation: bool = True
    enable_technical_matching: bool = True
    output_dir: str = "../data/integrated_output"
    
    def __post_init__(self):
        if self.similarity_thresholds is None:
            self.similarity_thresholds = {
                'exact_match': 0.95, 'high_similarity': 0.85,
                'medium_similarity': 0.70, 'low_similarity': 0.50
            }

@dataclass
class IntegrationResults:
    """Structured results from the integrated pipeline execution."""
    total_concepts_extracted: int
    concepts_mapped_to_existing: int
    concepts_extending_ontology: int
    concepts_requiring_review: int
    confidence_scores: List[float]
    processing_time: float
    decisions: List[ExtensionResult]
    costs: Dict[str, Any]  
    
    @property
    def automation_rate(self) -> float:
        total_automated = self.concepts_mapped_to_existing + self.concepts_extending_ontology
        return (total_automated / self.total_concepts_extracted * 100) if self.total_concepts_extracted > 0 else 0.0
    
    @property
    def average_confidence(self) -> float:
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0