#!/usr/bin/env python3
"""
Subgraph extraction module implementing OLLM's approach for electrotechnical documents.
Extracts coherent ontological subgraphs rather than isolated concepts/relations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import re
from typing import Dict, List, Tuple, Set, Optional
from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENAI_API_KEY
from tiktoken import get_encoding
from utils import setup_logging
import json

class ElectrotechnicalSubgraphExtractor:
    """Extract ontological subgraphs from electrotechnical documents using OLLM's approach."""
    
    def __init__(self, model_name: str = LLM_MODEL):
        setup_logging("../logs", "subgraph_extractor")
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
        self.tokenizer = get_encoding("cl100k_base")
        self.cost_per_1k_tokens = 0.00336  # GPT-4o cost
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Domain-specific root concepts for electrotechnical ontology
        self.domain_roots = [
            "Electronic Components",
            "Electrical Systems", 
            "Communication Devices",
            "RF Components",
            "Power Components",
            "Signal Processing Components",
            "Connectors and Interfaces"
        ]
    
    def extract_document_subgraphs(self, chunks: List, max_path_length: int = 4) -> Dict[str, List[Dict]]:
        """
        Extract relevant ontological subgraphs for each document chunk.
        
        Args:
            chunks: List of document chunks
            max_path_length: Maximum path length from root to specific concept
            
        Returns:
            Dictionary mapping chunk IDs to lists of extracted subgraphs
        """
        all_subgraphs = {}
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            print(f"\nProcessing {chunk_id}: {chunk.page_content[:100]}...")
            
            # Extract subgraphs for this chunk
            subgraphs = self._extract_chunk_subgraphs(chunk, max_path_length)
            all_subgraphs[chunk_id] = subgraphs
            
            print(f"Extracted {len(subgraphs)} subgraphs for {chunk_id}")
        
        print(f"\nTotal API cost: ${self.total_cost:.6f}")
        return all_subgraphs
    
    def _extract_chunk_subgraphs(self, chunk, max_path_length: int) -> List[Dict]:
        """Extract subgraphs for a single document chunk."""
        
        prompt = f"""
        You are an expert in electrotechnical ontology construction. Analyze this technical document segment and extract relevant ontological subgraphs.

        TASK: Extract hierarchical paths from general domain roots to specific concepts mentioned in the document.

        DOMAIN ROOTS (choose relevant ones):
        {', '.join(self.domain_roots)}

        DOCUMENT SEGMENT:
        {chunk.page_content}

        INSTRUCTIONS:
        1. Identify specific electrotechnical concepts mentioned in the document
        2. For each concept, create a hierarchical path of maximum {max_path_length} levels
        3. Start from an appropriate domain root and work down to the specific concept
        4. Each path should represent a taxonomic "is-a" relationship
        5. Focus on technical components, systems, and their classifications

        OUTPUT FORMAT - List of hierarchical paths:
        Electronic Components -> RF Components -> Antennas -> WiFi Antennas
        Electronic Components -> Connectors -> RF Connectors -> SMA Connectors
        Electrical Systems -> Power Systems -> DC Power -> Battery Systems

        GUIDELINES:
        - Each path should be 2-{max_path_length} levels deep
        - Use specific technical terminology from the document
        - Avoid generic terms like "device" or "system" at leaf nodes
        - Ensure paths represent clear taxonomic relationships
        - Include 3-8 relevant paths per document segment

        Generate relevant hierarchical paths for this document:
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
            
            # Parse paths from response
            paths = self._parse_hierarchical_paths(response.content)
            
            # Convert paths to subgraph format
            subgraphs = self._paths_to_subgraphs(paths)
            
            return subgraphs
            
        except Exception as e:
            print(f"Error extracting subgraphs: {e}")
            return []
    
    def _parse_hierarchical_paths(self, response: str) -> List[List[str]]:
        """Parse hierarchical paths from LLM response."""
        paths = []
        
        # Look for arrow-separated paths
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            # Match paths with arrows
            if '->' in line:
                # Remove leading dashes or numbers
                clean_line = re.sub(r'^[-\*\d\.\s]+', '', line)
                
                # Split on arrows and clean up
                parts = [part.strip() for part in clean_line.split('->')]
                
                # Filter out empty parts and ensure minimum path length
                parts = [part for part in parts if part]
                if len(parts) >= 2:
                    paths.append(parts)
        
        return paths
    
    def _paths_to_subgraphs(self, paths: List[List[str]]) -> List[Dict]:
        """Convert hierarchical paths to subgraph dictionaries."""
        subgraphs = []
        
        for path in paths:
            if len(path) < 2:
                continue
                
            # Create nodes (concepts)
            nodes = []
            for concept in path:
                nodes.append({
                    "name": concept,
                    "type": "Class",
                    "level": path.index(concept)
                })
            
            # Create edges (taxonomic relations)
            edges = []
            for i in range(len(path) - 1):
                edges.append({
                    "source": path[i],
                    "target": path[i + 1], 
                    "relation": "subclass_of",
                    "direction": "parent_to_child"
                })
            
            subgraph = {
                "path": path,
                "nodes": nodes,
                "edges": edges,
                "root": path[0],
                "leaf": path[-1],
                "depth": len(path)
            }
            
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def aggregate_subgraphs(self, all_subgraphs: Dict[str, List[Dict]], 
                           min_frequency: int = 1) -> Dict:
        """
        Aggregate subgraphs across documents to build unified ontology.
        Similar to OLLM's post-processing step.
        
        Args:
            all_subgraphs: Subgraphs from all document chunks
            min_frequency: Minimum frequency for concept/relation inclusion
            
        Returns:
            Unified ontology graph
        """
        # Count concept and relation frequencies
        concept_freq = {}
        relation_freq = {}
        all_nodes = {}
        all_edges = []
        
        # Aggregate all subgraphs
        for chunk_id, subgraphs in all_subgraphs.items():
            for subgraph in subgraphs:
                # Count nodes
                for node in subgraph["nodes"]:
                    concept = node["name"]
                    concept_freq[concept] = concept_freq.get(concept, 0) + 1
                    all_nodes[concept] = node
                
                # Count edges
                for edge in subgraph["edges"]:
                    edge_key = f"{edge['source']}->{edge['target']}"
                    relation_freq[edge_key] = relation_freq.get(edge_key, 0) + 1
                    all_edges.append(edge)
        
        # Filter by frequency
        filtered_concepts = {k: v for k, v in concept_freq.items() if v >= min_frequency}
        filtered_relations = {k: v for k, v in relation_freq.items() if v >= min_frequency}
        
        # Build unified graph
        unified_nodes = [all_nodes[concept] for concept in filtered_concepts.keys()]
        unified_edges = [edge for edge in all_edges 
                        if f"{edge['source']}->{edge['target']}" in filtered_relations]
        
        # Remove duplicate edges
        seen_edges = set()
        unique_edges = []
        for edge in unified_edges:
            edge_key = f"{edge['source']}->{edge['target']}"
            if edge_key not in seen_edges:
                edge["frequency"] = filtered_relations[edge_key]
                unique_edges.append(edge)
                seen_edges.add(edge_key)
        
        unified_graph = {
            "nodes": unified_nodes,
            "edges": unique_edges,
            "concept_frequencies": filtered_concepts,
            "relation_frequencies": filtered_relations,
            "stats": {
                "total_concepts": len(filtered_concepts),
                "total_relations": len(unique_edges),
                "avg_concept_frequency": sum(filtered_concepts.values()) / len(filtered_concepts),
                "total_subgraphs_processed": sum(len(sg) for sg in all_subgraphs.values())
            }
        }
        
        return unified_graph
    
    def export_subgraphs(self, unified_graph: Dict, output_path: str):
        """Export unified ontology to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unified_graph, f, indent=2, ensure_ascii=False)
        print(f"Exported unified ontology to {output_path}")
    
    def print_ontology_summary(self, unified_graph: Dict):
        """Print summary of extracted ontology."""
        stats = unified_graph["stats"]
        
        print(f"\n{'='*50}")
        print("EXTRACTED ONTOLOGY SUMMARY")
        print(f"{'='*50}")
        print(f"Total Concepts: {stats['total_concepts']}")
        print(f"Total Relations: {stats['total_relations']}")
        print(f"Average Concept Frequency: {stats['avg_concept_frequency']:.2f}")
        print(f"Subgraphs Processed: {stats['total_subgraphs_processed']}")
        
        print(f"\nTOP CONCEPTS (by frequency):")
        top_concepts = sorted(unified_graph["concept_frequencies"].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        for concept, freq in top_concepts:
            print(f"  {concept}: {freq}")
        
        print(f"\nSAMPLE TAXONOMIC RELATIONS:")
        for edge in unified_graph["edges"][:10]:
            print(f"  {edge['source']} -> {edge['target']} (freq: {edge['frequency']})")

def extract_ontological_subgraphs(chunks: List, max_path_length: int = 4) -> Dict:
    """
    Main function to extract ontological subgraphs using OLLM's approach.
    
    Args:
        chunks: Document chunks from data_loader
        max_path_length: Maximum taxonomic path length
        
    Returns:
        Unified ontology graph
    """
    extractor = ElectrotechnicalSubgraphExtractor()
    
    # Extract subgraphs for each document
    all_subgraphs = extractor.extract_document_subgraphs(chunks, max_path_length)
    
    # Aggregate into unified ontology
    unified_graph = extractor.aggregate_subgraphs(all_subgraphs, min_frequency=1)
    
    # Print summary
    extractor.print_ontology_summary(unified_graph)
    
    return unified_graph

if __name__ == "__main__":
    from data_loader import load_and_split_data
    
    # Test with sample data
    chunks = load_and_split_data()
    
    # Extract ontological subgraphs
    ontology = extract_ontological_subgraphs(chunks[:10])
    
    # Export results
    extractor = ElectrotechnicalSubgraphExtractor()
    extractor.export_subgraphs(ontology, "../data/extracted_ontology_subgraphs.json")
