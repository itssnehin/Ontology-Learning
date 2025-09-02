#!/usr/bin/env python3
"""
Integrated ontology extraction pipeline combining existing approach with OLLM's subgraph technique.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
from typing import Dict, List, Any
from datetime import datetime
from utils import setup_logging

# Import existing modules
from data_loader import load_and_split_data
from idea_extractor import extract_ideas
from relation_extractor import extract_relations
from graph_builder import build_subgraphs

# Import new subgraph module
from subgraph_extractor import extract_ontological_subgraphs, ElectrotechnicalSubgraphExtractor

# Schema.org modules (if using both approaches)
try:
    from schema_org_extractor import extract_schema_org_markup
    from schema_org_relation_extractor import extract_schema_org_relations
    SCHEMA_ORG_AVAILABLE = True
except ImportError:
    SCHEMA_ORG_AVAILABLE = False
    print("Schema.org modules not available - running without Schema.org integration")

class IntegratedOntologyPipeline:
    """
    Integrated pipeline combining traditional extraction with OLLM-style subgraph generation.
    """
    
    def __init__(self, output_dir: str = "../data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging("../logs", "integrated_ontology_pipeline")
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "traditional_extraction": {},
            "subgraph_extraction": {},
            "schema_org_extraction": {},
            "comparison": {}
        }
    
    def run_integrated_pipeline(self, max_chunks: int = None, compare_approaches: bool = True) -> Dict[str, Any]:
        """
        Run integrated ontology extraction using multiple approaches.
        
        Args:
            max_chunks: Maximum number of chunks to process
            compare_approaches: Whether to compare different extraction methods
            
        Returns:
            Dictionary containing results from all approaches
        """
        print("=" * 60)
        print("INTEGRATED ONTOLOGY EXTRACTION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n📄 Loading and chunking documents...")
        chunks = load_and_split_data()
        if max_chunks:
            chunks = chunks[:max_chunks]
        print(f"   Processing {len(chunks)} document chunks")
        
        # Step 2: Traditional extraction approach
        print("\n🔧 Running traditional extraction approach...")
        traditional_results = self._run_traditional_extraction(chunks)
        self.results["traditional_extraction"] = traditional_results
        
        # Step 3: OLLM-style subgraph extraction
        print("\n🌐 Running OLLM-style subgraph extraction...")
        subgraph_results = self._run_subgraph_extraction(chunks)
        self.results["subgraph_extraction"] = subgraph_results
        
        # Step 4: Schema.org extraction (if available)
        if SCHEMA_ORG_AVAILABLE:
            print("\n🏗️ Running Schema.org extraction...")
            schema_results = self._run_schema_org_extraction(chunks, traditional_results)
            self.results["schema_org_extraction"] = schema_results
        
        # Step 5: Compare approaches
        if compare_approaches:
            print("\n📊 Comparing extraction approaches...")
            comparison = self._compare_approaches()
            self.results["comparison"] = comparison
        
        # Step 6: Save results
        print("\n💾 Saving results...")
        self._save_integrated_results()
        
        # Step 7: Generate report
        print("\n📋 Generating integration report...")
        self._generate_integration_report()
        
        print("\n" + "=" * 60)
        print("INTEGRATED PIPELINE COMPLETED!")
        print("=" * 60)
        
        return self.results
    
    def _run_traditional_extraction(self, chunks: List) -> Dict[str, Any]:
        """Run the original extraction approach."""
        # Extract concepts and relations
        concepts = extract_ideas(chunks)
        relations = extract_relations(chunks)
        
        # Build graph (but don't visualize to avoid dependencies)
        print(f"   Extracted {len(concepts)} concepts and {len(relations)} relations")
        
        return {
            "concepts": concepts,
            "relations": relations,
            "concept_count": len(concepts),
            "relation_count": len(relations)
        }
    
    def _run_subgraph_extraction(self, chunks: List) -> Dict[str, Any]:
        """Run OLLM-style subgraph extraction."""
        # Extract hierarchical subgraphs
        ontology = extract_ontological_subgraphs(chunks, max_path_length=4)
        
        # Save subgraph ontology
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subgraph_file = self.output_dir / f"ontology_subgraphs_{timestamp}.json"
        
        extractor = ElectrotechnicalSubgraphExtractor()
        extractor.export_subgraphs(ontology, str(subgraph_file))
        
        return {
            "ontology": ontology,
            "output_file": str(subgraph_file),
            "stats": ontology["stats"]
        }
    
    def _run_schema_org_extraction(self, chunks: List, traditional_results: Dict) -> Dict[str, Any]:
        """Run Schema.org extraction approach."""
        concepts = traditional_results["concepts"]
        
        # Generate Schema.org markup
        schema_objects = extract_schema_org_markup(chunks, concepts)
        
        # Extract additional properties and relations
        relations_data = extract_schema_org_relations(chunks, concepts)
        
        return {
            "schema_objects": schema_objects,
            "relations_data": relations_data,
            "object_count": len(schema_objects),
            "enhanced_count": len(relations_data)
        }
    
    def _compare_approaches(self) -> Dict[str, Any]:
        """Compare different extraction approaches."""
        comparison = {
            "concept_overlap": {},
            "coverage_analysis": {},
            "structural_differences": {},
            "recommendations": []
        }
        
        # Get concepts from each approach
        traditional_concepts = set(self.results["traditional_extraction"]["concepts"])
        
        subgraph_concepts = set()
        for node in self.results["subgraph_extraction"]["ontology"]["nodes"]:
            subgraph_concepts.add(node["name"])
        
        # Calculate overlaps
        overlap = traditional_concepts.intersection(subgraph_concepts)
        traditional_only = traditional_concepts - subgraph_concepts
        subgraph_only = subgraph_concepts - traditional_concepts
        
        comparison["concept_overlap"] = {
            "total_traditional": len(traditional_concepts),
            "total_subgraph": len(subgraph_concepts),
            "overlap_count": len(overlap),
            "overlap_percentage": len(overlap) / len(traditional_concepts.union(subgraph_concepts)) * 100,
            "traditional_only": list(traditional_only)[:10],  # Sample
            "subgraph_only": list(subgraph_only)[:10]  # Sample
        }
        
        # Analyze coverage
        comparison["coverage_analysis"] = {
            "traditional_relations": self.results["traditional_extraction"]["relation_count"],
            "subgraph_relations": self.results["subgraph_extraction"]["stats"]["total_relations"],
            "hierarchical_structure": "present" if self.results["subgraph_extraction"]["stats"]["total_relations"] > 0 else "absent"
        }
        
        # Generate recommendations
        recommendations = []
        
        if len(overlap) / len(traditional_concepts) < 0.3:
            recommendations.append("Low concept overlap suggests approaches are capturing different aspects - consider hybrid approach")
        
        if self.results["subgraph_extraction"]["stats"]["total_relations"] > self.results["traditional_extraction"]["relation_count"]:
            recommendations.append("Subgraph approach extracted more hierarchical relations - useful for taxonomic structure")
        
        if SCHEMA_ORG_AVAILABLE and self.results["schema_org_extraction"]["object_count"] > 0:
            recommendations.append("Schema.org approach provides web-compatible structured data")
        
        comparison["recommendations"] = recommendations
        
        return comparison
    
    def _save_integrated_results(self):
        """Save integrated pipeline results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results_file = self.output_dir / f"integrated_results_{timestamp}.json"
        
        # Create serializable version
        serializable_results = {
            "timestamp": self.results["timestamp"],
            "traditional_extraction": {
                "concept_count": self.results["traditional_extraction"]["concept_count"],
                "relation_count": self.results["traditional_extraction"]["relation_count"],
                "concepts": self.results["traditional_extraction"]["concepts"],
                "sample_relations": self.results["traditional_extraction"]["relations"][:10]
            },
            "subgraph_extraction": {
                "stats": self.results["subgraph_extraction"]["stats"],
                "output_file": self.results["subgraph_extraction"]["output_file"]
            },
            "comparison": self.results["comparison"]
        }
        
        if SCHEMA_ORG_AVAILABLE and self.results["schema_org_extraction"]:
            serializable_results["schema_org_extraction"] = {
                "object_count": self.results["schema_org_extraction"]["object_count"],
                "enhanced_count": self.results["schema_org_extraction"]["enhanced_count"]
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"   Saved integrated results: {results_file}")
    
    def _generate_integration_report(self):
        """Generate comprehensive integration report."""
        report_lines = [
            "INTEGRATED ONTOLOGY EXTRACTION REPORT",
            "=" * 60,
            f"Generated: {self.results['timestamp']}",
            "",
            "APPROACH COMPARISON:",
            f"  🔧 Traditional Extraction:",
            f"     Concepts: {self.results['traditional_extraction']['concept_count']}",
            f"     Relations: {self.results['traditional_extraction']['relation_count']}",
            "",
            f"  🌐 Subgraph Extraction (OLLM-style):",
            f"     Concepts: {self.results['subgraph_extraction']['stats']['total_concepts']}",
            f"     Relations: {self.results['subgraph_extraction']['stats']['total_relations']}",
            f"     Avg Concept Frequency: {self.results['subgraph_extraction']['stats']['avg_concept_frequency']:.2f}",
            ""
        ]
        
        if SCHEMA_ORG_AVAILABLE and self.results["schema_org_extraction"]:
            report_lines.extend([
                f"  🏗️ Schema.org Extraction:",
                f"     Objects: {self.results['schema_org_extraction']['object_count']}",
                f"     Enhanced: {self.results['schema_org_extraction']['enhanced_count']}",
                ""
            ])
        
        if self.results["comparison"]:
            comp = self.results["comparison"]
            report_lines.extend([
                "CONCEPT OVERLAP ANALYSIS:",
                f"  Overlap: {comp['concept_overlap']['overlap_count']} concepts",
                f"  Overlap rate: {comp['concept_overlap']['overlap_percentage']:.1f}%",
                f"  Traditional-only: {len(comp['concept_overlap']['traditional_only'])} concepts",
                f"  Subgraph-only: {len(comp['concept_overlap']['subgraph_only'])} concepts",
                ""
            ])
            
            if comp["recommendations"]:
                report_lines.extend([
                    "RECOMMENDATIONS:",
                    *[f"  • {rec}" for rec in comp["recommendations"]],
                    ""
                ])
        
        report_lines.extend([
            "KEY INSIGHTS:",
            "  • Traditional extraction: Good for explicit concept-relation pairs",
            "  • Subgraph extraction: Better for hierarchical taxonomic structure", 
            "  • Schema.org: Web-compatible structured data format",
            "  • Hybrid approach recommended for comprehensive ontology",
            "",
            "NEXT STEPS:",
            "  1. Review concept overlap to identify complementary strengths",
            "  2. Consider merging taxonomic structure with explicit relations",
            "  3. Use Schema.org format for web deployment",
            "  4. Evaluate ontology quality using domain expert validation",
            "",
            "=" * 60
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save and display report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"integration_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"Full report saved: {report_file}")

def run_integrated_ontology_extraction(max_chunks: int = None) -> Dict[str, Any]:
    """
    Main function to run the integrated ontology extraction pipeline.
    
    Args:
        max_chunks: Maximum chunks to process (for testing)
        
    Returns:
        Dictionary containing results from all approaches
    """
    pipeline = IntegratedOntologyPipeline()
    return pipeline.run_integrated_pipeline(max_chunks)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run integrated ontology extraction pipeline")
    parser.add_argument("--max-chunks", type=int, help="Maximum chunks to process")
    parser.add_argument("--no-comparison", action="store_true", help="Skip approach comparison")
    
    args = parser.parse_args()
    
    try:
        results = run_integrated_ontology_extraction(
            max_chunks=args.max_chunks
        )
        print(f"\nIntegrated pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
