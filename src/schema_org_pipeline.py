#!/usr/bin/env python3
"""
Complete Schema.org ontology extraction pipeline.
Integrates all modules to create a full Schema.org knowledge graph from technical documents.
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

# Import new Schema.org modules
from schema_org_extractor import extract_schema_org_markup
from schema_org_relation_extractor import extract_schema_org_relations
from schema_org_graph_builder import build_schema_org_knowledge_graph

class SchemaOrgPipeline:
    """Complete pipeline for Schema.org ontology extraction and knowledge graph creation."""
    
    def __init__(self, output_dir: str = "../data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging("../logs", "schema_org_pipeline")
        
        # Pipeline results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "chunks": [],
            "concepts": [],
            "relations": [],
            "schema_objects": [],
            "enhanced_schema_objects": [],
            "graph_stats": {}
        }
    
    def run_complete_pipeline(self, max_chunks: int = None) -> Dict[str, Any]:
        """
        Run the complete Schema.org extraction and knowledge graph creation pipeline.
        
        Args:
            max_chunks: Maximum number of chunks to process (None for all)
            
        Returns:
            Dictionary containing all pipeline results and statistics
        """
        print("=" * 60)
        print("SCHEMA.ORG ONTOLOGY EXTRACTION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and chunk documents
        print("\n📄 Step 1: Loading and chunking documents...")
        chunks = load_and_split_data()
        if max_chunks:
            chunks = chunks[:max_chunks]
        self.results["chunks"] = len(chunks)
        print(f"   Loaded {len(chunks)} document chunks")
        
        # Step 2: Extract concepts using existing LLM pipeline
        print("\n🧠 Step 2: Extracting concepts...")
        concepts = extract_ideas(chunks)
        self.results["concepts"] = concepts
        print(f"   Extracted {len(concepts)} unique concepts: {concepts[:10]}...")
        
        # Step 3: Extract traditional relations (for comparison)
        print("\n🔗 Step 3: Extracting traditional relations...")
        relations = extract_relations(chunks)
        self.results["relations"] = relations
        print(f"   Extracted {len(relations)} relations: {relations[:5]}...")
        
        # Step 4: Generate Schema.org markup
        print("\n🌐 Step 4: Generating Schema.org markup...")
        schema_objects = extract_schema_org_markup(chunks, concepts)
        self.results["schema_objects"] = schema_objects
        print(f"   Generated {len(schema_objects)} Schema.org objects")
        
        # Step 5: Extract Schema.org properties and relations
        print("\n⚙️ Step 5: Extracting Schema.org properties and relations...")
        relations_data = extract_schema_org_relations(chunks, concepts)
        print(f"   Extracted detailed data for {len(relations_data)} concepts")
        
        # Step 6: Enhance Schema.org objects with properties and relations
        print("\n✨ Step 6: Enhancing Schema.org objects...")
        from schema_org_relation_extractor import SchemaOrgRelationExtractor
        extractor = SchemaOrgRelationExtractor()
        enhanced_objects = extractor.generate_enhanced_schema_objects(schema_objects, relations_data)
        self.results["enhanced_schema_objects"] = enhanced_objects
        print(f"   Enhanced {len(enhanced_objects)} Schema.org objects with properties")
        
        # Step 7: Build Neo4j knowledge graph
        print("\n🗃️ Step 7: Building Neo4j knowledge graph...")
        try:
            graph_stats = build_schema_org_knowledge_graph(enhanced_objects)
            self.results["graph_stats"] = graph_stats
            print(f"   Created knowledge graph with {graph_stats.get('totals', {}).get('nodes', 0)} nodes")
        except Exception as e:
            print(f"   Warning: Graph creation failed: {e}")
            self.results["graph_stats"] = {"error": str(e)}
        
        # Step 8: Save all outputs
        print("\n💾 Step 8: Saving results...")
        self._save_pipeline_outputs()
        
        # Step 9: Generate summary report
        print("\n📊 Step 9: Generating summary report...")
        self._generate_summary_report()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return self.results
    
    # In schema_org_pipeline.py, around line 130-150, replace the individual object saving section:

    # Save individual objects for inspection - FIXED FILENAME HANDLING
    objects_dir = self.output_dir / f"schema_objects_{timestamp}"
    objects_dir.mkdir(exist_ok=True)

    def sanitize_filename(name):
        """Create Windows-safe filename"""
        import re
        # Replace all invalid Windows filename characters
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name))
        # Remove multiple consecutive underscores
        safe_name = re.sub(r'_{2,}', '_', safe_name)
        # Remove leading/trailing dots and underscores
        safe_name = safe_name.strip('._')
        # Limit length
        safe_name = safe_name[:80]
        return safe_name if safe_name else "component"

    for i, obj in enumerate(self.results["enhanced_schema_objects"]):
        obj_name = obj.get('name', 'unknown')
        safe_name = sanitize_filename(obj_name)
        
        obj_file = objects_dir / f"object_{i:03d}_{safe_name}.json"
        
        try:
            with open(obj_file, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Ultimate fallback - use just the number
            fallback_file = objects_dir / f"object_{i:03d}.json"
            print(f"   Using fallback filename for object {i}: {e}")
            with open(fallback_file, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

print(f"   Saved {len(self.results['enhanced_schema_objects'])} individual objects")
    def _generate_summary_report(self) -> None:
        """Generate a human-readable summary report."""
        report_lines = [
            "SCHEMA.ORG ONTOLOGY EXTRACTION REPORT",
            "=" * 50,
            f"Generated: {self.results['timestamp']}",
            "",
            "PROCESSING SUMMARY:",
            f"  📄 Document chunks processed: {self.results['chunks']}",
            f"  🧠 Concepts extracted: {len(self.results['concepts'])}",
            f"  🔗 Relations extracted: {len(self.results['relations'])}",
            f"  🌐 Schema.org objects created: {len(self.results['schema_objects'])}",
            f"  ✨ Enhanced objects: {len(self.results['enhanced_schema_objects'])}",
            "",
            "EXTRACTED CONCEPTS:",
        ]
        
        for i, concept in enumerate(self.results['concepts'][:20], 1):
            report_lines.append(f"  {i:2d}. {concept}")
        
        if len(self.results['concepts']) > 20:
            report_lines.append(f"  ... and {len(self.results['concepts']) - 20} more")
        
        report_lines.extend([
            "",
            "SAMPLE SCHEMA.ORG OBJECTS:",
        ])
        
        for i, obj in enumerate(self.results['enhanced_schema_objects'][:3], 1):
            report_lines.append(f"  {i}. {obj.get('name', 'Unknown')}")
            report_lines.append(f"     Type: {obj.get('@type', 'Unknown')}")
            report_lines.append(f"     Category: {obj.get('category', 'N/A')}")
            if 'additionalType' in obj:
                report_lines.append(f"     Additional Type: {obj['additionalType']}")
            
            # Show some properties
            props = []
            for key, value in obj.items():
                if key.startswith('elec:') and len(props) < 3:
                    props.append(f"{key}: {value}")
            if props:
                report_lines.append(f"     Properties: {', '.join(props)}")
            report_lines.append("")
        
        if "error" not in self.results["graph_stats"]:
            report_lines.extend([
                "KNOWLEDGE GRAPH STATISTICS:",
                f"  📊 Total nodes: {self.results['graph_stats'].get('totals', {}).get('nodes', 0)}",
                f"  🔗 Total relationships: {self.results['graph_stats'].get('totals', {}).get('relationships', 0)}",
                "",
                "  Node types:"
            ])
            
            for node_type, count in self.results['graph_stats'].get('nodes', {}).items():
                report_lines.append(f"    {node_type}: {count}")
            
            report_lines.extend(["", "  Relationship types:"])
            for rel_type, count in self.results['graph_stats'].get('relationships', {}).items():
                report_lines.append(f"    {rel_type}: {count}")
        
        report_lines.extend([
            "",
            "NEXT STEPS:",
            "  1. Review generated Schema.org objects in the output files",
            "  2. Explore the Neo4j knowledge graph using Cypher queries", 
            "  3. Validate Schema.org markup using online validators",
            "  4. Consider extending with additional domain-specific properties",
            "",
            "=" * 50
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"extraction_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Print to console
        print(report_text)
        print(f"\nFull report saved: {report_file}")

def run_schema_org_pipeline(max_chunks: int = None) -> Dict[str, Any]:
    """
    Main function to run the complete Schema.org pipeline.
    
    Args:
        max_chunks: Maximum number of chunks to process (for testing)
        
    Returns:
        Dictionary containing pipeline results
    """
    pipeline = SchemaOrgPipeline()
    return pipeline.run_complete_pipeline(max_chunks)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Schema.org ontology extraction pipeline")
    parser.add_argument("--max-chunks", type=int, help="Maximum chunks to process (for testing)")
    parser.add_argument("--output-dir", type=str, default="../data", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results = run_schema_org_pipeline(max_chunks=args.max_chunks)
        print(f"\nPipeline completed successfully!")
        print(f"Processed {results['chunks']} chunks")
        print(f"Generated {len(results['enhanced_schema_objects'])} Schema.org objects")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
