#!/usr/bin/env python3
"""
Cached Schema.org ontology extraction pipeline with resume functionality.
Saves expensive LLM results to avoid re-running on debugging/iteration.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import pickle
import hashlib
from typing import Dict, List, Any
from datetime import datetime
from utils import setup_logging
import re

# Import existing modules
from data_loader import load_and_split_data
from idea_extractor import extract_ideas
from relation_extractor import extract_relations

# Import Schema.org modules
from schema_org_extractor import extract_schema_org_markup
from schema_org_relation_extractor import extract_schema_org_relations
from schema_org_graph_builder import build_schema_org_knowledge_graph

class CachedSchemaOrgPipeline:
    """Schema.org pipeline with intelligent caching and resume functionality."""
    
    def __init__(self, output_dir: str = "../data", cache_dir: str = "../cache"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        setup_logging("../logs", "cached_schema_org_pipeline")
        
        # Pipeline results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "chunks": 0,
            "concepts": [],
            "relations": [],
            "schema_objects": [],
            "enhanced_schema_objects": [],
            "graph_stats": {}
        }
        
        # Cache settings
        self.cache_enabled = True
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🔄 Cache directory: {self.cache_dir}")
        print(f"💾 Output directory: {self.output_dir}")
    
    def _get_cache_key(self, step_name: str, data_signature: str) -> str:
        """Generate unique cache key for a pipeline step."""
        return f"{step_name}_{data_signature}.pkl"
    
    def _create_data_signature(self, data, max_length: int = 1000) -> str:
        """Create a hash signature of input data for cache validation."""
        try:
            if isinstance(data, list):
                if len(data) == 0:
                    content = "empty_list"
                elif hasattr(data[0], 'page_content'):
                    # Document chunks
                    content = ''.join([chunk.page_content[:100] for chunk in data[:10]])
                elif isinstance(data[0], str):
                    # String list (concepts)
                    content = ''.join(data[:20])
                else:
                    content = str(data)[:max_length]
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True)[:max_length]
            else:
                content = str(data)[:max_length]
            
            return hashlib.md5(content.encode()).hexdigest()[:12]
        except Exception as e:
            print(f"   ⚠️ Warning: Could not create data signature: {e}")
            return f"sig_{datetime.now().strftime('%H%M%S')}"
    
    def _save_to_cache(self, step_name: str, data, input_signature: str, metadata: dict = None):
        """Save step results to cache file."""
        if not self.cache_enabled:
            return False
            
        cache_key = self._get_cache_key(step_name, input_signature)
        cache_file = self.cache_dir / cache_key
        
        cache_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "input_signature": input_signature,
            "data": data,
            "metadata": metadata or {},
            "data_size": len(data) if isinstance(data, (list, dict)) else 1
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f)
            
            size_info = f"{cache_entry['data_size']} items" if isinstance(data, list) else "1 object"
            print(f"   💾 Cached {step_name}: {size_info} -> {cache_file.name}")
            return True
            
        except Exception as e:
            print(f"   ⚠️ Failed to save cache for {step_name}: {e}")
            return False
    
    def _load_from_cache(self, step_name: str, input_signature: str):
        """Load step results from cache if available and valid."""
        if not self.cache_enabled:
            return None
            
        cache_key = self._get_cache_key(step_name, input_signature)
        cache_file = self.cache_dir / cache_key
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_entry = pickle.load(f)
            
            # Verify signatures match
            if cache_entry.get("input_signature") == input_signature:
                age_hours = (datetime.now() - datetime.fromisoformat(cache_entry["timestamp"])).total_seconds() / 3600
                size_info = f"{cache_entry.get('data_size', '?')} items" if isinstance(cache_entry['data'], list) else "1 object"
                print(f"   🔄 Using cached {step_name}: {size_info} (cached {age_hours:.1f}h ago)")
                return cache_entry["data"]
            else:
                print(f"   ⚠️ Cache signature mismatch for {step_name}, will regenerate")
                return None
                
        except Exception as e:
            print(f"   ⚠️ Failed to load cache for {step_name}: {e}")
            return None
    
    def run_complete_pipeline(self, max_chunks: int = None, resume_from: str = None, 
                            force_refresh: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline with intelligent caching.
        
        Args:
            max_chunks: Maximum number of chunks to process
            resume_from: Step to resume from ('concepts', 'relations', 'schema', 'enhance', 'graph')
            force_refresh: List of steps to force refresh (ignore cache)
        """
        force_refresh = force_refresh or []
        
        print("=" * 70)
        print("🚀 CACHED SCHEMA.ORG ONTOLOGY EXTRACTION PIPELINE")
        print(f"📅 Session ID: {self.session_id}")
        print(f"🎯 Max chunks: {max_chunks or 'All'}")
        print(f"⏯️  Resume from: {resume_from or 'Beginning'}")
        if force_refresh:
            print(f"🔄 Force refresh: {', '.join(force_refresh)}")
        print("=" * 70)
        
        # Step 1: Load documents (always run first, but cache results)
        if resume_from is None:
            chunks, chunks_signature = self._cached_load_documents(max_chunks, 'documents' in force_refresh)
            self.results["chunks"] = len(chunks)
        else:
            print(f"\n⏯️  Resuming from step: {resume_from}")
            chunks, chunks_signature = self._load_cached_chunks()
            self.results["chunks"] = len(chunks)
            if not chunks:
                print("❌ No cached chunks found! Run full pipeline first.")
                return self.results
        
        # Step 2: Extract concepts
        if resume_from is None or resume_from == 'concepts':
            concepts = self._cached_extract_concepts(chunks, chunks_signature, 'concepts' in force_refresh)
            self.results["concepts"] = concepts
        else:
            concepts = self._load_cached_step_data("concepts", chunks_signature)
            self.results["concepts"] = concepts or []
        
        # Step 3: Extract traditional relations
        if resume_from is None or resume_from == 'relations':
            relations = self._cached_extract_relations(chunks, chunks_signature, 'relations' in force_refresh)
            self.results["relations"] = relations
        else:
            relations = self._load_cached_step_data("relations", chunks_signature)
            self.results["relations"] = relations or []
        
        # Step 4: Generate Schema.org markup
        if resume_from is None or resume_from == 'schema':
            schema_signature = self._create_data_signature(chunks + self.results["concepts"])
            schema_objects = self._cached_generate_schema(chunks, self.results["concepts"], 
                                                        schema_signature, 'schema' in force_refresh)
            self.results["schema_objects"] = schema_objects
        else:
            schema_signature = self._create_data_signature(chunks + self.results["concepts"])
            schema_objects = self._load_cached_step_data("schema_objects", schema_signature)
            self.results["schema_objects"] = schema_objects or []
        
        # Step 5: Extract Schema.org properties and relations
        if resume_from is None or resume_from in ['enhance', 'properties']:
            relations_data = self._cached_extract_schema_relations(chunks, self.results["concepts"], 
                                                                 schema_signature, 'properties' in force_refresh)
        else:
            relations_data = self._load_cached_step_data("relations_data", schema_signature)
            if not relations_data:
                relations_data = {}
        
        # Step 6: Enhance Schema.org objects
        if resume_from is None or resume_from == 'enhance':
            enhanced_objects = self._cached_enhance_objects(self.results["schema_objects"], relations_data,
                                                          'enhance' in force_refresh)
            self.results["enhanced_schema_objects"] = enhanced_objects
        else:
            enhanced_signature = self._create_data_signature(str(self.results["schema_objects"]) + str(relations_data))
            enhanced_objects = self._load_cached_step_data("enhanced_objects", enhanced_signature)
            self.results["enhanced_schema_objects"] = enhanced_objects or []
        
        # Step 7: Build Neo4j knowledge graph (can always be re-run)
        if resume_from is None or resume_from == 'graph':
            self._build_knowledge_graph(self.results["enhanced_schema_objects"])
        
        # Step 8 & 9: Always save results and generate reports
        print("\n💾 Step 8: Saving results...")
        self._save_pipeline_outputs()
        
        print("\n📊 Step 9: Generating summary report...")
        self._generate_summary_report()
        
        print(f"\n🎉 Pipeline completed successfully! Session: {self.session_id}")
        print("=" * 70)
        return self.results
    
    def _cached_load_documents(self, max_chunks: int, force_refresh: bool = False):
        """Load and chunk documents with caching."""
        print("\n📄 Step 1: Loading and chunking documents...")
        
        doc_signature = f"docs_{max_chunks or 'all'}"
        
        if not force_refresh:
            cached_chunks = self._load_from_cache("documents", doc_signature)
            if cached_chunks:
                return cached_chunks, doc_signature
        
        # Load fresh documents
        print("   📖 Loading documents from disk...")
        chunks = load_and_split_data()
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        print(f"   ✅ Loaded {len(chunks)} document chunks")
        self._save_to_cache("documents", chunks, doc_signature, 
                          {"max_chunks": max_chunks, "total_chunks": len(chunks)})
        
        return chunks, doc_signature
    
    def _cached_extract_concepts(self, chunks, chunks_signature: str, force_refresh: bool = False):
        """Extract concepts with caching."""
        print("\n🧠 Step 2: Extracting concepts...")
        
        if not force_refresh:
            cached_concepts = self._load_from_cache("concepts", chunks_signature)
            if cached_concepts:
                print(f"   ✅ Using {len(cached_concepts)} cached concepts")
                return cached_concepts
        
        # Extract fresh concepts
        print("   🤖 Running LLM concept extraction...")
        concepts = extract_ideas(chunks)
        
        print(f"   ✅ Extracted {len(concepts)} unique concepts")
        self._save_to_cache("concepts", concepts, chunks_signature)
        
        return concepts
    
    def _cached_extract_relations(self, chunks, chunks_signature: str, force_refresh: bool = False):
        """Extract relations with caching."""
        print("\n🔗 Step 3: Extracting traditional relations...")
        
        if not force_refresh:
            cached_relations = self._load_from_cache("relations", chunks_signature)
            if cached_relations:
                print(f"   ✅ Using {len(cached_relations)} cached relations")
                return cached_relations
        
        # Extract fresh relations
        print("   🤖 Running LLM relation extraction...")
        relations = extract_relations(chunks)
        
        print(f"   ✅ Extracted {len(relations)} relations")
        self._save_to_cache("relations", relations, chunks_signature)
        
        return relations
    
    def _cached_generate_schema(self, chunks, concepts, schema_signature: str, force_refresh: bool = False):
        """Generate Schema.org objects with caching."""
        print("\n🌐 Step 4: Generating Schema.org markup...")
        
        if not force_refresh:
            cached_schema = self._load_from_cache("schema_objects", schema_signature)
            if cached_schema:
                print(f"   ✅ Using {len(cached_schema)} cached Schema.org objects")
                return cached_schema
        
        # Generate fresh schema objects
        print("   🤖 Running LLM Schema.org generation...")
        schema_objects = extract_schema_org_markup(chunks, concepts)
        
        print(f"   ✅ Generated {len(schema_objects)} Schema.org objects")
        self._save_to_cache("schema_objects", schema_objects, schema_signature)
        
        return schema_objects
    
    def _cached_extract_schema_relations(self, chunks, concepts, schema_signature: str, force_refresh: bool = False):
        """Extract Schema.org properties and relations with caching."""
        print("\n⚙️ Step 5: Extracting Schema.org properties and relations...")
        
        if not force_refresh:
            cached_relations_data = self._load_from_cache("relations_data", schema_signature)
            if cached_relations_data:
                print(f"   ✅ Using cached property data for {len(cached_relations_data)} concepts")
                return cached_relations_data
        
        # Extract fresh relations data
        print("   🤖 Running LLM property extraction...")
        relations_data = extract_schema_org_relations(chunks, concepts)
        
        print(f"   ✅ Extracted detailed data for {len(relations_data)} concepts")
        self._save_to_cache("relations_data", relations_data, schema_signature)
        
        return relations_data
    
    def _cached_enhance_objects(self, schema_objects, relations_data, force_refresh: bool = False):
        """Enhance Schema.org objects with caching."""
        print("\n✨ Step 6: Enhancing Schema.org objects...")
        
        enhance_signature = self._create_data_signature(str(schema_objects) + str(relations_data))
        
        if not force_refresh:
            cached_enhanced = self._load_from_cache("enhanced_objects", enhance_signature)
            if cached_enhanced:
                print(f"   ✅ Using {len(cached_enhanced)} cached enhanced objects")
                return cached_enhanced
        
        # Enhance fresh objects
        print("   ⚡ Enhancing Schema.org objects with properties...")
        from schema_org_relation_extractor import SchemaOrgRelationExtractor
        extractor = SchemaOrgRelationExtractor()
        enhanced_objects = extractor.generate_enhanced_schema_objects(schema_objects, relations_data)
        
        print(f"   ✅ Enhanced {len(enhanced_objects)} Schema.org objects")
        self._save_to_cache("enhanced_objects", enhanced_objects, enhance_signature)
        
        return enhanced_objects
    
    def _build_knowledge_graph(self, enhanced_objects):
        """Build Neo4j knowledge graph (always runs fresh)."""
        print("\n🗃️ Step 7: Building Neo4j knowledge graph...")
        
        try:
            graph_stats = build_schema_org_knowledge_graph(enhanced_objects)
            self.results["graph_stats"] = graph_stats
            node_count = graph_stats.get('totals', {}).get('nodes', 0)
            rel_count = graph_stats.get('totals', {}).get('relationships', 0)
            print(f"   ✅ Created knowledge graph: {node_count} nodes, {rel_count} relationships")
        except Exception as e:
            print(f"   ⚠️ Graph creation failed: {e}")
            self.results["graph_stats"] = {"error": str(e)}
    
    def _load_cached_chunks(self):
        """Load cached document chunks."""
        cache_files = list(self.cache_dir.glob("documents_*.pkl"))
        if not cache_files:
            return [], ""
        
        # Use most recent cache
        latest_cache = max(cache_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_cache, 'rb') as f:
                cache_entry = pickle.load(f)
            chunks = cache_entry["data"]
            signature = cache_entry["input_signature"]
            print(f"   🔄 Loaded {len(chunks)} cached chunks from {latest_cache.name}")
            return chunks, signature
        except Exception as e:
            print(f"   ⚠️ Failed to load cached chunks: {e}")
            return [], ""
    
    def _load_cached_step_data(self, step_name: str, expected_signature: str):
        """Generic method to load cached step data."""
        cached_data = self._load_from_cache(step_name, expected_signature)
        if cached_data:
            return cached_data
        
        # Try to find any cache file for this step
        cache_files = list(self.cache_dir.glob(f"{step_name}_*.pkl"))
        if cache_files:
            latest_cache = max(cache_files, key=lambda f: f.stat().st_mtime)
            print(f"   ⚠️ Using latest available {step_name} cache: {latest_cache.name}")
            try:
                with open(latest_cache, 'rb') as f:
                    cache_entry = pickle.load(f)
                return cache_entry["data"]
            except Exception as e:
                print(f"   ❌ Failed to load fallback cache: {e}")
        
        return None
    
    def _save_pipeline_outputs(self):
        """Save pipeline outputs with safe filename handling."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main Schema.org JSON-LD file
        schema_file = self.output_dir / f"schema_org_objects_{timestamp}.jsonld"
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump({
                "@context": "https://schema.org/",
                "@graph": self.results["enhanced_schema_objects"]
            }, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved main Schema.org file: {schema_file.name}")
        
        # Save individual objects with safe filenames
        objects_dir = self.output_dir / f"schema_objects_{timestamp}"
        objects_dir.mkdir(exist_ok=True)
        
        def sanitize_filename(name: str) -> str:
            """Create safe filename for Windows/Unix."""
            safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name))
            safe_name = re.sub(r'_{2,}', '_', safe_name)
            safe_name = safe_name.strip('._')
            return safe_name[:80] if safe_name else "component"
        
        saved_count = 0
        for i, obj in enumerate(self.results["enhanced_schema_objects"]):
            obj_name = obj.get('name', f'component_{i}')
            safe_name = sanitize_filename(obj_name)
            obj_file = objects_dir / f"object_{i:03d}_{safe_name}.json"
            
            try:
                with open(obj_file, 'w', encoding='utf-8') as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
                saved_count += 1
            except Exception as e:
                # Ultimate fallback
                fallback_file = objects_dir / f"object_{i:03d}.json"
                with open(fallback_file, 'w', encoding='utf-8') as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
                saved_count += 1
                print(f"   ⚠️ Used fallback filename for object {i}")
        
        print(f"   ✅ Saved {saved_count} individual objects")
        
        # Save pipeline summary
        results_file = self.output_dir / f"pipeline_results_{timestamp}.json"
        summary = {
            "timestamp": self.results["timestamp"],
            "session_id": self.session_id,
            "chunks_processed": self.results["chunks"],
            "concepts_extracted": len(self.results["concepts"]),
            "concepts": self.results["concepts"],
            "relations_extracted": len(self.results["relations"]),
            "schema_objects_created": len(self.results["schema_objects"]),
            "enhanced_objects_created": len(self.results["enhanced_schema_objects"]),
            "graph_statistics": self.results["graph_stats"]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved pipeline summary: {results_file.name}")
    
    def _generate_summary_report(self):
        """Generate human-readable summary report."""
        concepts_count = len(self.results["concepts"])
        objects_count = len(self.results["enhanced_schema_objects"])
        
        report_lines = [
            "🚀 CACHED SCHEMA.ORG PIPELINE REPORT",
            "=" * 60,
            f"📅 Generated: {self.results['timestamp']}",
            f"🆔 Session ID: {self.session_id}",
            "",
            "📊 PROCESSING SUMMARY:",
            f"  📄 Document chunks: {self.results['chunks']}",
            f"  🧠 Concepts extracted: {concepts_count}",
            f"  🔗 Relations extracted: {len(self.results['relations'])}",
            f"  🌐 Schema.org objects: {len(self.results['schema_objects'])}",
            f"  ✨ Enhanced objects: {objects_count}",
        ]
        
        # Add graph stats if available
        if "error" not in str(self.results["graph_stats"]):
            stats = self.results["graph_stats"]
            if "totals" in stats:
                report_lines.extend([
                    "",
                    "🗃️ KNOWLEDGE GRAPH:",
                    f"  📊 Total nodes: {stats['totals'].get('nodes', 0)}",
                    f"  🔗 Total relationships: {stats['totals'].get('relationships', 0)}"
                ])
        
        # Add sample concepts
        if self.results["concepts"]:
            report_lines.extend([
                "",
                "🏷️ SAMPLE CONCEPTS:",
                *[f"  • {concept}" for concept in self.results["concepts"][:10]]
            ])
            if len(self.results["concepts"]) > 10:
                report_lines.append(f"  ... and {len(self.results['concepts']) - 10} more")
        
        # Add cache info
        cache_files = list(self.cache_dir.glob("*.pkl"))
        report_lines.extend([
            "",
            "💾 CACHE STATUS:",
            f"  🗂️ Cache files: {len(cache_files)}",
            f"  📁 Cache directory: {self.cache_dir}"
        ])
        
        report_lines.extend([
            "",
            "🎯 NEXT STEPS:",
            "  1. Review Schema.org objects in output files",
            "  2. Query Neo4j graph for component relationships",
            "  3. Use --resume-from to iterate on specific steps",
            "  4. Validate Schema.org markup online",
            "",
            "=" * 60
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"pipeline_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"📋 Full report saved: {report_file.name}")
    
    def list_cache(self):
        """List all available cache files with details."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        if not cache_files:
            print("📂 No cache files found")
            return
        
        print(f"📋 CACHE STATUS ({len(cache_files)} files)")
        print("-" * 60)
        
        cache_info = {}
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_entry = pickle.load(f)
                
                step_name = cache_entry.get("step_name", "unknown")
                timestamp = cache_entry.get("timestamp", "unknown")
                data_size = cache_entry.get("data_size", "?")
                
                if step_name not in cache_info:
                    cache_info[step_name] = []
                
                cache_info[step_name].append({
                    "file": cache_file.name,
                    "timestamp": timestamp,
                    "size": data_size,
                    "age_hours": (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() / 3600
                })
                
            except Exception as e:
                print(f"⚠️  {cache_file.name}: Error reading ({e})")
        
        for step_name, entries in cache_info.items():
            print(f"\n🔸 {step_name.upper()}:")
            for entry in sorted(entries, key=lambda x: x["age_hours"]):
                print(f"   📄 {entry['file']}")
                print(f"      Size: {entry['size']} items")
                print(f"      Age: {entry['age_hours']:.1f} hours")
        
        print(f"\n💡 Use --resume-from <step> to resume from cached results")
        print(f"💡 Use --force-refresh <step> to regenerate specific steps")
    
    def clear_cache(self, step_name: str = None):
        """Clear cache files."""
        if step_name and step_name != 'all':
            pattern = f"{step_name}_*.pkl"
            cache_files = list(self.cache_dir.glob(pattern))
            action = f"Clearing {step_name} cache"
        else:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            action = "Clearing all cache"
        
        print(f"🗑️  {action} ({len(cache_files)} files)...")
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"   ⚠️ Failed to delete {cache_file.name}: {e}")
        
        print(f"   ✅ Cleared {len(cache_files)} cache files")

def run_cached_pipeline(max_chunks: int = None, resume_from: str = None, 
                       force_refresh: str = None, list_cache: bool = False, 
                       clear_cache: str = None) -> Dict[str, Any]:
    """
    Main function to run the cached pipeline.
    
    Args:
        max_chunks: Maximum chunks to process (None for all)
        resume_from: Step to resume from
        force_refresh: Comma-separated steps to force refresh
        list_cache: Show cache status
        clear_cache: Clear cache ('all' or step name)
    """
    pipeline = CachedSchemaOrgPipeline()
    
    if list_cache:
        pipeline.list_cache()
        return {}
    
    if clear_cache:
        pipeline.clear_cache(clear_cache)
        return {}
    
    # Parse force refresh list
    force_refresh_list = []
    if force_refresh:
        force_refresh_list = [s.strip() for s in force_refresh.split(',')]
    
    return pipeline.run_complete_pipeline(
        max_chunks=max_chunks,
        resume_from=resume_from,
        force_refresh=force_refresh_list
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cached Schema.org ontology extraction pipeline")
    parser.add_argument("--max-chunks", type=int, help="Maximum chunks to process")
    parser.add_argument("--resume-from", 
                       choices=['concepts', 'relations', 'schema', 'enhance', 'graph'],
                       help="Step to resume from")
    parser.add_argument("--force-refresh", type=str,
                       help="Comma-separated steps to force refresh (e.g., 'concepts,schema')")
    parser.add_argument("--list-cache", action="store_true", help="List cached steps")
    parser.add_argument("--clear-cache", type=str, help="Clear cache ('all' or step name)")
    
    args = parser.parse_args()
    
    try:
        results = run_cached_pipeline(
            max_chunks=args.max_chunks,
            resume_from=args.resume_from,
            force_refresh=args.force_refresh,
            list_cache=args.list_cache,
            clear_cache=args.clear_cache
        )
        
        if results and results.get("enhanced_schema_objects"):
            print(f"\n🎉 SUCCESS! Generated {len(results['enhanced_schema_objects'])} enhanced Schema.org objects")
            
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()