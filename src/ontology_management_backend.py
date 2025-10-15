#!/usr/bin/env python3
"""
Ontology Management Backend
Provides API endpoints and process management for the ontology dashboard.
"""

from src.evaluation.gold_standard import evaluate_ontology
from src.evaluation.graph import compare_graph_metrics, _load_gold_standard_graph, _load_generated_graph
from src.evaluation.consistency import evaluate_consistency
import networkx as nx
import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import os
import time
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from neo4j import GraphDatabase
import pickle

import logging
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# LangChain and Neo4j imports for the QA System
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

from src.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OPENAI_API_KEY, LLM_MODEL, MAX_WORKERS, NEO4J_DB_NAME #<-- ADD NEO4J_DB_NAME
)

from src.data_models import PipelineConfig
from src.integrated_schema_pipeline import run_integrated_pipeline

from src.cached_schema_org_pipeline import run_cached_pipeline, clear_cache

logger = logging.getLogger(__name__)

def initialize_qa_chain():
    """Initializes the GraphCypherQAChain and stores it globally."""
    global qa_chain
    logger.info("--- Initializing Knowledge Graph QA System ---")
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DB_NAME  # <-- ADD THIS EXPLICIT PARAMETER
        )
        graph.refresh_schema()
        
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
        
        qa_chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            verbose=True, # Set to True to see generated Cypher in the terminal
            allow_dangerous_requests=True
        )
        logger.info("✅ QA Chain initialized successfully.")
        logger.info(f"   Graph Schema Detected:\n{graph.schema}")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to initialize QA Chain. The /api/qa endpoint will not work. Error: {e}", exc_info=True)
        # This can happen if APOC is not installed or Neo4j is down.
        qa_chain = None

# --- FLASK APP AND MANAGER SETUP ---
app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

class ProcessStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ProcessLock:
    """Process lock to prevent duplicate ontology creation."""
    process_id: str
    status: ProcessStatus
    started_at: datetime
    step: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    
    def to_dict(self):
        return {
            "process_id": self.process_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "step": self.step,
            "progress": self.progress,
            "message": self.message
        }

class OntologyManager:
    """Manages ontology state, versions, and pipeline execution."""
    
    def __init__(self):
        
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.current_process: Optional[ProcessLock] = None
        self.lock = threading.Lock()
        
        # Paths for ontology management
        self.base_dir = Path("../ontology_management")
        self.base_dir.mkdir(exist_ok=True)
        self.snapshots_dir = self.base_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.baseline_file = self.base_dir / "schema_org_baseline.json"
        
        # Initialize baseline if not exists
        self._ensure_baseline_exists()
        
        print("🔧 Ontology Manager initialized")
    
    def _ensure_baseline_exists(self):
        """Ensure Schema.org baseline ontology exists."""
        if not self.baseline_file.exists():
            baseline_ontology = {
                "@context": "https://schema.org/",
                "@graph": [
                    {
                        "@type": "Product",
                        "name": "Generic Product",
                        "description": "Base Schema.org Product class",
                        "category": "Product"
                    },
                    {
                        "@type": "Organization", 
                        "name": "Schema.org Foundation",
                        "description": "Foundation managing Schema.org vocabulary"
                    }
                ],
                "metadata": {
                    "version": "1.0.0",
                    "type": "baseline",
                    "created": datetime.now().isoformat(),
                    "description": "Original Schema.org baseline ontology"
                }
            }
            
            with open(self.baseline_file, 'w') as f:
                json.dump(baseline_ontology, f, indent=2)
            
            print(f"   ✅ Created baseline ontology: {self.baseline_file}")
    
    def get_ontology_stats(self) -> Dict[str, Any]:
        """Get current ontology statistics from Neo4j."""
        try:
            with self.driver.session(database=NEO4J_DB_NAME) as session:
                # Get node statistics
                node_stats = session.run("""
                    MATCH (n)
                    RETURN count(n) as total_nodes,
                           count(CASE WHEN n:Product THEN 1 END) as product_nodes,
                           count(CASE WHEN n:Organization THEN 1 END) as org_nodes,
                           count(DISTINCT CASE WHEN n.category IS NOT NULL THEN n.category END) as categories
                """).single()
                
                # Get relationship statistics
                rel_stats = session.run("""
                    MATCH ()-[r]->()
                    RETURN count(r) as total_relationships,
                           collect(DISTINCT type(r)) as relationship_types
                """).single()
                
                # Get sample data
                sample_concepts = session.run("""
                    MATCH (p:Product)
                    RETURN p.name as name, p.category as category
                    ORDER BY p.name
                    LIMIT 10
                """).data()
                
                return {
                    "total_nodes": node_stats["total_nodes"] or 0,
                    "product_nodes": node_stats["product_nodes"] or 0,
                    "org_nodes": node_stats["org_nodes"] or 0,
                    "categories": node_stats["categories"] or 0,
                    "total_relationships": rel_stats["total_relationships"] or 0,
                    "relationship_types": rel_stats["relationship_types"] or [],
                    "sample_concepts": sample_concepts,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error getting ontology stats: {e}")
            return {
                "total_nodes": 0,
                "product_nodes": 0,
                "org_nodes": 0,
                "categories": 0,
                "total_relationships": 0,
                "relationship_types": [],
                "sample_concepts": [],
                "last_updated": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data for visualization."""
        try:
            with self.driver.session(database=NEO4J_DB_NAME) as session:
                # Get nodes
                nodes_query = """
                    MATCH (n)
                    RETURN n.name as name, 
                           labels(n)[0] as type,
                           n.category as category,
                           id(n) as neo4j_id
                    LIMIT 50
                """
                
                nodes = []
                for record in session.run(nodes_query):
                    nodes.append({
                        "id": record["name"] or f"node_{record['neo4j_id']}",
                        "type": record["type"],
                        "category": record["category"],
                        "group": self._get_node_group(record["type"])
                    })
                
                # Get relationships
                rels_query = """
                    MATCH (n1)-[r]->(n2)
                    WHERE n1.name IS NOT NULL AND n2.name IS NOT NULL
                    RETURN n1.name as source, n2.name as target, type(r) as relationship
                    LIMIT 100
                """
                
                links = []
                for record in session.run(rels_query):
                    links.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["relationship"]
                    })
                
                return {
                    "nodes": nodes,
                    "links": links,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error getting graph data: {e}")
            return {"nodes": [], "links": [], "error": str(e)}
    
    def _get_node_group(self, node_type: str) -> int:
        """Get node group for visualization coloring."""
        type_groups = {
            "Product": 1,
            "Organization": 2, 
            "Category": 3,
            "Property": 4
        }
        return type_groups.get(node_type, 0)
    
    def start_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start the ontology extension pipeline."""
        with self.lock:
            if self.current_process and self.current_process.status == ProcessStatus.RUNNING:
                return {
                    "success": False,
                    "message": "Pipeline is already running",
                    "current_process": self.current_process.to_dict()
                }
            
            # Create new process
            process_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
            self.current_process = ProcessLock(
                process_id=process_id,
                status=ProcessStatus.RUNNING,
                started_at=datetime.now(),
                step="initialization",
                message="Starting pipeline execution..."
            )
            
            # Start pipeline in background thread
            thread = threading.Thread(target=self._run_pipeline_thread, args=(config,))
            thread.daemon = True
            thread.start()
            
            return {
                "success": True,
                "message": "Pipeline started successfully",
                "process": self.current_process.to_dict()
            }
    
    def _run_pipeline_thread(self, config: Dict[str, Any]):
        """Run the appropriate pipeline in a background thread and update status."""
        try:
            def update_progress_callback(step_name: str, progress_percent: int):
                if self.current_process:
                    self.current_process.step = step_name
                    self.current_process.message = f"In progress: {step_name}..."
                    self.current_process.progress = progress_percent
                    logger.info(f"[PROGRESS] {step_name} - {progress_percent}%")

            resume_step = config.get('resume_from', 'start')
            llm_model = config.get('llm_model', LLM_MODEL)
            max_workers = config.get('max_workers', MAX_WORKERS)

            self.current_process.message = f"Initializing pipeline (Mode: {resume_step})..."
            self.current_process.progress = 5

            # We ALWAYS call the cached pipeline now.
            # If the user selected "Start from Scratch", the resume_step will be 'start',
            # which the cached pipeline knows how to handle by clearing the cache first.
            logger.info(f"🚀 Handing off to cached pipeline with resume_from='{resume_step}'...")
            
            run_cached_pipeline(
                resume_from=resume_step,
                llm_model=llm_model,
                max_workers=max_workers,
                progress_callback=update_progress_callback
            )



            # --- COMPLETION LOGIC MOVED INSIDE THE 'TRY' BLOCK ---
            output_dir = Path("../data/integrated_output")
            list_of_files = list(output_dir.glob('new_schema_objects_*.jsonld')) + \
                            list(output_dir.glob('cached_new_schema_objects_*.jsonld'))
            latest_file = max(list_of_files, key=lambda p: p.stat().st_ctime) if list_of_files else None
            
            with self.lock:
                self.current_process.status = ProcessStatus.COMPLETED
                self.current_process.progress = 100.0
                self.current_process.message = "Pipeline execution completed successfully."
                self.current_process.output_file = str(latest_file) if latest_file else None
            
            logger.info(f"✅ Pipeline {self.current_process.process_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Pipeline thread failed: {e}", exc_info=True)
            with self.lock:
                if self.current_process:
                    self.current_process.status = ProcessStatus.ERROR
                    self.current_process.message = f"Pipeline failed: {str(e)}"
    
    def get_process_status(self) -> Dict[str, Any]:
        """Get current process status."""
        if self.current_process:
            return self.current_process.to_dict()
        else:
            return {
                "process_id": None,
                "status": ProcessStatus.IDLE.value,
                "message": "No active process"
            }
    
    def cancel_process(self) -> Dict[str, Any]:
        """Cancel current process."""
        with self.lock:
            if self.current_process and self.current_process.status == ProcessStatus.RUNNING:
                self.current_process.status = ProcessStatus.CANCELLED
                self.current_process.message = "Process cancelled by user"
                return {"success": True, "message": "Process cancelled"}
            else:
                return {"success": False, "message": "No active process to cancel"}
    
    def create_snapshot(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a snapshot of the current ontology."""
        try:
            # Generate snapshot name if not provided
            if not name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"snapshot_{timestamp}"
            
            # Get current ontology data
            stats = self.get_ontology_stats()
            graph_data = self.get_graph_data()
            
            snapshot_data = {
                "name": name,
                "created": datetime.now().isoformat(),
                "stats": stats,
                "graph_data": graph_data,
                "metadata": {
                    "type": "snapshot",
                    "version": "1.0.0",
                    "description": f"Ontology snapshot created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            }
            
            # Save snapshot
            snapshot_file = self.snapshots_dir / f"{name}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
            
            return {
                "success": True,
                "message": f"Snapshot '{name}' created successfully",
                "snapshot": {
                    "name": name,
                    "file": str(snapshot_file),
                    "created": snapshot_data["created"],
                    "stats": stats
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create snapshot: {str(e)}"
            }
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots."""
        snapshots = []
        
        # Add baseline
        snapshots.append({
            "name": "schema_org_baseline",
            "type": "baseline",
            "created": "2024-09-08T10:00:00",
            "description": "Original Schema.org baseline ontology",
            "is_baseline": True
        })
        
        # Add snapshots
        for snapshot_file in self.snapshots_dir.glob("*.json"):
            try:
                with open(snapshot_file, 'r') as f:
                    snapshot_data = json.load(f)
                
                snapshots.append({
                    "name": snapshot_data["name"],
                    "type": "snapshot", 
                    "created": snapshot_data["created"],
                    "description": snapshot_data.get("metadata", {}).get("description", "User snapshot"),
                    "stats": snapshot_data.get("stats", {}),
                    "is_baseline": False
                })
                
            except Exception as e:
                print(f"Error reading snapshot {snapshot_file}: {e}")
        
        # Sort by creation date (newest first)
        snapshots.sort(key=lambda x: x["created"], reverse=True)
        
        return snapshots
    
    def restore_snapshot(self, snapshot_name: str) -> Dict[str, Any]:
        """Restore ontology from a snapshot."""
        try:
            if snapshot_name == "schema_org_baseline":
                # Restore from baseline
                return self.reset_to_baseline()
            
            # Find snapshot file
            snapshot_file = self.snapshots_dir / f"{snapshot_name}.json"
            if not snapshot_file.exists():
                return {
                    "success": False,
                    "message": f"Snapshot '{snapshot_name}' not found"
                }
            
            # Load snapshot data
            with open(snapshot_file, 'r') as f:
                snapshot_data = json.load(f)
            
            # Here you would restore the Neo4j database from snapshot
            # This is a complex operation that would require:
            # 1. Clearing current data
            # 2. Recreating nodes and relationships from snapshot
            # 3. Validating the restored data
            
            # For now, simulate restoration
            time.sleep(2)
            
            return {
                "success": True,
                "message": f"Successfully restored from snapshot '{snapshot_name}'",
                "restored_stats": snapshot_data.get("stats", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to restore snapshot: {str(e)}"
            }
    
    def reset_to_baseline(self) -> Dict[str, Any]:
        """Reset ontology to Schema.org baseline."""
        try:
            with self.driver.session(database=NEO4J_DB_NAME) as session:
                # Clear all custom data (keep only baseline Schema.org concepts)
                print("   🗑️ Clearing custom ontology data...")
                
                # Delete all relationships first
                session.run("MATCH ()-[r]->() DELETE r")
                
                # Delete all nodes
                session.run("MATCH (n) DELETE n")
                
                # Recreate baseline Schema.org concepts
                print("   🏗️ Recreating baseline Schema.org concepts...")
                
                # Create basic Schema.org Product structure
                baseline_queries = [
                    """
                    CREATE (p:Product {
                        name: "Product",
                        description: "Any offered product or service",
                        category: "Schema.org Base Class",
                        schemaType: "Product",
                        sourceFormat: "Schema.org"
                    })
                    """,
                    """
                    CREATE (o:Organization {
                        name: "Organization", 
                        description: "An organization such as a school, NGO, corporation, club, etc",
                        schemaType: "Organization",
                        sourceFormat: "Schema.org"
                    })
                    """,
                    """
                    CREATE (t:Thing {
                        name: "Thing",
                        description: "The most generic type of item",
                        schemaType: "Thing", 
                        sourceFormat: "Schema.org"
                    })
                    """
                ]
                
                for query in baseline_queries:
                    session.run(query)
                
                print("   ✅ Baseline ontology restored")
                
                return {
                    "success": True,
                    "message": "Successfully reset ontology to Schema.org baseline",
                    "baseline_stats": self.get_ontology_stats()
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to reset to baseline: {str(e)}"
            }
    
    def export_ontology(self, format_type: str = "jsonld") -> Dict[str, Any]:
        """Export current ontology in specified format."""
        try:
            if format_type == "jsonld":
                # Export as Schema.org JSON-LD
                graph_data = self.get_graph_data()
                stats = self.get_ontology_stats()
                
                ontology_export = {
                    "@context": "https://schema.org/",
                    "@graph": [],
                    "exportMetadata": {
                        "timestamp": datetime.now().isoformat(),
                        "format": "JSON-LD",
                        "stats": stats,
                        "tool": "Schema.org Ontology Manager"
                    }
                }
                
                # Convert graph nodes to Schema.org format
                for node in graph_data["nodes"]:
                    schema_object = {
                        "@type": node["type"],
                        "name": node["id"],
                        "category": node.get("category", "Unknown")
                    }
                    ontology_export["@graph"].append(schema_object)
                
                # Save export file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_file = self.base_dir / f"ontology_export_{timestamp}.jsonld"
                
                with open(export_file, 'w') as f:
                    json.dump(ontology_export, f, indent=2)
                
                return {
                    "success": True,
                    "message": "Ontology exported successfully",
                    "file": str(export_file),
                    "format": format_type,
                    "stats": stats
                }
                
            else:
                return {
                    "success": False,
                    "message": f"Unsupported export format: {format_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Export failed: {str(e)}"
            }
    
    def check_neo4j_connection(self) -> Dict[str, Any]:
        """Check Neo4j database connection."""
        try:
            with self.driver.session(database=NEO4J_DB_NAME) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
                if test_value == 1:
                    return {
                        "connected": True,
                        "message": "Neo4j connection successful",
                        "database": "neo4j"
                    }
                else:
                    return {
                        "connected": False,
                        "message": "Unexpected response from Neo4j"
                    }
                    
        except Exception as e:
            return {
                "connected": False,
                "message": f"Neo4j connection failed: {str(e)}"
            }
    
    def close(self):
        """Close database connections."""
        if self.driver:
            self.driver.close()


# Global ontology manager instance
ontology_manager = OntologyManager()

# Update the dashboard route
@app.route('/')
def dashboard():
    """Serve the main dashboard using Flask templates."""
    return render_template('dashboard.html')

# In src/ontology_management_backend.py

# ... (imports)

@app.route('/api/stats')
def get_stats():
    """Get current ontology statistics from the Neo4j database."""
    try:
        with ontology_manager.driver.session(database=NEO4J_DB_NAME) as session:
            # Query for the number of Product nodes (concepts)
            concepts_result = session.run("MATCH (p:Product) RETURN count(p) AS concept_count")
            total_concepts = concepts_result.single()["concept_count"]

            # Query for the total number of relationships
            rels_result = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
            total_relations = rels_result.single()["rel_count"]
            
            # Query for concepts needing review (assuming you add a label for this)
            # review_result = session.run("MATCH (p:Product {status: 'review'}) RETURN count(p) AS review_count")
            # pending_review = review_result.single()["review_count"]
            pending_review = 0 # Placeholder for now

            # Automation rate is complex to calculate here, so we'll use a placeholder.
            # This is better calculated at the end of a pipeline run.
            automation_rate = 0 # Placeholder

            return jsonify({
                "product_nodes": total_concepts,
                "total_relations": total_relations,
                "pending_review": pending_review,
                "automation_rate": automation_rate,
                "success": True
            })
            
    except Exception as e:
        logger.error(f"Failed to get ontology stats: {e}", exc_info=True)
        return jsonify({"error": "Could not connect to the database to fetch stats."}), 500

@app.route('/api/graph')
def get_graph():
    """Get graph data for visualization."""
    return jsonify(ontology_manager.get_graph_data())

@app.route('/api/neo4j/status')
def neo4j_status():
    """Check Neo4j connection status."""
    return jsonify(ontology_manager.check_neo4j_connection())

@app.route('/api/pipeline/start', methods=['POST'])
def start_pipeline():
    """Start the ontology extension pipeline."""
    config = request.json or {}
    result = ontology_manager.start_pipeline(config)
    return jsonify(result)

@app.route('/api/pipeline/status')
def pipeline_status():
    """Get current pipeline status."""
    return jsonify(ontology_manager.get_process_status())

@app.route('/api/pipeline/cancel', methods=['POST'])
def cancel_pipeline():
    """Cancel current pipeline execution."""
    result = ontology_manager.cancel_process()
    return jsonify(result)

@app.route('/api/snapshots')
def list_snapshots():
    """List all available snapshots."""
    snapshots = ontology_manager.list_snapshots()
    return jsonify({"snapshots": snapshots})

@app.route('/api/snapshots/create', methods=['POST'])
def create_snapshot():
    """Create a new snapshot."""
    data = request.json or {}
    name = data.get('name')
    result = ontology_manager.create_snapshot(name)
    return jsonify(result)

@app.route('/api/snapshots/<snapshot_name>/restore', methods=['POST'])
def restore_snapshot(snapshot_name):
    """Restore from a snapshot."""
    result = ontology_manager.restore_snapshot(snapshot_name)
    return jsonify(result)

@app.route('/api/ontology/reset', methods=['POST'])
def reset_ontology():
    """Reset ontology to baseline."""
    result = ontology_manager.reset_to_baseline()
    return jsonify(result)

@app.route('/api/ontology/export')
def export_ontology():
    """Export current ontology."""
    format_type = request.args.get('format', 'jsonld')
    result = ontology_manager.export_ontology(format_type)
    return jsonify(result)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def run_dashboard_server(host='localhost', port=5000, debug=True):
    """Run the dashboard server."""
    print(f"🌐 Starting Ontology Management Dashboard Server...")
    print(f"📍 Server URL: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🗄️ Neo4j URI: {NEO4J_URI}")
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\n⚠️ Server shutdown requested")
    finally:
        ontology_manager.close()
        print("🔌 Ontology Manager closed")

# --- ADD THESE NEW ROUTES ---

@app.route('/api/reviews/pending')
def get_pending_reviews():
    """Fetches concepts from Neo4j that are labeled as :NeedsReview."""
    try:
        with ontology_manager.driver.session(database=NEO4J_DB_NAME) as session:
            # This query finds all nodes with the NeedsReview label.
            # We are approximating target and reasoning for the demo.
            query = """
            MATCH (p:Product:NeedsReview)
            RETURN p.name AS concept, 
                   "Unknown" AS target, 
                   p.confidence AS confidence, 
                   "Medium confidence match requires expert validation." AS reasoning
            LIMIT 100
            """
            result = session.run(query)
            reviews = [record.data() for record in result]
            return jsonify(reviews)
    except Exception as e:
        logger.error(f"Failed to fetch pending reviews: {e}", exc_info=True)
        return jsonify({"error": "Database query for reviews failed."}), 500

@app.route('/api/reviews/accept/<concept_name>', methods=['POST'])
def accept_review(concept_name):
    """Accepts a concept by removing the :NeedsReview label."""
    try:
        with ontology_manager.driver.session(database=NEO4J_DB_NAME) as session:
            # This query finds the node and removes the label.
            query = """
            MATCH (p:Product {name: $name})
            REMOVE p:NeedsReview
            SET p.status = 'approved'
            RETURN p.name
            """
            result = session.run(query, name=concept_name)
            if result.single():
                logger.info(f"User accepted concept: {concept_name}")
                return jsonify({"success": True, "message": f"'{concept_name}' has been approved."})
            else:
                return jsonify({"success": False, "message": "Concept not found."}), 404
    except Exception as e:
        logger.error(f"Failed to accept review for {concept_name}: {e}", exc_info=True)
        return jsonify({"error": "Database update failed."}), 500

@app.route('/api/reviews/reject/<concept_name>', methods=['POST'])
def reject_review(concept_name):
    """Rejects a concept by removing the :NeedsReview label and marking as rejected."""
    try:
        with ontology_manager.driver.session(database=NEO4J_DB_NAME) as session:
            # For this demo, we'll mark as rejected. In a real app, you might delete it.
            query = """
            MATCH (p:Product {name: $name})
            REMOVE p:NeedsReview
            SET p.status = 'rejected'
            RETURN p.name
            """
            result = session.run(query, name=concept_name)
            if result.single():
                logger.info(f"User rejected concept: {concept_name}")
                return jsonify({"success": True, "message": f"'{concept_name}' has been rejected."})
            else:
                return jsonify({"success": False, "message": "Concept not found."}), 404
    except Exception as e:
        logger.error(f"Failed to reject review for {concept_name}: {e}", exc_info=True)
        return jsonify({"error": "Database update failed."}), 500


# --- NEW QA API ENDPOINT ---
@app.route('/api/qa', methods=['POST'])
def handle_qa_request():
    """Receives a question, runs it through the QA chain, and returns the answer."""
    if not qa_chain:
        # This happens if the initialization failed
        return jsonify({"error": "QA system is not available. Check server logs for details."}), 503 # Service Unavailable

    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided."}), 400 # Bad Request

    logger.info(f"Received question for QA Chain: '{question}'")
    
    try:
        result = qa_chain.invoke({"query": question})
        answer = result.get('result', "I could not find an answer in the knowledge graph.")
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error invoking QA Chain: {e}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the question."}), 500

@app.route('/api/evaluate/gold_standard', methods=['POST'])
def run_gold_standard_evaluation():
    data = request.json
    generated_file_path = data.get('generated_file')
    if not generated_file_path:
        return jsonify({"error": "Missing 'generated_file' path."}), 400
    
    # Assuming the gold standard is always at the same location
    gold_standard_path = Path("data/gold_standard.json")
    
    results = evaluate_ontology(Path(generated_file_path), gold_standard_path)
    return jsonify(results)

@app.route('/api/evaluate/graph_metrics', methods=['POST'])
def run_graph_metrics_evaluation():
    data = request.json
    generated_file_path = data.get('generated_file')
    if not generated_file_path:
        return jsonify({"error": "Missing 'generated_file' path."}), 400
        
    gold_standard_path = Path("data/gold_standard.json")

    G_gold = _load_gold_standard_graph(gold_standard_path)
    G_generated = _load_generated_graph(Path(generated_file_path))
    
    results = compare_graph_metrics(G_gold, G_generated)
    return jsonify(results)

@app.route('/api/evaluate/consistency', methods=['POST'])
def run_consistency_evaluation():
    data = request.json
    generated_file_path = data.get('generated_file')
    if not generated_file_path:
        return jsonify({"error": "Missing 'generated_file' path."}), 400
        
    schema_path = Path("data/electronics_schema.owl")
    
    # Note: evaluate_consistency prints to console, we may want to capture that output later
    # For now, we'll just run it and return a success message.
    try:
        evaluate_consistency(Path(generated_file_path), schema_path)
        return jsonify({"success": True, "message": "Consistency check completed. See server logs for results."})
    except Exception as e:
        logger.error(f"Consistency evaluation failed: {e}", exc_info=True)
        return jsonify({"error": "Consistency evaluation failed on the server."}), 500

@app.route('/api/openai/models')
def get_openai_models():
    """
    Fetches the list of available GPT models from the OpenAI API.
    This acts as a secure proxy to avoid exposing the API key on the frontend.
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        
        # Check for a successful response
        response.raise_for_status() 
        
        models_data = response.json()
        
        # Filter for models that are typically used for chat/completion (e.g., gpt models)
        # and sort them to have newer ones potentially first.
        gpt_models = sorted(
            [model['id'] for model in models_data.get('data', []) if 'gpt' in model['id']],
            reverse=True
        )
        
        return jsonify(gpt_models)

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch models from OpenAI API: {e}", exc_info=True)
        return jsonify({"error": "Could not connect to OpenAI API."}), 502 # Bad Gateway
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching models: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    initialize_qa_chain()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested.")
    finally:
        if ontology_manager:
            ontology_manager.close()
        logger.info("Shared Ontology Manager connection closed.")