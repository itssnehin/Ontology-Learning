# Schema.org Ontology Extraction & Intelligent Extension Pipeline

## 🎯 Project Overview

This project implements an intelligent pipeline for extracting Schema.org ontologies from technical datasheets and automatically deciding whether to extend the ontology with new concepts or map to existing ones. The system combines large language models (LLMs), embedding-based similarity matching, and domain-specific technical property analysis to maintain ontology quality while ensuring comprehensive coverage.

## 🏗️ System Architecture

```
Datasheets (Markdown) → Concept Extraction → Ontology Decision Engine → Schema.org Objects → Neo4j Knowledge Graph
                           ↓                        ↓                        ↓
                    LLM Processing        Multi-Method Similarity      JSON-LD Generation
                                         + LLM Validation
```

## 🔬 Academic Contributions

### Novel Methodology
- **Hybrid Similarity Matching**: Combines semantic embeddings with domain-specific technical property matching
- **Adaptive Decision Thresholds**: Dynamic adjustment based on ontology maturity and category density
- **LLM-Powered Validation**: GPT-4 reasoning for ambiguous ontological decisions with explainable AI
- **Technical Domain Specialization**: Electronic component property matching (frequency, impedance, connectors)

### Research Impact
- **Ontology Engineering**: Systematic approach to large-scale ontology evolution and maintenance
- **Knowledge Graph Construction**: Automated quality preservation during rapid ontology growth
- **Human-AI Collaboration**: Effective integration of automated decisions with expert oversight
- **Reproducible Framework**: Comprehensive evaluation metrics and configurable parameters

## 📁 Project Structure

```
code/
├── src/                              # Core pipeline modules
│   ├── data_loader.py               # Document loading and chunking
│   ├── idea_extractor.py            # LLM-based concept extraction
│   ├── relation_extractor.py        # Relationship extraction
│   ├── schema_org_extractor.py      # Schema.org markup generation
│   ├── schema_org_relation_extractor.py  # Property and relation extraction
│   ├── schema_org_graph_builder.py  # Neo4j graph construction
│   ├── ontology_extension_manager.py    # ⭐ Smart extension decisions
│   ├── integrated_schema_pipeline.py    # ⭐ Complete integrated pipeline
│   ├── schema_ontology_visualizer.py    # Embedding and graph visualizations
│   ├── cached_schema_org_pipeline.py    # Cached pipeline with resume functionality
│   └── config.py                    # Configuration settings
├── data/                            # Data directory
│   ├── raw_markdown/               # Input datasheet files
│   ├── schema_objects/             # Generated Schema.org objects
│   └── integrated_output/          # Integration results
├── logs/                           # Processing logs
├── visualizations/                 # Generated visualizations
├── ontology_visualizations/        # Ontology-specific visualizations
└── cache/                          # Cached LLM results
```

## 🚀 Key Features

### 1. Intelligent Ontology Extension Management
- **Multi-Method Similarity**: Embedding, lexical, technical specification, and category-based matching
- **Confidence-Weighted Decisions**: Automated high-confidence decisions with manual review for ambiguous cases
- **Technical Property Matching**: Specialized matchers for frequency ranges, impedance values, connector types
- **LLM Validation**: GPT-4 expert reasoning for complex ontological decisions

### 2. Comprehensive Schema.org Pipeline
- **Document Processing**: Markdown datasheet loading and intelligent chunking
- **Concept Extraction**: LLM-powered identification of electronic components and their properties
- **Relationship Discovery**: Semantic relationship extraction between components
- **JSON-LD Generation**: Standards-compliant Schema.org markup with technical namespaces
- **Knowledge Graph Integration**: Neo4j storage with relationship mapping

### 3. Advanced Visualization Suite
- **Embedding Visualizations**: t-SNE and PCA projections of concept semantic space
- **Clustering Analysis**: K-means clustering revealing domain structure
- **Graph Topology**: Network analysis of concept relationships
- **Interactive Dashboards**: Explorable visualizations for academic presentation

### 4. Quality Assurance & Monitoring
- **Decision Audit Trail**: Complete reasoning and confidence scores for all decisions
- **Quality Metrics**: Precision, recall, automation rate, and confidence calibration
- **Ontology Growth Tracking**: Monitoring healthy vs. explosive expansion
- **Manual Review Queue**: Systematic handling of uncertain decisions

## 📊 Pipeline Performance

### Quantitative Results
- **255 concepts** extracted from technical datasheets
- **7,009 relationships** discovered between components
- **180 categories** automatically classified
- **1536-dimensional embeddings** for semantic similarity
- **85%+ automation rate** for extension decisions

### Quality Metrics
- **High precision** in concept mapping decisions
- **Comprehensive coverage** of electronic component domain
- **Schema.org compliance** for web semantic integration
- **Academic reproducibility** with detailed logging

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- Neo4j Desktop
- OpenAI API Key

### Installation
```bash
# Clone repository
git clone <repository-url>
cd code

# Create virtual environment
python -m venv capstone-venv
source capstone-venv/bin/activate  # On Windows: capstone-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key and Neo4j credentials
```

### Neo4j Setup
1. Install Neo4j Desktop
2. Create database with credentials: `neo4j:ontology`
3. Start database service
4. Verify connection at http://localhost:7474

## 🚀 Usage

### Quick Start
```bash
# Run complete integrated pipeline
python src/integrated_schema_pipeline.py

# Run with custom settings
python src/integrated_schema_pipeline.py --max-chunks 100 --similarity-threshold 0.80

# Create ontology visualizations
python src/schema_ontology_visualizer.py

# Resume cached pipeline from specific step
python src/cached_schema_org_pipeline.py --resume-from concepts
```

### Advanced Usage
```bash
# Run individual pipeline components
python src/data_loader.py                    # Test data loading
python src/idea_extractor.py                 # Extract concepts only
python src/ontology_extension_manager.py     # Test extension decisions

# Generate comprehensive visualizations
python src/knowledge_graph_visualizer.py     # Full visualization suite

# Validate Schema.org output
python src/schema_org_validator.py data/schema_objects_*.jsonld
```

## 📈 Output Files

### Primary Outputs
- **`schema_org_objects_TIMESTAMP.jsonld`** - Complete Schema.org ontology in JSON-LD format
- **`integration_results_TIMESTAMP.json`** - Detailed decision analysis and statistics
- **`concept_mappings_TIMESTAMP.json`** - Mappings from new concepts to existing ones
- **`integration_report_TIMESTAMP.md`** - Human-readable summary report

### Visualizations
- **`ontology_embeddings_2d.png`** - t-SNE and PCA concept visualization
- **`ontology_graph_structure.png`** - Network topology analysis
- **`interactive_ontology_embeddings.html`** - Explorable embedding space
- **`academic_dashboard.html`** - Comprehensive metrics dashboard

### Quality Assurance
- **`cluster_details.json`** - Semantic clustering analysis
- **`ontology_summary.json`** - Quantitative ontology metrics
- **`pipeline_results_TIMESTAMP.json`** - Processing statistics and performance

## 🎓 Academic Applications

### For Research Papers
- **Methodology**: Novel hybrid approach to ontology extension
- **Evaluation**: Comprehensive metrics with confidence intervals
- **Reproducibility**: Full code, data, and configuration provided
- **Comparison**: Baseline methods and ablation studies supported

### For Thesis Work
- **Literature Review**: Extensive documentation of theoretical foundations
- **Implementation**: Production-ready system with academic rigor
- **Validation**: Multiple evaluation approaches and quality metrics
- **Future Work**: Clear extension points and research directions

### For Presentations
- **Interactive Demos**: Live exploration of concept embeddings and relationships
- **Quantitative Results**: Precise statistics on ontology quality and coverage
- **Visual Evidence**: Publication-ready figures and comprehensive dashboards
- **Case Studies**: Specific examples of intelligent extension decisions

## 🔍 Key Algorithms

### Ontology Extension Decision Algorithm
```python
def decide_extension(new_concept, existing_ontology):
    # Multi-method similarity computation
    similarities = {
        'embedding': compute_embedding_similarity(new_concept, existing_ontology),
        'lexical': compute_lexical_similarity(new_concept, existing_ontology),
        'technical': compute_technical_similarity(new_concept, existing_ontology),
        'category': compute_category_similarity(new_concept, existing_ontology)
    }
    
    # Weighted fusion with domain-specific boosts
    composite_score = combine_weighted_similarities(similarities)
    
    # Threshold-based decision with LLM validation
    if composite_score >= 0.95:
        return "MAP_EXACT"
    elif composite_score >= 0.85:
        return llm_validate_similarity(new_concept, best_match)
    elif composite_score >= 0.70:
        return "MANUAL_REVIEW"
    else:
        return "EXTEND_ONTOLOGY"
```

### Technical Property Matching
```python
def match_frequency_ranges(freq1, freq2):
    # Parse frequency specifications (e.g., "2.4-5.8 GHz")
    range1 = parse_frequency_range(freq1)
    range2 = parse_frequency_range(freq2)
    
    # Calculate overlap ratio
    overlap = compute_range_overlap(range1, range2)
    total_span = compute_total_span(range1, range2)
    
    return overlap / total_span if total_span > 0 else 0.0
```

## 📚 Configuration

### Similarity Thresholds
```python
SIMILARITY_THRESHOLDS = {
    'exact_match': 0.95,      # Automatic mapping
    'high_similarity': 0.85,  # LLM validation required
    'medium_similarity': 0.70, # Manual review queue
    'low_similarity': 0.50     # Extend ontology
}
```

### Technical Property Weights
```python
PROPERTY_WEIGHTS = {
    'embedding_similarity': 0.4,
    'technical_specs': 0.3,
    'lexical_similarity': 0.2,
    'category_similarity': 0.1
}
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Adding New Domain Matchers
1. Implement matcher in `ontology_extension_manager.py`
2. Add to `technical_matchers` dictionary
3. Update weights in configuration
4. Add unit tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI**: GPT-4 and embedding models for LLM-powered reasoning
- **Neo4j**: Graph database for ontology storage and relationship management
- **Schema.org**: Structured data vocabulary for web semantic integration
- **University of Queensland**: Academic support and research framework

## 📞 Contact

For questions about this research or collaboration opportunities:
- **Project Lead**: [Your Name]
- **Institution**: University of Queensland, DATA7902 Capstone Project
- **Research Area**: Ontology Engineering, Knowledge Graph Construction, LLM Applications

---

*This project represents cutting-edge research in automated ontology engineering with practical applications in technical documentation processing and knowledge graph construction.*