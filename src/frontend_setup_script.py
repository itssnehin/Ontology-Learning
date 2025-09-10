#!/usr/bin/env python3
"""
Simple Frontend Setup Script - No Unicode Characters
Creates frontend directory structure for the ontology dashboard.
"""

import os
from pathlib import Path

def create_frontend_structure():
    """Create frontend directory structure."""
    
    # Get project root
    current_dir = Path.cwd()
    project_root = current_dir.parent if current_dir.name == 'src' else current_dir
    
    print(f"Setting up frontend in: {project_root}")
    
    # Create directories
    directories = [
        "frontend",
        "frontend/templates", 
        "frontend/static",
        "frontend/static/css",
        "frontend/static/js",
        "frontend/static/img"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {dir_path}")
    
    # Create CSS file
    css_content = """/* Ontology Dashboard Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.dashboard-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 25px;
    cursor: pointer;
    font-weight: 600;
    margin: 5px;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}
"""
    
    css_file = project_root / "frontend/static/css/dashboard.css"
    with open(css_file, 'w', encoding='utf-8') as f:
        f.write(css_content)
    print(f"   Created: CSS file")
    
    # Create JavaScript file
    js_content = """// Dashboard JavaScript
console.log('Dashboard loaded');

function initializeDashboard() {
    console.log('Dashboard initialized');
}

document.addEventListener('DOMContentLoaded', initializeDashboard);
"""
    
    js_file = project_root / "frontend/static/js/dashboard.js"
    with open(js_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"   Created: JavaScript file")
    
    # Create HTML template
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ontology Management Dashboard</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/dashboard.css') }}">
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>Schema.org Ontology Management Dashboard</h1>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Process Control</h3>
                <p>Pipeline management</p>
                <button class="btn">Run Pipeline</button>
            </div>
            
            <div class="card">
                <h3>Statistics</h3>
                <p>Ontology metrics</p>
                <button class="btn">Refresh</button>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='js/dashboard.js') }}"></script>
</body>
</html>"""
    
    html_file = project_root / "frontend/templates/dashboard.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"   Created: HTML template")
    
    # Create backend update instructions
    instructions = """# Backend Update Instructions

## Update ontology_management_backend.py:

1. Change the Flask app initialization:
app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')

2. Update imports:
from flask import Flask, jsonify, request, render_template

3. Update the dashboard route:
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

## Then run:
python ontology_management_backend.py
"""
    
    instructions_file = project_root / "frontend/SETUP_INSTRUCTIONS.txt"
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    print(f"   Created: Setup instructions")
    
    print(f"\nFrontend structure created successfully!")
    print(f"Directory: {project_root / 'frontend'}")
    print(f"Next: Follow instructions in frontend/SETUP_INSTRUCTIONS.txt")

if __name__ == "__main__":
    create_frontend_structure()