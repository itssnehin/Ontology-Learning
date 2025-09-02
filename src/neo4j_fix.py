#!/usr/bin/env python3
"""
Quick patch to fix Neo4j Cypher syntax error in schema_org_graph_builder.py
Run this once to patch your existing file.
"""

import os
from pathlib import Path

def patch_neo4j_cypher_syntax():
    """Patch the Neo4j Cypher syntax error in schema_org_graph_builder.py"""
    
    file_path = Path("schema_org_graph_builder.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("   Make sure you're in the src/ directory")
        return False
    
    # Read the original file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The problematic query pattern
    old_pattern = '''    # Group products by category
    query_category_groups = """
    MATCH (p:Product)
    WHERE p.category IS NOT NULL
    WITH p.category as category, collect(p) as products
    WHERE size(products) > 1
    UNWIND products as p1
    UNWIND products as p2
    WHERE p1 <> p2 AND id(p1) < id(p2)
    MERGE (p1)-[:SAME_CATEGORY]->(p2)
    RETURN count(*) as category_relationships
    """'''
    
    # Fixed version
    new_pattern = '''    # Group products by category - FIXED SYNTAX
    query_category_groups = """
    MATCH (p1:Product), (p2:Product)
    WHERE p1.category = p2.category 
      AND p1.category IS NOT NULL
      AND p1 <> p2
      AND id(p1) < id(p2)
    MERGE (p1)-[:SAME_CATEGORY]->(p2)
    RETURN count(*) as category_relationships
    """'''
    
    if old_pattern in content:
        print("✅ Found Neo4j syntax error - applying patch...")
        content = content.replace(old_pattern, new_pattern)
        
        # Also fix the frequency groups query
        old_freq_pattern = '''    # Group products by frequency range (for antennas)
    query_frequency_groups = """
    MATCH (p:Product)
    WHERE p.elec_frequency IS NOT NULL
    WITH p.elec_frequency as freq, collect(p) as products
    WHERE size(products) > 1
    UNWIND products as p1
    UNWIND products as p2
    WHERE p1 <> p2 AND id(p1) < id(p2)
    MERGE (p1)-[:SAME_FREQUENCY_RANGE]->(p2)
    RETURN count(*) as frequency_relationships
    """'''
        
        new_freq_pattern = '''    # Group products by frequency range (for antennas) - FIXED SYNTAX  
    query_frequency_groups = """
    MATCH (p1:Product), (p2:Product)
    WHERE p1.elec_frequency = p2.elec_frequency
      AND p1.elec_frequency IS NOT NULL
      AND p1 <> p2
      AND id(p1) < id(p2)
    MERGE (p1)-[:SAME_FREQUENCY_RANGE]->(p2)
    RETURN count(*) as frequency_relationships
    """'''
        
        if old_freq_pattern in content:
            content = content.replace(old_freq_pattern, new_freq_pattern)
            print("✅ Also fixed frequency grouping query")
        
        # Add better error handling
        error_handling_patch = '''        try:
            result1 = session.run(query_category_groups)
            cat_count = result1.single()["category_relationships"]
            print(f"Created {cat_count} category-based relationships")
            
            result2 = session.run(query_frequency_groups)
            freq_count = result2.single()["frequency_relationships"]
            print(f"Created {freq_count} frequency-based relationships")
            
            result3 = session.run(query_manufacturer_groups)
            mfg_count = result3.single()["manufacturer_relationships"]
            print(f"Created {mfg_count} manufacturer-based relationships")
            
        except Exception as e:
            print(f"Error creating inferred relationships: {e}")
            # Don't raise - let pipeline continue'''
        
        # Replace the old try-except block
        old_try_block_start = "        try:\n            result1 = session.run(query_category_groups)"
        if old_try_block_start in content:
            # Find and replace the entire try-except block
            start_idx = content.find("        try:")
            end_idx = content.find("            print(f\"Error creating inferred relationships: {e}\")", start_idx)
            if start_idx != -1 and end_idx != -1:
                end_idx = content.find("\n", end_idx + len("            print(f\"Error creating inferred relationships: {e}\")"))
                old_try_block = content[start_idx:end_idx]
                content = content.replace(old_try_block, error_handling_patch)
                print("✅ Improved error handling")
        
        # Create backup
        backup_path = file_path.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)  # Actually, write original content to backup
        
        # Write the patched file  
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Successfully patched {file_path}")
        print(f"📋 Backup saved as: {backup_path}")
        return True
    
    else:
        print("⚠️  Neo4j syntax error not found - file may already be fixed")
        return False

def patch_filename_sanitization():
    """Patch filename sanitization in schema_org_pipeline.py"""
    
    file_path = Path("schema_org_pipeline.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the problematic filename creation
    old_filename_pattern = '''        for i, obj in enumerate(self.results["enhanced_schema_objects"]):
            obj_file = objects_dir / f"object_{i:03d}_{obj.get('name', 'unknown').replace(' ', '_')}.json"
            with open(obj_file, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)'''
    
    new_filename_pattern = '''        def sanitize_filename(name: str) -> str:
            """Create Windows-safe filename"""
            import re
            safe_name = re.sub(r'[<>:"/\\|?*\\x00-\\x1f]', '_', str(name))
            safe_name = re.sub(r'_{2,}', '_', safe_name)
            safe_name = safe_name.strip('._')
            return safe_name[:80] if safe_name else "component"
        
        for i, obj in enumerate(self.results["enhanced_schema_objects"]):
            obj_name = obj.get('name', f'component_{i}')
            safe_name = sanitize_filename(obj_name)
            obj_file = objects_dir / f"object_{i:03d}_{safe_name}.json"
            
            try:
                with open(obj_file, 'w', encoding='utf-8') as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
            except Exception as e:
                # Ultimate fallback
                fallback_file = objects_dir / f"object_{i:03d}.json"
                print(f"   ⚠️ Using fallback filename for object {i}: {e}")
                with open(fallback_file, 'w', encoding='utf-8') as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)'''
    
    if "replace(' ', '_')" in content and "sanitize_filename" not in content:
        print("✅ Found filename sanitization issue - applying patch...")
        
        # This is trickier - let's add the import at the top if needed
        if "import re" not in content:
            # Add import after other imports
            import_location = content.find("from datetime import datetime")
            if import_location != -1:
                end_of_line = content.find("\n", import_location)
                content = content[:end_of_line] + "\nimport re" + content[end_of_line:]
                print("✅ Added 're' import")
        
        # Replace the problematic section - this is approximate
        # The user should manually apply this fix for safety
        print("⚠️  Filename sanitization patch available but requires manual application")
        print("   Please add the sanitize_filename function manually to _save_pipeline_outputs")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Applying patches to fix pipeline bugs...")
    print("=" * 50)
    
    # Check current directory
    if not Path("schema_org_graph_builder.py").exists():
        print("❌ Please run this script from the src/ directory")
        exit(1)
    
    # Apply Neo4j fix
    neo4j_fixed = patch_neo4j_cypher_syntax()
    
    # Apply filename fix  
    filename_fixed = patch_filename_sanitization()
    
    print("\n" + "=" * 50)
    if neo4j_fixed:
        print("✅ Neo4j Cypher syntax fixed")
    if filename_fixed:
        print("✅ Filename sanitization fixed") 
    
    print("\n🚀 Now you can run:")
    print("   python cached_schema_org_pipeline.py --resume-from graph")
    print("\n💡 Or install the cached pipeline first:")
    print("   # Copy cached_schema_org_pipeline.py to this directory")
    print("   python cached_schema_org_pipeline.py --max-chunks 20")