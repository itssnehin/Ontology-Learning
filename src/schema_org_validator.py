#!/usr/bin/env python3
"""
Schema.org validation and quality checking utilities.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import re
from typing import Dict, List, Any, Tuple
import requests
from urllib.parse import urlparse
import logging

class SchemaOrgValidator:
    """Validate and check quality of Schema.org JSON-LD objects."""
    
    def __init__(self):
        self.required_properties = {
            "Product": ["@context", "@type", "name"],
            "Organization": ["@context", "@type", "name"]
        }
        
        self.recommended_properties = {
            "Product": ["description", "category", "manufacturer"],
            "Organization": ["description"]
        }
        
        self.validation_results = {
            "total_objects": 0,
            "valid_objects": 0,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
    
    def validate_schema_objects(self, schema_objects: List[Dict]) -> Dict[str, Any]:
        """
        Validate a list of Schema.org objects.
        
        Args:
            schema_objects: List of Schema.org JSON-LD objects
            
        Returns:
            Validation results dictionary
        """
        print(f"Validating {len(schema_objects)} Schema.org objects...")
        
        self.validation_results = {
            "total_objects": len(schema_objects),
            "valid_objects": 0,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        for i, obj in enumerate(schema_objects):
            self._validate_single_object(obj, i)
        
        # Calculate validation rate
        self.validation_results["validation_rate"] = (
            self.validation_results["valid_objects"] / 
            self.validation_results["total_objects"] * 100
        )
        
        return self.validation_results
    
    def _validate_single_object(self, obj: Dict, index: int) -> None:
        """Validate a single Schema.org object."""
        object_id = f"Object {index} ({obj.get('name', 'Unknown')})"
        is_valid = True
        
        # Check required properties
        schema_type = obj.get("@type", "Unknown")
        required_props = self.required_properties.get(schema_type, ["@context", "@type", "name"])
        
        for prop in required_props:
            if prop not in obj or not obj[prop]:
                self.validation_results["errors"].append(
                    f"{object_id}: Missing required property '{prop}'"
                )
                is_valid = False
        
        # Check @context validity
        if "@context" in obj:
            if not self._validate_context(obj["@context"]):
                self.validation_results["warnings"].append(
                    f"{object_id}: Invalid or non-standard @context"
                )
        
        # Check for recommended properties
        recommended_props = self.recommended_properties.get(schema_type, [])
        missing_recommended = []
        for prop in recommended_props:
            if prop not in obj or not obj[prop]:
                missing_recommended.append(prop)
        
        if missing_recommended:
            self.validation_results["recommendations"].append(
                f"{object_id}: Consider adding properties: {', '.join(missing_recommended)}"
            )
        
        # Check additionalType URIs
        if "additionalType" in obj:
            if not self._validate_uri(obj["additionalType"]):
                self.validation_results["warnings"].append(
                    f"{object_id}: additionalType URI may be invalid: {obj['additionalType']}"
                )
        
        # Check for empty or invalid values
        empty_values = []
        for key, value in obj.items():
            if value == "" or value is None:
                empty_values.append(key)
        
        if empty_values:
            self.validation_results["warnings"].append(
                f"{object_id}: Empty values found in properties: {', '.join(empty_values)}"
            )
        
        # Check namespace prefixes
        self._validate_namespaced_properties(obj, object_id)
        
        if is_valid:
            self.validation_results["valid_objects"] += 1
    
    def _validate_context(self, context: Any) -> bool:
        """Validate @context property."""
        if isinstance(context, str):
            return context == "https://schema.org/" or context == "http://schema.org/"
        elif isinstance(context, dict):
            vocab = context.get("@vocab", "")
            return vocab == "https://schema.org/" or vocab == "http://schema.org/"
        return False
    
    def _validate_uri(self, uri: str) -> bool:
        """Validate URI format."""
        try:
            result = urlparse(uri)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _validate_namespaced_properties(self, obj: Dict, object_id: str) -> None:
        """Validate custom namespaced properties."""
        context = obj.get("@context", {})
        if isinstance(context, dict):
            defined_namespaces = {k: v for k, v in context.items() if not k.startswith("@")}
        else:
            defined_namespaces = {}
        
        for key in obj.keys():
            if ":" in key and not key.startswith("@"):
                namespace = key.split(":")[0]
                if namespace not in defined_namespaces:
                    self.validation_results["warnings"].append(
                        f"{object_id}: Property '{key}' uses undefined namespace '{namespace}'"
                    )
    
    def print_validation_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted validation report."""
        print("\n" + "=" * 60)
        print("SCHEMA.ORG VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\nSUMMARY:")
        print(f"  Total objects: {results['total_objects']}")
        print(f"  Valid objects: {results['valid_objects']}")
        print(f"  Validation rate: {results['validation_rate']:.1f}%")
        
        if results['errors']:
            print(f"\nERRORS ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  ❌ {error}")
        
        if results['warnings']:
            print(f"\nWARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings'][:10]:  # Limit to first 10
                print(f"  ⚠️  {warning}")
            if len(results['warnings']) > 10:
                print(f"  ... and {len(results['warnings']) - 10} more warnings")
        
        if results['recommendations']:
            print(f"\nRECOMMENDATIONS ({len(results['recommendations'])}):")
            for rec in results['recommendations'][:10]:  # Limit to first 10
                print(f"  💡 {rec}")
            if len(results['recommendations']) > 10:
                print(f"  ... and {len(results['recommendations']) - 10} more recommendations")
        
        print("\n" + "=" * 60)
    
    def check_product_ontology_uris(self, schema_objects: List[Dict]) -> Dict[str, Any]:
        """Check if Product Types Ontology URIs are accessible."""
        print("\nChecking Product Types Ontology URI accessibility...")
        
        uri_results = {
            "total_uris": 0,
            "accessible_uris": 0,
            "failed_uris": [],
            "timeout_uris": []
        }
        
        for obj in schema_objects:
            if "additionalType" in obj:
                uri = obj["additionalType"]
                if "productontology.org" in uri:
                    uri_results["total_uris"] += 1
                    
                    try:
                        response = requests.head(uri, timeout=5)
                        if response.status_code == 200:
                            uri_results["accessible_uris"] += 1
                        else:
                            uri_results["failed_uris"].append(f"{uri} (Status: {response.status_code})")
                    
                    except requests.exceptions.Timeout:
                        uri_results["timeout_uris"].append(uri)
                    
                    except Exception as e:
                        uri_results["failed_uris"].append(f"{uri} (Error: {str(e)})")
        
        if uri_results["total_uris"] > 0:
            uri_results["accessibility_rate"] = (
                uri_results["accessible_uris"] / uri_results["total_uris"] * 100
            )
            
            print(f"  Checked {uri_results['total_uris']} Product Ontology URIs")
            print(f"  Accessible: {uri_results['accessible_uris']} ({uri_results['accessibility_rate']:.1f}%)")
            
            if uri_results["failed_uris"]:
                print(f"  Failed URIs: {len(uri_results['failed_uris'])}")
            if uri_results["timeout_uris"]:
                print(f"  Timeout URIs: {len(uri_results['timeout_uris'])}")
        else:
            print("  No Product Ontology URIs found to check")
        
        return uri_results

def validate_schema_org_file(file_path: str) -> Dict[str, Any]:
    """
    Validate Schema.org objects from a JSON-LD file.
    
    Args:
        file_path: Path to JSON-LD file
        
    Returns:
        Validation results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single objects and @graph arrays
        if "@graph" in data:
            schema_objects = data["@graph"]
        elif isinstance(data, list):
            schema_objects = data
        else:
            schema_objects = [data]
        
        validator = SchemaOrgValidator()
        results = validator.validate_schema_objects(schema_objects)
        validator.print_validation_report(results)
        
        # Check URI accessibility
        uri_results = validator.check_product_ontology_uris(schema_objects)
        results["uri_check"] = uri_results
        
        return results
        
    except Exception as e:
        print(f"Error validating file {file_path}: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Schema.org JSON-LD files")
    parser.add_argument("file", help="Path to JSON-LD file to validate")
    parser.add_argument("--check-uris", action="store_true", 
                       help="Check URI accessibility (requires internet)")
    
    args = parser.parse_args()
    
    results = validate_schema_org_file(args.file)
    
    if "error" not in results:
        print(f"\nValidation completed: {results['validation_rate']:.1f}% objects valid")
    else:
        print(f"Validation failed: {results['error']}")
