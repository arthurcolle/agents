#!/usr/bin/env python3
"""
Test script to register all default APIs
This script can be run independently to test the API registration
"""

import os
import sys
import json
from pprint import pprint

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenAPI tools
from modules.openapi_tools_functions import (
    register_all_default_apis,
    list_openapi_services,
    get_openapi_service_functions
)

def main():
    """Test registering all APIs and list their functions"""
    print("🔍 API Registration Test 🔍")
    print("-" * 50)
    
    # Register all default APIs
    print("\n1. Registering all default APIs...")
    result = register_all_default_apis()
    
    if result["status"] == "success":
        print(f"✅ {result['message']}")
        
        # Check results for each API
        for api_name, api_result in result["results"].items():
            if api_result.get("status") == "success":
                print(f"  ✓ {api_name}: {api_result.get('message', 'Registered successfully')}")
            else:
                print(f"  ✗ {api_name}: {api_result.get('message', 'Registration failed')}")
    else:
        print(f"❌ Failed: {result['message']}")
        return
        
    # List all registered services
    print("\n2. Listing all registered APIs...")
    list_result = list_openapi_services()
    
    if list_result["status"] == "success":
        print(f"✅ Found {list_result['count']} registered APIs:")
        for api in list_result["apis"]:
            print(f"\n  📌 {api['name']}: {api['title']} v{api['version']}")
            print(f"     Endpoints: {api['endpoints']}")
            print(f"     Description: {api['description'][:100]}..." if len(api['description']) > 100 else f"     Description: {api['description']}")
            
            # Get functions for this API
            funcs_result = get_openapi_service_functions(api['name'])
            if funcs_result["status"] == "success" and funcs_result["count"] > 0:
                print(f"\n     Top 3 Functions:")
                for i, func in enumerate(funcs_result["functions"][:3], 1):
                    print(f"       {i}. {func['name']}: {func['method']} {func['path']}")
                    print(f"          Description: {func['description'][:50]}..." if func['description'] and len(func['description']) > 50 else f"          Description: {func['description'] or 'None'}")
                
                if funcs_result["count"] > 3:
                    print(f"       ... and {funcs_result['count'] - 3} more functions")
            else:
                print("     No functions found or error retrieving functions")
    else:
        print(f"❌ Failed to list APIs: {list_result['message']}")
    
    print("\n✨ Test complete! ✨")

if __name__ == "__main__":
    main()