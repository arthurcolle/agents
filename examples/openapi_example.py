#!/usr/bin/env python3
"""
Example script for using the OpenAPI tools
Demonstrates how to register and call API endpoints
"""

import os
import sys
import json
from pprint import pprint

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenAPI tools
from modules.openapi_tools_functions import (
    register_openapi_service,
    list_openapi_services,
    get_openapi_service_functions,
    call_openapi_function,
    register_tool_management_api,
    register_tool_management_api_dev,
    generate_openapi_client_code
)

def main():
    """Example of using OpenAPI tools"""
    print("🔍 OpenAPI Tools Example 🔍")
    print("-" * 50)
    
    # Register the Tool Management API
    print("\n1. Registering Tool Management API...")
    result = register_tool_management_api()
    if result["status"] == "success":
        print(f"✅ Successfully registered API: {result['api']['title']}")
        print(f"   Functions available: {result['api']['endpoints']}")
    else:
        print(f"❌ Failed to register API: {result['message']}")
        return
    
    # List all registered APIs
    print("\n2. Listing registered APIs...")
    result = list_openapi_services()
    if result["status"] == "success":
        print(f"✅ Found {result['count']} registered APIs:")
        for api in result["apis"]:
            print(f"   - {api['name']}: {api['title']} (v{api['version']})")
            print(f"     Endpoints: {api['endpoints']}")
    else:
        print(f"❌ Failed to list APIs: {result['message']}")
    
    # Get functions for the API
    print("\n3. Getting functions for Tool Management API...")
    result = get_openapi_service_functions("tool_management_api")
    if result["status"] == "success":
        print(f"✅ Found {result['count']} functions:")
        for i, func in enumerate(result["functions"][:5], 1):
            print(f"   {i}. {func['name']}: {func['method']} {func['path']}")
            if i == 5 and result["count"] > 5:
                print(f"   ... and {result['count'] - 5} more")
                break
    else:
        print(f"❌ Failed to get functions: {result['message']}")
    
    # Generate client code
    print("\n4. Generating Python client code...")
    result = generate_openapi_client_code("tool_management_api")
    if result["status"] == "success":
        print(f"✅ Successfully generated {result['language']} client code")
        print("\nExample code snippet:")
        code_lines = result["code"].split("\n")
        for line in code_lines[:10]:  # Print first 10 lines
            print(f"   {line}")
        print("   ...")
        
        # Save the generated code to a file
        code_file = os.path.join(os.path.dirname(__file__), "tool_management_client.py")
        with open(code_file, "w") as f:
            f.write(result["code"])
        print(f"\n📄 Full client code saved to: {code_file}")
    else:
        print(f"❌ Failed to generate client code: {result['message']}")
    
    # Try calling a function (if available)
    print("\n5. Testing API call (health check endpoint)...")
    # Find health check or simple GET endpoint
    health_function = None
    result = get_openapi_service_functions("tool_management_api")
    if result["status"] == "success":
        for func in result["functions"]:
            if "health" in func["name"].lower() or (func["method"] == "GET" and len(func["parameters"]) == 0):
                health_function = func["name"]
                break
    
    if health_function:
        print(f"   Calling function: {health_function}")
        result = call_openapi_function(health_function)
        if result["status"] == "success":
            print("✅ API call successful!")
            print("\nResponse:")
            pprint(result["result"])
        else:
            print(f"❌ API call failed: {result['message']}")
    else:
        print("❌ No suitable health check endpoint found")
    
    print("\n✨ Example complete! ✨")

if __name__ == "__main__":
    main()