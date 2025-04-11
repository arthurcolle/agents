"""
OpenAPI Tools Functions - Functions for interacting with OpenAPI-based services
These functions will be registered with the agent kernel.
"""

from typing import Dict, List, Any, Optional
from modules.openapi_tools import openapi_registry
import json
import inspect

def register_openapi_service(url: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Register an OpenAPI service from a URL
    
    Args:
        url: URL to the OpenAPI specification or service
        name: Optional name for the API (derived from spec if not provided)
        
    Returns:
        Dictionary with registration result
    """
    result = openapi_registry.register_api_from_url(url, name)
    return result

def list_openapi_services() -> Dict[str, Any]:
    """
    List all registered OpenAPI services
    
    Returns:
        Dictionary with list of registered APIs
    """
    apis = openapi_registry.list_apis()
    return {
        "status": "success",
        "count": len(apis),
        "apis": apis
    }

def get_openapi_service_functions(api_name: str) -> Dict[str, Any]:
    """
    Get functions for a specific OpenAPI service
    
    Args:
        api_name: Name of the API
        
    Returns:
        Dictionary with function information
    """
    functions = openapi_registry.get_api_functions(api_name)
    
    if not functions:
        return {
            "status": "error",
            "message": f"No functions found for API '{api_name}'"
        }
    
    return {
        "status": "success",
        "api": api_name,
        "count": len(functions),
        "functions": functions
    }

def get_openapi_function_info(function_name: str) -> Dict[str, Any]:
    """
    Get information about a specific OpenAPI function
    
    Args:
        function_name: Name of the function
        
    Returns:
        Dictionary with function details
    """
    info = openapi_registry.get_function_info(function_name)
    
    if not info:
        return {
            "status": "error",
            "message": f"Function '{function_name}' not found"
        }
    
    return {
        "status": "success",
        "function": info
    }

def call_openapi_function(function_name: str, **kwargs) -> Dict[str, Any]:
    """
    Call an OpenAPI function
    
    Args:
        function_name: Name of the function to call
        **kwargs: Arguments to pass to the function
        
    Returns:
        Dictionary with function result
    """
    function = openapi_registry.get_function(function_name)
    
    if not function:
        return {
            "status": "error",
            "message": f"Function '{function_name}' not found"
        }
    
    try:
        # Check if all required parameters are provided
        info = openapi_registry.get_function_info(function_name)
        if info:
            for param_name in info.get("required_params", []):
                if param_name not in kwargs:
                    return {
                        "status": "error",
                        "message": f"Missing required parameter: {param_name}"
                    }
        
        # Call the function
        result = function(**kwargs)
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            return {
                "status": "error",
                "message": result["error"],
                "details": result
            }
        
        return {
            "status": "success",
            "function": function_name,
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error calling function: {str(e)}"
        }

def refresh_openapi_service(api_name: str) -> Dict[str, Any]:
    """
    Refresh an OpenAPI service by re-fetching its specification
    
    Args:
        api_name: Name of the API to refresh
        
    Returns:
        Dictionary with refresh result
    """
    # Check if API exists
    apis = openapi_registry.list_apis()
    api_found = False
    url = None
    
    for api in apis:
        if api["name"] == api_name:
            api_found = True
            url = openapi_registry.apis[api_name]["url"]
            break
    
    if not api_found or not url:
        return {
            "status": "error",
            "message": f"API '{api_name}' not found"
        }
    
    # Re-register the API
    return register_openapi_service(url, api_name)

def generate_openapi_client_code(api_name: str, language: str = "python") -> Dict[str, Any]:
    """
    Generate client code for an OpenAPI service
    
    Args:
        api_name: Name of the API
        language: Programming language for the client code (currently only Python supported)
        
    Returns:
        Dictionary with generated code
    """
    # Check if API exists
    apis = openapi_registry.list_apis()
    api_found = False
    api_info = None
    
    for api in apis:
        if api["name"] == api_name:
            api_found = True
            api_info = openapi_registry.apis[api_name]
            break
    
    if not api_found or not api_info:
        return {
            "status": "error",
            "message": f"API '{api_name}' not found"
        }
    
    # Get functions for the API
    functions = openapi_registry.get_api_functions(api_name)
    if not functions:
        return {
            "status": "error",
            "message": f"No functions found for API '{api_name}'"
        }
    
    # Generate Python code
    if language.lower() == "python":
        code = [
            f"# Client for {api_info['title']} API (v{api_info['version']})",
            "# Generated by Llama4 OpenAPI Tools",
            "",
            "import requests",
            "import json",
            "from typing import Dict, List, Any, Optional, Union",
            "",
            f"class {api_name.capitalize()}Client:",
            f'    """Client for {api_info["title"]} API"""',
            "",
            f'    def __init__(self, base_url: str = "{api_info["url"]}", timeout: int = 30):',
            '        """Initialize the client"""',
            "        self.base_url = base_url",
            "        self.timeout = timeout",
            "        self.headers = {",
            '            "Content-Type": "application/json",',
            '            "Accept": "application/json"',
            "        }",
            ""
        ]
        
        # Add a method for each function
        for func in functions:
            method_name = func["name"].split('_', 1)[1] if '_' in func["name"] else func["name"]
            
            # Function signature
            params = []
            for param_name, param in func["parameters"].items():
                param_type = "str"
                if param["type"] == "integer":
                    param_type = "int"
                elif param["type"] == "number":
                    param_type = "float"
                elif param["type"] == "boolean":
                    param_type = "bool"
                elif param["type"] == "array":
                    param_type = "List[Any]"
                elif param["type"] == "object":
                    param_type = "Dict[str, Any]"
                
                # Add default value for optional parameters
                if not param["required"]:
                    params.append(f"{param_name}: Optional[{param_type}] = None")
                else:
                    params.append(f"{param_name}: {param_type}")
            
            # Function definition
            code.append(f"    def {method_name}({', '.join(params)}) -> Dict[str, Any]:")
            code.append(f'        """{func["description"]}"""')
            
            # Function implementation
            url_path = func["path"]
            method = func["method"]
            
            # URL with path parameters
            code.append(f'        url = f"{url_path}"')
            
            # Query parameters
            query_params = []
            for param_name, param in func["parameters"].items():
                if param["in"] == "query":
                    query_params.append(param_name)
            
            if query_params:
                code.append("        # Prepare query parameters")
                code.append("        params = {}")
                for param_name in query_params:
                    code.append(f"        if {param_name} is not None:")
                    code.append(f"            params['{param_name}'] = {param_name}")
            else:
                code.append("        params = {}")
            
            # Body parameter
            body_param = None
            for param_name, param in func["parameters"].items():
                if param["in"] == "body":
                    body_param = param_name
                    break
            
            # Make the request
            code.append("        # Make the request")
            if method == "GET":
                code.append("        try:")
                code.append(f"            response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)")
            elif method == "POST":
                code.append("        # Prepare request body")
                if body_param:
                    code.append(f"        json_data = {body_param}")
                else:
                    code.append("        json_data = {}")
                code.append("        try:")
                code.append(f"            response = requests.post(url, params=params, json=json_data, headers=self.headers, timeout=self.timeout)")
            elif method == "PUT":
                code.append("        # Prepare request body")
                if body_param:
                    code.append(f"        json_data = {body_param}")
                else:
                    code.append("        json_data = {}")
                code.append("        try:")
                code.append(f"            response = requests.put(url, params=params, json=json_data, headers=self.headers, timeout=self.timeout)")
            elif method == "DELETE":
                code.append("        try:")
                code.append(f"            response = requests.delete(url, params=params, headers=self.headers, timeout=self.timeout)")
            elif method == "PATCH":
                code.append("        # Prepare request body")
                if body_param:
                    code.append(f"        json_data = {body_param}")
                else:
                    code.append("        json_data = {}")
                code.append("        try:")
                code.append(f"            response = requests.patch(url, params=params, json=json_data, headers=self.headers, timeout=self.timeout)")
            
            # Handle response
            code.append("            # Process response")
            code.append("            if response.status_code >= 200 and response.status_code < 300:")
            code.append("                try:")
            code.append("                    return {")
            code.append('                        "status": "success",')
            code.append('                        "status_code": response.status_code,')
            code.append('                        "data": response.json()')
            code.append("                    }")
            code.append("                except ValueError:")
            code.append("                    return {")
            code.append('                        "status": "success",')
            code.append('                        "status_code": response.status_code,')
            code.append('                        "text": response.text')
            code.append("                    }")
            code.append("            else:")
            code.append("                return {")
            code.append('                    "status": "error",')
            code.append('                    "status_code": response.status_code,')
            code.append('                    "error": f"API Error: {response.status_code}",')
            code.append('                    "data": response.json() if response.headers.get("content-type") == "application/json" else response.text')
            code.append("                }")
            code.append("        except requests.RequestException as e:")
            code.append("            return {")
            code.append('                "status": "error",')
            code.append('                "error": f"Request failed: {str(e)}"')
            code.append("            }")
            code.append("")
        
        # Example usage
        code.append("# Example usage:")
        code.append(f"# client = {api_name.capitalize()}Client()")
        
        # Get a sample function
        if functions:
            example_func = functions[0]
            method_name = example_func["name"].split('_', 1)[1] if '_' in example_func["name"] else example_func["name"]
            
            # Example call
            example_call = f"# result = client.{method_name}("
            for param_name, param in example_func["parameters"].items():
                if param["required"]:
                    if param["type"] == "string":
                        example_call += f'{param_name}="example", '
                    elif param["type"] == "integer":
                        example_call += f"{param_name}=1, "
                    elif param["type"] == "boolean":
                        example_call += f"{param_name}=True, "
                    elif param["type"] == "object":
                        example_call += f"{param_name}={{}}, "
            
            if example_call.endswith(", "):
                example_call = example_call[:-2]
            example_call += ")"
            code.append(example_call)
        
        return {
            "status": "success",
            "language": language,
            "api": api_name,
            "code": "\n".join(code)
        }
    else:
        return {
            "status": "error",
            "message": f"Language '{language}' not supported for code generation"
        }

def register_tool_management_api() -> Dict[str, Any]:
    """
    Register the Tool Management API (shortcut function)
    
    Returns:
        Dictionary with registration result
    """
    url = "https://arthurcolle--registry.modal.run/openapi.json"
    return register_openapi_service(url, "tool_management_api")

def register_tool_management_api_dev() -> Dict[str, Any]:
    """
    Register the Tool Management API Dev version (shortcut function)
    
    Returns:
        Dictionary with registration result
    """
    url = "https://arthurcolle--registry-dev.modal.run/openapi.json"
    return register_openapi_service(url, "tool_management_api_dev")

def register_embeddings_api() -> Dict[str, Any]:
    """
    Register the Embeddings API (shortcut function)
    
    Returns:
        Dictionary with registration result
    """
    url = "https://arthurcolle--embeddings.modal.run"
    return register_openapi_service(url, "embeddings_api")

def register_dynamic_schema_api() -> Dict[str, Any]:
    """
    Register the Dynamic Schema API (shortcut function)
    
    Returns:
        Dictionary with registration result
    """
    url = "https://arthurcolle--dynamic-schema.modal.run"
    return register_openapi_service(url, "dynamic_schema_api")

def register_realtime_relay_api() -> Dict[str, Any]:
    """
    Register the Realtime Relay API (shortcut function)
    
    Returns:
        Dictionary with registration result
    """
    url = "https://arthurcolle--realtime-relay.modal.run"
    return register_openapi_service(url, "realtime_relay_api")

def register_all_default_apis() -> Dict[str, Any]:
    """
    Register all default APIs: Tool Management, Embeddings, Dynamic Schema, and Realtime Relay
    
    Returns:
        Dictionary with registration results
    """
    results = {}
    
    # Register each API
    results["tool_management_api"] = register_tool_management_api()
    results["embeddings_api"] = register_embeddings_api()
    results["dynamic_schema_api"] = register_dynamic_schema_api()
    results["realtime_relay_api"] = register_realtime_relay_api()
    
    # Count successful registrations
    success_count = sum(1 for result in results.values() if result.get("status") == "success")
    
    return {
        "status": "success" if success_count > 0 else "error",
        "message": f"Successfully registered {success_count} out of 4 APIs",
        "results": results
    }