"""
OpenAPI Tools Module - Dynamically generate tool wrappers from OpenAPI specifications
Fetches OpenAPI schemas from endpoints and creates callable functions for the agent
"""

import json
import re
import os
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from urllib.parse import urljoin, urlparse
import logging
import inspect
import uuid
import hashlib

# Configure logging
logger = logging.getLogger("openapi_tools")

class OpenAPIRegistry:
    """
    Registry for OpenAPI-based tools
    Fetches, parses and generates callable functions from OpenAPI specs
    """
    def __init__(self):
        self.apis = {}  # Dictionary of name -> API info
        self.endpoints = {}  # Dictionary of function_name -> endpoint info
        self.generated_functions = {}  # Dictionary of function_name -> function
        self.api_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.api_cache_dir, exist_ok=True)
        
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as a function name"""
        # Replace non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it starts with a letter
        if not sanitized[0].isalpha():
            sanitized = 'f_' + sanitized
        return sanitized.lower()
    
    def _get_operation_id(self, path: str, method: str, path_item: Dict[str, Any]) -> str:
        """Extract or generate an operationId from path item"""
        operation = path_item.get(method.lower(), {})
        
        # Use provided operationId if available
        if "operationId" in operation:
            return self._sanitize_name(operation["operationId"])
            
        # Generate an operationId from path and method
        path_parts = [p for p in path.split('/') if p and not p.startswith('{')]
        if path_parts:
            # Use last significant path part
            name_part = path_parts[-1]
            # If it's a common CRUD operation, use the part before it
            if name_part in ['create', 'read', 'update', 'delete', 'list'] and len(path_parts) > 1:
                name_part = f"{path_parts[-2]}_{name_part}"
        else:
            name_part = "root"
            
        return self._sanitize_name(f"{method.lower()}_{name_part}")
    
    def _parse_openapi_spec(self, spec: Dict[str, Any], base_url: str) -> Dict[str, Any]:
        """
        Parse an OpenAPI specification and extract endpoint information
        Returns a dictionary of function_name -> endpoint info
        """
        endpoints = {}
        
        # Get API title for namespacing
        api_title = self._sanitize_name(spec.get("info", {}).get("title", "api"))
        
        # Extract server URL if available
        servers = spec.get("servers", [])
        if servers and "url" in servers[0]:
            # If server URL is relative, join with base_url
            server_url = servers[0]["url"]
            if not server_url.startswith(('http://', 'https://')):
                base_url = urljoin(base_url, server_url)
            else:
                base_url = server_url
        
        # Ensure base_url ends with a slash
        if not base_url.endswith('/'):
            base_url += '/'
        
        # Process paths
        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            # Process HTTP methods (operations)
            for method in ['get', 'post', 'put', 'delete', 'patch']:
                if method in path_item:
                    operation = path_item[method]
                    
                    # Get or generate operation ID
                    operation_id = self._get_operation_id(path, method, path_item)
                    
                    # Ensure unique function names by adding API title prefix
                    function_name = f"{api_title}_{operation_id}"
                    
                    # Extract parameters
                    parameters = []
                    required_params = []
                    
                    # Path parameters
                    for param in path_item.get("parameters", []) + operation.get("parameters", []):
                        if param.get("in") in ["path", "query"]:
                            param_name = param.get("name")
                            param_type = param.get("schema", {}).get("type", "string")
                            param_desc = param.get("description", f"{param_name} parameter")
                            param_required = param.get("required", False)
                            
                            parameters.append({
                                "name": param_name,
                                "type": param_type,
                                "description": param_desc,
                                "required": param_required,
                                "in": param.get("in")
                            })
                            
                            if param_required:
                                required_params.append(param_name)
                    
                    # Request body for non-GET methods
                    if method != 'get' and 'requestBody' in operation:
                        content = operation["requestBody"].get("content", {})
                        if 'application/json' in content:
                            body_schema = content['application/json'].get("schema", {})
                            
                            # If it's a reference, resolve it
                            if "$ref" in body_schema:
                                ref = body_schema["$ref"]
                                # Extract component name from reference (e.g., "#/components/schemas/Pet" -> "Pet")
                                ref_name = ref.split('/')[-1]
                                # Simplified approach - in real implementation, would need to fully resolve the reference
                                parameters.append({
                                    "name": "body",
                                    "type": "object",
                                    "description": f"Request body ({ref_name})",
                                    "required": operation["requestBody"].get("required", True),
                                    "in": "body"
                                })
                                
                                if operation["requestBody"].get("required", True):
                                    required_params.append("body")
                            else:
                                # Handle inline schema
                                parameters.append({
                                    "name": "body",
                                    "type": "object",
                                    "description": "Request body",
                                    "required": operation["requestBody"].get("required", True),
                                    "in": "body"
                                })
                                
                                if operation["requestBody"].get("required", True):
                                    required_params.append("body")
                    
                    # Store endpoint information
                    endpoint_url = urljoin(base_url, path.lstrip('/'))
                    endpoints[function_name] = {
                        "method": method.upper(),
                        "url": endpoint_url,
                        "path": path,
                        "parameters": parameters,
                        "required_params": required_params,
                        "summary": operation.get("summary", ""),
                        "description": operation.get("description", operation.get("summary", "")),
                        "operation_id": operation_id,
                        "api_title": api_title
                    }
        
        return endpoints
    
    def fetch_openapi_spec(self, url: str, name: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Fetch and process an OpenAPI specification from a URL
        Returns (success, message_or_spec)
        """
        try:
            # Try to fetch the specification
            if url.endswith(('/openapi.json', '/swagger.json')):
                # Direct URL to spec
                spec_url = url
            elif url.endswith(('/docs', '/swagger', '/redoc')):
                # Docs URL, try to find the spec URL
                if url.endswith('/'):
                    spec_url = url + 'openapi.json'
                else:
                    spec_url = url + '/openapi.json'
            else:
                # Try to append standard paths
                parsed_url = urlparse(url)
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                
                # Common OpenAPI spec endpoints
                common_paths = [
                    '/openapi.json',
                    '/swagger.json',
                    '/api/openapi.json',
                    '/api/swagger.json',
                    '/api/docs/openapi.json'
                ]
                
                # Try each common path
                for path in common_paths:
                    spec_url = urljoin(base_url, path.lstrip('/'))
                    try:
                        response = requests.get(spec_url, timeout=10)
                        if response.status_code == 200:
                            break
                    except requests.RequestException:
                        continue
                else:
                    # If no common path works, use the provided URL
                    spec_url = url
            
            # Fetch the specification
            response = requests.get(spec_url, timeout=10)
            if response.status_code != 200:
                return False, {"error": f"Failed to fetch OpenAPI spec: HTTP {response.status_code}"}
            
            # Parse the JSON specification
            spec = response.json()
            
            # Validate that it's an OpenAPI specification
            if "openapi" not in spec and "swagger" not in spec:
                return False, {"error": "Invalid OpenAPI specification"}
            
            # Generate a name if not provided
            if name is None:
                name = self._sanitize_name(spec.get("info", {}).get("title", "api"))
            else:
                name = self._sanitize_name(name)
            
            # Store the API information
            self.apis[name] = {
                "name": name,
                "url": url,
                "spec_url": spec_url,
                "title": spec.get("info", {}).get("title", name),
                "version": spec.get("info", {}).get("version", "1.0.0"),
                "description": spec.get("info", {}).get("description", ""),
                "spec": spec
            }
            
            # Parse the specification and extract endpoints
            base_url = spec_url.rsplit('/', 1)[0]  # Remove the last part of the URL
            endpoints = self._parse_openapi_spec(spec, base_url)
            
            # Update the endpoints dictionary
            self.endpoints.update(endpoints)
            
            # Generate the functions
            self._generate_functions(name, endpoints)
            
            # Cache the specification
            self._cache_api_spec(name, spec)
            
            return True, {
                "api": name,
                "title": spec.get("info", {}).get("title", name),
                "version": spec.get("info", {}).get("version", "1.0.0"),
                "endpoints": len(endpoints),
                "functions": list(endpoints.keys())
            }
            
        except requests.RequestException as e:
            return False, {"error": f"Failed to fetch OpenAPI spec: {str(e)}"}
        except json.JSONDecodeError as e:
            return False, {"error": f"Failed to parse OpenAPI spec: {str(e)}"}
        except Exception as e:
            return False, {"error": f"Error processing OpenAPI spec: {str(e)}"}
    
    def _cache_api_spec(self, name: str, spec: Dict[str, Any]) -> None:
        """Cache the API specification to a file"""
        try:
            # Generate a filename from the API name
            filename = os.path.join(self.api_cache_dir, f"{name}.json")
            
            # Write the specification to the file
            with open(filename, 'w') as f:
                json.dump(spec, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache API spec: {str(e)}")
    
    def _load_cached_api_spec(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a cached API specification"""
        try:
            # Generate a filename from the API name
            filename = os.path.join(self.api_cache_dir, f"{name}.json")
            
            # Check if the file exists
            if not os.path.exists(filename):
                return None
                
            # Read the specification from the file
            with open(filename, 'r') as f:
                spec = json.load(f)
                
            return spec
        except Exception as e:
            logger.warning(f"Failed to load cached API spec: {str(e)}")
            return None
    
    def _generate_functions(self, api_name: str, endpoints: Dict[str, Any]) -> None:
        """Generate callable functions for the endpoints"""
        for function_name, endpoint in endpoints.items():
            # Create a function that calls the endpoint
            def _make_api_function(endpoint_info):
                def api_function(**kwargs):
                    # Prepare the request
                    method = endpoint_info["method"]
                    url = endpoint_info["url"]
                    
                    # Replace path parameters
                    for param in endpoint_info["parameters"]:
                        if param["in"] == "path" and param["name"] in kwargs:
                            placeholder = f"{{{param['name']}}}"
                            url = url.replace(placeholder, str(kwargs[param["name"]]))
                    
                    # Prepare query parameters
                    query_params = {}
                    for param in endpoint_info["parameters"]:
                        if param["in"] == "query" and param["name"] in kwargs:
                            query_params[param["name"]] = kwargs[param["name"]]
                    
                    # Prepare the body
                    body = None
                    for param in endpoint_info["parameters"]:
                        if param["in"] == "body" and param["name"] in kwargs:
                            body = kwargs[param["name"]]
                    
                    # Make the request
                    headers = {"Content-Type": "application/json"}
                    try:
                        if method == "GET":
                            response = requests.get(url, params=query_params, headers=headers, timeout=30)
                        elif method == "POST":
                            response = requests.post(url, params=query_params, json=body, headers=headers, timeout=30)
                        elif method == "PUT":
                            response = requests.put(url, params=query_params, json=body, headers=headers, timeout=30)
                        elif method == "DELETE":
                            response = requests.delete(url, params=query_params, json=body, headers=headers, timeout=30)
                        elif method == "PATCH":
                            response = requests.patch(url, params=query_params, json=body, headers=headers, timeout=30)
                        else:
                            return {"error": f"Unsupported HTTP method: {method}"}
                        
                        # Parse the response
                        if response.status_code >= 200 and response.status_code < 300:
                            # Try to parse as JSON first
                            try:
                                result = response.json()
                                return {
                                    "status_code": response.status_code,
                                    "data": result
                                }
                            except json.JSONDecodeError:
                                # If not JSON, return text
                                return {
                                    "status_code": response.status_code,
                                    "data": response.text
                                }
                        else:
                            # Error response
                            try:
                                error_data = response.json()
                            except json.JSONDecodeError:
                                error_data = response.text
                                
                            return {
                                "error": f"API Error: {response.status_code}",
                                "status_code": response.status_code,
                                "data": error_data
                            }
                    except requests.RequestException as e:
                        return {"error": f"Request failed: {str(e)}"}
                
                # Set the function name and docstring
                api_function.__name__ = function_name
                api_function.__doc__ = f"{endpoint_info['description']}\n\n" \
                                      f"API: {endpoint_info['api_title']}\n" \
                                      f"Method: {endpoint_info['method']} {endpoint_info['path']}\n\n" \
                                      f"Parameters:\n" + \
                                      '\n'.join([f"  {p['name']}: {p['description']} ({p['type']})" +
                                                 (" [REQUIRED]" if p['required'] else "")
                                                 for p in endpoint_info['parameters']])
                
                return api_function
            
            # Create the function
            self.generated_functions[function_name] = _make_api_function(endpoint)
    
    def get_function(self, function_name: str) -> Optional[Callable]:
        """Get a generated function by name"""
        return self.generated_functions.get(function_name)
    
    def list_apis(self) -> List[Dict[str, Any]]:
        """List all registered APIs"""
        return [
            {
                "name": api["name"],
                "title": api["title"],
                "version": api["version"],
                "description": api["description"],
                "endpoints": len([f for f in self.endpoints if f.startswith(api["name"])])
            }
            for api in self.apis.values()
        ]
    
    def get_api_functions(self, api_name: str) -> List[Dict[str, Any]]:
        """Get all functions for a specific API"""
        api_name = self._sanitize_name(api_name)
        functions = []
        
        for function_name, endpoint in self.endpoints.items():
            if function_name.startswith(f"{api_name}_"):
                params = {}
                for param in endpoint["parameters"]:
                    params[param["name"]] = {
                        "type": param["type"],
                        "description": param["description"],
                        "required": param["required"],
                        "in": param["in"]
                    }
                
                functions.append({
                    "name": function_name,
                    "description": endpoint["description"] or endpoint["summary"],
                    "method": endpoint["method"],
                    "path": endpoint["path"],
                    "parameters": params,
                    "required_params": endpoint["required_params"]
                })
        
        return functions
    
    def get_function_info(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific function"""
        if function_name not in self.endpoints:
            return None
            
        endpoint = self.endpoints[function_name]
        params = {}
        
        for param in endpoint["parameters"]:
            params[param["name"]] = {
                "type": param["type"],
                "description": param["description"],
                "required": param["required"],
                "in": param["in"]
            }
            
        return {
            "name": function_name,
            "description": endpoint["description"] or endpoint["summary"],
            "method": endpoint["method"],
            "path": endpoint["path"],
            "parameters": params,
            "required_params": endpoint["required_params"],
            "api": endpoint["api_title"]
        }
    
    def register_api_from_url(self, url: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Register an API from a URL"""
        success, result = self.fetch_openapi_spec(url, name)
        
        if success:
            return {
                "status": "success",
                "message": f"API '{result['api']}' registered successfully",
                "api": result
            }
        else:
            return {
                "status": "error",
                "message": result.get("error", "Unknown error"),
                "details": result
            }

# Create a global instance
openapi_registry = OpenAPIRegistry()