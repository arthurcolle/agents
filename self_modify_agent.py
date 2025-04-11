#!/usr/bin/env python3
"""
Self-Modifying Agent

An agent that can analyze, modify, and improve its own code using:
- OpenRouter Llama 4 Maverick for code analysis and generation
- Protected kernel architecture to prevent self-destruction
- Code editor with AST-based analysis for safe modifications
- Hot module reloading to apply changes without restarting

Usage:
  python self_modify_agent.py [--target-file FILE] [--function FUNCTION] [--interactive]
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union

# Import our modules
from openrouter_kernel import OpenRouterKernel, register_kernel_function
from agent_editor import AgentEditor

# === CONFIG ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "meta-llama/llama-4-maverick")
SYSTEM_PROMPT = """You are an AI assistant specialized in analyzing and improving Python code.
Your task is to analyze code, identify potential improvements, and implement those changes safely.
Focus on:
1. Performance optimizations
2. Bug fixes and edge case handling
3. Code organization and readability
4. Adding helpful docstrings
5. Simplifying complex logic

Always explain your reasoning and the benefits of your changes. Be specific and precise in your explanations.
"""

# === AGENT CAPABILITIES ===
@register_kernel_function(
    name="analyze_code",
    description="Analyze a code snippet and provide insights"
)
def analyze_code(code: str) -> Dict:
    """
    Analyze a code snippet and provide insights
    
    Args:
        code: Python code to analyze
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Basic static analysis
        import ast
        tree = ast.parse(code)
        
        # Count functions and classes
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Count imports
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # Look for potential issues
        issues = []
        
        # Check for unused imports
        imported_names = set()
        for node in imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name if alias.asname is None else alias.asname)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name if alias.asname is None else alias.asname)
                    
        # Check for usage of imported names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
                
        unused_imports = imported_names - used_names
        if unused_imports:
            issues.append(f"Unused imports: {', '.join(unused_imports)}")
            
        # Check for TODO comments
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                if "TODO" in node.value.s:
                    issues.append(f"TODO found: {node.value.s.strip()}")
                    
        # Check for complex functions (too many lines)
        for func in functions:
            if hasattr(func, 'end_lineno') and func.end_lineno - func.lineno > 50:
                issues.append(f"Function '{func.name}' is very long ({func.end_lineno - func.lineno} lines)")
                
        # Check for missing docstrings
        for func in functions:
            if not (func.body and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Str)):
                issues.append(f"Function '{func.name}' missing docstring")
                
        for cls in classes:
            if not (cls.body and isinstance(cls.body[0], ast.Expr) and isinstance(cls.body[0].value, ast.Str)):
                issues.append(f"Class '{cls.name}' missing docstring")
                
        return {
            "functions": len(functions),
            "classes": len(classes),
            "imports": len(imports),
            "lines": len(code.split('\n')),
            "issues": issues,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@register_kernel_function(
    name="suggest_improvements",
    description="Suggest improvements for a code snippet"
)
def suggest_improvements(code: str, analysis: Optional[Dict] = None) -> Dict:
    """
    This is a placeholder function that will be handled by the LLM.
    The actual implementation will be provided by the agent's reasoning.
    """
    # This will be replaced with LLM-based analysis
    return {
        "status": "success",
        "message": "This function is handled by the LLM. Use the agent to process this."
    }

@register_kernel_function(
    name="improve_code",
    description="Generate improved version of code"
)
def improve_code(code: str, issues: List[str]) -> Dict:
    """
    This is a placeholder function that will be handled by the LLM.
    The actual implementation will be provided by the agent's reasoning.
    """
    # This will be replaced with LLM-based code improvement
    return {
        "status": "success",
        "message": "This function is handled by the LLM. Use the agent to process this."
    }

# === SELF-MODIFYING AGENT ===
class SelfModifyingAgent:
    """
    Agent that can analyze and modify its own code
    """
    def __init__(self, api_key: str = None, model: str = MODEL):
        """Initialize the agent with necessary components"""
        self.kernel = OpenRouterKernel(api_key, model)
        self.editor = AgentEditor()
        
        # Track modifications
        self.modifications = []
        
        # Register LLM handlers for certain functions
        self._register_llm_handlers()
        
    def _register_llm_handlers(self):
        """Register LLM-powered function handlers"""
        # Override suggest_improvements with LLM-powered version
        self.kernel.register_function(
            "suggest_improvements",
            self._llm_suggest_improvements,
            "Suggest improvements for a code snippet"
        )
        
        # Override improve_code with LLM-powered version
        self.kernel.register_function(
            "improve_code",
            self._llm_improve_code,
            "Generate improved version of code"
        )
        
    def _llm_suggest_improvements(self, code: str, analysis: Optional[Dict] = None) -> Dict:
        """LLM-powered code improvement suggestions"""
        if not analysis:
            # Get analysis if not provided
            analysis_result = analyze_code(code)
            analysis = analysis_result if analysis_result["status"] == "success" else None
            
        # Prepare prompt
        prompt = f"""Analyze the following Python code and suggest specific improvements:

```python
{code}
```

{"Analysis results:" + json.dumps(analysis, indent=2) if analysis else ""}

Provide a detailed list of suggested improvements, focusing on:
1. Performance optimizations
2. Bug fixes and edge case handling 
3. Code organization and readability
4. Better docstrings and comments
5. Simplifying complex logic

Format your response as JSON with:
- "improvements": A list of specific improvements
- "reasoning": Short explanation of why these changes would be beneficial
"""
        
        # Get suggestions from LLM
        responses = []
        for chunk in self.kernel.stream_chat(prompt):
            if chunk["type"] == "content":
                responses.append(chunk["content"])
                
        # Join responses and extract JSON
        response = "".join(responses)
        
        # Try to extract JSON from the response
        try:
            # Look for JSON object in the response
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
            else:
                # Try to parse the whole response as JSON
                result = json.loads(response)
                
            # Ensure required fields exist
            if "improvements" not in result:
                result["improvements"] = []
            if "reasoning" not in result:
                result["reasoning"] = "No specific reasoning provided"
                
            result["status"] = "success"
            return result
        except Exception as e:
            # If JSON parsing fails, try to extract improvements manually
            improvements = []
            lines = response.split('\n')
            current_improvement = ""
            
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    # New numbered improvement
                    if current_improvement:
                        improvements.append(current_improvement.strip())
                    current_improvement = line
                elif current_improvement:
                    current_improvement += " " + line
                    
            # Add the last improvement
            if current_improvement:
                improvements.append(current_improvement.strip())
                
            return {
                "status": "success",
                "improvements": improvements,
                "reasoning": "Extracted from non-JSON response",
                "parse_error": str(e)
            }
            
    def _llm_improve_code(self, code: str, issues: List[str]) -> Dict:
        """LLM-powered code improvement implementation"""
        # Prepare prompt
        prompt = f"""Improve the following Python code by addressing these specific issues:

ISSUES TO ADDRESS:
{chr(10).join(f"- {issue}" for issue in issues)}

ORIGINAL CODE:
```python
{code}
```

Provide a new, improved version of the code that fixes these issues.
Keep the function signature the same unless absolutely necessary.
Include clear, descriptive docstrings with parameter descriptions.
Focus on making minimal changes to fix the issues while preserving functionality.

Return ONLY the improved code in a code block, nothing else.
"""
        
        # Get improved code from LLM
        responses = []
        for chunk in self.kernel.stream_chat(prompt):
            if chunk["type"] == "content":
                responses.append(chunk["content"])
                
        # Join responses and extract code
        response = "".join(responses)
        
        # Try to extract code from the response
        try:
            # Look for code block in the response
            import re
            code_match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', response)
            if code_match:
                improved_code = code_match.group(1)
            else:
                # If no code block, use the whole response
                improved_code = response
                
            return {
                "status": "success",
                "improved_code": improved_code,
                "original_code": code
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "raw_response": response
            }
            
    def analyze_file(self, file_path: str) -> Dict:
        """Analyze a Python file"""
        # First use editor to get code structure
        structure = self.editor.analyze_file(file_path)
        
        # Get the file content
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Run static analysis
        analysis = analyze_code(content)
        
        # Get improvement suggestions
        suggestions = self._llm_suggest_improvements(content, analysis)
        
        return {
            "file_path": file_path,
            "structure": structure,
            "analysis": analysis,
            "suggestions": suggestions
        }
        
    def improve_function(self, file_path: str, function_name: str) -> Dict:
        """Analyze and improve a specific function"""
        # Get the function code
        function_code = self.editor.extract_function(file_path, function_name)
        if not function_code:
            return {
                "status": "error",
                "message": f"Function '{function_name}' not found in {file_path}"
            }
            
        # Analyze the function
        analysis = analyze_code(function_code)
        
        # Get improvement suggestions
        suggestions = self._llm_suggest_improvements(function_code, analysis)
        
        # Generate improved code
        if "improvements" in suggestions and suggestions["improvements"]:
            result = self._llm_improve_code(function_code, suggestions["improvements"])
            
            if result["status"] == "success" and "improved_code" in result:
                # Apply the changes
                success = self.editor.replace_function(file_path, function_name, result["improved_code"])
                
                if success:
                    # Track the modification
                    self.modifications.append({
                        "timestamp": time.time(),
                        "file": file_path,
                        "function": function_name,
                        "improvements": suggestions["improvements"],
                        "before": function_code,
                        "after": result["improved_code"]
                    })
                    
                    # Reload the module
                    reload_success, module = self.editor.reload_module(file_path)
                    
                    return {
                        "status": "success",
                        "function": function_name,
                        "file": file_path,
                        "improvements": suggestions["improvements"],
                        "reload_success": reload_success,
                        "modified": True
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to apply changes to function '{function_name}'",
                        "improvements": suggestions["improvements"]
                    }
            else:
                return {
                    "status": "error",
                    "message": "Failed to generate improved code",
                    "error": result.get("error", "Unknown error")
                }
        else:
            return {
                "status": "info",
                "message": f"No improvements needed for function '{function_name}'",
                "function": function_name,
                "file": file_path,
                "modified": False
            }
            
    def improve_method(self, file_path: str, class_name: str, method_name: str) -> Dict:
        """Analyze and improve a specific class method"""
        # Get the method code
        method_code = self.editor.extract_method(file_path, class_name, method_name)
        if not method_code:
            return {
                "status": "error",
                "message": f"Method '{class_name}.{method_name}' not found in {file_path}"
            }
            
        # Analyze the method
        analysis = analyze_code(method_code)
        
        # Get improvement suggestions
        suggestions = self._llm_suggest_improvements(method_code, analysis)
        
        # Generate improved code
        if "improvements" in suggestions and suggestions["improvements"]:
            result = self._llm_improve_code(method_code, suggestions["improvements"])
            
            if result["status"] == "success" and "improved_code" in result:
                # Apply the changes
                success = self.editor.replace_method(file_path, class_name, method_name, result["improved_code"])
                
                if success:
                    # Track the modification
                    self.modifications.append({
                        "timestamp": time.time(),
                        "file": file_path,
                        "class": class_name,
                        "method": method_name,
                        "improvements": suggestions["improvements"],
                        "before": method_code,
                        "after": result["improved_code"]
                    })
                    
                    # Reload the module
                    reload_success, module = self.editor.reload_module(file_path)
                    
                    return {
                        "status": "success",
                        "class": class_name,
                        "method": method_name,
                        "file": file_path,
                        "improvements": suggestions["improvements"],
                        "reload_success": reload_success,
                        "modified": True
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to apply changes to method '{class_name}.{method_name}'",
                        "improvements": suggestions["improvements"]
                    }
            else:
                return {
                    "status": "error",
                    "message": "Failed to generate improved code",
                    "error": result.get("error", "Unknown error")
                }
        else:
            return {
                "status": "info",
                "message": f"No improvements needed for method '{class_name}.{method_name}'",
                "class": class_name,
                "method": method_name,
                "file": file_path,
                "modified": False
            }
            
    def get_modification_history(self) -> List[Dict]:
        """Get history of all modifications made by the agent"""
        return self.modifications
        
    def undo_last_modification(self) -> Dict:
        """Undo the last modification"""
        if not self.modifications:
            return {
                "status": "error",
                "message": "No modifications to undo"
            }
            
        # Get the last modification
        last_mod = self.modifications.pop()
        
        # Check the type of modification
        if "method" in last_mod:
            # It's a method modification
            success = self.editor.replace_method(
                last_mod["file"],
                last_mod["class"],
                last_mod["method"],
                last_mod["before"]
            )
        else:
            # It's a function modification
            success = self.editor.replace_function(
                last_mod["file"],
                last_mod["function"],
                last_mod["before"]
            )
            
        if success:
            # Reload the module
            reload_success, module = self.editor.reload_module(last_mod["file"])
            
            return {
                "status": "success",
                "message": "Successfully undid last modification",
                "modification": last_mod,
                "reload_success": reload_success
            }
        else:
            # Put the modification back in the list
            self.modifications.append(last_mod)
            
            return {
                "status": "error",
                "message": "Failed to undo last modification"
            }
    
    def run_interactive(self):
        """Run the agent in interactive mode"""
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  🤖 Self-Modifying Agent using Llama 4 Maverick                 ║
║  Code Analysis • Self-Improvement • Safe Modifications         ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
        """)
        
        print("\n💡 Interactive mode. Type 'help' for commands, 'exit' to quit.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("> ")
                
                # Check for exit command
                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Goodbye!")
                    break
                    
                # Check for help command
                if user_input.lower() == "help":
                    print("\nAvailable commands:")
                    print("  help              - Show this help message")
                    print("  analyze <file>    - Analyze a Python file")
                    print("  improve <file> <function> - Improve a specific function")
                    print("  improve-method <file> <class> <method> - Improve a class method")
                    print("  history           - Show modification history")
                    print("  undo              - Undo the last modification")
                    print("  exit              - Exit the program")
                    continue
                    
                # Check for analyze command
                if user_input.lower().startswith("analyze "):
                    file_path = user_input[8:].strip()
                    if not os.path.exists(file_path):
                        print(f"❌ File not found: {file_path}")
                        continue
                        
                    print(f"🔍 Analyzing {file_path}...")
                    result = self.analyze_file(file_path)
                    
                    print(f"\n📊 Analysis Results for {file_path}:")
                    print(f"  Functions: {result['analysis']['functions']}")
                    print(f"  Classes: {result['analysis']['classes']}")
                    print(f"  Lines: {result['analysis']['lines']}")
                    
                    if result['analysis']['issues']:
                        print("\n⚠️ Issues Found:")
                        for issue in result['analysis']['issues']:
                            print(f"  - {issue}")
                            
                    if 'improvements' in result['suggestions']:
                        print("\n💡 Suggested Improvements:")
                        for i, improvement in enumerate(result['suggestions']['improvements'], 1):
                            print(f"  {i}. {improvement}")
                            
                    continue
                    
                # Check for improve command
                if user_input.lower().startswith("improve "):
                    parts = user_input[8:].strip().split(" ", 1)
                    if len(parts) != 2:
                        print("❌ Invalid format. Use: improve <file> <function>")
                        continue
                        
                    file_path, function_name = parts
                    
                    if not os.path.exists(file_path):
                        print(f"❌ File not found: {file_path}")
                        continue
                        
                    print(f"🔧 Improving function '{function_name}' in {file_path}...")
                    result = self.improve_function(file_path, function_name)
                    
                    if result["status"] == "success" and result.get("modified", False):
                        print(f"✅ Successfully improved function '{function_name}':")
                        for i, improvement in enumerate(result['improvements'], 1):
                            print(f"  {i}. {improvement}")
                    elif result["status"] == "info":
                        print(f"ℹ️ {result['message']}")
                    else:
                        print(f"❌ {result['message']}")
                        
                    continue
                    
                # Check for improve-method command
                if user_input.lower().startswith("improve-method "):
                    parts = user_input[14:].strip().split(" ", 2)
                    if len(parts) != 3:
                        print("❌ Invalid format. Use: improve-method <file> <class> <method>")
                        continue
                        
                    file_path, class_name, method_name = parts
                    
                    if not os.path.exists(file_path):
                        print(f"❌ File not found: {file_path}")
                        continue
                        
                    print(f"🔧 Improving method '{class_name}.{method_name}' in {file_path}...")
                    result = self.improve_method(file_path, class_name, method_name)
                    
                    if result["status"] == "success" and result.get("modified", False):
                        print(f"✅ Successfully improved method '{class_name}.{method_name}':")
                        for i, improvement in enumerate(result['improvements'], 1):
                            print(f"  {i}. {improvement}")
                    elif result["status"] == "info":
                        print(f"ℹ️ {result['message']}")
                    else:
                        print(f"❌ {result['message']}")
                        
                    continue
                    
                # Check for history command
                if user_input.lower() == "history":
                    modifications = self.get_modification_history()
                    
                    if not modifications:
                        print("ℹ️ No modifications have been made yet")
                        continue
                        
                    print(f"\n📜 Modification History ({len(modifications)} changes):")
                    for i, mod in enumerate(modifications, 1):
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod["timestamp"]))
                        if "method" in mod:
                            print(f"  {i}. [{timestamp}] Modified method '{mod['class']}.{mod['method']}' in {mod['file']}")
                        else:
                            print(f"  {i}. [{timestamp}] Modified function '{mod['function']}' in {mod['file']}")
                            
                    continue
                    
                # Check for undo command
                if user_input.lower() == "undo":
                    result = self.undo_last_modification()
                    
                    if result["status"] == "success":
                        mod = result["modification"]
                        if "method" in mod:
                            print(f"✅ Undid change to method '{mod['class']}.{mod['method']}' in {mod['file']}")
                        else:
                            print(f"✅ Undid change to function '{mod['function']}' in {mod['file']}")
                    else:
                        print(f"❌ {result['message']}")
                        
                    continue
                    
                # Unknown command
                print(f"❓ Unknown command: {user_input}")
                print("Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\n⚠️ Operation interrupted. Type 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue

# === MAIN FUNCTION ===
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Self-Modifying Agent")
    parser.add_argument("--target-file", help="Target file to analyze and modify")
    parser.add_argument("--function", help="Specific function to improve")
    parser.add_argument("--class", dest="class_name", help="Class containing method to improve")
    parser.add_argument("--method", help="Method to improve (requires --class)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check API key
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key_here")
        return 1
        
    # Create the agent
    agent = SelfModifyingAgent(OPENROUTER_API_KEY)
    
    # Run in interactive mode if requested
    if args.interactive:
        agent.run_interactive()
        return 0
        
    # Otherwise, process the command-line arguments
    if args.target_file:
        if not os.path.exists(args.target_file):
            print(f"❌ Target file not found: {args.target_file}")
            return 1
            
        if args.function:
            # Improve a specific function
            print(f"🔧 Improving function '{args.function}' in {args.target_file}...")
            result = agent.improve_function(args.target_file, args.function)
            
            if result["status"] == "success" and result.get("modified", False):
                print(f"✅ Successfully improved function '{args.function}':")
                for i, improvement in enumerate(result['improvements'], 1):
                    print(f"  {i}. {improvement}")
                return 0
            elif result["status"] == "info":
                print(f"ℹ️ {result['message']}")
                return 0
            else:
                print(f"❌ {result['message']}")
                return 1
                
        elif args.class_name and args.method:
            # Improve a specific method
            print(f"🔧 Improving method '{args.class_name}.{args.method}' in {args.target_file}...")
            result = agent.improve_method(args.target_file, args.class_name, args.method)
            
            if result["status"] == "success" and result.get("modified", False):
                print(f"✅ Successfully improved method '{args.class_name}.{args.method}':")
                for i, improvement in enumerate(result['improvements'], 1):
                    print(f"  {i}. {improvement}")
                return 0
            elif result["status"] == "info":
                print(f"ℹ️ {result['message']}")
                return 0
            else:
                print(f"❌ {result['message']}")
                return 1
                
        else:
            # Just analyze the file
            print(f"🔍 Analyzing {args.target_file}...")
            result = agent.analyze_file(args.target_file)
            
            print(f"\n📊 Analysis Results for {args.target_file}:")
            print(f"  Functions: {result['analysis']['functions']}")
            print(f"  Classes: {result['analysis']['classes']}")
            print(f"  Lines: {result['analysis']['lines']}")
            
            if result['analysis']['issues']:
                print("\n⚠️ Issues Found:")
                for issue in result['analysis']['issues']:
                    print(f"  - {issue}")
                    
            if 'improvements' in result['suggestions']:
                print("\n💡 Suggested Improvements:")
                for i, improvement in enumerate(result['suggestions']['improvements'], 1):
                    print(f"  {i}. {improvement}")
                    
            return 0
    else:
        # No target file specified, default to interactive mode
        print("ℹ️ No target file specified. Running in interactive mode...")
        agent.run_interactive()
        return 0

if __name__ == "__main__":
    sys.exit(main())