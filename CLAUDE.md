# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run: `python <script_name>.py --interactive`
- Analyze file: `python self_modify_agent.py --target-file <file_path>`
- Improve function: `python self_modify_agent.py --target-file <file_path> --function <function_name>`
- Improve method: `python self_modify_agent.py --target-file <file_path> --class <class_name> --method <method_name>`

## Environment Setup
- Weather API: Set the OpenWeatherMap API key with `export OPENWEATHERMAP_API_KEY=your_api_key_here`

## Code Style Guidelines
- **Imports**: Standard library first, third-party imports second (with try/except for optional deps), relative imports last
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Types**: Use typing module annotations (Dict, List, Optional, etc.) consistently
- **Documentation**: Detailed docstrings for modules, classes, and functions
- **Error Handling**: Use try/except blocks with specific exceptions, return tuple(success, result/error)
- **Formatting**: 4-space indentation, 100 character line length
- **Code Structure**: Clear separation between core functionality and interfaces