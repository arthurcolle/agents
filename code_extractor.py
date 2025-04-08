import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CodeBlock:
    """Represents an extracted code block with metadata"""
    language: str
    code: str
    line_count: int
    start_line: int
    end_line: int

class CodeExtractor:
    """Extracts code blocks from text using structured output parsing"""
    
    def __init__(self):
        # Regex patterns for different code block formats
        self.patterns = {
            # Markdown-style code blocks: ```python\ncode\n```
            'markdown': re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL),
            
            # Indented code blocks (4 spaces or tab)
            'indented': re.compile(r'(?:^(?:[ ]{4}|\t).*?$\n?)+', re.MULTILINE),
            
            # Custom tags: <|python_start|>code<|python_end|>
            'custom_tags': re.compile(r'<\|(\w+)_start\|\>(.*?)<\|(\w+)_end\|\>', re.DOTALL),
            
            # Inline code: `code`
            'inline': re.compile(r'`([^`]+)`'),
            
            # HTML code tags: <code>code</code>
            'html': re.compile(r'<code(?:\s+class="(\w+)")?>([^<]+)</code>', re.DOTALL),
        }
    
    def extract_code_blocks(self, text: str) -> List[CodeBlock]:
        """
        Extract all code blocks from the given text
        
        Args:
            text: The text to extract code blocks from
            
        Returns:
            List of CodeBlock objects containing the extracted code
        """
        blocks = []
        
        # Process markdown-style code blocks
        for match in self.patterns['markdown'].finditer(text):
            language = match.group(1) or "text"
            code = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers
            start_line = text[:start_pos].count('\n') + 1
            end_line = start_line + code.count('\n') + 1
            
            blocks.append(CodeBlock(
                language=language,
                code=code,
                line_count=code.count('\n') + 1,
                start_line=start_line,
                end_line=end_line
            ))
        
        # Process custom tag blocks
        for match in self.patterns['custom_tags'].finditer(text):
            start_tag = match.group(1)
            code = match.group(2)
            end_tag = match.group(3)
            
            # Verify matching tags
            if start_tag == end_tag:
                language = start_tag
                start_pos = match.start()
                end_pos = match.end()
                
                # Calculate line numbers
                start_line = text[:start_pos].count('\n') + 1
                end_line = start_line + code.count('\n') + 1
                
                blocks.append(CodeBlock(
                    language=language,
                    code=code,
                    line_count=code.count('\n') + 1,
                    start_line=start_line,
                    end_line=end_line
                ))
        
        # Process HTML code tags
        for match in self.patterns['html'].finditer(text):
            language = match.group(1) or "text"
            code = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate line numbers
            start_line = text[:start_pos].count('\n') + 1
            end_line = start_line + code.count('\n') + 1
            
            blocks.append(CodeBlock(
                language=language,
                code=code,
                line_count=code.count('\n') + 1,
                start_line=start_line,
                end_line=end_line
            ))
        
        return blocks
    
    def extract_python_code(self, text: str) -> List[str]:
        """
        Extract only Python code blocks from the given text
        
        Args:
            text: The text to extract Python code from
            
        Returns:
            List of Python code strings
        """
        blocks = self.extract_code_blocks(text)
        return [block.code for block in blocks if block.language.lower() in ('python', 'py')]
    
    def extract_and_execute(self, text: str) -> Dict[str, Any]:
        """
        Extract Python code blocks and execute them
        
        Args:
            text: The text containing code to extract and execute
            
        Returns:
            Dictionary with execution results
        """
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        code_blocks = self.extract_python_code(text)
        if not code_blocks:
            return {"success": False, "error": "No Python code blocks found"}
        
        # Join all code blocks
        full_code = "\n\n".join(code_blocks)
        
        # Execute the code
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        local_vars = {}
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(full_code, globals(), local_vars)
                
            return {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "local_vars": {k: v for k, v in local_vars.items() if not k.startswith('_')},
                "code_blocks": code_blocks,
                "line_count": full_code.count('\n') + 1
            }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "code_blocks": code_blocks
            }

# Example usage
if __name__ == "__main__":
    extractor = CodeExtractor()
    
    sample_text = """
Here's a simple Python function:

```python
def hello_world():
    print("Hello, world!")
    return 42
```

And another one:

<|python_start|>
import numpy as np

def calculate_mean(numbers):
    return np.mean(numbers)
<|python_end|>
"""
    
    blocks = extractor.extract_code_blocks(sample_text)
    for i, block in enumerate(blocks):
        print(f"Block {i+1} ({block.language}):")
        print(f"Lines: {block.line_count} (from {block.start_line} to {block.end_line})")
        print(block.code)
        print("-" * 40)
