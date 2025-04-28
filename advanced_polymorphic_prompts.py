#!/usr/bin/env python3
"""
Advanced Prompt Management System for AI Agents

This module extends the polymorphic_prompts.py implementation with advanced capabilities:
- Prompt chaining and composition
- Adaptive prompt generation
- Retrieval-augmented prompting
- Dynamic template substitution
- Conditional prompt branching
- Advanced prompt evaluation and optimization
- Multi-modal prompt integration
- Bidirectional prompt feedback

Features:
- Create, chain, and compose specialized prompts
- Dynamic prompt generation based on context
- Track prompt performance and optimize over time
- Manage complex prompt workflows and pipelines
"""

import os
import sys
import json
import yaml
import time
import shlex
import asyncio
import argparse
import traceback
import numpy as np
import networkx as nx
from uuid import uuid4
from glob import glob
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, Union, TypedDict, Generator

# Import embedding functions if available
try:
    # Try the new embedding service first
    from embedding_service import compute_embedding
    has_embedding = True
except ImportError:
    try:
        # Fall back to holographic memory if available
        from holographic_memory import compute_embedding
        has_embedding = True
    except ImportError:
        has_embedding = False
        print("Warning: Embedding functionality not available. Some features will be limited.")

# Optional LLM integration
try:
    import openai
    has_llm = True
except ImportError:
    has_llm = False
    print("Warning: LLM functionality not available. Some features will be limited.")

# Optional graph visualization
try:
    import matplotlib.pyplot as plt
    has_visualization = True
except ImportError:
    has_visualization = False

# Types and Data Classes
@dataclass
class PromptMetadata:
    """Metadata for tracking prompt information"""
    uuid: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    tags: List[str] = field(default_factory=list)
    prompt_type: str = "standard"
    author: str = ""
    description: str = ""
    usage_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

@dataclass
class Prompt:
    """Core prompt class with content and metadata"""
    name: str
    content: str
    metadata: PromptMetadata = field(default_factory=PromptMetadata)
    variables: Dict[str, str] = field(default_factory=dict)
    embeddings: Dict[str, Any] = field(default_factory=dict)

class PromptChain:
    """A sequence of prompts that can be executed in order"""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.nodes = []  # List of prompt nodes
        self.edges = []  # List of (from_idx, to_idx, condition) tuples
        self.metadata = PromptMetadata()
        self.metadata.prompt_type = "chain"
        self.graph = nx.DiGraph()
        
    def add_node(self, prompt: Prompt, node_type: str = "prompt") -> int:
        """Add a prompt node to the chain"""
        node_id = len(self.nodes)
        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "prompt": prompt
        })
        self.graph.add_node(node_id, type=node_type, name=prompt.name)
        return node_id
        
    def add_edge(self, from_id: int, to_id: int, condition: str = None) -> None:
        """Add a directed edge between prompts with optional condition"""
        if from_id >= len(self.nodes) or to_id >= len(self.nodes):
            raise ValueError("Invalid node IDs")
            
        self.edges.append((from_id, to_id, condition))
        self.graph.add_edge(from_id, to_id, condition=condition)
        
    def validate(self) -> bool:
        """Check if the prompt chain is valid (connected, no cycles if strict pipeline)"""
        if not self.nodes:
            return False
            
        # Check if graph is connected
        if not nx.is_weakly_connected(self.graph):
            return False
            
        return True
        
    def visualize(self, output_file: str = None) -> None:
        """Create a visualization of the prompt chain"""
        if not has_visualization:
            print("Visualization requires matplotlib. Please install it to use this feature.")
            return
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes with different colors based on type
        node_colors = []
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node].get('type', 'prompt')
            if node_type == 'prompt':
                node_colors.append('skyblue')
            elif node_type == 'conditional':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
                
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors)
        nx.draw_networkx_edges(self.graph, pos, arrows=True)
        
        # Draw node labels (prompt names)
        labels = {node: self.graph.nodes[node].get('name', str(node)) for node in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=labels)
        
        # Draw edge labels (conditions)
        edge_labels = {(u, v): d.get('condition', '') for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        plt.title(f"Prompt Chain: {self.name}")
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
            
    def to_dict(self) -> Dict:
        """Convert the prompt chain to a dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": asdict(self.metadata),
            "nodes": [
                {
                    "id": node["id"],
                    "type": node["type"],
                    "prompt": {
                        "name": node["prompt"].name,
                        "content": node["prompt"].content,
                        "metadata": asdict(node["prompt"].metadata),
                        "variables": node["prompt"].variables
                    }
                } for node in self.nodes
            ],
            "edges": [
                {
                    "from": edge[0],
                    "to": edge[1],
                    "condition": edge[2]
                } for edge in self.edges
            ]
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'PromptChain':
        """Create a prompt chain from a dictionary representation"""
        chain = cls(data["name"], data.get("description", ""))
        
        # Set metadata
        if "metadata" in data:
            metadata_dict = data["metadata"]
            chain.metadata = PromptMetadata(**metadata_dict)
            
        # Create nodes
        for node_data in data["nodes"]:
            prompt_data = node_data["prompt"]
            prompt = Prompt(
                name=prompt_data["name"],
                content=prompt_data["content"]
            )
            
            if "metadata" in prompt_data:
                prompt.metadata = PromptMetadata(**prompt_data["metadata"])
                
            if "variables" in prompt_data:
                prompt.variables = prompt_data["variables"]
                
            chain.add_node(prompt, node_data["type"])
            
        # Create edges
        for edge_data in data["edges"]:
            chain.add_edge(
                edge_data["from"],
                edge_data["to"],
                edge_data.get("condition")
            )
            
        return chain

class PromptEvaluator:
    """Evaluates prompt performance using predefined metrics"""
    def __init__(self):
        self.metrics = {
            "clarity": self._evaluate_clarity,
            "specificity": self._evaluate_specificity,
            "execution_time": self._evaluate_execution_time,
            "token_efficiency": self._evaluate_token_efficiency,
            "success_rate": self._evaluate_success_rate
        }
        
    def evaluate(self, prompt: Union[Prompt, PromptChain], metric_names: List[str] = None) -> Dict[str, float]:
        """Evaluate a prompt or prompt chain using selected metrics"""
        if metric_names is None:
            # Use all available metrics
            metric_names = list(self.metrics.keys())
            
        results = {}
        for metric in metric_names:
            if metric in self.metrics:
                try:
                    results[metric] = self.metrics[metric](prompt)
                except Exception as e:
                    print(f"Error evaluating {metric}: {e}")
                    results[metric] = 0.0
                    
        return results
        
    def _evaluate_clarity(self, prompt: Union[Prompt, PromptChain]) -> float:
        """Evaluate prompt clarity (placeholder implementation)"""
        # This would typically use an LLM to evaluate clarity
        # Placeholder scoring based on simple heuristics
        if isinstance(prompt, Prompt):
            content = prompt.content
        else:
            # For chains, evaluate the first prompt
            if not prompt.nodes:
                return 0.0
            content = prompt.nodes[0]["prompt"].content
            
        # Simple heuristics for clarity
        score = 0.0
        
        # Check for structure (paragraphs, bullets, numbering)
        if any(marker in content for marker in ['1.', '•', '-', '*', '\n\n']):
            score += 0.2
            
        # Check for clear instructions
        instruction_words = ['analyze', 'identify', 'evaluate', 'describe', 'explain', 'compare', 'list']
        if any(word in content.lower() for word in instruction_words):
            score += 0.3
            
        # Check for examples
        if 'example' in content.lower() or 'for instance' in content.lower():
            score += 0.2
            
        # Check length (not too short, not too long)
        words = content.split()
        if 20 <= len(words) <= 500:
            score += 0.3
        elif len(words) > 500:
            score += 0.1
            
        return min(score, 1.0)
        
    def _evaluate_specificity(self, prompt: Union[Prompt, PromptChain]) -> float:
        """Evaluate prompt specificity (placeholder implementation)"""
        # Placeholder scoring based on simple heuristics
        if isinstance(prompt, Prompt):
            content = prompt.content
        else:
            # For chains, evaluate the first prompt
            if not prompt.nodes:
                return 0.0
            content = prompt.nodes[0]["prompt"].content
            
        # Simple heuristics for specificity
        score = 0.0
        
        # Check for specific details or parameters
        if any(char in content for char in [':', '%', '#', '@']):
            score += 0.2
            
        # Check for quantitative specifications
        numbers = sum(1 for c in content if c.isdigit())
        score += min(numbers / 20, 0.3)
        
        # Check for specific domain terms or jargon
        # This is a very simple approximation
        words = content.lower().split()
        unusual_word_count = sum(1 for word in words if len(word) > 8)
        score += min(unusual_word_count / 10, 0.3)
        
        # Check for constraints or requirements
        constraint_phrases = ['must be', 'should include', 'required', 'necessary', 'ensure that']
        if any(phrase in content.lower() for phrase in constraint_phrases):
            score += 0.2
            
        return min(score, 1.0)
        
    def _evaluate_execution_time(self, prompt: Union[Prompt, PromptChain]) -> float:
        """Measure the execution time of a prompt (placeholder implementation)"""
        # In a real implementation, this would time the execution of the prompt through an LLM
        # For now, return a synthetic score based on prompt complexity
        
        if isinstance(prompt, Prompt):
            content = prompt.content
            complexity = min(len(content) / 2000, 1.0)  # Longer prompts are more complex
            execution_time_score = 1.0 - complexity  # Higher score for faster execution
        else:
            # For chains, estimate based on number of nodes and edges
            num_nodes = len(prompt.nodes)
            num_edges = len(prompt.edges)
            complexity = min((num_nodes + num_edges) / 20, 1.0)
            execution_time_score = 1.0 - complexity
            
        return max(0.1, execution_time_score)
        
    def _evaluate_token_efficiency(self, prompt: Union[Prompt, PromptChain]) -> float:
        """Evaluate the token efficiency of a prompt (placeholder implementation)"""
        # In a real implementation, this would count tokens and evaluate output/input token ratio
        
        if isinstance(prompt, Prompt):
            content = prompt.content
            # Approximate token count (words + punctuation)
            words = content.split()
            punctuation_count = sum(1 for c in content if c in '.,;:!?-()[]{}')
            token_count = len(words) + punctuation_count
            
            # Score inversely proportional to token count (fewer is better)
            efficiency_score = max(0.1, min(1.0, 200 / token_count))
        else:
            # For chains, sum the token counts of all prompts
            total_tokens = 0
            for node in prompt.nodes:
                content = node["prompt"].content
                words = content.split()
                punctuation_count = sum(1 for c in content if c in '.,;:!?-()[]{}')
                total_tokens += len(words) + punctuation_count
                
            efficiency_score = max(0.1, min(1.0, 500 / total_tokens))
            
        return efficiency_score
        
    def _evaluate_success_rate(self, prompt: Union[Prompt, PromptChain]) -> float:
        """Evaluate the success rate of a prompt (placeholder implementation)"""
        # In a real implementation, this would track historical success rates
        # For now, return a placeholder value or use metadata if available
        
        if isinstance(prompt, Prompt):
            return prompt.metadata.performance_metrics.get("success_rate", 0.75)
        else:
            # For chains, use the chain's success rate or a placeholder
            return prompt.metadata.performance_metrics.get("success_rate", 0.7)

class AdvancedPromptManager:
    """Manages loading, editing, and saving prompts with advanced features"""
    def __init__(self, prompts_dir="./prompts", chains_dir="./prompt_chains"):
        self.prompts_dir = prompts_dir
        self.chains_dir = chains_dir
        self.current_prompt = None
        self.current_chain = None
        self.prompt_history = []
        self.evaluator = PromptEvaluator()
        
        # Create directories if they don't exist
        os.makedirs(prompts_dir, exist_ok=True)
        os.makedirs(chains_dir, exist_ok=True)
        
    def load_prompt(self, prompt_name: str) -> Optional[Prompt]:
        """Load a prompt from file"""
        # Try both .txt and .yaml formats
        prompt_path_txt = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        prompt_path_yaml = os.path.join(self.prompts_dir, f"{prompt_name}.yaml")
        
        if os.path.exists(prompt_path_yaml):
            # Load YAML format (with metadata)
            with open(prompt_path_yaml, 'r') as f:
                prompt_data = yaml.safe_load(f)
                
            prompt = Prompt(
                name=prompt_data.get("name", prompt_name),
                content=prompt_data.get("content", "")
            )
            
            if "metadata" in prompt_data:
                prompt.metadata = PromptMetadata(**prompt_data["metadata"])
                
            if "variables" in prompt_data:
                prompt.variables = prompt_data["variables"]
                
            self.current_prompt = prompt
            return prompt
            
        elif os.path.exists(prompt_path_txt):
            # Simple text format (content only)
            with open(prompt_path_txt, 'r') as f:
                content = f.read()
                
            prompt = Prompt(name=prompt_name, content=content)
            self.current_prompt = prompt
            return prompt
            
        return None
        
    def save_prompt(self, prompt: Prompt, format: str = "yaml") -> bool:
        """Save a prompt to file"""
        if format == "yaml":
            prompt_path = os.path.join(self.prompts_dir, f"{prompt.name}.yaml")
            
            # Update the modified time
            prompt.metadata.modified_at = datetime.now().isoformat()
            
            prompt_data = {
                "name": prompt.name,
                "content": prompt.content,
                "metadata": asdict(prompt.metadata),
                "variables": prompt.variables
            }
            
            with open(prompt_path, 'w') as f:
                yaml.dump(prompt_data, f, default_flow_style=False)
        else:
            # Simple text format (content only)
            prompt_path = os.path.join(self.prompts_dir, f"{prompt.name}.txt")
            with open(prompt_path, 'w') as f:
                f.write(prompt.content)
                
        self.current_prompt = prompt
        return True
        
    def load_chain(self, chain_name: str) -> Optional[PromptChain]:
        """Load a prompt chain from file"""
        chain_path = os.path.join(self.chains_dir, f"{chain_name}.yaml")
        
        if os.path.exists(chain_path):
            with open(chain_path, 'r') as f:
                chain_data = yaml.safe_load(f)
                
            chain = PromptChain.from_dict(chain_data)
            self.current_chain = chain
            return chain
            
        return None
        
    def save_chain(self, chain: PromptChain) -> bool:
        """Save a prompt chain to file"""
        chain_path = os.path.join(self.chains_dir, f"{chain.name}.yaml")
        
        # Update the modified time
        chain.metadata.modified_at = datetime.now().isoformat()
        
        chain_data = chain.to_dict()
        
        with open(chain_path, 'w') as f:
            yaml.dump(chain_data, f, default_flow_style=False)
            
        self.current_chain = chain
        return True
        
    def list_prompts(self, prompt_type: str = None) -> List[str]:
        """List all available prompts, optionally filtered by type"""
        if not os.path.exists(self.prompts_dir):
            return []
            
        # Get all prompt files (both .txt and .yaml)
        prompt_files = glob(os.path.join(self.prompts_dir, "*.txt"))
        prompt_files.extend(glob(os.path.join(self.prompts_dir, "*.yaml")))
        
        # Extract names without extensions
        prompt_names = [os.path.splitext(os.path.basename(f))[0] for f in prompt_files]
        
        # Remove duplicates (prompts that exist in both formats)
        prompt_names = list(set(prompt_names))
        
        # Filter by type if specified
        if prompt_type:
            filtered_names = []
            for name in prompt_names:
                prompt = self.load_prompt(name)
                if prompt and prompt.metadata.prompt_type == prompt_type:
                    filtered_names.append(name)
            return filtered_names
        
        return prompt_names
        
    def list_chains(self) -> List[str]:
        """List all available prompt chains"""
        if not os.path.exists(self.chains_dir):
            return []
            
        chain_files = glob(os.path.join(self.chains_dir, "*.yaml"))
        chain_names = [os.path.splitext(os.path.basename(f))[0] for f in chain_files]
        
        return chain_names
        
    def create_prompt(self, name: str, content: str, prompt_type: str = "standard", 
                     variables: Dict[str, str] = None, description: str = "", 
                     tags: List[str] = None) -> Prompt:
        """Create a new prompt with metadata"""
        metadata = PromptMetadata(
            prompt_type=prompt_type,
            description=description,
            tags=tags or []
        )
        
        prompt = Prompt(
            name=name,
            content=content,
            metadata=metadata,
            variables=variables or {}
        )
        
        return prompt
        
    def create_chain(self, name: str, description: str = "") -> PromptChain:
        """Create a new prompt chain"""
        return PromptChain(name, description)
        
    def detect_prompt_type(self, content: str) -> str:
        """Auto-detect the prompt type based on content structure"""
        # Check for XML tags to determine prompt type
        if "<holographic_prompt>" in content:
            return "holographic"
        elif "<temporal_prompt>" in content:
            return "temporal"
        elif "<multi_agent_prompt>" in content:
            return "multi-agent"
        elif "<self_calibrating_prompt>" in content:
            return "self-calibrating"
        elif "<knowledge_synthesis_prompt>" in content:
            return "knowledge-synthesis"
        elif "<counterfactual_prompt>" in content:
            return "counterfactual"
        elif "<emergent_prompt>" in content:
            return "emergent"
        elif "<bias_aware_prompt>" in content:
            return "bias-aware"
            
        return "standard"
        
    def extract_variables(self, content: str) -> Dict[str, str]:
        """Extract template variables from prompt content"""
        import re
        
        # Look for variables in the format {{variable_name}}
        variable_pattern = r'\{\{([a-zA-Z0-9_]+)\}\}'
        matches = re.findall(variable_pattern, content)
        
        # Create a dictionary of variable names with empty values
        variables = {var_name: "" for var_name in matches}
        
        return variables
        
    def substitute_variables(self, content: str, variables: Dict[str, str]) -> str:
        """Substitute variables in a prompt template"""
        result = content
        
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            result = result.replace(placeholder, var_value)
            
        return result
        
    def create_composition(self, prompts: List[Prompt], composition_type: str = "sequence") -> Prompt:
        """Create a composite prompt by combining multiple prompts"""
        if not prompts:
            raise ValueError("No prompts provided for composition")
            
        # Handle different composition types
        if composition_type == "sequence":
            # Sequential composition (one after another)
            combined_content = "\n\n".join(p.content for p in prompts)
            name = f"Composition_{'-'.join(p.name for p in prompts)}"
            
        elif composition_type == "nested":
            # Nested composition (innermost first)
            content = prompts[-1].content
            for prompt in reversed(prompts[:-1]):
                content = f"{prompt.content}\n\nFor the above task, use the following:\n\n{content}"
            combined_content = content
            name = f"Nested_{'-'.join(p.name for p in prompts)}"
            
        elif composition_type == "parallel":
            # Parallel composition (side by side with roles)
            combined_content = "<parallel_prompt>\n"
            for i, prompt in enumerate(prompts):
                role = f"role_{i+1}"
                combined_content += f"<component role=\"{role}\">\n{prompt.content}\n</component>\n"
            combined_content += "</parallel_prompt>"
            name = f"Parallel_{'-'.join(p.name for p in prompts)}"
            
        else:
            raise ValueError(f"Unknown composition type: {composition_type}")
            
        # Create new prompt with combined content
        new_prompt = self.create_prompt(
            name=name,
            content=combined_content,
            prompt_type=f"composition-{composition_type}",
            description=f"Composition of {len(prompts)} prompts using {composition_type} method"
        )
        
        # Track dependencies
        new_prompt.metadata.dependencies = [p.name for p in prompts]
        
        return new_prompt
        
    def evaluate_prompt(self, prompt: Union[Prompt, PromptChain], metrics: List[str] = None) -> Dict[str, float]:
        """Evaluate a prompt using the prompt evaluator"""
        results = self.evaluator.evaluate(prompt, metrics)
        
        # Update the prompt's performance metrics
        if isinstance(prompt, Prompt):
            prompt.metadata.performance_metrics.update(results)
        else:
            prompt.metadata.performance_metrics.update(results)
            
        return results
        
    def optimize_prompt(self, prompt: Prompt, target_metrics: Dict[str, float], 
                       max_iterations: int = 5) -> Prompt:
        """Optimize a prompt to improve specific metrics"""
        if not has_llm:
            print("LLM functionality not available. Cannot optimize prompts.")
            return prompt
            
        # Create a copy of the original prompt
        optimized = Prompt(
            name=f"{prompt.name}_optimized",
            content=prompt.content,
            metadata=PromptMetadata(**asdict(prompt.metadata)),
            variables=prompt.variables.copy()
        )
        
        # Perform multiple iterations of optimization
        for i in range(max_iterations):
            # Evaluate current state
            current_metrics = self.evaluate_prompt(optimized)
            
            # Check if we've reached the targets
            targets_achieved = all(
                current_metrics.get(metric, 0) >= value 
                for metric, value in target_metrics.items()
            )
            
            if targets_achieved:
                break
                
            # Generate optimization instructions
            instructions = "Optimize this prompt to improve the following metrics:\n"
            for metric, target in target_metrics.items():
                current = current_metrics.get(metric, 0)
                if current < target:
                    instructions += f"- {metric}: currently {current:.2f}, target is {target:.2f}\n"
                    
            # Use LLM to optimize the prompt
            try:
                response = openai.Completion.create(
                    model="gpt-4",
                    prompt=f"{instructions}\n\nOriginal prompt:\n{optimized.content}\n\nOptimized version:",
                    max_tokens=1500,
                    temperature=0.7
                )
                
                optimized_content = response.choices[0].text.strip()
                if optimized_content:
                    optimized.content = optimized_content
                    optimized.metadata.version += 1
                    optimized.metadata.modified_at = datetime.now().isoformat()
            except Exception as e:
                print(f"Error during optimization: {e}")
                break
                
        return optimized
        
    def generate_prompt(self, prompt_type: str, topic: str, complexity: str = "medium") -> Optional[Prompt]:
        """Generate a new prompt of the specified type using an LLM"""
        if not has_llm:
            print("LLM functionality not available. Cannot generate prompts.")
            return None
            
        # Create instructions based on prompt type
        instructions = f"Generate a {prompt_type} prompt about {topic} with {complexity} complexity.\n\n"
        
        if prompt_type == "holographic":
            instructions += "Structure the prompt with <holographic_prompt> tags and multiple detail levels."
        elif prompt_type == "temporal":
            instructions += "Structure the prompt with <temporal_prompt> tags and multiple timeframes."
        elif prompt_type == "multi-agent":
            instructions += "Structure the prompt with <multi_agent_prompt> tags and multiple agent roles."
        elif prompt_type == "self-calibrating":
            instructions += "Structure the prompt with <self_calibrating_prompt> tags, including confidence requirements."
        elif prompt_type == "knowledge-synthesis":
            instructions += "Structure the prompt with <knowledge_synthesis_prompt> tags, including multiple domains."
            
        try:
            response = openai.Completion.create(
                model="gpt-4",
                prompt=instructions,
                max_tokens=1000,
                temperature=0.7
            )
            
            content = response.choices[0].text.strip()
            
            if content:
                # Generate a suitable name
                name = f"{prompt_type}_{topic.replace(' ', '_').lower()}"
                
                # Create the prompt
                prompt = self.create_prompt(
                    name=name,
                    content=content,
                    prompt_type=prompt_type,
                    description=f"Generated {prompt_type} prompt about {topic}"
                )
                
                return prompt
        except Exception as e:
            print(f"Error generating prompt: {e}")
            
        return None
        
    def export_all(self, output_dir: str) -> Tuple[int, int]:
        """Export all prompts and chains to a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export prompts
        prompts_output_dir = os.path.join(output_dir, "prompts")
        os.makedirs(prompts_output_dir, exist_ok=True)
        
        prompt_count = 0
        for prompt_name in self.list_prompts():
            prompt = self.load_prompt(prompt_name)
            if prompt:
                self.save_prompt(prompt, format="yaml")
                prompt_path = os.path.join(prompts_output_dir, f"{prompt.name}.yaml")
                
                prompt_data = {
                    "name": prompt.name,
                    "content": prompt.content,
                    "metadata": asdict(prompt.metadata),
                    "variables": prompt.variables
                }
                
                with open(prompt_path, 'w') as f:
                    yaml.dump(prompt_data, f, default_flow_style=False)
                    
                prompt_count += 1
                
        # Export chains
        chains_output_dir = os.path.join(output_dir, "chains")
        os.makedirs(chains_output_dir, exist_ok=True)
        
        chain_count = 0
        for chain_name in self.list_chains():
            chain = self.load_chain(chain_name)
            if chain:
                chain_path = os.path.join(chains_output_dir, f"{chain.name}.yaml")
                
                with open(chain_path, 'w') as f:
                    yaml.dump(chain.to_dict(), f, default_flow_style=False)
                    
                chain_count += 1
                
        return prompt_count, chain_count
        
    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for semantically similar prompts using embeddings"""
        if not has_embedding:
            print("Embedding functionality not available. Using text search instead.")
            return self.search_text(query)
            
        # Get embeddings for the query
        query_embedding = compute_embedding(query)
        
        results = []
        for prompt_name in self.list_prompts():
            prompt = self.load_prompt(prompt_name)
            if not prompt:
                continue
                
            # Check if the prompt has an embedding, otherwise compute it
            if "main" not in prompt.embeddings:
                prompt.embeddings["main"] = compute_embedding(prompt.content)
                self.save_prompt(prompt)
                
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, prompt.embeddings["main"])
            results.append((prompt_name, similarity))
            
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
        
    def search_text(self, query: str) -> List[Tuple[str, int]]:
        """Simple text-based search in prompts"""
        results = []
        
        for prompt_name in self.list_prompts():
            prompt = self.load_prompt(prompt_name)
            if not prompt:
                continue
                
            content = prompt.content.lower()
            query_lower = query.lower()
            
            # Count occurrences of the query
            count = content.count(query_lower)
            
            if count > 0:
                results.append((prompt_name, count))
                
        # Sort by occurrence count (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)

class AdvancedPromptCLI:
    """Command-line interface for the advanced prompt management system"""
    def __init__(self, prompts_dir="./prompts", chains_dir="./prompt_chains", kb_connection=None):
        self.prompt_manager = AdvancedPromptManager(prompts_dir, chains_dir)
        if has_embedding and kb_connection:
            self.embedding_system = PolymorphicEmbedding(kb_connection)
        else:
            self.embedding_system = None
        self.conversation_id = None
        
    def do_prompt(self, args):
        """
        Manage prompts: prompt [list|load|save|edit|new|analyze|convert|search|compose|evaluate|optimize|generate]
        
        Commands:
          list [type]                     - List all available prompts, optionally filtered by type
          load <name>                     - Load a prompt
          save <name> [format]            - Save current prompt (format: txt or yaml)
          edit <name>                     - Edit a prompt
          new <name> [type]               - Create a new prompt
          analyze <name>                  - Analyze prompt structure and suggest improvements
          convert <name> <type>           - Convert a prompt to a specialized format
          search <query>                  - Search for prompts
          compose <prompt1,prompt2> [type] - Compose multiple prompts
          evaluate <name> [metrics]       - Evaluate a prompt using specified metrics
          optimize <name> <target_metric> - Optimize a prompt for a specific metric
          generate <type> <topic>         - Generate a new prompt using an LLM
          export <directory>              - Export all prompts to a directory
        """
        if not args:
            print("Usage: prompt [list|load|save|edit|new|analyze|convert|search|compose|evaluate|optimize|generate|export]")
            return
            
        command = args[0].lower()
        
        if command == "list":
            prompt_type = args[1] if len(args) > 1 else None
            prompts = self.prompt_manager.list_prompts(prompt_type)
            
            if prompts:
                print(f"\nAvailable prompts{' of type '+prompt_type if prompt_type else ''}:")
                for i, name in enumerate(prompts):
                    prompt = self.prompt_manager.load_prompt(name)
                    if prompt:
                        type_str = prompt.metadata.prompt_type
                        desc = prompt.metadata.description[:30] + "..." if len(prompt.metadata.description) > 30 else prompt.metadata.description
                        print(f"{i+1}. {name} ({type_str}) - {desc}")
                    else:
                        print(f"{i+1}. {name}")
            else:
                print("No prompts found.")
                
        elif command == "load":
            if len(args) < 2:
                print("Usage: prompt load <name>")
                return
                
            prompt_name = args[1]
            prompt = self.prompt_manager.load_prompt(prompt_name)
            
            if prompt:
                print(f"Prompt '{prompt_name}' loaded:")
                print("\n---\n")
                print(prompt.content)
                print("\n---\n")
                
                # Show metadata if available
                if prompt.metadata:
                    print("Metadata:")
                    print(f"  Type: {prompt.metadata.prompt_type}")
                    if prompt.metadata.description:
                        print(f"  Description: {prompt.metadata.description}")
                    if prompt.metadata.tags:
                        print(f"  Tags: {', '.join(prompt.metadata.tags)}")
                    print(f"  Created: {prompt.metadata.created_at.split('T')[0]}")
                    print(f"  Modified: {prompt.metadata.modified_at.split('T')[0]}")
                    print(f"  Version: {prompt.metadata.version}")
                    
                # Show variables if available
                if prompt.variables:
                    print("\nVariables:")
                    for var_name, var_value in prompt.variables.items():
                        print(f"  {var_name}: {var_value or '(empty)'}")
            else:
                print(f"Prompt '{prompt_name}' not found.")
                
        elif command == "save":
            if len(args) < 2:
                print("Usage: prompt save <name> [format]")
                return
                
            prompt_name = args[1]
            format = args[2] if len(args) > 2 else "yaml"
            
            if not self.prompt_manager.current_prompt:
                print("No prompt currently loaded or edited.")
                return
                
            # Update the name if it's different
            self.prompt_manager.current_prompt.name = prompt_name
            
            success = self.prompt_manager.save_prompt(self.prompt_manager.current_prompt, format)
            if success:
                print(f"Prompt saved as '{prompt_name}' in {format} format.")
            else:
                print("Error saving prompt.")
                
        elif command == "edit":
            if len(args) < 2:
                print("Usage: prompt edit <name>")
                return
                
            prompt_name = args[1]
            prompt = self.prompt_manager.load_prompt(prompt_name)
            
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            print(f"Editing prompt '{prompt_name}'.")
            print("Enter your changes (type '###' on a line by itself to end):")
            
            lines = []
            print("\n--- Current content ---")
            print(prompt.content)
            print("--- Enter changes below ---")
            
            while True:
                try:
                    line = input()
                    if line.strip() == '###':
                        break
                    lines.append(line)
                except EOFError:
                    break
                    
            if lines:
                new_content = '\n'.join(lines)
                prompt.content = new_content
                prompt.metadata.version += 1
                prompt.metadata.modified_at = datetime.now().isoformat()
                
                # Auto-detect variables
                prompt.variables = self.prompt_manager.extract_variables(new_content)
                
                # Auto-detect prompt type if not already set or if standard
                if prompt.metadata.prompt_type == "standard":
                    detected_type = self.prompt_manager.detect_prompt_type(new_content)
                    if detected_type != "standard":
                        prompt.metadata.prompt_type = detected_type
                        print(f"Detected prompt type: {detected_type}")
                
                self.prompt_manager.save_prompt(prompt)
                print(f"Prompt '{prompt_name}' updated (version {prompt.metadata.version}).")
                
                if prompt.variables:
                    print("\nDetected variables:")
                    for var_name in prompt.variables:
                        print(f"  - {var_name}")
            else:
                print("No changes made.")
                
        elif command == "new":
            if len(args) < 2:
                print("Usage: prompt new <name> [type]")
                return
                
            prompt_name = args[1]
            prompt_type = args[2] if len(args) > 2 else "standard"
            
            print(f"Creating new {prompt_type} prompt '{prompt_name}'.")
            print("Enter prompt content (type '###' on a line by itself to end):")
            
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == '###':
                        break
                    lines.append(line)
                except EOFError:
                    break
                    
            if lines:
                content = '\n'.join(lines)
                
                # Extract variables
                variables = self.prompt_manager.extract_variables(content)
                
                # Create prompt
                prompt = self.prompt_manager.create_prompt(
                    name=prompt_name,
                    content=content,
                    prompt_type=prompt_type,
                    variables=variables
                )
                
                # Save prompt
                self.prompt_manager.save_prompt(prompt)
                print(f"Prompt '{prompt_name}' created.")
                
                if variables:
                    print("\nDetected variables:")
                    for var_name in variables:
                        print(f"  - {var_name}")
            else:
                print("No content provided. Prompt not created.")
                
        elif command == "analyze":
            if len(args) < 2:
                print("Usage: prompt analyze <name>")
                return
                
            prompt_name = args[1]
            prompt = self.prompt_manager.load_prompt(prompt_name)
            
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            print(f"Analyzing prompt '{prompt_name}'...")
            
            # Perform basic analysis
            word_count = len(prompt.content.split())
            line_count = prompt.content.count('\n') + 1
            char_count = len(prompt.content)
            
            print("\n=== Prompt Analysis ===\n")
            print(f"Type: {prompt.metadata.prompt_type}")
            print(f"Size: {word_count} words, {line_count} lines, {char_count} characters")
            
            # Extract structure based on prompt type
            if prompt.metadata.prompt_type == "holographic":
                import re
                levels = re.findall(r'<level depth="(\d+)">', prompt.content)
                print(f"Structure: Holographic prompt with {len(levels)} detail levels")
                
            elif prompt.metadata.prompt_type == "temporal":
                import re
                timeframes = re.findall(r'<timeframe period="([^"]+)"', prompt.content)
                print(f"Structure: Temporal prompt with {len(timeframes)} timeframes ({', '.join(timeframes)})")
                
            elif prompt.metadata.prompt_type == "multi-agent":
                import re
                agents = re.findall(r'<agent role="([^"]+)">', prompt.content)
                print(f"Structure: Multi-agent prompt with {len(agents)} roles ({', '.join(agents)})")
                has_integration = "<integration>" in prompt.content
                print(f"  Integration section: {'Yes' if has_integration else 'No'}")
                
            # Evaluate prompt metrics
            print("\nMetrics:")
            metrics = self.prompt_manager.evaluate_prompt(prompt)
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")
                
            # Variables
            if prompt.variables:
                print("\nVariables:")
                for var_name in prompt.variables:
                    print(f"  - {var_name}")
            
        elif command == "convert":
            if len(args) < 3:
                print("Usage: prompt convert <name> <type>")
                print("Available types: holographic, temporal, multi-agent, self-calibrating, knowledge-synthesis")
                return
                
            prompt_name = args[1]
            prompt_type = args[2].lower()
            
            prompt = self.prompt_manager.load_prompt(prompt_name)
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            # Valid prompt types
            valid_types = ["holographic", "temporal", "multi-agent", "self-calibrating", "knowledge-synthesis"]
            
            if prompt_type not in valid_types:
                print(f"Invalid prompt type: {prompt_type}")
                print(f"Available types: {', '.join(valid_types)}")
                return
                
            print(f"Converting prompt '{prompt_name}' to {prompt_type} format...")
            
            # Placeholder for conversion
            # In a real implementation, this would use an LLM for conversion
            new_prompt_name = f"{prompt_name}_{prompt_type}"
            
            # Add some basic structure based on the prompt type
            if prompt_type == "holographic":
                converted_content = f"""<holographic_prompt>
  <level depth="1">
    {prompt.content[:100]}...
  </level>
  <level depth="2">
    {prompt.content[:200]}...
  </level>
  <level depth="3">
    {prompt.content}
  </level>
</holographic_prompt>"""
            elif prompt_type == "temporal":
                converted_content = f"""<temporal_prompt>
  <timeframe period="immediate">
    {prompt.content[:150]}...
  </timeframe>
  <timeframe period="recent" range="past-30-days">
    Considering recent developments and information from the past month...
  </timeframe>
  <timeframe period="historical" range="all-time">
    Taking into account historical context and long-term trends...
  </timeframe>
  <timeframe period="future" range="next-6-months">
    Looking ahead to potential implications and developments...
  </timeframe>
</temporal_prompt>"""
            elif prompt_type == "multi-agent":
                converted_content = f"""<multi_agent_prompt>
  <agent role="analyst">
    {prompt.content[:150]}...
  </agent>
  <agent role="critic">
    Evaluate the strengths and weaknesses of approaches to this task...
  </agent>
  <agent role="creative">
    Consider innovative and unexpected approaches to this challenge...
  </agent>
  <agent role="implementer">
    Develop practical steps to execute the solution effectively...
  </agent>
  <integration>
    Synthesize insights from all perspectives to create a comprehensive response...
  </integration>
</multi_agent_prompt>"""
            elif prompt_type == "self-calibrating":
                converted_content = f"""<self_calibrating_prompt>
  <instruction>
    {prompt.content}
  </instruction>
  <confidence_requirements>
    <requirement>Estimate your confidence for each part of your answer on a scale of 1-10</requirement>
    <requirement>For any confidence below 7, provide alternative possibilities</requirement>
    <requirement>For any confidence below 5, explicitly state what information would improve confidence</requirement>
  </confidence_requirements>
  <verification_steps>
    <step>Check factual claims against reliable sources</step>
    <step>Verify logical consistency of all arguments</step>
    <step>Ensure comprehensive coverage of all aspects of the question</step>
  </verification_steps>
  <output_format>
    <answer>Main response with confidence markers</answer>
    <alternatives>Alternative possibilities for low-confidence items</alternatives>
    <information_needs>Additional information that would improve confidence</information_needs>
  </output_format>
</self_calibrating_prompt>"""
            elif prompt_type == "knowledge-synthesis":
                converted_content = f"""<knowledge_synthesis_prompt>
  <domain name="primary">
    {prompt.content[:150]}...
  </domain>
  <domain name="secondary">
    Related concepts from adjacent fields...
  </domain>
  <domain name="tertiary">
    Broader contextual knowledge...
  </domain>
  <connection_points>
    <connection>
      <from>Primary concept from main domain</from>
      <to>Related concept in secondary domain</to>
      <relationship>Analogical mapping</relationship>
    </connection>
    <connection>
      <from>Another primary concept</from>
      <to>Related concept in tertiary domain</to>
      <relationship>Causal relationship</relationship>
    </connection>
  </connection_points>
  <synthesis_goal>
    Create an integrated understanding that leverages insights from all domains...
  </synthesis_goal>
</knowledge_synthesis_prompt>"""
            
            # Create new prompt with converted content
            new_prompt = self.prompt_manager.create_prompt(
                name=new_prompt_name,
                content=converted_content,
                prompt_type=prompt_type,
                description=f"Converted from {prompt_name}"
            )
            
            # Add dependency
            new_prompt.metadata.dependencies.append(prompt_name)
            
            # Save the converted prompt
            self.prompt_manager.save_prompt(new_prompt)
            
            print(f"\nConverted prompt saved as '{new_prompt_name}'.")
            print("\nPreview of converted prompt:")
            print("\n---\n")
            preview = converted_content[:500] + "..." if len(converted_content) > 500 else converted_content
            print(preview)
            print("\n---\n")
            
        elif command == "search":
            if len(args) < 2:
                print("Usage: prompt search <query>")
                return
                
            query = " ".join(args[1:])
            
            # Try semantic search if embeddings are available
            if has_embedding:
                print(f"Performing semantic search for: {query}")
                results = self.prompt_manager.search_semantic(query)
                
                if results:
                    print(f"\nFound {len(results)} semantically similar prompts:")
                    for i, (name, similarity) in enumerate(results):
                        print(f"{i+1}. {name} (similarity: {similarity:.2f})")
                        
                        # Show a snippet
                        prompt = self.prompt_manager.load_prompt(name)
                        if prompt:
                            content = prompt.content
                            snippet = content[:150] + "..." if len(content) > 150 else content
                            print(f"   Snippet: {snippet}")
                else:
                    print("No similar prompts found.")
            else:
                # Fall back to text search
                print(f"Performing text search for: {query}")
                results = self.prompt_manager.search_text(query)
                
                if results:
                    print(f"\nFound {len(results)} prompts containing '{query}':")
                    for i, (name, count) in enumerate(results):
                        print(f"{i+1}. {name} ({count} occurrences)")
                else:
                    print(f"No prompts found containing '{query}'.")
                    
        elif command == "compose":
            if len(args) < 2:
                print("Usage: prompt compose <prompt1,prompt2,...> [type]")
                print("Composition types: sequence, nested, parallel")
                return
                
            prompt_names = args[1].split(',')
            composition_type = args[2] if len(args) > 2 else "sequence"
            
            if composition_type not in ["sequence", "nested", "parallel"]:
                print(f"Invalid composition type: {composition_type}")
                print("Available types: sequence, nested, parallel")
                return
                
            prompts = []
            for name in prompt_names:
                prompt = self.prompt_manager.load_prompt(name)
                if prompt:
                    prompts.append(prompt)
                else:
                    print(f"Prompt '{name}' not found. Aborting composition.")
                    return
                    
            if len(prompts) < 2:
                print("At least two prompts are required for composition.")
                return
                
            # Create the composition
            composite = self.prompt_manager.create_composition(prompts, composition_type)
            
            # Save the composition
            self.prompt_manager.save_prompt(composite)
            
            print(f"Created composite prompt '{composite.name}' using {composition_type} composition.")
            print("\nPreview:")
            print("\n---\n")
            preview = composite.content[:500] + "..." if len(composite.content) > 500 else composite.content
            print(preview)
            print("\n---\n")
            
        elif command == "evaluate":
            if len(args) < 2:
                print("Usage: prompt evaluate <name> [metrics]")
                return
                
            prompt_name = args[1]
            metrics = args[2].split(',') if len(args) > 2 else None
            
            prompt = self.prompt_manager.load_prompt(prompt_name)
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            print(f"Evaluating prompt '{prompt_name}'...")
            results = self.prompt_manager.evaluate_prompt(prompt, metrics)
            
            print("\nEvaluation results:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.2f}")
                
            # Update and save the prompt with new metrics
            self.prompt_manager.save_prompt(prompt)
            
        elif command == "optimize":
            if len(args) < 3:
                print("Usage: prompt optimize <name> <target_metric:target_value,...>")
                print("Example: prompt optimize my_prompt clarity:0.8,specificity:0.7")
                return
                
            prompt_name = args[1]
            targets_str = args[2]
            
            prompt = self.prompt_manager.load_prompt(prompt_name)
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            # Parse target metrics
            try:
                target_metrics = {}
                for target in targets_str.split(','):
                    metric, value = target.split(':')
                    target_metrics[metric] = float(value)
            except ValueError:
                print("Invalid target format. Use metric:value,metric:value")
                return
                
            print(f"Optimizing prompt '{prompt_name}' for {len(target_metrics)} metrics...")
            
            if not has_llm:
                print("LLM functionality not available. Cannot optimize prompts.")
                return
                
            optimized = self.prompt_manager.optimize_prompt(prompt, target_metrics)
            
            # Save the optimized prompt
            self.prompt_manager.save_prompt(optimized)
            
            print(f"Created optimized prompt '{optimized.name}' (version {optimized.metadata.version}).")
            print("\nNew metrics:")
            for metric, value in optimized.metadata.performance_metrics.items():
                print(f"  {metric}: {value:.2f}")
                
        elif command == "generate":
            if len(args) < 3:
                print("Usage: prompt generate <type> <topic> [complexity]")
                return
                
            prompt_type = args[1]
            topic = args[2]
            complexity = args[3] if len(args) > 3 else "medium"
            
            if not has_llm:
                print("LLM functionality not available. Cannot generate prompts.")
                return
                
            print(f"Generating {prompt_type} prompt about {topic} with {complexity} complexity...")
            
            generated = self.prompt_manager.generate_prompt(prompt_type, topic, complexity)
            
            if generated:
                # Save the generated prompt
                self.prompt_manager.save_prompt(generated)
                
                print(f"Created prompt '{generated.name}'.")
                print("\nPreview:")
                print("\n---\n")
                preview = generated.content[:500] + "..." if len(generated.content) > 500 else generated.content
                print(preview)
                print("\n---\n")
            else:
                print("Failed to generate prompt.")
                
        elif command == "export":
            if len(args) < 2:
                print("Usage: prompt export <directory>")
                return
                
            output_dir = args[1]
            
            prompt_count, chain_count = self.prompt_manager.export_all(output_dir)
            
            print(f"Exported {prompt_count} prompts and {chain_count} chains to {output_dir}")
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: prompt [list|load|save|edit|new|analyze|convert|search|compose|evaluate|optimize|generate|export]")
    
    def do_chain(self, args):
        """
        Manage prompt chains: chain [list|create|add|connect|visualize|execute|save|load]
        
        Commands:
          list                        - List all available chains
          create <name> [description] - Create a new chain
          add <chain> <prompt>        - Add a prompt to a chain
          connect <chain> <from> <to> - Connect prompts in a chain
          visualize <chain> [output]  - Visualize a chain
          execute <chain>             - Execute a chain
          save <chain>                - Save a chain
          load <chain>                - Load a chain
        """
        if not args:
            print("Usage: chain [list|create|add|connect|visualize|execute|save|load]")
            return
            
        command = args[0].lower()
        
        if command == "list":
            chains = self.prompt_manager.list_chains()
            
            if chains:
                print("\nAvailable prompt chains:")
                for i, name in enumerate(chains):
                    print(f"{i+1}. {name}")
            else:
                print("No prompt chains found.")
                
        elif command == "create":
            if len(args) < 2:
                print("Usage: chain create <name> [description]")
                return
                
            chain_name = args[1]
            description = " ".join(args[2:]) if len(args) > 2 else ""
            
            chain = self.prompt_manager.create_chain(chain_name, description)
            self.prompt_manager.current_chain = chain
            
            print(f"Created new prompt chain '{chain_name}'.")
            
        elif command == "add":
            if len(args) < 3:
                print("Usage: chain add <chain> <prompt>")
                return
                
            chain_name = args[1]
            prompt_name = args[2]
            
            chain = self.prompt_manager.load_chain(chain_name)
            if not chain:
                print(f"Chain '{chain_name}' not found.")
                return
                
            prompt = self.prompt_manager.load_prompt(prompt_name)
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            node_id = chain.add_node(prompt)
            self.prompt_manager.current_chain = chain
            
            print(f"Added prompt '{prompt_name}' to chain '{chain_name}' as node {node_id}.")
            
        elif command == "connect":
            if len(args) < 4:
                print("Usage: chain connect <chain> <from_id> <to_id> [condition]")
                return
                
            chain_name = args[1]
            try:
                from_id = int(args[2])
                to_id = int(args[3])
            except ValueError:
                print("Node IDs must be integers.")
                return
                
            condition = " ".join(args[4:]) if len(args) > 4 else None
            
            chain = self.prompt_manager.load_chain(chain_name)
            if not chain:
                print(f"Chain '{chain_name}' not found.")
                return
                
            try:
                chain.add_edge(from_id, to_id, condition)
                self.prompt_manager.current_chain = chain
                
                print(f"Connected node {from_id} to node {to_id} in chain '{chain_name}'.")
                if condition:
                    print(f"Condition: {condition}")
            except ValueError as e:
                print(f"Error connecting nodes: {e}")
                
        elif command == "visualize":
            if len(args) < 2:
                print("Usage: chain visualize <chain> [output_file]")
                return
                
            chain_name = args[1]
            output_file = args[2] if len(args) > 2 else None
            
            chain = self.prompt_manager.load_chain(chain_name)
            if not chain:
                print(f"Chain '{chain_name}' not found.")
                return
                
            if not has_visualization:
                print("Visualization requires matplotlib. Please install it to use this feature.")
                return
                
            print(f"Visualizing chain '{chain_name}'...")
            chain.visualize(output_file)
            
            if output_file:
                print(f"Visualization saved to {output_file}")
                
        elif command == "save":
            if len(args) < 2:
                print("Usage: chain save <chain>")
                return
                
            chain_name = args[1]
            
            if self.prompt_manager.current_chain and self.prompt_manager.current_chain.name == chain_name:
                chain = self.prompt_manager.current_chain
            else:
                chain = self.prompt_manager.load_chain(chain_name)
                
            if not chain:
                print(f"Chain '{chain_name}' not found.")
                return
                
            if not chain.validate():
                print("Chain validation failed. Please check that the chain is properly connected.")
                return
                
            success = self.prompt_manager.save_chain(chain)
            if success:
                print(f"Chain '{chain_name}' saved.")
            else:
                print("Error saving chain.")
                
        elif command == "load":
            if len(args) < 2:
                print("Usage: chain load <chain>")
                return
                
            chain_name = args[1]
            chain = self.prompt_manager.load_chain(chain_name)
            
            if chain:
                print(f"Chain '{chain_name}' loaded.")
                print(f"Description: {chain.description}")
                print(f"Nodes: {len(chain.nodes)}")
                print(f"Edges: {len(chain.edges)}")
            else:
                print(f"Chain '{chain_name}' not found.")
                
        else:
            print(f"Unknown command: {command}")
            print("Usage: chain [list|create|add|connect|visualize|execute|save|load]")
    
    def do_variable(self, args):
        """
        Manage prompt variables: variable [list|set|clear|apply]
        
        Commands:
          list <prompt>              - List variables in a prompt
          set <prompt> <name> <value> - Set a variable value
          clear <prompt> [name]      - Clear a variable or all variables
          apply <prompt>             - Apply variables to a prompt template
        """
        if not args:
            print("Usage: variable [list|set|clear|apply]")
            return
            
        command = args[0].lower()
        
        if command == "list":
            if len(args) < 2:
                print("Usage: variable list <prompt>")
                return
                
            prompt_name = args[1]
            prompt = self.prompt_manager.load_prompt(prompt_name)
            
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            if prompt.variables:
                print(f"\nVariables in prompt '{prompt_name}':")
                for name, value in prompt.variables.items():
                    print(f"  {name}: {value or '(empty)'}")
            else:
                print(f"No variables found in prompt '{prompt_name}'.")
                
        elif command == "set":
            if len(args) < 4:
                print("Usage: variable set <prompt> <name> <value>")
                return
                
            prompt_name = args[1]
            var_name = args[2]
            var_value = " ".join(args[3:])
            
            prompt = self.prompt_manager.load_prompt(prompt_name)
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            if var_name not in prompt.variables:
                print(f"Warning: Variable '{var_name}' is not defined in the prompt template.")
                print("Available variables:", ", ".join(prompt.variables.keys()) if prompt.variables else "none")
                
            prompt.variables[var_name] = var_value
            self.prompt_manager.save_prompt(prompt)
            
            print(f"Set variable '{var_name}' in prompt '{prompt_name}'.")
            
        elif command == "clear":
            if len(args) < 2:
                print("Usage: variable clear <prompt> [name]")
                return
                
            prompt_name = args[1]
            var_name = args[2] if len(args) > 2 else None
            
            prompt = self.prompt_manager.load_prompt(prompt_name)
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            if var_name:
                if var_name in prompt.variables:
                    prompt.variables[var_name] = ""
                    print(f"Cleared variable '{var_name}' in prompt '{prompt_name}'.")
                else:
                    print(f"Variable '{var_name}' not found in prompt '{prompt_name}'.")
            else:
                prompt.variables = {k: "" for k in prompt.variables.keys()}
                print(f"Cleared all variables in prompt '{prompt_name}'.")
                
            self.prompt_manager.save_prompt(prompt)
            
        elif command == "apply":
            if len(args) < 2:
                print("Usage: variable apply <prompt>")
                return
                
            prompt_name = args[1]
            prompt = self.prompt_manager.load_prompt(prompt_name)
            
            if not prompt:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            if not prompt.variables:
                print(f"No variables found in prompt '{prompt_name}'.")
                return
                
            # Check if all variables have values
            empty_vars = [name for name, value in prompt.variables.items() if not value]
            if empty_vars:
                print("Warning: Some variables have no values:")
                for name in empty_vars:
                    print(f"  {name}")
                print("Use 'variable set' to set values before applying.")
                return
                
            # Apply variables to the template
            result = self.prompt_manager.substitute_variables(prompt.content, prompt.variables)
            
            print("\nApplied variables to prompt template:")
            print("\n---\n")
            print(result)
            print("\n---\n")
    
    def do_help(self, args):
        """Show help information for commands"""
        if not args:
            print("\nAvailable commands:")
            print("  prompt - Manage prompts")
            print("  chain - Manage prompt chains")
            print("  variable - Manage prompt variables")
            print("  help - Show help information")
            print("  exit - Exit the program")
            print("\nType 'help <command>' for detailed help on a command.")
            return
            
        command = args[0].lower()
        
        if command == "prompt":
            print(self.do_prompt.__doc__)
        elif command == "chain":
            print(self.do_chain.__doc__)
        elif command == "variable":
            print(self.do_variable.__doc__)
        elif command == "help":
            print(self.do_help.__doc__)
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for a list of available commands.")

def main():
    """Main entry point for advanced prompt manager CLI"""
    parser = argparse.ArgumentParser(description="Advanced Prompt Management System")
    parser.add_argument('--prompts-dir', type=str, default='./prompts', 
                       help='Directory to store prompts (default: ./prompts)')
    parser.add_argument('--chains-dir', type=str, default='./prompt_chains',
                       help='Directory to store prompt chains (default: ./prompt_chains)')
    parser.add_argument('--command', type=str, nargs='+',
                       help='Command to execute directly')
    parser.add_argument('--list', action='store_true',
                       help='List all available prompts')
    parser.add_argument('--create', type=str,
                       help='Create a new prompt with the given name')
    parser.add_argument('--evaluate', type=str,
                       help='Evaluate a prompt with the given name')
    parser.add_argument('--visualize-chain', type=str,
                       help='Visualize a prompt chain with the given name')
    
    args = parser.parse_args()
    
    # Initialize the CLI
    cli = AdvancedPromptCLI(prompts_dir=args.prompts_dir, chains_dir=args.chains_dir)
    
    # Handle direct command execution
    if args.list:
        cli.do_prompt(['list'])
    elif args.create:
        cli.do_prompt(['new', args.create])
    elif args.evaluate:
        cli.do_prompt(['evaluate', args.evaluate])
    elif args.visualize_chain:
        cli.do_chain(['visualize', args.visualize_chain])
    elif args.command:
        if args.command[0] in ['prompt', 'chain', 'variable']:
            if args.command[0] == 'prompt':
                cli.do_prompt(args.command[1:])
            elif args.command[0] == 'chain':
                cli.do_chain(args.command[1:])
            elif args.command[0] == 'variable':
                cli.do_variable(args.command[1:])
        else:
            print(f"Unknown command: {args.command[0]}")
    else:
        # Interactive mode
        print("Advanced Prompt Management System")
        print("Type 'help' for usage information")
        print("Type 'exit' to quit")
        
        while True:
            try:
                cmd = input("\nprompt-manager> ")
                if cmd.lower() in ['exit', 'quit']:
                    break
                elif cmd.lower() == 'help':
                    cli.do_help([])
                elif cmd.lower().startswith('help '):
                    cli.do_help([cmd.split(' ', 1)[1]])
                else:
                    parts = shlex.split(cmd)
                    if not parts:
                        continue
                        
                    if parts[0] == 'prompt':
                        cli.do_prompt(parts[1:])
                    elif parts[0] == 'chain':
                        cli.do_chain(parts[1:])
                    elif parts[0] == 'variable':
                        cli.do_variable(parts[1:])
                    else:
                        print(f"Unknown command: {parts[0]}")
                        print("Available commands: prompt, chain, variable, help, exit")
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nOperation cancelled")
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
    
    print("Goodbye!")

if __name__ == "__main__":
    main()