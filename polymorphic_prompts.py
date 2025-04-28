#!/usr/bin/env python3
"""
Prompt Management System for AI Agents

This module provides CLI utilities for managing, analyzing, converting, and embedding 
various types of prompts including holographic, temporal-context, multi-agent perspective,
self-calibrating, and knowledge-synthesis prompts.

Features:
- Create, edit, save, and load prompts
- Convert prompts to specialized formats
- Analyze prompt structure and suggest improvements
- Generate and manage embeddings for semantic retrieval
- Search for semantically similar prompts
"""

import os
import sys
import json
import shlex
import asyncio
import traceback
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union

# Import embedding functions if available
try:
    from holographic_memory import compute_embedding
    has_embedding = True
except ImportError:
    has_embedding = False
    print("Warning: Embedding functionality not available. Some features will be limited.")

class PromptManager:
    """Manages loading, editing, and saving prompts"""
    def __init__(self, prompts_dir="./prompts"):
        self.prompts_dir = prompts_dir
        self.current_prompt = None
        self.prompt_history = []
        
        # Create prompts directory if it doesn't exist
        os.makedirs(prompts_dir, exist_ok=True)
        
    def load_prompt(self, prompt_name):
        """Load a prompt from file"""
        prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                content = f.read()
            self.current_prompt = {"name": prompt_name, "content": content}
            return content
        return None
        
    def save_prompt(self, prompt_name, content):
        """Save a prompt to file"""
        prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        with open(prompt_path, 'w') as f:
            f.write(content)
        self.current_prompt = {"name": prompt_name, "content": content}
        return True
        
    def list_prompts(self):
        """List all available prompts"""
        if not os.path.exists(self.prompts_dir):
            return []
        return [f[:-4] for f in os.listdir(self.prompts_dir) if f.endswith('.txt')]
        
    def add_to_history(self, prompt_name, content):
        """Add a prompt to history"""
        self.prompt_history.append({
            "name": prompt_name,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_prompt_metadata(self):
        """Get metadata for all prompts"""
        prompts = self.list_prompts()
        metadata = []
        
        for prompt_name in prompts:
            prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
            stat = os.stat(prompt_path)
            
            metadata.append({
                "name": prompt_name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
            
        return metadata
        
    def search_prompts(self, query):
        """Simple text-based search in prompts"""
        results = []
        
        for prompt_name in self.list_prompts():
            content = self.load_prompt(prompt_name)
            if query.lower() in content.lower():
                results.append({
                    "name": prompt_name,
                    "content": content,
                    "match_index": content.lower().index(query.lower())
                })
                
        # Sort by match position (earlier matches first)
        results.sort(key=lambda x: x["match_index"])
        return results
        
    def export_prompts(self, output_file):
        """Export all prompts to a JSON file"""
        prompts_data = {}
        
        for prompt_name in self.list_prompts():
            content = self.load_prompt(prompt_name)
            prompts_data[prompt_name] = content
            
        with open(output_file, 'w') as f:
            json.dump(prompts_data, f, indent=2)
            
        return len(prompts_data)
        
    def import_prompts(self, input_file):
        """Import prompts from a JSON file"""
        with open(input_file, 'r') as f:
            prompts_data = json.load(f)
            
        count = 0
        for prompt_name, content in prompts_data.items():
            self.save_prompt(prompt_name, content)
            count += 1
            
        return count

class PolymorphicEmbedding:
    """
    Advanced embedding system for specialized prompt types including
    holographic, temporal-context, multi-agent, and others.
    """
    def __init__(self, kb_connection=None, base_dimension=1536):
        self.kb = kb_connection
        self.base_dimension = base_dimension
        self.dimensions = {
            "content": base_dimension,
            "detail_levels": 3,
            "temporal": 4,
            "agent_perspectives": 4,
            "confidence": 2,
            "domains": 5
        }
        
    async def embed_holographic(self, text, detail_levels=3):
        """Generate embeddings at multiple detail levels"""
        # Create summarized versions at different detail levels
        summaries = await self._generate_detail_levels(text, detail_levels)
        # Embed each detail level
        embeddings = []
        for summary in summaries:
            emb = await compute_embedding(summary, db_conn=self.kb)
            embeddings.append(emb)
        return embeddings
        
    async def embed_temporal(self, text):
        """Generate embeddings with temporal markers"""
        # Extract temporal segments if available
        segments = self._extract_temporal_segments(text)
        
        if segments:
            embeddings = {}
            for period, content in segments.items():
                emb = await compute_embedding(content, db_conn=self.kb)
                embeddings[period] = emb
            return embeddings
        else:
            # Fall back to standard embedding
            return await compute_embedding(text, db_conn=self.kb)
            
    async def embed_multi_agent(self, text):
        """Generate embeddings for multi-agent perspectives"""
        # Extract agent perspectives if available
        perspectives = self._extract_agent_perspectives(text)
        
        if perspectives:
            embeddings = {}
            for role, content in perspectives.items():
                emb = await compute_embedding(content, db_conn=self.kb)
                embeddings[role] = emb
            return embeddings
        else:
            # Fall back to standard embedding
            return await compute_embedding(text, db_conn=self.kb)
            
    async def embed_self_calibrating(self, text):
        """Generate embeddings with confidence bands"""
        # Extract main instruction and confidence requirements
        instruction, requirements = self._extract_calibration_components(text)
        
        # Generate embeddings for both
        instruction_emb = await compute_embedding(instruction, db_conn=self.kb)
        requirements_emb = await compute_embedding(requirements, db_conn=self.kb)
        
        return {
            "instruction": instruction_emb,
            "requirements": requirements_emb
        }
        
    async def embed_knowledge_synthesis(self, text):
        """Generate embeddings for cross-domain knowledge"""
        # Extract domains and connections
        domains, connections = self._extract_knowledge_domains(text)
        
        domain_embeddings = {}
        for domain_name, content in domains.items():
            emb = await compute_embedding(content, db_conn=self.kb)
            domain_embeddings[domain_name] = emb
            
        connection_embeddings = []
        for connection in connections:
            connection_text = f"{connection['from']} {connection['relationship']} {connection['to']}"
            emb = await compute_embedding(connection_text, db_conn=self.kb)
            connection_embeddings.append(emb)
            
        return {
            "domains": domain_embeddings,
            "connections": connection_embeddings
        }
        
    async def retrieve_polymorphic(self, query, prompt_type, **params):
        """Retrieve using specialized metrics for each prompt type"""
        base_emb = await compute_embedding(query, db_conn=self.kb)
        
        if prompt_type == "holographic":
            # Get query at different detail levels
            query_levels = await self._generate_detail_levels(query)
            query_embs = [await compute_embedding(q) for q in query_levels]
            
            # Hierarchical matching against stored holographic embeddings
            results = await self._holographic_search(query_embs, params.get("level_weights"))
            return results
            
        elif prompt_type == "temporal":
            # Extract time references from query
            time_focus = self._extract_time_focus(query)
            
            # Time-weighted search
            results = await self._temporal_search(base_emb, time_focus)
            return results
            
        elif prompt_type == "multi-agent":
            # Determine which agent perspective is most relevant
            perspective = self._determine_query_perspective(query)
            
            # Perspective-focused search
            results = await self._perspective_search(base_emb, perspective)
            return results
            
        elif prompt_type == "self-calibrating":
            # Extract confidence requirements from query
            confidence_level = self._extract_confidence_requirements(query)
            
            # Confidence-aware search
            results = await self._confidence_search(base_emb, confidence_level)
            return results
            
        elif prompt_type == "knowledge-synthesis":
            # Extract domain references from query
            domains = self._extract_query_domains(query)
            
            # Cross-domain search
            results = await self._cross_domain_search(base_emb, domains)
            return results
            
        else:
            # Default to standard embedding search
            # Implementation would depend on the knowledge base interface
            return []
            
    async def _generate_detail_levels(self, text, levels=3):
        """Generate different summary levels of the text"""
        # This would typically use an LLM to create summaries
        # Placeholder implementation
        summaries = [text]
        
        # Find XML tags if they exist
        import re
        level_matches = re.findall(r'<level depth="(\d+)">(.*?)</level>', text, re.DOTALL)
        
        if level_matches:
            # Use existing levels
            level_map = {int(depth): content for depth, content in level_matches}
            summaries = [level_map.get(i+1, "") for i in range(levels)]
        else:
            # Generate simple summaries based on text length
            if len(text) > 1000:
                # Level 1: Very brief summary (first paragraph)
                first_para = text.split('\n\n')[0] if '\n\n' in text else text[:200]
                summaries = [first_para, text[:len(text)//2], text]
                
        return summaries[:levels]
        
    def _extract_temporal_segments(self, text):
        """Extract time-based segments from a temporal prompt"""
        segments = {}
        
        # Look for temporal XML tags
        import re
        time_matches = re.findall(r'<timeframe period="([^"]+)"[^>]*>(.*?)</timeframe>', text, re.DOTALL)
        
        if time_matches:
            for period, content in time_matches:
                segments[period] = content.strip()
                
        return segments
        
    def _extract_agent_perspectives(self, text):
        """Extract different agent perspectives from a multi-agent prompt"""
        perspectives = {}
        
        # Look for agent XML tags
        import re
        agent_matches = re.findall(r'<agent role="([^"]+)">(.*?)</agent>', text, re.DOTALL)
        
        if agent_matches:
            for role, content in agent_matches:
                perspectives[role] = content.strip()
                
        # Check for integration section
        integration_match = re.search(r'<integration>(.*?)</integration>', text, re.DOTALL)
        if integration_match:
            perspectives["integration"] = integration_match.group(1).strip()
            
        return perspectives
        
    def _extract_calibration_components(self, text):
        """Extract instruction and confidence requirements from a self-calibrating prompt"""
        instruction = ""
        requirements = ""
        
        # Look for instruction and requirements XML tags
        import re
        instruction_match = re.search(r'<instruction>(.*?)</instruction>', text, re.DOTALL)
        if instruction_match:
            instruction = instruction_match.group(1).strip()
            
        requirements_match = re.search(r'<confidence_requirements>(.*?)</confidence_requirements>', text, re.DOTALL)
        if requirements_match:
            requirements = requirements_match.group(1).strip()
            
        if not instruction:
            # Fall back to using the whole text as instruction
            instruction = text
            
        return instruction, requirements
        
    def _extract_knowledge_domains(self, text):
        """Extract domains and connections from a knowledge-synthesis prompt"""
        domains = {}
        connections = []
        
        # Look for domain XML tags
        import re
        domain_matches = re.findall(r'<domain name="([^"]+)">(.*?)</domain>', text, re.DOTALL)
        
        if domain_matches:
            for name, content in domain_matches:
                domains[name] = content.strip()
                
        # Look for connection points
        connection_matches = re.findall(r'<connection>.*?<from>(.*?)</from>.*?<to>(.*?)</to>.*?<relationship>(.*?)</relationship>.*?</connection>', text, re.DOTALL)
        
        if connection_matches:
            for from_item, to_item, relationship in connection_matches:
                connections.append({
                    "from": from_item.strip(),
                    "to": to_item.strip(),
                    "relationship": relationship.strip()
                })
                
        return domains, connections
        
    def _extract_time_focus(self, query):
        """Extract time focus from query for temporal search"""
        # Simple keyword-based extraction
        time_focus = "present"  # Default
        
        if any(term in query.lower() for term in ["history", "past", "previous", "before", "ago"]):
            time_focus = "past"
        elif any(term in query.lower() for term in ["future", "upcoming", "next", "plan", "will"]):
            time_focus = "future"
        elif any(term in query.lower() for term in ["now", "current", "present", "today"]):
            time_focus = "present"
            
        return time_focus
        
    def _determine_query_perspective(self, query):
        """Determine which agent perspective is most relevant to query"""
        perspectives = {
            "analyst": ["analyze", "data", "information", "statistics", "trends"],
            "critic": ["evaluate", "assess", "critique", "problems", "issues"],
            "creative": ["ideas", "creative", "innovative", "possibilities", "imagine"],
            "implementer": ["implement", "execute", "steps", "process", "how to"]
        }
        
        # Count keywords for each perspective
        scores = {role: 0 for role in perspectives}
        for role, keywords in perspectives.items():
            for keyword in keywords:
                if keyword in query.lower():
                    scores[role] += 1
                    
        # Return the perspective with highest score, or "all" if tied
        max_score = max(scores.values())
        if max_score == 0:
            return "all"  # No clear perspective
            
        top_perspectives = [role for role, score in scores.items() if score == max_score]
        if len(top_perspectives) == 1:
            return top_perspectives[0]
        else:
            return "all"  # Multiple perspectives tied
            
    def _extract_confidence_requirements(self, query):
        """Extract confidence requirements from query"""
        # Simple keyword-based extraction
        confidence_level = "medium"  # Default
        
        if any(term in query.lower() for term in ["certain", "confident", "sure", "definitely"]):
            confidence_level = "high"
        elif any(term in query.lower() for term in ["uncertain", "unsure", "maybe", "perhaps"]):
            confidence_level = "low"
            
        return confidence_level
        
    def _extract_query_domains(self, query):
        """Extract domain references from query"""
        # This would typically use an LLM or classifier to determine domains
        # Simple placeholder implementation
        domains = ["primary"]
        
        return domains

class PromptCLI:
    """Command-line interface for managing prompts"""
    def __init__(self, prompts_dir="./prompts", kb_connection=None):
        self.prompt_manager = PromptManager(prompts_dir)
        if has_embedding and kb_connection:
            self.embedding_system = PolymorphicEmbedding(kb_connection)
        else:
            self.embedding_system = None
        self.conversation_id = None
        
    def do_prompt(self, args):
        """
        Manage prompts: prompt [list|load|save|edit|new|analyze|convert]
        
        Commands:
          list                 - List all available prompts
          load <name>          - Load a prompt
          save <name>          - Save current prompt
          edit <name>          - Edit a prompt
          new <name>           - Create a new prompt
          analyze <name>       - Analyze prompt structure and suggest improvements
          convert <name> <type> - Convert a prompt to a specialized format
          export <filename>    - Export all prompts to a JSON file
          import <filename>    - Import prompts from a JSON file
          search <text>        - Search for prompts containing text
          
        Specialized prompt types:
          holographic         - Multi-level detail prompt
          temporal            - Time-aware prompt
          multi-agent         - Multiple perspective prompt
          self-calibrating    - Confidence-aware prompt
          knowledge-synthesis - Cross-domain prompt
        """
        if not args:
            print("Usage: prompt [list|load|save|edit|new|analyze|convert|export|import|search]")
            return
            
        command = args[0].lower()
        
        if command == "list":
            prompts = self.prompt_manager.list_prompts()
            metadata = self.prompt_manager.get_prompt_metadata()
            
            if prompts:
                print("\nAvailable prompts:")
                print(f"{'Name':<30} {'Size':>8} {'Last Modified':<20}")
                print("-" * 60)
                
                for meta in metadata:
                    name = meta["name"]
                    size = meta["size"]
                    modified = meta["modified"].split("T")[0]  # Just the date part
                    print(f"{name:<30} {size:>8} {modified:<20}")
            else:
                print("No prompts found.")
                
        elif command == "load":
            if len(args) < 2:
                print("Usage: prompt load <name>")
                return
                
            prompt_name = args[1]
            content = self.prompt_manager.load_prompt(prompt_name)
            
            if content:
                print(f"Prompt '{prompt_name}' loaded:")
                print("\n---\n")
                print(content)
                print("\n---\n")
            else:
                print(f"Prompt '{prompt_name}' not found.")
                
        elif command == "save":
            if len(args) < 2:
                print("Usage: prompt save <name>")
                return
                
            prompt_name = args[1]
            
            if not self.prompt_manager.current_prompt:
                print("No prompt currently loaded or edited.")
                return
                
            self.prompt_manager.save_prompt(prompt_name, self.prompt_manager.current_prompt["content"])
            print(f"Prompt saved as '{prompt_name}'.")
            
        elif command == "edit":
            if len(args) < 2:
                print("Usage: prompt edit <name>")
                return
                
            prompt_name = args[1]
            content = self.prompt_manager.load_prompt(prompt_name)
            
            if not content:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            print(f"Editing prompt '{prompt_name}'.")
            print("Enter your changes (type '###' on a line by itself to end):")
            
            # Store original for history
            original_content = content
            
            lines = []
            print("\n--- Current content ---")
            print(content)
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
                self.prompt_manager.save_prompt(prompt_name, new_content)
                self.prompt_manager.add_to_history(prompt_name, original_content)
                print(f"Prompt '{prompt_name}' updated.")
            else:
                print("No changes made.")
                
        elif command == "new":
            if len(args) < 2:
                print("Usage: prompt new <name>")
                return
                
            prompt_name = args[1]
            
            print(f"Creating new prompt '{prompt_name}'.")
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
                self.prompt_manager.save_prompt(prompt_name, content)
                print(f"Prompt '{prompt_name}' created.")
            else:
                print("No content provided. Prompt not created.")
                
        elif command == "analyze":
            if len(args) < 2:
                print("Usage: prompt analyze <name>")
                return
                
            prompt_name = args[1]
            content = self.prompt_manager.load_prompt(prompt_name)
            
            if not content:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            print(f"Analyzing prompt '{prompt_name}'...")
            
            # Placeholder for agent-based analysis
            # In a real implementation, this would connect to an LLM
            print("\n=== Prompt Analysis ===\n")
            print("This is a placeholder for prompt analysis.")
            print("In a real implementation, this would use an LLM to analyze the prompt structure,")
            print("identify potential ambiguities, and suggest improvements.")
            
        elif command == "convert":
            if len(args) < 3:
                print("Usage: prompt convert <name> <type>")
                print("Available types: holographic, temporal, multi-agent, self-calibrating, knowledge-synthesis")
                return
                
            prompt_name = args[1]
            prompt_type = args[2].lower()
            
            content = self.prompt_manager.load_prompt(prompt_name)
            if not content:
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
    {content[:100]}...
  </level>
  <level depth="2">
    {content[:200]}...
  </level>
  <level depth="3">
    {content}
  </level>
</holographic_prompt>"""
            elif prompt_type == "temporal":
                converted_content = f"""<temporal_prompt>
  <timeframe period="immediate">
    {content[:150]}...
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
    {content[:150]}...
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
    {content}
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
    {content[:150]}...
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
            
            self.prompt_manager.save_prompt(new_prompt_name, converted_content)
            
            print(f"\nConverted prompt saved as '{new_prompt_name}'.")
            print("\nPreview of converted prompt:")
            print("\n---\n")
            preview = converted_content[:500] + "..." if len(converted_content) > 500 else converted_content
            print(preview)
            print("\n---\n")
            
        elif command == "export":
            if len(args) < 2:
                print("Usage: prompt export <filename>")
                return
                
            output_file = args[1]
            if not output_file.endswith('.json'):
                output_file += '.json'
                
            count = self.prompt_manager.export_prompts(output_file)
            print(f"Exported {count} prompts to {output_file}")
            
        elif command == "import":
            if len(args) < 2:
                print("Usage: prompt import <filename>")
                return
                
            input_file = args[1]
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}")
                return
                
            try:
                count = self.prompt_manager.import_prompts(input_file)
                print(f"Imported {count} prompts from {input_file}")
            except Exception as e:
                print(f"Error importing prompts: {e}")
                
        elif command == "search":
            if len(args) < 2:
                print("Usage: prompt search <text>")
                return
                
            query = " ".join(args[1:])
            results = self.prompt_manager.search_prompts(query)
            
            if results:
                print(f"\nFound {len(results)} prompts containing '{query}':")
                for i, result in enumerate(results):
                    print(f"\n{i+1}. {result['name']}")
                    
                    # Show context around the match
                    content = result["content"]
                    match_pos = result["match_index"]
                    start = max(0, match_pos - 50)
                    end = min(len(content), match_pos + len(query) + 50)
                    context = content[start:end]
                    
                    if start > 0:
                        context = "..." + context
                    if end < len(content):
                        context += "..."
                        
                    print(f"   Context: {context}")
            else:
                print(f"No prompts found containing '{query}'.")
        else:
            print(f"Unknown command: {command}")
            print("Usage: prompt [list|load|save|edit|new|analyze|convert|export|import|search]")
    
    def do_embed_prompt(self, args):
        """
        Create and manage prompt embeddings: embed_prompt [create|search|batch|view]
        
        Commands:
          create <name> [type]   - Create embeddings for a prompt
          search <query>         - Search for prompts similar to query
          batch                  - Batch embed all prompts
          view <name>            - View stored embeddings for a prompt
          
        Examples:
          embed_prompt create task_completion holographic
          embed_prompt search "how to analyze data"
          embed_prompt batch
          embed_prompt view customer_service
        """
        if not self.embedding_system:
            print("Embedding functionality not available. Make sure the required dependencies are installed.")
            return
            
        if not args:
            print("Usage: embed_prompt [create|search|batch|view]")
            return
            
        command = args[0].lower()
        
        if command == "create":
            if len(args) < 2:
                print("Usage: embed_prompt create <name> [type]")
                return
                
            prompt_name = args[1]
            prompt_type = args[2] if len(args) > 2 else "standard"
            
            content = self.prompt_manager.load_prompt(prompt_name)
            if not content:
                print(f"Prompt '{prompt_name}' not found.")
                return
                
            print(f"Creating embeddings for prompt '{prompt_name}' as {prompt_type} type...")
            print("This is a placeholder for embedding creation.")
            print("In a real implementation, this would use an embedding model to generate and store embeddings.")
            
        elif command == "search":
            if len(args) < 2:
                print("Usage: embed_prompt search <query>")
                return
                
            query = " ".join(args[1:])
            print(f"Searching for prompts similar to: {query}")
            print("This is a placeholder for semantic search.")
            print("In a real implementation, this would use embeddings to find semantically similar prompts.")
            
        elif command == "batch":
            print("Batch embedding all prompts...")
            print("This is a placeholder for batch embedding.")
            print("In a real implementation, this would create embeddings for all prompts.")
            
        elif command == "view":
            if len(args) < 2:
                print("Usage: embed_prompt view <name>")
                return
                
            prompt_name = args[1]
            print(f"Embedding details for prompt '{prompt_name}':")
            print("This is a placeholder for embedding visualization.")
            print("In a real implementation, this would show statistics about the stored embeddings.")
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: embed_prompt [create|search|batch|view]")

def main():
    """Main entry point for prompt manager CLI"""
    parser = argparse.ArgumentParser(description="Advanced Prompt Management System")
    parser.add_argument('--prompts-dir', type=str, default='./prompts', 
                       help='Directory to store prompts (default: ./prompts)')
    parser.add_argument('--command', type=str, nargs='+',
                       help='Command to execute directly')
    parser.add_argument('--list', action='store_true',
                       help='List all available prompts')
    parser.add_argument('--create', type=str,
                       help='Create a new prompt with the given name')
    parser.add_argument('--edit', type=str,
                       help='Edit an existing prompt with the given name')
    parser.add_argument('--convert', nargs=2, metavar=('PROMPT_NAME', 'TYPE'),
                       help='Convert a prompt to a specialized format')
    
    args = parser.parse_args()
    
    # Initialize the CLI
    cli = PromptCLI(prompts_dir=args.prompts_dir)
    
    # Handle direct command execution
    if args.list:
        cli.do_prompt(['list'])
    elif args.create:
        cli.do_prompt(['new', args.create])
    elif args.edit:
        cli.do_prompt(['edit', args.edit])
    elif args.convert:
        cli.do_prompt(['convert', args.convert[0], args.convert[1]])
    elif args.command:
        if args.command[0] in ['prompt', 'embed_prompt']:
            if args.command[0] == 'prompt':
                cli.do_prompt(args.command[1:])
            else:
                cli.do_embed_prompt(args.command[1:])
        else:
            print(f"Unknown command: {args.command[0]}")
    else:
        # Interactive mode
        print("Advanced Prompt Management System")
        print("Type 'help prompt' or 'help embed_prompt' for usage information")
        print("Type 'exit' to quit")
        
        while True:
            try:
                cmd = input("\nprompt-manager> ")
                if cmd.lower() in ['exit', 'quit']:
                    break
                elif cmd.lower() == 'help prompt':
                    print(cli.do_prompt.__doc__)
                elif cmd.lower() == 'help embed_prompt':
                    print(cli.do_embed_prompt.__doc__)
                elif cmd.lower() == 'help':
                    print("Available commands: prompt, embed_prompt, exit")
                    print("Type 'help prompt' or 'help embed_prompt' for detailed help")
                else:
                    parts = shlex.split(cmd)
                    if not parts:
                        continue
                        
                    if parts[0] == 'prompt':
                        cli.do_prompt(parts[1:])
                    elif parts[0] == 'embed_prompt':
                        cli.do_embed_prompt(parts[1:])
                    else:
                        print(f"Unknown command: {parts[0]}")
                        print("Available commands: prompt, embed_prompt, exit")
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