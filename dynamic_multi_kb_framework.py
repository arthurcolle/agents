#!/usr/bin/env python3
"""
Dynamic Multi-Knowledge Base Framework - A meta-framework that dynamically constructs
specialized multi-agent systems tailored to specific problem domains and requirements.

This framework serves as a factory that can dynamically assemble agent systems with
the appropriate architecture, knowledge domains, and interaction patterns based on
the nature of the problem to be solved.
"""

import os
import json
import asyncio
import logging
import importlib
import inspect
import tempfile
import argparse
from typing import Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple
from pathlib import Path
import time
import uuid
import sys
import re
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dynamic_framework.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dynamic-framework")


class AgentArchitecture:
    """Base class for agent system architectures that can be dynamically composed"""
    
    def __init__(self, name: str, description: str):
        """Initialize an agent architecture"""
        self.name = name
        self.description = description
        self.components = {}
        self.required_dependencies = []
        self.optional_dependencies = []
        self.configuration_schema = {}
    
    def add_component(self, component_id: str, component_type: str, 
                     required: bool = True) -> None:
        """Add a component to the architecture"""
        self.components[component_id] = {
            "type": component_type,
            "required": required,
            "connections": []
        }
    
    def add_connection(self, source_id: str, target_id: str, 
                      connection_type: str = "default") -> None:
        """Add a connection between components"""
        if source_id not in self.components:
            raise ValueError(f"Source component {source_id} not found")
        if target_id not in self.components:
            raise ValueError(f"Target component {target_id} not found")
        
        self.components[source_id]["connections"].append({
            "target": target_id,
            "type": connection_type
        })
    
    def add_dependency(self, module_name: str, required: bool = True) -> None:
        """Add a dependency to the architecture"""
        if required:
            self.required_dependencies.append(module_name)
        else:
            self.optional_dependencies.append(module_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "components": self.components,
            "required_dependencies": self.required_dependencies,
            "optional_dependencies": self.optional_dependencies,
            "configuration_schema": self.configuration_schema
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentArchitecture':
        """Create architecture from dictionary"""
        arch = cls(data["name"], data["description"])
        arch.components = data["components"]
        arch.required_dependencies = data["required_dependencies"]
        arch.optional_dependencies = data["optional_dependencies"]
        arch.configuration_schema = data.get("configuration_schema", {})
        return arch


class KnowledgeDomainRegistry:
    """Registry of available knowledge domains and their characteristics"""
    
    def __init__(self, knowledge_base_dir: str = "knowledge_bases"):
        """Initialize the knowledge domain registry"""
        self.knowledge_base_dir = knowledge_base_dir
        self.domains = {}
        self.domain_relationships = {}
        self.domain_metadata = {}
        self.domain_stats = {}
        self._load_domains()
    
    def _load_domains(self) -> None:
        """Load available knowledge domains"""
        kb_path = Path(self.knowledge_base_dir)
        if not kb_path.exists() or not kb_path.is_dir():
            logger.warning(f"Knowledge base directory not found: {self.knowledge_base_dir}")
            return
        
        # Find all JSON files in the knowledge_bases directory
        kb_files = list(kb_path.glob("*.json"))
        logger.info(f"Found {len(kb_files)} knowledge base files")
        
        # Load basic information about each domain
        for kb_file in kb_files:
            try:
                domain_name = kb_file.stem
                self.domains[domain_name] = {
                    "file_path": str(kb_file),
                    "last_modified": kb_file.stat().st_mtime,
                    "size": kb_file.stat().st_size
                }
                
                # Try to load metadata from the file
                try:
                    with open(kb_file, 'r', encoding='utf-8') as f:
                        # Just read the first 8KB to extract metadata without loading the entire file
                        header = f.read(8192)
                        
                        # Try to parse as JSON
                        try:
                            # Check if it's a valid JSON object with opening/closing braces
                            if re.match(r'^\s*\{.*\}\s*$', header, re.DOTALL):
                                data = json.loads(header)
                                if isinstance(data, dict) and "metadata" in data:
                                    self.domain_metadata[domain_name] = data["metadata"]
                            else:
                                # If not a full object, look for metadata key within first part of file
                                metadata_match = re.search(r'"metadata"\s*:\s*(\{[^}]*\})', header)
                                if metadata_match:
                                    try:
                                        metadata = json.loads(metadata_match.group(1))
                                        self.domain_metadata[domain_name] = metadata
                                    except json.JSONDecodeError:
                                        pass
                        except json.JSONDecodeError:
                            pass
                except Exception as e:
                    logger.debug(f"Could not extract metadata from {kb_file}: {e}")
                
                logger.debug(f"Registered knowledge domain: {domain_name}")
            except Exception as e:
                logger.error(f"Error loading domain from {kb_file.name}: {e}")
        
        # Analyze relationships between domains
        self._analyze_domain_relationships()
    
    def _analyze_domain_relationships(self) -> None:
        """Analyze relationships between knowledge domains"""
        # This is a simplified analysis - in a real system, this would be more sophisticated
        # and based on actual content analysis or predefined taxonomies
        
        for domain_name in self.domains:
            self.domain_relationships[domain_name] = {}
            
            # Simple relationship analysis based on name similarity
            domain_words = set(domain_name.lower().replace('_', ' ').split())
            
            for other_domain in self.domains:
                if other_domain == domain_name:
                    continue
                
                other_words = set(other_domain.lower().replace('_', ' ').split())
                
                # Calculate simple Jaccard similarity
                intersection = len(domain_words.intersection(other_words))
                union = len(domain_words.union(other_words))
                
                if union > 0:
                    similarity = intersection / union
                    
                    # Only record relationships with some meaningful similarity
                    if similarity > 0.1:
                        self.domain_relationships[domain_name][other_domain] = {
                            "type": "related",
                            "strength": similarity
                        }
    
    def get_domains(self) -> List[str]:
        """Get list of available domains"""
        return list(self.domains.keys())
    
    def get_domain_info(self, domain_name: str) -> Dict[str, Any]:
        """Get information about a specific domain"""
        if domain_name not in self.domains:
            return {"error": f"Domain {domain_name} not found"}
        
        info = self.domains[domain_name].copy()
        
        # Add metadata if available
        if domain_name in self.domain_metadata:
            info["metadata"] = self.domain_metadata[domain_name]
        
        # Add relationships if available
        if domain_name in self.domain_relationships:
            info["relationships"] = self.domain_relationships[domain_name]
        
        # Add usage statistics if available
        if domain_name in self.domain_stats:
            info["stats"] = self.domain_stats[domain_name]
        
        return info
    
    def get_related_domains(self, domain_name: str, min_strength: float = 0.2) -> List[Dict[str, Any]]:
        """Get domains related to a specific domain"""
        if domain_name not in self.domain_relationships:
            return []
        
        related = []
        for related_domain, rel_info in self.domain_relationships[domain_name].items():
            if rel_info["strength"] >= min_strength:
                related.append({
                    "domain": related_domain,
                    "type": rel_info["type"],
                    "strength": rel_info["strength"]
                })
        
        # Sort by relationship strength
        related.sort(key=lambda x: x["strength"], reverse=True)
        
        return related
    
    def find_domains_for_problem(self, problem_statement: str) -> List[Dict[str, Any]]:
        """
        Find relevant domains for a given problem statement.
        
        This uses simple keyword matching. In a real implementation, this would
        use more sophisticated NLP techniques, embeddings, etc.
        """
        problem_words = set(problem_statement.lower().split())
        relevant_domains = []
        
        for domain_name in self.domains:
            domain_text = domain_name.lower().replace('_', ' ')
            
            # Check word overlap
            domain_words = set(domain_text.split())
            overlap = problem_words.intersection(domain_words)
            
            relevance_score = len(overlap) / len(domain_words) if domain_words else 0
            
            # Check for exact domain name in problem
            if domain_text in problem_statement.lower():
                relevance_score += 0.3
            
            # If some relevance found, add to results
            if relevance_score > 0:
                relevant_domains.append({
                    "domain": domain_name,
                    "relevance": min(1.0, relevance_score)  # Cap at 1.0
                })
        
        # Sort by relevance
        relevant_domains.sort(key=lambda x: x["relevance"], reverse=True)
        
        return relevant_domains
    
    def update_domain_stats(self, domain_name: str, usage_type: str) -> None:
        """Update usage statistics for a domain"""
        if domain_name not in self.domain_stats:
            self.domain_stats[domain_name] = {
                "queries": 0,
                "solutions": 0,
                "insights": 0,
                "last_used": None
            }
        
        # Update appropriate counter
        if usage_type in self.domain_stats[domain_name]:
            self.domain_stats[domain_name][usage_type] += 1
        
        # Update last used timestamp
        self.domain_stats[domain_name]["last_used"] = time.time()


class SystemComponentRegistry:
    """Registry of available system components that can be dynamically loaded"""
    
    def __init__(self):
        """Initialize the component registry"""
        self.components = {}
        self.architectures = {}
        self.agent_types = {}
        self.communication_protocols = {}
    
    def register_component(self, component_id: str, component_type: str, 
                          module_path: str, class_name: str,
                          description: str = "") -> None:
        """Register a system component"""
        self.components[component_id] = {
            "type": component_type,
            "module_path": module_path,
            "class_name": class_name,
            "description": description
        }
        
        logger.debug(f"Registered component: {component_id} ({component_type})")
    
    def register_architecture(self, architecture: AgentArchitecture) -> None:
        """Register an agent system architecture"""
        self.architectures[architecture.name] = architecture
        logger.debug(f"Registered architecture: {architecture.name}")
    
    def register_agent_type(self, agent_type: str, module_path: str, 
                           class_name: str, capabilities: List[str]) -> None:
        """Register an agent type"""
        self.agent_types[agent_type] = {
            "module_path": module_path,
            "class_name": class_name,
            "capabilities": capabilities
        }
        
        logger.debug(f"Registered agent type: {agent_type}")
    
    def register_communication_protocol(self, protocol_id: str, 
                                       module_path: str, class_name: str) -> None:
        """Register a communication protocol"""
        self.communication_protocols[protocol_id] = {
            "module_path": module_path,
            "class_name": class_name
        }
        
        logger.debug(f"Registered communication protocol: {protocol_id}")
    
    def get_component(self, component_id: str) -> Dict[str, Any]:
        """Get information about a component"""
        if component_id not in self.components:
            return {"error": f"Component {component_id} not found"}
        
        return self.components[component_id]
    
    def get_architecture(self, architecture_name: str) -> Optional[AgentArchitecture]:
        """Get an architecture by name"""
        return self.architectures.get(architecture_name)
    
    def get_agent_type(self, agent_type: str) -> Dict[str, Any]:
        """Get information about an agent type"""
        if agent_type not in self.agent_types:
            return {"error": f"Agent type {agent_type} not found"}
        
        return self.agent_types[agent_type]
    
    def load_component_class(self, component_id: str) -> Optional[Type]:
        """Dynamically load a component class"""
        if component_id not in self.components:
            logger.error(f"Component {component_id} not found")
            return None
        
        component_info = self.components[component_id]
        
        try:
            module = importlib.import_module(component_info["module_path"])
            component_class = getattr(module, component_info["class_name"])
            return component_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading component {component_id}: {e}")
            return None
    
    def load_agent_class(self, agent_type: str) -> Optional[Type]:
        """Dynamically load an agent class"""
        if agent_type not in self.agent_types:
            logger.error(f"Agent type {agent_type} not found")
            return None
        
        agent_info = self.agent_types[agent_type]
        
        try:
            module = importlib.import_module(agent_info["module_path"])
            agent_class = getattr(module, agent_info["class_name"])
            return agent_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Error loading agent type {agent_type}: {e}")
            return None

    def discover_and_register_components(self, base_dir: str = None) -> int:
        """
        Automatically discover and register components from the codebase.
        
        Returns the number of components registered.
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        registered_count = 0
        
        # Find Python files
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    
                    # Get module path
                    rel_path = os.path.relpath(file_path, os.path.dirname(base_dir))
                    module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                    
                    try:
                        # Try to import the module
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Look for classes that could be components
                            for name, obj in inspect.getmembers(module, inspect.isclass):
                                # Check if this is a component class (simplified check)
                                if hasattr(obj, '__module__') and obj.__module__ == module_name:
                                    # Determine component type from class name or inheritance
                                    component_type = None
                                    
                                    if 'Agent' in name:
                                        component_type = 'agent'
                                        
                                        # Register agent type with capabilities
                                        capabilities = []
                                        if hasattr(obj, 'capabilities'):
                                            capabilities = obj.capabilities
                                        elif hasattr(obj, 'get_capabilities') and callable(getattr(obj, 'get_capabilities')):
                                            # Try to call get_capabilities as a static method
                                            try:
                                                capabilities = obj.get_capabilities()
                                            except:
                                                pass
                                        
                                        self.register_agent_type(
                                            name,
                                            module_name,
                                            name,
                                            capabilities
                                        )
                                        
                                    elif 'Protocol' in name or 'Connector' in name or 'PubSub' in name:
                                        component_type = 'communication'
                                        
                                        # Register communication protocol
                                        self.register_communication_protocol(
                                            name,
                                            module_name,
                                            name
                                        )
                                    
                                    # Register as general component
                                    if component_type:
                                        self.register_component(
                                            name,
                                            component_type,
                                            module_name,
                                            name,
                                            obj.__doc__ or ""
                                        )
                                        registered_count += 1
                    
                    except (ImportError, AttributeError, ValueError) as e:
                        logger.debug(f"Could not process module {module_name}: {e}")
        
        return registered_count


class DynamicSystemFactory:
    """
    Factory for dynamically constructing multi-agent knowledge systems
    tailored to specific problems and requirements.
    """
    
    def __init__(self, component_registry: SystemComponentRegistry, 
                domain_registry: KnowledgeDomainRegistry):
        """Initialize the dynamic system factory"""
        self.component_registry = component_registry
        self.domain_registry = domain_registry
        self.system_instances = {}
        self.instance_configs = {}
        self.build_history = []
    
    async def analyze_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Analyze a problem statement to determine the best system configuration.
        
        Args:
            problem_statement: The problem to analyze
            
        Returns:
            System configuration recommendations
        """
        logger.info(f"Analyzing problem: {problem_statement[:100]}...")
        
        # Find relevant knowledge domains
        relevant_domains = self.domain_registry.find_domains_for_problem(problem_statement)
        
        # Determine complexity and characteristics
        complexity = self._assess_problem_complexity(problem_statement)
        characteristics = self._identify_problem_characteristics(problem_statement)
        
        # Select appropriate architecture
        architecture = self._select_architecture(complexity, characteristics)
        
        # Determine appropriate mode
        mode = self._determine_operation_mode(characteristics)
        
        # Generate configuration
        config = {
            "problem_statement": problem_statement,
            "architecture": architecture,
            "relevant_domains": relevant_domains[:10],  # Top 10 most relevant domains
            "complexity": complexity,
            "characteristics": characteristics,
            "operation_mode": mode,
            "timestamp": time.time()
        }
        
        logger.info(f"Problem analysis complete. Selected architecture: {architecture}, Mode: {mode}")
        
        return config
    
    def _assess_problem_complexity(self, problem_statement: str) -> Dict[str, Any]:
        """
        Assess the complexity of a problem.
        
        This is a simplified implementation - a real system would use more 
        sophisticated NLP techniques.
        """
        # Basic complexity indicators
        word_count = len(problem_statement.split())
        sentence_count = len(re.split(r'[.!?]', problem_statement))
        
        # Number of distinct concepts (simplified using unique words)
        unique_words = set(re.sub(r'[^\w\s]', '', problem_statement.lower()).split())
        concept_count = len(unique_words)
        
        # Interdisciplinary indicators
        interdisciplinary_terms = [
            "interdisciplinary", "cross-domain", "multi-faceted", "integrated",
            "holistic", "cross-functional", "intersection", "across", "between"
        ]
        
        interdisciplinary_score = sum(term in problem_statement.lower() for term in interdisciplinary_terms) / len(interdisciplinary_terms)
        
        # Calculate overall complexity
        length_complexity = min(1.0, word_count / 150)  # Normalize by typical problem length
        structural_complexity = min(1.0, concept_count / 50)  # Normalize by typical concept count
        
        overall_complexity = (length_complexity * 0.3) + (structural_complexity * 0.5) + (interdisciplinary_score * 0.2)
        
        return {
            "overall": overall_complexity,
            "length": length_complexity,
            "structural": structural_complexity,
            "interdisciplinary": interdisciplinary_score,
            "word_count": word_count,
            "concept_count": concept_count
        }
    
    def _identify_problem_characteristics(self, problem_statement: str) -> Dict[str, bool]:
        """
        Identify characteristics of the problem.
        
        This is a simplified implementation - a real system would use more
        sophisticated NLP techniques.
        """
        lower_problem = problem_statement.lower()
        
        # Define characteristic keywords
        characteristics = {
            "creative": ["creative", "novel", "innovative", "design", "invent", "new approach"],
            "analytical": ["analyze", "examine", "investigate", "assess", "evaluate"],
            "decision_making": ["decide", "choice", "select", "determine the best", "optimal"],
            "planning": ["plan", "strategy", "roadmap", "timeline", "schedule", "steps"],
            "prediction": ["predict", "forecast", "estimate", "future", "trend", "will happen"],
            "explanation": ["explain", "why", "reason", "cause", "understand", "clarify"],
            "optimization": ["optimize", "improve", "enhance", "efficiency", "maximize", "minimize"],
            "comparison": ["compare", "contrast", "difference", "versus", "pros and cons"],
            "ethical": ["ethical", "moral", "right", "wrong", "should", "ought", "fair"],
            "uncertain": ["uncertain", "unclear", "probability", "chance", "risk", "might", "could"],
            "technical": ["technical", "technology", "system", "process", "mechanism", "how to"],
            "social": ["social", "people", "community", "group", "society", "interaction"],
            "educational": ["learn", "teach", "education", "training", "skill", "knowledge"],
            "temporal": ["time", "duration", "period", "when", "history", "evolution", "future"]
        }
        
        # Check for each characteristic
        results = {}
        for char, keywords in characteristics.items():
            # Check if any keyword is present
            presence = any(keyword in lower_problem for keyword in keywords)
            results[char] = presence
        
        return results
    
    def _select_architecture(self, complexity: Dict[str, Any], 
                            characteristics: Dict[str, bool]) -> str:
        """Select the most appropriate architecture based on problem analysis"""
        # Get available architectures
        available_architectures = list(self.component_registry.architectures.keys())
        
        if not available_architectures:
            return "default"  # Fallback
        
        # For high complexity or creative problems, use more advanced architectures
        if complexity["overall"] > 0.7 or characteristics.get("creative", False):
            for arch_name in ["emergent_insights", "recursive_decomposition", "multi_modal"]:
                if arch_name in available_architectures:
                    return arch_name
        
        # For analytical or optimization problems, use specialized architectures
        if characteristics.get("analytical", False) or characteristics.get("optimization", False):
            for arch_name in ["analytical", "optimization_focused"]:
                if arch_name in available_architectures:
                    return arch_name
        
        # Default to collaborative architecture
        if "collaborative" in available_architectures:
            return "collaborative"
        
        # Fallback to first available
        return available_architectures[0]
    
    def _determine_operation_mode(self, characteristics: Dict[str, bool]) -> str:
        """Determine the best operation mode based on problem characteristics"""
        # Creative problems benefit from emergent mode
        if characteristics.get("creative", False) or characteristics.get("prediction", False):
            return "emergent"
        
        # Analytical, comparison, or decision-making problems benefit from competitive mode
        if characteristics.get("analytical", False) or characteristics.get("comparison", False) or characteristics.get("decision_making", False):
            return "competitive"
        
        # Default to collaborative mode
        return "collaborative"
    
    async def create_system(self, config: Dict[str, Any]) -> str:
        """
        Dynamically create a multi-agent knowledge system based on configuration.
        
        Args:
            config: System configuration
            
        Returns:
            System instance ID
        """
        architecture_name = config.get("architecture", "default")
        architecture = self.component_registry.get_architecture(architecture_name)
        
        if not architecture:
            logger.warning(f"Architecture {architecture_name} not found, using fallback")
            # Create a basic default architecture
            architecture = AgentArchitecture("default", "Default architecture")
            architecture.add_component("knowledge_dispatcher", "dispatcher", True)
            architecture.add_component("pubsub", "communication", True)
            architecture.add_component("coordinator", "agent", True)
        
        # Generate system ID
        system_id = f"system_{uuid.uuid4()}"
        
        # Create working directory for this system
        system_dir = Path(f"./system_instances/{system_id}")
        system_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = system_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create system instance
        try:
            # Import required modules
            from multi_kb_agent_system_extended import MultiKBAgentSystem
            from knowledge_base_dispatcher import KnowledgeBaseDispatcher
            from pubsub_service import PubSubService
            
            # Database path for this instance
            db_path = f"./knowledge/system_{system_id}.db"
            
            # Initialize system components based on architecture
            system = MultiKBAgentSystem(epistemic_db_path=db_path)
            
            # Store system instance
            self.system_instances[system_id] = {
                "instance": system,
                "created_at": time.time(),
                "status": "initialized"
            }
            
            # Store configuration
            self.instance_configs[system_id] = config
            
            # Record in build history
            self.build_history.append({
                "system_id": system_id,
                "created_at": time.time(),
                "config": config
            })
            
            logger.info(f"Created system instance {system_id} with architecture {architecture_name}")
            
            # Initialize the system
            await system.setup_domain_agents()
            
            # Update status
            self.system_instances[system_id]["status"] = "ready"
            
            return system_id
            
        except Exception as e:
            logger.error(f"Error creating system instance: {e}", exc_info=True)
            
            # Clean up in case of failure
            if system_id in self.system_instances:
                del self.system_instances[system_id]
            
            if system_id in self.instance_configs:
                del self.instance_configs[system_id]
            
            # Remove system directory
            shutil.rmtree(system_dir, ignore_errors=True)
            
            raise RuntimeError(f"Failed to create system instance: {e}")
    
    async def solve_problem_with_system(self, system_id: str, 
                                      problem_statement: Optional[str] = None,
                                      mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a problem using an existing system instance.
        
        Args:
            system_id: System instance ID
            problem_statement: Problem to solve (overrides the one in config)
            mode: Operation mode (overrides the one in config)
            
        Returns:
            Solution details
        """
        if system_id not in self.system_instances:
            raise ValueError(f"System instance {system_id} not found")
        
        system_info = self.system_instances[system_id]
        if system_info["status"] != "ready":
            raise ValueError(f"System instance {system_id} is not ready (status: {system_info['status']})")
        
        # Get system instance
        system = system_info["instance"]
        
        # Get configuration
        config = self.instance_configs[system_id]
        
        # Use provided problem statement or get from config
        if problem_statement is None:
            problem_statement = config.get("problem_statement", "")
            if not problem_statement:
                raise ValueError("No problem statement provided or found in configuration")
        
        # Use provided mode or get from config
        if mode is None:
            mode = config.get("operation_mode", "collaborative")
        
        logger.info(f"Solving problem with system {system_id} in {mode} mode")
        
        # Update status
        system_info["status"] = "solving"
        system_info["problem"] = problem_statement
        system_info["started_at"] = time.time()
        
        try:
            # Solve the problem
            solution = await system.solve_problem(problem_statement, mode=mode)
            
            # Update status
            system_info["status"] = "solved"
            system_info["completed_at"] = time.time()
            system_info["duration"] = system_info["completed_at"] - system_info["started_at"]
            
            # Update domain statistics
            for domain in solution.get("domains_utilized", []):
                self.domain_registry.update_domain_stats(domain, "solutions")
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving problem with system {system_id}: {e}", exc_info=True)
            
            # Update status
            system_info["status"] = "error"
            system_info["error"] = str(e)
            
            raise RuntimeError(f"Failed to solve problem with system {system_id}: {e}")
    
    async def create_system_and_solve(self, problem_statement: str, 
                                    mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a system tailored to a problem and solve it in one operation.
        
        Args:
            problem_statement: Problem to solve
            mode: Operation mode (optional)
            
        Returns:
            Solution details
        """
        # Analyze the problem
        config = await self.analyze_problem(problem_statement)
        
        # Override mode if provided
        if mode:
            config["operation_mode"] = mode
        
        # Create system
        system_id = await self.create_system(config)
        
        # Solve problem
        solution = await self.solve_problem_with_system(system_id, problem_statement)
        
        # Add system_id to solution
        solution["system_id"] = system_id
        
        return solution
    
    def get_system_instance(self, system_id: str) -> Dict[str, Any]:
        """Get information about a system instance"""
        if system_id not in self.system_instances:
            return {"error": f"System instance {system_id} not found"}
        
        system_info = self.system_instances[system_id].copy()
        
        # Remove actual instance object from result
        if "instance" in system_info:
            del system_info["instance"]
        
        # Add configuration
        if system_id in self.instance_configs:
            system_info["config"] = self.instance_configs[system_id]
        
        return system_info
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall statistics for the factory"""
        return {
            "systems_created": len(self.build_history),
            "active_systems": sum(1 for info in self.system_instances.values() if info["status"] in ["ready", "solving"]),
            "domains_used": len(self.domain_registry.domain_stats),
            "architectures_used": {},
            "average_solution_time": None
        }
    
    async def close_system(self, system_id: str) -> bool:
        """
        Close and clean up a system instance.
        
        Args:
            system_id: System instance ID
            
        Returns:
            True if successful, False otherwise
        """
        if system_id not in self.system_instances:
            logger.warning(f"System instance {system_id} not found for closing")
            return False
        
        try:
            # Get system instance
            system_info = self.system_instances[system_id]
            system = system_info["instance"]
            
            # Close the system
            system.close()
            
            # Update status
            system_info["status"] = "closed"
            system_info["closed_at"] = time.time()
            
            logger.info(f"Closed system instance {system_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing system instance {system_id}: {e}")
            
            # Update status
            self.system_instances[system_id]["status"] = "error"
            self.system_instances[system_id]["error"] = str(e)
            
            return False
    
    async def close_all_systems(self) -> Dict[str, bool]:
        """
        Close and clean up all system instances.
        
        Returns:
            Dictionary mapping system IDs to close status
        """
        results = {}
        
        for system_id in list(self.system_instances.keys()):
            results[system_id] = await self.close_system(system_id)
        
        return results


class DynamicFrameworkAPI:
    """API for interacting with the dynamic multi-KB framework"""
    
    def __init__(self):
        """Initialize the framework API"""
        # Create component registry
        self.component_registry = SystemComponentRegistry()
        
        # Create domain registry
        self.domain_registry = KnowledgeDomainRegistry()
        
        # Create system factory
        self.system_factory = DynamicSystemFactory(
            self.component_registry,
            self.domain_registry
        )
        
        # Register built-in components
        self._register_builtin_components()
        
        # Discover and register additional components
        discovered = self.component_registry.discover_and_register_components()
        logger.info(f"Discovered and registered {discovered} components")
    
    def _register_builtin_components(self):
        """Register built-in components and architectures"""
        # Register collaborative architecture
        collab_arch = AgentArchitecture(
            "collaborative",
            "Collaborative architecture with incremental problem solving"
        )
        collab_arch.add_component("kb_dispatcher", "dispatcher", True)
        collab_arch.add_component("pubsub", "communication", True)
        collab_arch.add_component("coordinator", "agent", True)
        collab_arch.add_component("researcher", "agent", True)
        collab_arch.add_component("architect", "agent", True)
        collab_arch.add_component("evaluator", "agent", True)
        
        collab_arch.add_connection("coordinator", "researcher", "delegates")
        collab_arch.add_connection("coordinator", "architect", "delegates")
        collab_arch.add_connection("researcher", "kb_dispatcher", "queries")
        collab_arch.add_connection("architect", "kb_dispatcher", "queries")
        collab_arch.add_connection("coordinator", "evaluator", "delegates")
        
        self.component_registry.register_architecture(collab_arch)
        
        # Register competitive architecture
        comp_arch = AgentArchitecture(
            "competitive",
            "Competitive architecture with solution voting"
        )
        comp_arch.add_component("kb_dispatcher", "dispatcher", True)
        comp_arch.add_component("pubsub", "communication", True)
        comp_arch.add_component("coordinator", "agent", True)
        comp_arch.add_component("domain_specialists", "agent_pool", True)
        comp_arch.add_component("consensus", "voting", True)
        comp_arch.add_component("evaluator", "agent", True)
        
        comp_arch.add_connection("coordinator", "domain_specialists", "assigns")
        comp_arch.add_connection("domain_specialists", "kb_dispatcher", "queries")
        comp_arch.add_connection("domain_specialists", "consensus", "proposes")
        comp_arch.add_connection("evaluator", "consensus", "votes")
        comp_arch.add_connection("coordinator", "consensus", "votes")
        
        self.component_registry.register_architecture(comp_arch)
        
        # Register emergent insights architecture
        emergent_arch = AgentArchitecture(
            "emergent_insights",
            "Emergent insights architecture for novel discoveries"
        )
        emergent_arch.add_component("kb_dispatcher", "dispatcher", True)
        emergent_arch.add_component("pubsub", "communication", True)
        emergent_arch.add_component("coordinator", "agent", True)
        emergent_arch.add_component("pattern_detector", "agent", True)
        emergent_arch.add_component("domain_specialists", "agent_pool", True)
        emergent_arch.add_component("integrator", "agent", True)
        
        emergent_arch.add_connection("coordinator", "domain_specialists", "assigns")
        emergent_arch.add_connection("domain_specialists", "kb_dispatcher", "queries")
        emergent_arch.add_connection("domain_specialists", "pattern_detector", "provides")
        emergent_arch.add_connection("pattern_detector", "integrator", "provides")
        emergent_arch.add_connection("integrator", "coordinator", "proposes")
        
        self.component_registry.register_architecture(emergent_arch)
    
    async def initialize(self):
        """Initialize the framework"""
        # Create system directories
        os.makedirs("./system_instances", exist_ok=True)
        os.makedirs("./knowledge", exist_ok=True)
        
        logger.info("Dynamic Multi-KB Framework initialized")
    
    async def solve_problem(self, problem_statement: str, 
                          mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a problem by dynamically creating an appropriate system.
        
        Args:
            problem_statement: Problem to solve
            mode: Override the operation mode
            
        Returns:
            Solution details
        """
        return await self.system_factory.create_system_and_solve(problem_statement, mode)
    
    async def analyze_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Analyze a problem without solving it.
        
        Args:
            problem_statement: Problem to analyze
            
        Returns:
            Analysis results
        """
        return await self.system_factory.analyze_problem(problem_statement)
    
    async def get_relevant_domains(self, problem_statement: str) -> List[Dict[str, Any]]:
        """
        Get domains relevant to a problem.
        
        Args:
            problem_statement: Problem to analyze
            
        Returns:
            List of relevant domains with scores
        """
        return self.domain_registry.find_domains_for_problem(problem_statement)
    
    async def shutdown(self):
        """Shut down the framework and clean up resources"""
        # Close all system instances
        await self.system_factory.close_all_systems()
        
        logger.info("Dynamic Multi-KB Framework shut down")


async def main():
    """Main function for demonstration"""
    # Initialize the framework
    framework = DynamicFrameworkAPI()
    await framework.initialize()
    
    try:
        # Demonstrative problems that exercise different system architectures
        problems = [
            # Creative problem (likely emergent architecture)
            "Design an innovative urban transportation system that combines renewable energy, "
            "autonomous vehicles, and shared mobility to reduce congestion and emissions while "
            "improving accessibility for all citizens.",
            
            # Analytical problem (likely competitive architecture)
            "Analyze the economic, environmental, and social impacts of transitioning from "
            "fossil fuels to renewable energy sources in developing countries, considering "
            "infrastructure requirements, job displacement, and international financing mechanisms.",
            
            # Planning problem (likely collaborative architecture)
            "Develop a comprehensive disaster response plan for a coastal city susceptible to "
            "hurricanes, flooding, and sea-level rise, addressing evacuation procedures, "
            "emergency services coordination, and long-term resilience strategies."
        ]
        
        # Solve each problem with automatic architecture selection
        for i, problem in enumerate(problems):
            print(f"\n{'='*80}")
            print(f"PROBLEM {i+1}:")
            print(problem)
            print(f"{'='*80}\n")
            
            print("Analyzing problem...")
            analysis = await framework.analyze_problem(problem)
            
            print(f"Recommended architecture: {analysis['architecture']}")
            print(f"Recommended mode: {analysis['operation_mode']}")
            print(f"Problem complexity: {analysis['complexity']['overall']:.2f}")
            
            print("\nRelevant knowledge domains:")
            for domain in analysis['relevant_domains'][:5]:
                print(f"- {domain['domain']} (relevance: {domain['relevance']:.2f})")
            
            print("\nSolving problem...")
            solution = await framework.solve_problem(problem)
            
            print(f"\n{'='*80}")
            print("SOLUTION:")
            print(solution["solution"])
            print(f"\nDomains utilized: {', '.join(solution['domains_utilized'])}")
            print(f"Resolution time: {solution.get('resolution_time', 0):.2f} seconds")
            print(f"{'='*80}\n")
    
    finally:
        # Shut down the framework
        await framework.shutdown()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dynamic Multi-Knowledge Base Framework")
    parser.add_argument("--problem", type=str, help="Problem to solve")
    parser.add_argument("--mode", type=str, choices=["collaborative", "competitive", "emergent"], 
                       help="Override operation mode")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze the problem without solving")
    
    args = parser.parse_args()
    
    if args.problem:
        async def process_problem():
            # Initialize the framework
            framework = DynamicFrameworkAPI()
            await framework.initialize()
            
            try:
                if args.analyze_only:
                    # Analyze the problem
                    analysis = await framework.analyze_problem(args.problem)
                    
                    print(f"\n{'='*80}")
                    print("PROBLEM ANALYSIS:")
                    print(f"Recommended architecture: {analysis['architecture']}")
                    print(f"Recommended mode: {analysis['operation_mode']}")
                    print(f"Problem complexity: {analysis['complexity']['overall']:.2f}")
                    
                    print("\nProblem characteristics:")
                    for char, present in analysis['characteristics'].items():
                        if present:
                            print(f"- {char}")
                    
                    print("\nRelevant knowledge domains:")
                    for domain in analysis['relevant_domains']:
                        print(f"- {domain['domain']} (relevance: {domain['relevance']:.2f})")
                    print(f"{'='*80}\n")
                    
                else:
                    # Solve the problem
                    solution = await framework.solve_problem(args.problem, args.mode)
                    
                    print(f"\n{'='*80}")
                    print("SOLUTION:")
                    print(solution["solution"])
                    print(f"\nDomains utilized: {', '.join(solution['domains_utilized'])}")
                    print(f"Resolution time: {solution.get('resolution_time', 0):.2f} seconds")
                    print(f"Operation mode: {solution.get('mode', 'unknown')}")
                    print(f"{'='*80}\n")
            
            finally:
                # Shut down the framework
                await framework.shutdown()
        
        asyncio.run(process_problem())
    else:
        # Run demonstration with multiple problems
        asyncio.run(main())