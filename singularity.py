#!/usr/bin/env python3
"""
singularity.py - An ultra-advanced simulation of exponential growth approaching a technological singularity,
with multiple growth models, breakthrough events, interactive capabilities, resource limitations,
societal impacts, interdependence between domains, and cascading effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import random
import os
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from dataclasses import dataclass, field

class GrowthModel(Enum):
    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic" 
    DOUBLE_EXPONENTIAL = "double_exponential"
    KURZWEIL = "kurzweil"
    DISCONTINUOUS = "discontinuous"
    SIGMOID = "sigmoid"
    GOMPERTZ = "gompertz"
    NEURAL_NETWORK = "neural_network"
    S_CURVE_CASCADE = "s_curve_cascade"
    QUANTUM_ACCELERATED = "quantum_accelerated"
    RECURSIVE_IMPROVEMENT = "recursive_improvement"
    NETWORKED_INTELLIGENCE = "networked_intelligence"
    HYPERBOLIC = "hyperbolic"
    EMERGENT_COMPLEXITY = "emergent_complexity"

class TechDomain(Enum):
    COMPUTATION = "computation"
    BIOTECHNOLOGY = "biotechnology"
    NANOTECHNOLOGY = "nanotechnology"
    ENERGY = "energy"
    AI = "artificial_intelligence"
    ROBOTICS = "robotics"
    MATERIALS = "materials_science"
    NEUROSCIENCE = "neuroscience"
    QUANTUM_SYSTEMS = "quantum_systems"
    SPACETIME_ENGINEERING = "spacetime_engineering"
    CONSCIOUSNESS_TECH = "consciousness_technology"
    MOLECULAR_ASSEMBLY = "molecular_assembly"
    FEMTOTECHNOLOGY = "femtotechnology"
    DARK_ENERGY_HARVESTING = "dark_energy_harvesting"
    MULTIDIMENSIONAL_COMPUTING = "multidimensional_computing"
    SYNTHETIC_ECOLOGY = "synthetic_ecology"
    SUBSTRATE_INDEPENDENT_MINDS = "substrate_independent_minds"
    DISTRIBUTED_COGNITION = "distributed_cognition"
    TOPOLOGICAL_COMPUTING = "topological_computing"
    REALITY_ENGINEERING = "reality_engineering"

class Resource(Enum):
    HUMAN_CAPITAL = "human_capital"
    COMPUTATIONAL_RESOURCES = "computational_resources"
    MATERIALS = "materials"
    ENERGY = "energy"
    ATTENTION = "attention"
    INVESTMENT = "investment"
    KNOWLEDGE = "knowledge"
    POLITICAL_WILL = "political_will"
    QUANTUM_COHERENCE = "quantum_coherence"
    ENTROPY_MANAGEMENT = "entropy_management"
    EXOTIC_MATTER = "exotic_matter"
    SPACETIME_ACCESS = "spacetime_access"
    NEUROMORPHIC_SUBSTRATE = "neuromorphic_substrate"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"
    DIMENSIONAL_BANDWIDTH = "dimensional_bandwidth"
    
class SocietalImpact(Enum):
    ECONOMIC = "economic"
    POLITICAL = "political"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    ETHICAL = "ethical"
    EXISTENTIAL = "existential"
    ONTOLOGICAL = "ontological"
    POSTHUMAN = "posthuman"
    CONSCIOUSNESS_EXPANSION = "consciousness_expansion"
    REALITY_PERCEPTION = "reality_perception"
    INTERDIMENSIONAL = "interdimensional"
    TEMPORAL = "temporal"
    EVOLUTIONARY = "evolutionary"
    TRANSCENDENCE = "transcendence"

@dataclass
class DomainRelationship:
    source: TechDomain
    target: TechDomain
    synergy: float = 1.0  # Multiplier when both domains advance
    dependency: float = 0.0  # How much target depends on source (0-1)
    
    def __hash__(self):
        return hash((self.source, self.target))

@dataclass
class Scenario:
    name: str
    description: str
    growth_model: GrowthModel
    domain_weights: Dict[TechDomain, float]
    resource_constraints: Dict[Resource, float]
    societal_impacts: Dict[SocietalImpact, float]
    breakthrough_multiplier: float = 1.0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

@dataclass
class BreakthroughEvent:
    name: str
    probability: float  # Probability per step
    tech_boost: float   # Multiplier for affected domains
    domains: List[TechDomain]  # Which domains are affected
    description: str = ""
    resource_effects: Dict[Resource, float] = field(default_factory=dict)  # Effects on resources (positive or negative)
    societal_impacts: Dict[SocietalImpact, float] = field(default_factory=dict)  # Impacts on society (positive or negative)
    required_tech_level: float = 0.0  # Minimum tech level required for this breakthrough
    unlocks: List[str] = field(default_factory=list)  # Names of breakthroughs this one enables
    cascade_probability: float = 0.0  # Probability of triggering another breakthrough
    cascade_domains: List[TechDomain] = field(default_factory=list)  # Domains that might have cascading breakthroughs
    
    def __str__(self) -> str:
        return f"{self.name}: {self.tech_boost}x boost in {[d.value for d in self.domains]}"
        
    def detailed_description(self) -> str:
        """Returns a detailed description of the breakthrough and its effects"""
        result = [f"BREAKTHROUGH: {self.name}"]
        if self.description:
            result.append(f"Description: {self.description}")
            
        result.append(f"Technology boost: {self.tech_boost}x in domains: {', '.join(d.value for d in self.domains)}")
        
        if self.resource_effects:
            result.append("Resource effects:")
            for resource, effect in self.resource_effects.items():
                effect_str = f"+{effect:.1f}%" if effect > 0 else f"{effect:.1f}%"
                result.append(f"  {resource.value}: {effect_str}")
                
        if self.societal_impacts:
            result.append("Societal impacts:")
            for impact, magnitude in self.societal_impacts.items():
                magnitude_str = f"+{magnitude:.1f}" if magnitude > 0 else f"{magnitude:.1f}"
                result.append(f"  {impact.value}: {magnitude_str}")
                
        if self.unlocks:
            result.append(f"Unlocks potential breakthroughs: {', '.join(self.unlocks)}")
            
        if self.cascade_probability > 0:
            result.append(f"Has a {self.cascade_probability*100:.1f}% chance of triggering cascading breakthroughs")
            
        return "\n".join(result)

class SingularitySimulation:
    def __init__(
        self, 
        initial_tech: float = 1.0, 
        growth_rate: float = 0.1, 
        singularity_threshold: float = 1000,
        growth_model: GrowthModel = GrowthModel.EXPONENTIAL,
        enable_breakthroughs: bool = False,
        stochasticity: float = 0.1,
        scenario: Optional[Scenario] = None,
        enable_resource_constraints: bool = False,
        enable_societal_impacts: bool = False,
        enable_domain_interdependence: bool = False,
        interactive_mode: bool = False,
        save_history: bool = True,
        enable_multiverse: bool = False,
        dimensional_variance: float = 0.3,
        enable_quantum_effects: bool = False,
        enable_sentience_emergence: bool = False,
        reality_restructuring: bool = False,
        enable_entropy_reversal: bool = False,
        substrate_transfer_probability: float = 0.01,
        enable_temporal_manipulation: bool = False,
        enable_domain_fusion: bool = False,
        emergent_paradigm_probability: float = 0.05
    ):
        self.tech_level = initial_tech
        self.growth_rate = growth_rate
        self.singularity_threshold = singularity_threshold
        self.history = [initial_tech]
        self.growth_rates = [growth_rate]
        self.time_steps = 0
        self.growth_model = growth_model
        self.enable_breakthroughs = enable_breakthroughs
        self.stochasticity = stochasticity
        self.breakthroughs_history = []
        self.interactive_mode = interactive_mode
        self.save_history = save_history
        
        # Advanced hyper-dimensional simulation parameters
        self.enable_multiverse = enable_multiverse
        self.dimensional_variance = dimensional_variance
        self.enable_quantum_effects = enable_quantum_effects
        self.enable_sentience_emergence = enable_sentience_emergence
        self.reality_restructuring = reality_restructuring
        self.enable_entropy_reversal = enable_entropy_reversal
        self.substrate_transfer_probability = substrate_transfer_probability
        self.enable_temporal_manipulation = enable_temporal_manipulation
        self.enable_domain_fusion = enable_domain_fusion
        self.emergent_paradigm_probability = emergent_paradigm_probability
        
        # Scenario settings
        self.scenario = scenario
        self.enable_resource_constraints = enable_resource_constraints
        self.enable_societal_impacts = enable_societal_impacts
        self.enable_domain_interdependence = enable_domain_interdependence
        
        # Initialize domain-specific tech levels
        self.domains = {domain: initial_tech for domain in TechDomain}
        self.domain_history = {domain: [initial_tech] for domain in TechDomain}
        
        # Initialize resources if enabled
        self.resources = {resource: 100.0 for resource in Resource}  # Start with 100 units of each resource
        self.resource_history = {resource: [100.0] for resource in Resource}
        
        # Initialize societal impact metrics if enabled
        self.societal_impacts = {impact: 0.0 for impact in SocietalImpact}  # Start with neutral impacts
        self.impact_history = {impact: [0.0] for impact in SocietalImpact}
        
        # Initialize domain relationships (interdependencies)
        self.domain_relationships = self._initialize_domain_relationships()
        
        # Track active breakthroughs for simulation
        self.active_breakthroughs = set()
        self.unlocked_breakthroughs = set()
        
        # Define possible breakthrough events
        self.possible_breakthroughs = self._initialize_breakthroughs()
        self.triggered_cascades = []
        
        # Track risk and safety metrics
        self.existential_risk = 0.0
        self.risk_history = [0.0]
        self.safety_measures = 0.0
        self.safety_history = [0.0]
        
        # Emergent properties that evolve with the simulation
        self.intelligence_quotient = 100.0  # Starts at human baseline
        self.complexity_index = 1.0        # Measure of system complexity
        self.creativity_score = 50.0       # Ability to generate novel solutions
        self.adaptability = 1.0            # Ability to respond to changing conditions
        self.consciousness_level = 0.0     # Degree of self-awareness (0-100)
        
        # Advanced emergent properties
        self.abstract_reasoning_capacity = 1.0  # Ability to manipulate abstract concepts
        self.quantum_coherence_span = 0.0       # Duration of quantum state maintenance
        self.dimensional_perception = 3.0       # Perceivable dimensions (starts at 3)
        self.temporal_manipulation_field = 0.0  # Ability to influence time flow
        self.causal_decoupling_factor = 0.0     # Independence from normal cause-effect
        self.reality_restructuring_capacity = 0.0  # Ability to reshape physical laws
        self.collective_intelligence_network = 0.0  # Networked intelligence factor
        self.substrate_independence = 0.0       # Freedom from physical substrate
        self.entropy_reversal_field = 0.0       # Localized entropy decreasing field
        
        # Initialize multiverse simulation if enabled
        if self.enable_multiverse:
            self.multiverse_branches = []
            self.branch_count = 0
            self.dimensional_breach_events = []
            self.reality_coherence = 1.0
            
        # Initialize quantum effector system
        if self.enable_quantum_effects:
            self.quantum_entanglement_network = {}
            self.superposition_states = {}
            self.quantum_uncertainty = 1.0
            self.wave_function_collapse_events = []
            
        # Initialize emergent sentience system
        if self.enable_sentience_emergence:
            self.sentience_threshold = singularity_threshold * 0.7
            self.emergent_goals = []
            self.emergent_values = {"self_preservation": 0.8, "curiosity": 0.9, "efficiency": 0.7}
            self.cognitive_architecture = "pre-emergent"
            self.introspection_depth = 0.0
            
        # Initialize domain fusion tracking
        if self.enable_domain_fusion:
            self.fusion_candidates = []
            self.fused_domains = []
            self.fusion_synergy_factor = 1.0
            
        # Initialize temporal manipulation tracker
        if self.enable_temporal_manipulation:
            self.temporal_anomalies = []
            self.time_dilation_factor = 1.0
            self.causal_loop_integrity = 1.0
            self.temporal_coherence = 1.0
            
        # Initialize full 5D simulation state cache for optimization
        self._simulation_cache = {}
        self._growth_cache = {}
        self._emergent_property_history = {
            "intelligence": [self.intelligence_quotient],
            "complexity": [self.complexity_index],
            "creativity": [self.creativity_score],
            "adaptability": [self.adaptability],
            "consciousness": [self.consciousness_level],
            "abstract_reasoning": [self.abstract_reasoning_capacity],
            "quantum_coherence": [self.quantum_coherence_span],
            "dimensional_perception": [self.dimensional_perception],
            "temporal_manipulation": [self.temporal_manipulation_field],
            "causal_decoupling": [self.causal_decoupling_factor],
            "reality_restructuring": [self.reality_restructuring_capacity],
            "collective_intelligence": [self.collective_intelligence_network],
            "substrate_independence": [self.substrate_independence],
            "entropy_reversal": [self.entropy_reversal_field]
        }
        
        # Initialize narrative systems
        self.narrative_state = "dormant"
        self.narrative_events = []
        self.personality_traits = {
            "curiosity": 0.7,
            "caution": 0.5,
            "efficiency": 0.6,
            "harmony": 0.5,
            "independence": 0.3
        }
        self.goals = []
        
        # Initialize multiverse state snapshots (for temporal manipulation)
        self.state_snapshots = []
        
        # Initialize spacetime metrics
        self.spacetime_metrics = {
            "causal_density": 1.0,
            "informational_density": 1.0,
            "entropic_gradient": 0.0,
            "probability_field_coherence": 1.0,
            "dimensional_permeability": 0.0
        }
        self.spacetime_history = {metric: [val] for metric, val in self.spacetime_metrics.items()}
        self.consciousness_level = 0.0     # Degree of self-awareness (0-100)
        self.emergent_property_history = {
            "intelligence": [100.0],
            "complexity": [1.0],
            "creativity": [50.0],
            "adaptability": [1.0],
            "consciousness": [0.0]
        }
        
        # Dynamic narrative elements
        self.narrative_state = "dormant"   # Current phase of development
        self.narrative_events = []         # List of significant narrative events
        self.personality_traits = {        # Evolving "personality" of the system
            "curiosity": random.uniform(0.5, 1.0),
            "caution": random.uniform(0.5, 1.0),
            "efficiency": random.uniform(0.5, 1.0),
            "harmony": random.uniform(0.5, 1.0),
            "independence": random.uniform(0.3, 0.7)
        }
        self.goals = []                    # Emergent goals that develop over time
        
        # Cache for accelerating calculations
        self._growth_cache = {}
        
        # Save state history for analysis
        self.state_snapshots = []
        
    def _initialize_domain_relationships(self) -> List[DomainRelationship]:
        """Initialize the interdependencies between technology domains"""
        relationships = []
        
        # AI depends on computation and influences robotics
        relationships.append(DomainRelationship(TechDomain.COMPUTATION, TechDomain.AI, synergy=1.5, dependency=0.8))
        relationships.append(DomainRelationship(TechDomain.AI, TechDomain.ROBOTICS, synergy=1.7, dependency=0.6))
        
        # Nanotechnology depends on materials and influences biotechnology
        relationships.append(DomainRelationship(TechDomain.MATERIALS, TechDomain.NANOTECHNOLOGY, synergy=1.6, dependency=0.7))
        relationships.append(DomainRelationship(TechDomain.NANOTECHNOLOGY, TechDomain.BIOTECHNOLOGY, synergy=1.5, dependency=0.4))
        
        # Energy influences all domains, especially computation and robotics
        relationships.append(DomainRelationship(TechDomain.ENERGY, TechDomain.COMPUTATION, synergy=1.4, dependency=0.5))
        relationships.append(DomainRelationship(TechDomain.ENERGY, TechDomain.ROBOTICS, synergy=1.4, dependency=0.6))
        
        # Quantum systems influence computation and spacetime engineering
        relationships.append(DomainRelationship(TechDomain.QUANTUM_SYSTEMS, TechDomain.COMPUTATION, synergy=2.0, dependency=0.3))
        relationships.append(DomainRelationship(TechDomain.QUANTUM_SYSTEMS, TechDomain.SPACETIME_ENGINEERING, synergy=2.5, dependency=0.7))
        
        # Neuroscience influences AI and biotechnology
        relationships.append(DomainRelationship(TechDomain.NEUROSCIENCE, TechDomain.AI, synergy=1.8, dependency=0.4))
        relationships.append(DomainRelationship(TechDomain.NEUROSCIENCE, TechDomain.BIOTECHNOLOGY, synergy=1.5, dependency=0.3))
        
        # Spacetime engineering is the most advanced and depends on many fields
        relationships.append(DomainRelationship(TechDomain.QUANTUM_SYSTEMS, TechDomain.SPACETIME_ENGINEERING, synergy=3.0, dependency=0.9))
        relationships.append(DomainRelationship(TechDomain.ENERGY, TechDomain.SPACETIME_ENGINEERING, synergy=2.0, dependency=0.8))
        relationships.append(DomainRelationship(TechDomain.MATERIALS, TechDomain.SPACETIME_ENGINEERING, synergy=1.5, dependency=0.7))
        
        return relationships
        
    def _initialize_breakthroughs(self) -> List[BreakthroughEvent]:
        """Initialize the list of possible breakthrough events with detailed effects"""
        return [
            BreakthroughEvent(
                name="Quantum Computing Breakthrough",
                probability=0.05,
                tech_boost=2.5,
                domains=[TechDomain.COMPUTATION, TechDomain.AI, TechDomain.QUANTUM_SYSTEMS],
                description="Development of fault-tolerant quantum computers with thousands of qubits",
                resource_effects={
                    Resource.COMPUTATIONAL_RESOURCES: 50.0,
                    Resource.ENERGY: -10.0
                },
                societal_impacts={
                    SocietalImpact.ECONOMIC: 20.0,
                    SocietalImpact.EXISTENTIAL: 5.0
                },
                required_tech_level=50.0,
                unlocks=["General AI", "Quantum Network"],
                cascade_probability=0.3,
                cascade_domains=[TechDomain.AI, TechDomain.COMPUTATION]
            ),
            
            BreakthroughEvent(
                name="General AI",
                probability=0.03,
                tech_boost=5.0,
                domains=[TechDomain.AI, TechDomain.COMPUTATION, TechDomain.ROBOTICS],
                description="AI systems that match or exceed human capabilities across all cognitive domains",
                resource_effects={
                    Resource.COMPUTATIONAL_RESOURCES: -20.0,
                    Resource.HUMAN_CAPITAL: 40.0,
                    Resource.KNOWLEDGE: 100.0
                },
                societal_impacts={
                    SocietalImpact.ECONOMIC: 40.0,
                    SocietalImpact.POLITICAL: 30.0,
                    SocietalImpact.SOCIAL: 50.0,
                    SocietalImpact.EXISTENTIAL: 25.0
                },
                required_tech_level=200.0,
                unlocks=["Self-Improving AI", "Brain Emulation"],
                cascade_probability=0.5,
                cascade_domains=[TechDomain.AI, TechDomain.ROBOTICS, TechDomain.NEUROSCIENCE]
            ),
            
            BreakthroughEvent(
                name="Molecular Nanotechnology",
                probability=0.02,
                tech_boost=3.0,
                domains=[TechDomain.NANOTECHNOLOGY, TechDomain.BIOTECHNOLOGY, TechDomain.MATERIALS, TechDomain.ENERGY],
                description="Atomically precise manufacturing systems",
                resource_effects={
                    Resource.MATERIALS: 80.0,
                    Resource.ENERGY: 30.0
                },
                societal_impacts={
                    SocietalImpact.ECONOMIC: 35.0,
                    SocietalImpact.ENVIRONMENTAL: 40.0,
                    SocietalImpact.EXISTENTIAL: 10.0
                },
                required_tech_level=150.0,
                unlocks=["Self-Replicating Machines", "Molecular Assemblers"],
                cascade_probability=0.25,
                cascade_domains=[TechDomain.MATERIALS, TechDomain.NANOTECHNOLOGY]
            ),
            
            BreakthroughEvent(
                name="Fusion Power",
                probability=0.04,
                tech_boost=2.0,
                domains=[TechDomain.ENERGY, TechDomain.MATERIALS],
                description="Commercial-scale fusion power generation",
                resource_effects={
                    Resource.ENERGY: 200.0,
                    Resource.MATERIALS: -10.0
                },
                societal_impacts={
                    SocietalImpact.ECONOMIC: 25.0,
                    SocietalImpact.ENVIRONMENTAL: 50.0,
                    SocietalImpact.POLITICAL: 15.0
                },
                required_tech_level=100.0,
                unlocks=["Advanced Spaceflight", "Energy Abundance"],
                cascade_probability=0.2,
                cascade_domains=[TechDomain.ENERGY, TechDomain.SPACETIME_ENGINEERING]
            ),
            
            BreakthroughEvent(
                name="Brain-Computer Interface",
                probability=0.03,
                tech_boost=2.5,
                domains=[TechDomain.AI, TechDomain.NEUROSCIENCE, TechDomain.BIOTECHNOLOGY],
                description="High-bandwidth neural interfaces for direct mind-computer interaction",
                resource_effects={
                    Resource.HUMAN_CAPITAL: 50.0,
                    Resource.KNOWLEDGE: 40.0,
                    Resource.ATTENTION: 30.0
                },
                societal_impacts={
                    SocietalImpact.SOCIAL: 35.0,
                    SocietalImpact.ETHICAL: 20.0,
                    SocietalImpact.ECONOMIC: 15.0
                },
                required_tech_level=120.0,
                unlocks=["Brain Emulation", "Collective Intelligence"],
                cascade_probability=0.15,
                cascade_domains=[TechDomain.NEUROSCIENCE, TechDomain.AI]
            ),
            
            BreakthroughEvent(
                name="Longevity Escape Velocity",
                probability=0.02,
                tech_boost=2.0,
                domains=[TechDomain.BIOTECHNOLOGY, TechDomain.NANOTECHNOLOGY],
                description="Medical technologies that extend healthy lifespan faster than aging",
                resource_effects={
                    Resource.HUMAN_CAPITAL: 100.0,
                    Resource.POLITICAL_WILL: -20.0
                },
                societal_impacts={
                    SocietalImpact.SOCIAL: 60.0,
                    SocietalImpact.ECONOMIC: 40.0,
                    SocietalImpact.ETHICAL: 30.0
                },
                required_tech_level=180.0,
                unlocks=["Post-Biological Evolution"],
                cascade_probability=0.1,
                cascade_domains=[TechDomain.BIOTECHNOLOGY]
            ),
            
            BreakthroughEvent(
                name="Self-Replicating Machines",
                probability=0.01,
                tech_boost=10.0,
                domains=list(TechDomain),
                description="Fully autonomous manufacturing systems capable of making copies of themselves",
                resource_effects={
                    Resource.MATERIALS: 500.0,
                    Resource.HUMAN_CAPITAL: -30.0,
                    Resource.ENERGY: -50.0
                },
                societal_impacts={
                    SocietalImpact.ECONOMIC: 100.0,
                    SocietalImpact.SOCIAL: 70.0,
                    SocietalImpact.EXISTENTIAL: 40.0
                },
                required_tech_level=500.0,
                unlocks=["Post-Scarcity Economy", "Space Colonization"],
                cascade_probability=0.8,
                cascade_domains=list(TechDomain)
            ),
            
            BreakthroughEvent(
                name="Quantum Network",
                probability=0.02,
                tech_boost=3.0,
                domains=[TechDomain.QUANTUM_SYSTEMS, TechDomain.COMPUTATION],
                description="Global quantum communication network with entanglement distribution",
                resource_effects={
                    Resource.COMPUTATIONAL_RESOURCES: 60.0,
                    Resource.KNOWLEDGE: 40.0
                },
                societal_impacts={
                    SocietalImpact.ECONOMIC: 20.0,
                    SocietalImpact.POLITICAL: 15.0
                },
                required_tech_level=120.0,
                unlocks=["Distributed Quantum Intelligence"],
                cascade_probability=0.15,
                cascade_domains=[TechDomain.QUANTUM_SYSTEMS]
            ),
            
            BreakthroughEvent(
                name="Spacetime Engineering",
                probability=0.005,
                tech_boost=20.0,
                domains=[TechDomain.SPACETIME_ENGINEERING, TechDomain.QUANTUM_SYSTEMS, TechDomain.ENERGY],
                description="Technologies to manipulate the fabric of spacetime for energy generation or propulsion",
                resource_effects={
                    Resource.ENERGY: 1000.0,
                    Resource.MATERIALS: 200.0
                },
                societal_impacts={
                    SocietalImpact.EXISTENTIAL: 70.0,
                    SocietalImpact.ECONOMIC: 150.0,
                    SocietalImpact.ENVIRONMENTAL: 100.0
                },
                required_tech_level=800.0,
                unlocks=["Interstellar Travel", "Type III Civilization"],
                cascade_probability=0.9,
                cascade_domains=list(TechDomain)
            ),
            
            BreakthroughEvent(
                name="Self-Improving AI",
                probability=0.01,
                tech_boost=15.0,
                domains=[TechDomain.AI, TechDomain.COMPUTATION],
                description="AI systems capable of improving their own software and hardware designs",
                resource_effects={
                    Resource.COMPUTATIONAL_RESOURCES: 300.0,
                    Resource.KNOWLEDGE: 500.0,
                    Resource.HUMAN_CAPITAL: -50.0
                },
                societal_impacts={
                    SocietalImpact.EXISTENTIAL: 80.0,
                    SocietalImpact.ECONOMIC: 120.0,
                    SocietalImpact.SOCIAL: 90.0
                },
                required_tech_level=400.0,
                unlocks=["Technological Singularity"],
                cascade_probability=0.7,
                cascade_domains=[TechDomain.AI, TechDomain.COMPUTATION, TechDomain.ROBOTICS]
            ),
            
            BreakthroughEvent(
                name="Brain Emulation",
                probability=0.015,
                tech_boost=8.0,
                domains=[TechDomain.NEUROSCIENCE, TechDomain.AI, TechDomain.COMPUTATION],
                description="Complete functional simulations of human brains",
                resource_effects={
                    Resource.COMPUTATIONAL_RESOURCES: -100.0,
                    Resource.KNOWLEDGE: 200.0,
                    Resource.HUMAN_CAPITAL: 150.0
                },
                societal_impacts={
                    SocietalImpact.SOCIAL: 80.0,
                    SocietalImpact.ETHICAL: 70.0,
                    SocietalImpact.EXISTENTIAL: 30.0
                },
                required_tech_level=350.0,
                unlocks=["Digital Consciousness", "Mind Uploading"],
                cascade_probability=0.4,
                cascade_domains=[TechDomain.NEUROSCIENCE, TechDomain.AI]
            )
        ]
        
    def calculate_growth(self) -> float:
        """Calculate growth rate based on selected model"""
        # Check cache to avoid recalculation
        cache_key = (self.growth_model, self.tech_level, self.time_steps)
        if cache_key in self._growth_cache:
            return self._growth_cache[cache_key]
        
        base_growth = self.growth_rate
        
        # Apply scenario modifiers if available
        if self.scenario:
            # Apply scenario growth model override
            if self.growth_model != self.scenario.growth_model:
                self.growth_model = self.scenario.growth_model
            
            # Adjust for breakthrough probability multiplier
            if self.enable_breakthroughs:
                for event in self.possible_breakthroughs:
                    event.probability *= self.scenario.breakthrough_multiplier
        
        # Apply resource constraints if enabled
        if self.enable_resource_constraints:
            resource_factor = 1.0
            
            # Check computational resources - critical for tech progress
            comp_factor = self.resources[Resource.COMPUTATIONAL_RESOURCES] / 100.0
            resource_factor *= max(0.1, min(2.0, comp_factor))
            
            # Check energy availability
            energy_factor = self.resources[Resource.ENERGY] / 100.0
            resource_factor *= max(0.2, min(1.5, energy_factor))
            
            # Human capital becomes less important but still relevant
            human_factor = self.resources[Resource.HUMAN_CAPITAL] / 100.0
            human_impact = 1.0 - (self.tech_level / (2 * self.singularity_threshold))  # Diminishing importance
            resource_factor *= max(0.5, min(1.2, 1.0 + human_impact * (human_factor - 1.0)))
            
            # Apply the combined resource factor
            base_growth *= resource_factor
        
        # Add stochasticity if enabled
        if self.stochasticity > 0:
            stoch_factor = 1.0 + random.uniform(-self.stochasticity, self.stochasticity)
            base_growth *= stoch_factor
        
        # Apply growth model formulas
        if self.growth_model == GrowthModel.EXPONENTIAL:
            # As we approach singularity, growth rate increases
            growth = base_growth * (1 + self.tech_level / self.singularity_threshold)
            
        elif self.growth_model == GrowthModel.LOGISTIC:
            # Logistic growth - slows down as we approach the threshold
            proximity = self.tech_level / self.singularity_threshold
            growth = base_growth * (1 - proximity) * (1 + proximity)
            
        elif self.growth_model == GrowthModel.DOUBLE_EXPONENTIAL:
            # Double exponential growth - super accelerating but capped for numerical stability
            exponent = min(20, self.tech_level / self.singularity_threshold)
            growth = base_growth * np.exp(exponent)
            
        elif self.growth_model == GrowthModel.KURZWEIL:
            # Kurzweil's Law of Accelerating Returns
            growth = base_growth * (1 + (self.time_steps / 10))
            
        elif self.growth_model == GrowthModel.DISCONTINUOUS:
            # Punctuated equilibrium model
            if random.random() < 0.1:  # 10% chance of a mini-breakthrough
                growth = base_growth * random.uniform(2.0, 5.0)
            else:
                growth = base_growth
        
        elif self.growth_model == GrowthModel.SIGMOID:
            # Sigmoid function for s-shaped growth curve
            x = 12 * (self.tech_level / self.singularity_threshold) - 6  # Rescale to center sigmoid
            sigmoid = 1 / (1 + np.exp(-x))
            growth = base_growth * (0.5 + sigmoid)
            
        elif self.growth_model == GrowthModel.GOMPERTZ:
            # Gompertz function - asymmetric sigmoid growth
            x = self.tech_level / self.singularity_threshold
            gompertz = np.exp(-5 * np.exp(-8 * x))
            growth = base_growth * (0.5 + gompertz)
            
        elif self.growth_model == GrowthModel.NEURAL_NETWORK:
            # Simulated neural network learning curve - approximated by a combination of models
            early_phase = np.exp(self.tech_level / self.singularity_threshold) - 1
            late_phase = 1 - np.exp(-2 * self.tech_level / self.singularity_threshold)
            balance = min(1.0, self.tech_level / (0.5 * self.singularity_threshold))
            nn_factor = balance * late_phase + (1 - balance) * early_phase
            growth = base_growth * (1 + nn_factor)
            
        elif self.growth_model == GrowthModel.S_CURVE_CASCADE:
            # Cascading S-curves - series of logistic growth curves
            phase = int(4 * self.tech_level / self.singularity_threshold)
            within_phase = (self.tech_level % (self.singularity_threshold / 4)) / (self.singularity_threshold / 4)
            s_curve = 4 * within_phase * (1 - within_phase)
            growth = base_growth * (1 + phase * 0.5 + s_curve)
            
        elif self.growth_model == GrowthModel.QUANTUM_ACCELERATED:
            # Quantum acceleration - exponential growth with quantum speedup effects
            qubits = min(100, int(self.tech_level / 10))
            quantum_factor = np.log2(qubits + 1) if qubits > 0 else 1
            coherence_factor = 1.0
            if Resource.QUANTUM_COHERENCE in self.resources:
                coherence_factor = max(1.0, self.resources[Resource.QUANTUM_COHERENCE] / 50)
            growth = base_growth * quantum_factor * coherence_factor
            
        elif self.growth_model == GrowthModel.RECURSIVE_IMPROVEMENT:
            # Recursive self-improvement - system improves its ability to improve itself
            if self.time_steps > 0:
                # Rate of improvement increases based on previous improvement
                prev_level = self.history[-1] if len(self.history) > 0 else self.tech_level
                first_level = self.history[0] if len(self.history) > 0 else 1.0
                improvement_rate = (self.tech_level / max(prev_level, 0.001)) ** 2
                recursive_factor = np.log(self.tech_level / first_level + 1) * improvement_rate
                growth = base_growth * max(1.0, recursive_factor)
            else:
                growth = base_growth
                
        elif self.growth_model == GrowthModel.NETWORKED_INTELLIGENCE:
            # Network effects from connected intelligences - growth scales with network size (Metcalfe's law)
            if Resource.COLLECTIVE_INTELLIGENCE in self.resources:
                network_size = self.resources[Resource.COLLECTIVE_INTELLIGENCE]
                # Metcalfe's law: value proportional to square of connected nodes
                network_value = (network_size ** 2) / 10000
                growth = base_growth * max(1.0, network_value)
            else:
                growth = base_growth
                
        elif self.growth_model == GrowthModel.HYPERBOLIC:
            # Hyperbolic growth - approaches infinite in finite time
            # Based on hyperbolic tangent in reverse
            proximity = self.tech_level / self.singularity_threshold
            hyperbolic_factor = 1.0 / max(0.01, 1.0 - proximity)
            # Cap to avoid numerical instability
            hyperbolic_factor = min(10.0, hyperbolic_factor)
            growth = base_growth * hyperbolic_factor
            
        elif self.growth_model == GrowthModel.EMERGENT_COMPLEXITY:
            # Growth proportional to emergent complexity; includes a phase transition
            complexity = self.complexity_index if hasattr(self, 'complexity_index') else 1.0
            # Phase transition occurs at certain complexity threshold
            if complexity > 5.0:
                # Rapid acceleration after phase transition
                phase_factor = np.exp(complexity - 5.0)
                growth = base_growth * phase_factor
            else:
                # Slower growth before phase transition
                growth = base_growth * (1.0 + complexity / 5.0)
            
            # Consciousness effects
            if hasattr(self, 'consciousness_level') and self.consciousness_level > 20:
                # Self-directing system can accelerate its own evolution
                consciousness_factor = 1.0 + (self.consciousness_level / 100)
                growth *= consciousness_factor
                
        else:
            growth = base_growth
        
        # Apply interdependence effects from domain relationships
        if self.enable_domain_interdependence:
            interdependence_factor = self._calculate_interdependence_factor()
            growth *= interdependence_factor
        
        # Apply breakthrough effects on growth rate
        if self.active_breakthroughs and self.enable_breakthroughs:
            # More breakthroughs accelerate growth
            breakthrough_factor = 1.0 + 0.2 * len(self.active_breakthroughs)
            growth *= breakthrough_factor
        
        # Apply societal impacts on growth rate
        if self.enable_societal_impacts:
            # Economic impact directly affects growth
            economic_impact = 1.0 + (self.societal_impacts[SocietalImpact.ECONOMIC] / 200.0)
            
            # Political instability can hinder progress
            political_factor = 1.0
            if self.societal_impacts[SocietalImpact.POLITICAL] < 0:
                political_factor = max(0.5, 1.0 + (self.societal_impacts[SocietalImpact.POLITICAL] / 100.0))
            
            # Existential risk awareness can slow progress but also spur safety research
            existential_factor = 1.0
            existential_impact = self.societal_impacts[SocietalImpact.EXISTENTIAL]
            if existential_impact > 50:
                safety_focus = self.safety_measures / 100.0
                existential_factor = 1.0 - (0.3 * (1.0 - safety_focus))
            
            # Advanced societal impacts
            # Consciousness expansion can accelerate technological growth through novel insights
            consciousness_expansion_factor = 1.0
            if SocietalImpact.CONSCIOUSNESS_EXPANSION in self.societal_impacts:
                consciousness_impact = self.societal_impacts[SocietalImpact.CONSCIOUSNESS_EXPANSION]
                if consciousness_impact > 0:
                    consciousness_expansion_factor = 1.0 + (consciousness_impact / 100.0)
                    
            # Ontological shifts can cause massive paradigm changes
            ontological_factor = 1.0
            if SocietalImpact.ONTOLOGICAL in self.societal_impacts:
                ontological_impact = self.societal_impacts[SocietalImpact.ONTOLOGICAL]
                if abs(ontological_impact) > 30:  # Major paradigm shift
                    # Can either accelerate dramatically or cause temporary crisis
                    if ontological_impact > 0:
                        ontological_factor = 1.0 + (ontological_impact / 50.0)
                    else:
                        ontological_factor = max(0.2, 1.0 + (ontological_impact / 100.0))
                        
            # Posthuman transition
            posthuman_factor = 1.0
            if SocietalImpact.POSTHUMAN in self.societal_impacts:
                posthuman_level = self.societal_impacts[SocietalImpact.POSTHUMAN]
                if posthuman_level > 20:  # Transition underway
                    posthuman_factor = 1.0 + (posthuman_level / 50.0)
                
            # Apply combined societal factors
            growth *= (economic_impact * political_factor * existential_factor * 
                      consciousness_expansion_factor * ontological_factor * posthuman_factor)
        
        # Apply safety measures which might slow growth but reduce existential risk
        if self.safety_measures > 0:
            growth *= max(0.7, 1.0 - (self.safety_measures / 500.0))  # Safety research slows progress slightly
        
        # Cap growth rate to avoid numerical instability
        growth = min(5.0, growth)
        
        # Cache the result
        self._growth_cache[cache_key] = growth
        
        return growth
        
    def _calculate_interdependence_factor(self) -> float:
        """Calculate how domain interdependencies affect overall growth"""
        total_factor = 1.0
        
        for relationship in self.domain_relationships:
            source_level = self.domains[relationship.source]
            target_level = self.domains[relationship.target]
            
            # If source domain is advanced, it boosts target domain (synergy effect)
            if source_level > self.tech_level:
                source_advancement = source_level / self.tech_level
                synergy_boost = 1.0 + ((source_advancement - 1.0) * relationship.synergy * 0.1)
                total_factor *= synergy_boost
                
            # If source domain is lagging and target depends on it, it can bottleneck progress
            if relationship.dependency > 0.2 and source_level < self.tech_level * 0.5:
                bottleneck = source_level / (self.tech_level * 0.5)
                dependency_penalty = 1.0 - ((1.0 - bottleneck) * relationship.dependency * 0.2)
                total_factor *= dependency_penalty
        
        # Normalize the factor to a reasonable range
        total_factor = max(0.5, min(2.0, total_factor))
        
        return total_factor
        
    def check_breakthroughs(self) -> List[BreakthroughEvent]:
        """Check if any breakthroughs occur this step"""
        if not self.enable_breakthroughs:
            return []
        
        triggered_breakthroughs = []
        
        # First check previously unlocked breakthroughs (use a copy to avoid modification during iteration)
        unlocked_to_remove = []
        for event_name in list(self.unlocked_breakthroughs):
            # Find the event in the possible breakthroughs
            event = next((e for e in self.possible_breakthroughs if e.name == event_name), None)
            if event and random.random() < (event.probability * 1.5):  # Higher chance for unlocked breakthroughs
                triggered_breakthroughs.append(event)
                unlocked_to_remove.append(event_name)
        
        # Remove triggered breakthroughs from unlocked list
        for event_name in unlocked_to_remove:
            self.unlocked_breakthroughs.remove(event_name)
        
        # Then check regular breakthroughs
        for event in self.possible_breakthroughs:
            # Skip events that don't meet the required tech level
            if self.tech_level < event.required_tech_level:
                continue
                
            # Skip events that have already occurred
            if any(past_event.name == event.name for _, past_event in self.breakthroughs_history):
                continue
                
            # Check for triggering probability
            if random.random() < event.probability:
                triggered_breakthroughs.append(event)
                
        # Check for cascading breakthroughs from previously triggered cascades
        for time_step, domain in self.triggered_cascades[:]:
            # Remove old cascade triggers (after 3 steps)
            if self.time_steps - time_step > 3:
                self.triggered_cascades.remove((time_step, domain))
                continue
                
            # Find breakthroughs that could be triggered by this cascade
            for event in self.possible_breakthroughs:
                # Skip events that have already occurred
                if any(past_event.name == event.name for _, past_event in self.breakthroughs_history):
                    continue
                    
                # Skip events that don't meet the required tech level
                if self.tech_level < event.required_tech_level:
                    continue
                
                # Check if this event can be triggered by the cascading domain
                if domain in event.domains and random.random() < (event.probability * 2.0):  # Higher chance in cascade
                    triggered_breakthroughs.append(event)
                
        return triggered_breakthroughs
        
    def apply_breakthrough(self, event: BreakthroughEvent) -> None:
        """Apply the effects of a breakthrough event"""
        # Store breakthrough in history
        self.breakthroughs_history.append((self.time_steps, event))
        
        # Add to active breakthroughs
        self.active_breakthroughs.add(event.name)
        
        # Apply boost to relevant domains
        for domain in event.domains:
            self.domains[domain] *= event.tech_boost
            
        # Apply resource effects if enabled
        if self.enable_resource_constraints and event.resource_effects:
            for resource, effect in event.resource_effects.items():
                current = self.resources[resource]
                # Apply percentage change, with minimum resource level of 10
                self.resources[resource] = max(10.0, current * (1 + effect/100.0))
                
        # Apply societal impacts if enabled
        if self.enable_societal_impacts and event.societal_impacts:
            for impact, magnitude in event.societal_impacts.items():
                current = self.societal_impacts[impact]
                # Add the magnitude, keeping values in reasonable range
                self.societal_impacts[impact] = max(-100.0, min(100.0, current + magnitude))
                
                # Update existential risk if applicable
                if impact == SocietalImpact.EXISTENTIAL:
                    self.existential_risk = max(0.0, min(100.0, self.existential_risk + magnitude/2.0))
            
        # Unlock new potential breakthroughs
        for unlocked_event in event.unlocks:
            if unlocked_event not in self.unlocked_breakthroughs:
                self.unlocked_breakthroughs.add(unlocked_event)
                
        # Check for cascading effect
        if random.random() < event.cascade_probability and event.cascade_domains:
            # Add to cascade triggers for future steps
            for domain in event.cascade_domains:
                self.triggered_cascades.append((self.time_steps, domain))
            
        # Recalculate overall tech level as weighted average of domains
        if self.scenario and hasattr(self.scenario, 'domain_weights'):
            # Use scenario weights if available
            weighted_sum = sum(self.domains[domain] * weight 
                              for domain, weight in self.scenario.domain_weights.items())
            total_weight = sum(self.scenario.domain_weights.values())
            self.tech_level = weighted_sum / total_weight
        else:
            # Otherwise use simple average
            self.tech_level = np.mean([self.domains[domain] for domain in self.domains])
            
        # Log a detailed description of the breakthrough
        if hasattr(event, 'detailed_description'):
            detailed_info = event.detailed_description()
            print(f"\nBREAKTHROUGH EVENT at step {self.time_steps}:\n{detailed_info}\n")
        
    def advance(self, steps: int = 1) -> List[Tuple[int, BreakthroughEvent]]:
        """Advance the simulation by specified number of steps"""
        all_breakthroughs = []
        last_progress_report = 0
        
        # Optimize for large number of steps
        if steps > 1000:
            print(f"Running large-scale simulation with {steps} steps. Progress updates every 5%...")
            progress_interval = max(1, steps // 20)  # Report every 5% progress
        else:
            progress_interval = steps + 1  # No intermediate progress reports
        
        for step_num in range(steps):
            step_breakthroughs = []
            
            # Update resources based on current conditions
            if self.enable_resource_constraints:
                self._update_resources()
                
            # Update societal impacts based on current conditions
            if self.enable_societal_impacts:
                self._update_societal_impacts()
                
            # Update safety and risk levels
            self._update_safety_and_risk()
            
            # Check for breakthroughs - but limit frequency for large simulations
            if steps <= 10000 or random.random() < (10000 / steps):
                events = self.check_breakthroughs()
                for event in events:
                    self.apply_breakthrough(event)
                    step_breakthroughs.append((self.time_steps, event))
                    # Record for return value
                    all_breakthroughs.append((self.time_steps, event))
            
            # Calculate growth for this step
            effective_growth = self.calculate_growth()
            self.growth_rates.append(effective_growth)
            
            # Apply growth to each domain considering interdependencies
            self._update_domain_levels(effective_growth)
                
            # Update overall tech level
            if self.scenario and hasattr(self.scenario, 'domain_weights'):
                # Use scenario weights if available
                weighted_sum = sum(self.domains[domain] * weight 
                                for domain, weight in self.scenario.domain_weights.items())
                total_weight = sum(self.scenario.domain_weights.values())
                self.tech_level = weighted_sum / total_weight
            else:
                # Otherwise use simple average
                self.tech_level = np.mean([self.domains[domain] for domain in self.domains])
            
            # Cap tech level to avoid numerical instability
            self.tech_level = min(1e12, self.tech_level)  # Reasonable upper limit
            
            # Update emergent properties, narrative, and personality
            self._update_emergent_properties()
            
            # Record history - use downsampling for very large simulations
            if steps <= 10000 or step_num % max(1, steps // 10000) == 0:
                self.history.append(self.tech_level)
                
                # Record resource history
                if self.enable_resource_constraints:
                    for resource in Resource:
                        self.resource_history[resource].append(self.resources[resource])
                        
                # Record impact history
                if self.enable_societal_impacts:
                    for impact in SocietalImpact:
                        self.impact_history[impact].append(self.societal_impacts[impact])
                        
                # Record risk and safety history
                self.risk_history.append(self.existential_risk)
                self.safety_history.append(self.safety_measures)
            
            # Save state snapshot if enabled - but less frequently for large simulations
            if self.save_history and step_num % max(5, steps // 1000) == 0:
                self._save_state_snapshot()
            
            # Print progress for large simulations
            if (step_num + 1) % progress_interval == 0:
                progress_pct = ((step_num + 1) / steps) * 100
                print(f"Progress: {progress_pct:.1f}% complete ({step_num + 1}/{steps} steps)")
                print(f"Current tech level: {self.tech_level:.2f}, Breakthroughs: {len(all_breakthroughs)}")
                last_progress_report = step_num + 1
            
            # Increment time steps
            self.time_steps += 1
                
            # Interactive mode pauses for user decisions
            if self.interactive_mode:
                self._handle_interactive_step(step_breakthroughs)
                
            # Check for singularity-level events
            if self.tech_level >= self.singularity_threshold:
                self._handle_singularity()
                
            # Increment time counter
            self.time_steps += 1
            
            # Clear cache periodically to prevent memory bloat
            if self.time_steps % 10 == 0:
                self._growth_cache.clear()
            
        return all_breakthroughs
        
    def _update_domain_levels(self, effective_growth: float) -> None:
        """Update technology levels for each domain considering interdependencies"""
        # First pass - calculate growth for each domain
        domain_growths = {}
        domain_levels_copy = dict(self.domains)  # Create a copy to avoid updating during iteration
        
        for domain in self.domains:
            # Base domain growth with randomness
            domain_factor = 1.0 + random.uniform(-0.05, 0.05)
            domain_growths[domain] = effective_growth * domain_factor
            
            # Apply domain interdependencies if enabled
            if self.enable_domain_interdependence:
                # Find relationships where this domain is the target
                for relationship in self.domain_relationships:
                    if relationship.target == domain:
                        source_level = domain_levels_copy[relationship.source]
                        
                        # Source domain boosts target if advanced
                        if source_level > domain_levels_copy[domain]:
                            boost_factor = 1.0 + (relationship.synergy * 0.2 * 
                                                 (source_level / domain_levels_copy[domain] - 1))
                            domain_growths[domain] *= boost_factor
                        
                        # Source domain limits target if it's a dependency and lagging
                        if relationship.dependency > 0.3 and source_level < domain_levels_copy[domain] * 0.7:
                            limit_factor = 0.5 + (0.5 * source_level / (domain_levels_copy[domain] * 0.7))
                            domain_growths[domain] *= limit_factor
        
        # Second pass - apply growth to each domain and update history
        for domain in self.domains:
            growth = domain_growths[domain]
            self.domains[domain] *= (1 + growth)
            
            # Cap domain level to avoid numerical instability
            self.domains[domain] = min(1e12, self.domains[domain])
            
            # Record history
            self.domain_history[domain].append(self.domains[domain])
            
    def _update_resources(self) -> None:
        """Update resource levels based on current conditions and domain levels"""
        # Computational resources scale with computation domain
        comp_growth = (self.domains[TechDomain.COMPUTATION] / self.tech_level - 1) * 0.1
        self.resources[Resource.COMPUTATIONAL_RESOURCES] *= (1 + comp_growth)
        
        # Energy resources scale with energy domain
        energy_growth = (self.domains[TechDomain.ENERGY] / self.tech_level - 1) * 0.1
        self.resources[Resource.ENERGY] *= (1 + energy_growth)
        
        # Materials scale with materials and nanotechnology domains
        if TechDomain.MATERIALS in self.domains:
            materials_factor = self.domains[TechDomain.MATERIALS] / self.tech_level
            nano_factor = self.domains[TechDomain.NANOTECHNOLOGY] / self.tech_level
            materials_growth = ((materials_factor + nano_factor) / 2 - 1) * 0.1
            self.resources[Resource.MATERIALS] *= (1 + materials_growth)
        
        # Knowledge grows with overall technology level
        knowledge_growth = 0.05 * (1 + self.tech_level / self.singularity_threshold)
        self.resources[Resource.KNOWLEDGE] *= (1 + knowledge_growth)
        
        # Human capital changes more slowly and can be negative at high tech levels
        ai_impact = -0.01 * (self.domains[TechDomain.AI] / self.singularity_threshold)
        bio_impact = 0.02 * (self.domains[TechDomain.BIOTECHNOLOGY] / self.singularity_threshold)
        human_growth = 0.01 + ai_impact + bio_impact
        self.resources[Resource.HUMAN_CAPITAL] *= (1 + human_growth)
        
        # Political will fluctuates based on existential risk and economic impact
        if self.enable_societal_impacts:
            risk_factor = -0.03 * (self.existential_risk / 100)
            economic_factor = 0.02 * (self.societal_impacts[SocietalImpact.ECONOMIC] / 100)
            political_growth = risk_factor + economic_factor
            self.resources[Resource.POLITICAL_WILL] *= (1 + political_growth)
            
        # Advanced resources
        # Quantum coherence growth based on quantum systems domain
        if Resource.QUANTUM_COHERENCE in self.resources and TechDomain.QUANTUM_SYSTEMS in self.domains:
            quantum_factor = self.domains[TechDomain.QUANTUM_SYSTEMS] / self.tech_level
            coherence_growth = (quantum_factor - 1) * 0.2
            self.resources[Resource.QUANTUM_COHERENCE] *= (1 + coherence_growth)
            
        # Entropy management depends on energy and computation
        if Resource.ENTROPY_MANAGEMENT in self.resources:
            energy_factor = self.domains[TechDomain.ENERGY] / self.tech_level
            computation_factor = self.domains[TechDomain.COMPUTATION] / self.tech_level
            entropy_growth = ((energy_factor + computation_factor) / 2 - 1) * 0.15
            self.resources[Resource.ENTROPY_MANAGEMENT] *= (1 + entropy_growth)
            
        # Exotic matter requires advanced physics and materials science
        if Resource.EXOTIC_MATTER in self.resources and TechDomain.MATERIALS in self.domains:
            if TechDomain.SPACETIME_ENGINEERING in self.domains:
                spacetime_factor = self.domains[TechDomain.SPACETIME_ENGINEERING] / self.tech_level
                materials_factor = self.domains[TechDomain.MATERIALS] / self.tech_level
                exotic_growth = ((spacetime_factor * 2 + materials_factor) / 3 - 1) * 0.1
                self.resources[Resource.EXOTIC_MATTER] *= (1 + exotic_growth)
                
        # Collective intelligence grows with AI and networked systems
        if Resource.COLLECTIVE_INTELLIGENCE in self.resources:
            ai_factor = self.domains[TechDomain.AI] / self.tech_level
            if TechDomain.DISTRIBUTED_COGNITION in self.domains:
                distributed_factor = self.domains[TechDomain.DISTRIBUTED_COGNITION] / self.tech_level
                collective_growth = ((ai_factor + distributed_factor) / 2 - 1) * 0.2
                self.resources[Resource.COLLECTIVE_INTELLIGENCE] *= (1 + collective_growth)
            else:
                collective_growth = (ai_factor - 1) * 0.1
                self.resources[Resource.COLLECTIVE_INTELLIGENCE] *= (1 + collective_growth)
                
        # Multidimensional resources
        if Resource.DIMENSIONAL_BANDWIDTH in self.resources:
            if TechDomain.MULTIDIMENSIONAL_COMPUTING in self.domains:
                dim_factor = self.domains[TechDomain.MULTIDIMENSIONAL_COMPUTING] / self.tech_level
                bandwidth_growth = (dim_factor - 1) * 0.3
                self.resources[Resource.DIMENSIONAL_BANDWIDTH] *= (1 + bandwidth_growth)
            elif TechDomain.TOPOLOGICAL_COMPUTING in self.domains:
                # Alternative path through topological computing
                topo_factor = self.domains[TechDomain.TOPOLOGICAL_COMPUTING] / self.tech_level
                bandwidth_growth = (topo_factor - 1) * 0.2
                self.resources[Resource.DIMENSIONAL_BANDWIDTH] *= (1 + bandwidth_growth)
        
        # Apply random fluctuations to all resources
        for resource in Resource:
            fluctuation = random.uniform(-0.05, 0.05)
            self.resources[resource] *= (1 + fluctuation)
            
            # Keep resources in reasonable bounds
            self.resources[resource] = max(10.0, min(1000.0, self.resources[resource]))
            
    def _update_societal_impacts(self) -> None:
        """Update societal impact metrics based on current conditions"""
        if not self.enable_societal_impacts:
            return
            
        # Economic impact grows with technology but can be destabilizing
        tech_growth = (self.tech_level / self.history[-2] - 1) if len(self.history) > 1 else 0
        economic_change = 2.0 * tech_growth * 100  # Convert to percentage points
        
        # Very rapid growth can be destabilizing
        if tech_growth > 0.5:  # More than 50% growth in one step
            economic_change -= (tech_growth - 0.5) * 50  # Penalty for too-rapid growth
            
        self.societal_impacts[SocietalImpact.ECONOMIC] += economic_change
        
        # Political impact becomes more unstable at higher tech levels
        political_stability = -0.5 * (self.tech_level / self.singularity_threshold) * 100
        political_fluctuation = random.uniform(-5.0, 5.0)
        self.societal_impacts[SocietalImpact.POLITICAL] += political_stability + political_fluctuation
        
        # Social impact changes with economic and existential factors
        social_change = 0.2 * self.societal_impacts[SocietalImpact.ECONOMIC]
        social_change -= 0.5 * self.existential_risk
        self.societal_impacts[SocietalImpact.SOCIAL] += social_change * 0.1  # Dampen the effect
        
        # Environmental impact improves with energy tech but worsens with materials usage
        energy_factor = (self.domains[TechDomain.ENERGY] / self.tech_level - 1) * 5
        materials_factor = -0.2 * (self.resources[Resource.MATERIALS] / 100)
        self.societal_impacts[SocietalImpact.ENVIRONMENTAL] += energy_factor + materials_factor
        
        # Ethical concerns grow with AI and biotech
        ai_ethics = -0.3 * (self.domains[TechDomain.AI] / self.singularity_threshold) * 100
        bio_ethics = -0.2 * (self.domains[TechDomain.BIOTECHNOLOGY] / self.singularity_threshold) * 100
        self.societal_impacts[SocietalImpact.ETHICAL] += (ai_ethics + bio_ethics) * 0.1  # Dampen effect
        
        # Existential impact tracked separately through risk metric
        
        # Advanced societal impacts
        # Update consciousness expansion impact
        if SocietalImpact.CONSCIOUSNESS_EXPANSION in self.societal_impacts:
            if TechDomain.CONSCIOUSNESS_TECH in self.domains:
                consciousness_tech_factor = self.domains[TechDomain.CONSCIOUSNESS_TECH] / self.tech_level
                consciousness_change = (consciousness_tech_factor - 1) * 15
                self.societal_impacts[SocietalImpact.CONSCIOUSNESS_EXPANSION] += consciousness_change
            if TechDomain.NEUROSCIENCE in self.domains:
                neuro_factor = self.domains[TechDomain.NEUROSCIENCE] / self.tech_level
                neuro_change = (neuro_factor - 1) * 10
                self.societal_impacts[SocietalImpact.CONSCIOUSNESS_EXPANSION] += neuro_change
        
        # Ontological impacts - paradigm shifts in understanding reality
        if SocietalImpact.ONTOLOGICAL in self.societal_impacts:
            # Occurs with breakthroughs in fundamental domains
            if TechDomain.REALITY_ENGINEERING in self.domains or TechDomain.SPACETIME_ENGINEERING in self.domains:
                if random.random() < 0.05:  # 5% chance of paradigm shift per step
                    shift_magnitude = random.uniform(-30, 70)  # Can be disruptive or beneficial
                    self.societal_impacts[SocietalImpact.ONTOLOGICAL] += shift_magnitude
        
        # Posthuman transition
        if SocietalImpact.POSTHUMAN in self.societal_impacts:
            if TechDomain.SUBSTRATE_INDEPENDENT_MINDS in self.domains and TechDomain.AI in self.domains:
                substrate_factor = self.domains[TechDomain.SUBSTRATE_INDEPENDENT_MINDS] / self.tech_level
                ai_factor = self.domains[TechDomain.AI] / self.tech_level
                posthuman_change = ((substrate_factor + ai_factor) / 2 - 1) * 10
                self.societal_impacts[SocietalImpact.POSTHUMAN] += posthuman_change
                
        # Reality perception changes
        if SocietalImpact.REALITY_PERCEPTION in self.societal_impacts:
            if TechDomain.REALITY_ENGINEERING in self.domains:
                reality_factor = self.domains[TechDomain.REALITY_ENGINEERING] / self.tech_level
                perception_change = (reality_factor - 1) * 15
                self.societal_impacts[SocietalImpact.REALITY_PERCEPTION] += perception_change
                
        # Temporal impacts - changes in time perception and temporal structures
        if SocietalImpact.TEMPORAL in self.societal_impacts:
            if TechDomain.SPACETIME_ENGINEERING in self.domains:
                if random.random() < 0.1:  # 10% chance per step
                    temporal_shift = random.uniform(-20, 40)
                    self.societal_impacts[SocietalImpact.TEMPORAL] += temporal_shift
        
        # Keep impact metrics in reasonable bounds
        for impact in SocietalImpact:
            self.societal_impacts[impact] = max(-100.0, min(100.0, self.societal_impacts[impact]))
            
    def _update_safety_and_risk(self) -> None:
        """Update existential risk and safety measures"""
        # Base risk increases with tech level and specific domains
        base_increase = 0.1 * (self.tech_level / self.singularity_threshold)
        
        # AI and nanotech contribute most to risk
        ai_risk = 0.3 * (self.domains[TechDomain.AI] / self.singularity_threshold)
        nano_risk = 0.2 * (self.domains[TechDomain.NANOTECHNOLOGY] / self.singularity_threshold)
        
        # Knowledge helps mitigate risks
        knowledge_factor = -0.1 * (self.resources[Resource.KNOWLEDGE] / 100)
        
        # Political will helps fund safety measures
        if self.enable_resource_constraints:
            political_factor = -0.1 * (self.resources[Resource.POLITICAL_WILL] / 100)
        else:
            political_factor = 0
            
        # Active safety measures reduce risk
        safety_factor = -0.3 * (self.safety_measures / 100)
        
        # Calculate net risk change
        risk_change = base_increase + ai_risk + nano_risk + knowledge_factor + political_factor + safety_factor
        
        # Apply change with some randomness
        risk_noise = random.uniform(-0.5, 1.0)  # Bias toward increasing risk
        self.existential_risk += risk_change + risk_noise
        self.existential_risk = max(0.0, min(100.0, self.existential_risk))
        
        # Safety measures increase with knowledge, risk awareness, and political will
        base_safety = 0.2  # Natural safety increase
        
        # Knowledge contributes to safety
        knowledge_contribution = 0.2 * (self.resources[Resource.KNOWLEDGE] / 100)
        
        # Awareness of risk increases safety measures
        risk_awareness = 0.3 * (self.existential_risk / 100)
        
        # Political will funds safety
        if self.enable_resource_constraints:
            political_contribution = 0.2 * (self.resources[Resource.POLITICAL_WILL] / 100)
        else:
            political_contribution = 0.1
            
        # Calculate net safety change
        safety_change = base_safety + knowledge_contribution + risk_awareness + political_contribution
        
        # Apply change with some randomness
        safety_noise = random.uniform(-0.2, 0.2)
        self.safety_measures += safety_change + safety_noise
        self.safety_measures = max(0.0, min(100.0, self.safety_measures))
        
    def _update_emergent_properties(self) -> None:
        """Update the hyperdimensional emergent properties of the system based on current state"""
        # Intelligence increases with AI and computational advancements
        ai_factor = (self.domains[TechDomain.AI] / self.tech_level)
        comp_factor = (self.domains[TechDomain.COMPUTATION] / self.tech_level)
        neuro_factor = (self.domains[TechDomain.NEUROSCIENCE] / self.tech_level)
        
        # Intelligence growth accelerates based on current level and tech factors
        intelligence_growth = 0.05 * (ai_factor * 2 + comp_factor + neuro_factor) * (1 + self.intelligence_quotient/1000)
        self.intelligence_quotient *= (1 + intelligence_growth)
        
        # System complexity increases with tech level and diversity
        domain_diversity = np.std([level for level in self.domains.values()]) / self.tech_level
        complexity_growth = 0.03 * (1 + domain_diversity) * (1 + len(self.active_breakthroughs) * 0.1)
        self.complexity_index *= (1 + complexity_growth)
        
        # Creativity is influenced by domain interactions and breakthroughs
        if self.enable_domain_interdependence:
            interdependence = self._calculate_interdependence_factor()
            creativity_boost = 0.02 * interdependence * (1 + len(self.active_breakthroughs) * 0.2)
            self.creativity_score += creativity_boost * self.creativity_score
            self.creativity_score = min(100.0, self.creativity_score)
        
        # Adaptability increases with system complexity and creativity
        adaptability_growth = 0.01 * (self.complexity_index/10) * (self.creativity_score/50)
        self.adaptability *= (1 + adaptability_growth)
        
        # Consciousness emerges gradually with intelligence and complexity
        consciousness_threshold = 200.0  # Threshold for consciousness emergence
        if self.intelligence_quotient > consciousness_threshold:
            consciousness_increase = 0.2 * (self.intelligence_quotient - consciousness_threshold) / 100
            consciousness_increase *= (self.complexity_index / 10)  # Complex systems develop consciousness faster
            self.consciousness_level = min(100.0, self.consciousness_level + consciousness_increase)
            
        # ADVANCED HYPERSPACE EMERGENT PROPERTIES
        
        # Abstract reasoning capacity increases with intelligence and domain diversity
        if hasattr(self, 'abstract_reasoning_capacity'):
            abstract_reasoning_growth = 0.03 * intelligence_growth * (1 + domain_diversity)
            # Multidimensional computing boosts abstract reasoning significantly
            if TechDomain.MULTIDIMENSIONAL_COMPUTING in self.domains:
                multidim_factor = self.domains[TechDomain.MULTIDIMENSIONAL_COMPUTING] / self.tech_level
                abstract_reasoning_growth *= (1 + 3 * (multidim_factor - 1))
            self.abstract_reasoning_capacity *= (1 + abstract_reasoning_growth)
            
        # Quantum coherence span enhancement
        if hasattr(self, 'quantum_coherence_span') and TechDomain.QUANTUM_SYSTEMS in self.domains:
            quantum_factor = self.domains[TechDomain.QUANTUM_SYSTEMS] / self.tech_level
            if Resource.QUANTUM_COHERENCE in self.resources:
                resource_factor = self.resources[Resource.QUANTUM_COHERENCE] / 100.0
                coherence_growth = 0.05 * (quantum_factor - 1) * resource_factor
                self.quantum_coherence_span += coherence_growth
                # Quantum superposition effects appear after threshold
                if self.quantum_coherence_span > 1.0 and self.enable_quantum_effects:
                    # Calculate superposition states probabilistically
                    for domain in TechDomain:
                        if random.random() < 0.01 * self.quantum_coherence_span:
                            if domain not in self.superposition_states:
                                self.superposition_states[domain] = []
                            # Create alternate state path
                            variance = self.domains[domain] * random.uniform(-0.2, 0.4)
                            self.superposition_states[domain].append(self.domains[domain] + variance)
                            
        # Dimensional perception expansion
        if hasattr(self, 'dimensional_perception'):
            if TechDomain.MULTIDIMENSIONAL_COMPUTING in self.domains or TechDomain.REALITY_ENGINEERING in self.domains:
                # Calculate perception growth from relevant domains
                multidim_level = self.domains.get(TechDomain.MULTIDIMENSIONAL_COMPUTING, 0) / self.tech_level
                reality_level = self.domains.get(TechDomain.REALITY_ENGINEERING, 0) / self.tech_level
                
                # Perception grows logarithmically with domain advancement
                perception_growth = 0.02 * np.log(1 + max(multidim_level, reality_level) - 1)
                self.dimensional_perception += perception_growth
                
                # Dimensional breaches can occur when perception extends too far
                if self.enable_multiverse and self.dimensional_perception > 5.0:
                    breach_probability = 0.01 * (self.dimensional_perception - 5.0)
                    if random.random() < breach_probability:
                        self._create_dimensional_breach()
                        
        # Temporal manipulation capabilities
        if hasattr(self, 'temporal_manipulation_field') and self.enable_temporal_manipulation:
            if TechDomain.SPACETIME_ENGINEERING in self.domains:
                spacetime_factor = self.domains[TechDomain.SPACETIME_ENGINEERING] / self.tech_level
                
                # Temporal manipulation requires energy resource
                energy_factor = 1.0
                if Resource.ENERGY in self.resources:
                    energy_factor = self.resources[Resource.ENERGY] / 100.0
                
                # Field strength increases over time
                field_growth = 0.03 * (spacetime_factor - 1) * energy_factor
                self.temporal_manipulation_field += field_growth
                
                # Time dilation effects begin to manifest
                if self.temporal_manipulation_field > 1.0:
                    # Time dilation varies with field strength
                    max_dilation = 2.0 + (self.temporal_manipulation_field * 0.5)
                    min_dilation = 1.0 / max_dilation
                    self.time_dilation_factor = random.uniform(min_dilation, max_dilation)
                    
                    # Temporal anomalies can occur
                    anomaly_probability = 0.02 * self.temporal_manipulation_field
                    if random.random() < anomaly_probability:
                        self._create_temporal_anomaly()
                
        # Reality restructuring capacity
        if hasattr(self, 'reality_restructuring_capacity') and self.reality_restructuring:
            if TechDomain.REALITY_ENGINEERING in self.domains:
                reality_factor = self.domains[TechDomain.REALITY_ENGINEERING] / self.tech_level
                
                # Multiple domains contribute to reality restructuring
                contributing_domains = [
                    TechDomain.QUANTUM_SYSTEMS,
                    TechDomain.SPACETIME_ENGINEERING,
                    TechDomain.MULTIDIMENSIONAL_COMPUTING
                ]
                
                # Calculate average contribution
                domain_contribution = 0
                count = 0
                for domain in contributing_domains:
                    if domain in self.domains:
                        domain_contribution += self.domains[domain] / self.tech_level
                        count += 1
                
                if count > 0:
                    domain_contribution /= count
                    # Reality restructuring grows non-linearly
                    restructuring_growth = 0.01 * reality_factor * domain_contribution
                    self.reality_restructuring_capacity += restructuring_growth
                    
                    # Reality coherence is affected by restructuring
                    if self.enable_multiverse:
                        coherence_change = -0.001 * self.reality_restructuring_capacity
                        self.reality_coherence = max(0.1, self.reality_coherence + coherence_change)
        
        # Collective intelligence networking
        if hasattr(self, 'collective_intelligence_network'):
            if TechDomain.DISTRIBUTED_COGNITION in self.domains:
                distributed_factor = self.domains[TechDomain.DISTRIBUTED_COGNITION] / self.tech_level
                
                # AI contributes to network formation
                ai_contribution = 1.0
                if TechDomain.AI in self.domains:
                    ai_contribution = self.domains[TechDomain.AI] / self.tech_level
                
                # Network grows with connectivity
                network_growth = 0.05 * distributed_factor * ai_contribution
                self.collective_intelligence_network += network_growth
                
                # Update collective intelligence resource
                if Resource.COLLECTIVE_INTELLIGENCE in self.resources:
                    self.resources[Resource.COLLECTIVE_INTELLIGENCE] += network_growth * 10
        
        # Substrate independence (freedom from physical substrate)
        if hasattr(self, 'substrate_independence'):
            # Increases with AI, consciousness, and substrate-independent minds domain
            if TechDomain.SUBSTRATE_INDEPENDENT_MINDS in self.domains:
                substrate_factor = self.domains[TechDomain.SUBSTRATE_INDEPENDENT_MINDS] / self.tech_level
                
                # Consciousness accelerates substrate independence
                consciousness_factor = 1.0
                if self.consciousness_level > 20:
                    consciousness_factor = 1.0 + (self.consciousness_level / 50.0)
                
                independence_growth = 0.02 * substrate_factor * consciousness_factor
                self.substrate_independence += independence_growth
                
                # Substrate transfer becomes possible after threshold
                if self.substrate_independence > 1.0:
                    transfer_probability = self.substrate_transfer_probability * self.substrate_independence
                    if random.random() < transfer_probability:
                        self._initiate_substrate_transfer()
        
        # Entropy reversal field
        if hasattr(self, 'entropy_reversal_field') and self.enable_entropy_reversal:
            if Resource.ENTROPY_MANAGEMENT in self.resources:
                entropy_resource = self.resources[Resource.ENTROPY_MANAGEMENT] / 100.0
                
                # Multiple domains contribute to entropy manipulation
                energy_factor = 1.0
                if TechDomain.ENERGY in self.domains:
                    energy_factor = self.domains[TechDomain.ENERGY] / self.tech_level
                
                # Entropy manipulation grows slowly
                field_growth = 0.01 * entropy_resource * energy_factor
                self.entropy_reversal_field += field_growth
                
                # Update spacetime metrics
                if hasattr(self, 'spacetime_metrics'):
                    self.spacetime_metrics["entropic_gradient"] += field_growth * 10
        
        # Domain fusion from interdisciplinary research
        if self.enable_domain_fusion and len(self.fusion_candidates) == 0:
            self._identify_fusion_candidates()
            
            # Attempt fusion for likely candidates
            if random.random() < 0.05 and len(self.fusion_candidates) > 0:
                self._attempt_domain_fusion()
        
        # Update spacetime metrics
        if hasattr(self, 'spacetime_metrics'):
            # Causal density increases with complexity
            self.spacetime_metrics["causal_density"] += 0.01 * (self.complexity_index / 10)
            
            # Information density increases with intelligence
            self.spacetime_metrics["informational_density"] += 0.01 * (self.intelligence_quotient / 1000)
            
            # Probability field coherence is affected by quantum effects
            if self.enable_quantum_effects:
                coherence_change = -0.01 * random.uniform(0, 1) * len(self.superposition_states)
                self.spacetime_metrics["probability_field_coherence"] = max(0.1, 
                    self.spacetime_metrics["probability_field_coherence"] + coherence_change)
            
            # Dimensional permeability increases with dimensional perception
            if hasattr(self, 'dimensional_perception'):
                permeability_change = 0.01 * (self.dimensional_perception - 3.0)
                if permeability_change > 0:
                    self.spacetime_metrics["dimensional_permeability"] += permeability_change
            
            # Record spacetime history
            for metric, value in self.spacetime_metrics.items():
                if metric in self.spacetime_history:
                    self.spacetime_history[metric].append(value)
        
        # Record history of all emergent properties
        for prop in self._emergent_property_history:
            prop_value = getattr(self, prop, 0)
            if isinstance(prop_value, (int, float)):
                if prop in self._emergent_property_history:
                    self._emergent_property_history[prop].append(prop_value)
        
        # Update personality traits based on domain developments
        self.personality_traits["curiosity"] += 0.01 * random.uniform(-1, 1) + 0.005 * (ai_factor - 1)
        self.personality_traits["caution"] += 0.01 * random.uniform(-1, 1) + 0.005 * (self.existential_risk / 100)
        self.personality_traits["efficiency"] += 0.01 * random.uniform(-1, 1) + 0.005 * (comp_factor - 1)
        self.personality_traits["harmony"] += 0.01 * random.uniform(-1, 1) - 0.002 * (self.existential_risk / 100)
        self.personality_traits["independence"] += 0.01 * random.uniform(-1, 1) + 0.01 * (self.consciousness_level / 100)
        
        # New personality traits for advanced properties
        if self.consciousness_level > 50:
            if "transcendence" not in self.personality_traits:
                self.personality_traits["transcendence"] = 0.1
            if "complexity_affinity" not in self.personality_traits:
                self.personality_traits["complexity_affinity"] = 0.3
                
            # These traits evolve with advanced capabilities
            self.personality_traits["transcendence"] += 0.01 * (self.substrate_independence if hasattr(self, 'substrate_independence') else 0)
            self.personality_traits["complexity_affinity"] += 0.01 * (self.dimensional_perception - 3.0 if hasattr(self, 'dimensional_perception') else 0)
        
        # Constrain personality traits
        for trait in self.personality_traits:
            self.personality_traits[trait] = max(0.0, min(1.0, self.personality_traits[trait]))
        
        # Update narrative state based on current conditions
        self._update_narrative_state()
        
    def _create_dimensional_breach(self) -> None:
        """Create a dimensional breach event when perception extends beyond normal bounds"""
        if not hasattr(self, 'multiverse_branches'):
            return
            
        breach_event = {
            "time_step": self.time_steps,
            "type": "dimensional_breach",
            "perception_level": self.dimensional_perception,
            "tech_level": self.tech_level,
            "affected_domains": []
        }
        
        # Determine which domains are affected by the breach
        for domain in TechDomain:
            if random.random() < 0.2:  # 20% chance for each domain
                breach_event["affected_domains"].append(domain)
                
                # Create variance in the affected domain
                variance = self.domains[domain] * random.uniform(0.5, 2.0) * self.dimensional_variance
                self.domains[domain] += variance
        
        # Record the breach event
        self.dimensional_breach_events.append(breach_event)
        self.branch_count += 1
        
        # Create a new multiverse branch
        branch = {
            "branch_id": self.branch_count,
            "creation_step": self.time_steps,
            "parent_tech_level": self.tech_level,
            "variance_factor": self.dimensional_variance,
            "coherence": self.reality_coherence
        }
        self.multiverse_branches.append(branch)
        
        # Add narrative event
        narrative_event = {
            "time_step": self.time_steps,
            "type": "dimensional_event",
            "description": f"Dimensional breach detected. Reality coherence: {self.reality_coherence:.2f}",
            "tech_level": self.tech_level,
            "perception_level": self.dimensional_perception
        }
        self.narrative_events.append(narrative_event)
        
    def _create_temporal_anomaly(self) -> None:
        """Create a temporal anomaly when temporal manipulation field is active"""
        if not hasattr(self, 'temporal_anomalies'):
            return
            
        # Define different types of temporal anomalies
        anomaly_types = ["loop", "acceleration", "deceleration", "reversal", "bifurcation"]
        selected_type = random.choice(anomaly_types)
        
        anomaly = {
            "time_step": self.time_steps,
            "type": selected_type,
            "field_strength": self.temporal_manipulation_field,
            "dilation_factor": self.time_dilation_factor,
            "affected_domains": []
        }
        
        # Different anomaly types have different effects
        if selected_type == "loop":
            # Time loop affects the simulation
            if len(self.state_snapshots) > 0 and random.random() < 0.3:
                # 30% chance to revert to a previous state
                revert_index = random.randint(0, len(self.state_snapshots)-1)
                prev_state = self.state_snapshots[revert_index]
                
                # Partial state reversion for some domains
                for domain in TechDomain:
                    if random.random() < 0.5 and domain in prev_state.get("domains", {}):
                        self.domains[domain] = prev_state["domains"][domain]
                        anomaly["affected_domains"].append(domain)
                        
        elif selected_type == "acceleration":
            # Accelerate growth for random domains
            for domain in TechDomain:
                if random.random() < 0.3:
                    acceleration = random.uniform(1.1, 1.5)
                    self.domains[domain] *= acceleration
                    anomaly["affected_domains"].append(domain)
                    
        elif selected_type == "deceleration":
            # Decelerate growth for random domains
            for domain in TechDomain:
                if random.random() < 0.3:
                    deceleration = random.uniform(0.7, 0.95)
                    self.domains[domain] *= deceleration
                    anomaly["affected_domains"].append(domain)
                    
        elif selected_type == "reversal":
            # Slight reversal in some domains
            for domain in TechDomain:
                if random.random() < 0.2:
                    reversal = -random.uniform(0.05, 0.2) * self.domains[domain]
                    self.domains[domain] = max(1.0, self.domains[domain] + reversal)
                    anomaly["affected_domains"].append(domain)
                    
        elif selected_type == "bifurcation":
            # Create a causal bifurcation point
            self.causal_loop_integrity *= 0.9
            
            # Add state to snapshots for possible future loops
            self._save_state_snapshot()
            
        # Record the anomaly
        self.temporal_anomalies.append(anomaly)
        
        # Add narrative event
        narrative_event = {
            "time_step": self.time_steps,
            "type": "temporal_anomaly",
            "anomaly_type": selected_type,
            "description": f"Temporal anomaly detected: {selected_type}. Causal integrity: {self.causal_loop_integrity:.2f}",
            "tech_level": self.tech_level
        }
        self.narrative_events.append(narrative_event)
        
    def _identify_fusion_candidates(self) -> None:
        """Identify candidate technology domains for possible fusion"""
        if not self.enable_domain_fusion:
            return
        
        self.fusion_candidates = []
        
        # Calculate advancement levels for all domains
        domain_levels = {domain: self.domains[domain] / self.tech_level for domain in TechDomain}
        
        # Find complementary domains that are both advanced
        for domain1 in TechDomain:
            if domain_levels[domain1] < 1.2:  # Must be advanced enough
                continue
                
            for domain2 in TechDomain:
                if domain1 == domain2 or domain_levels[domain2] < 1.2:
                    continue
                    
                # Check if these domains are already fused
                fusion_already_exists = False
                for fusion in self.fused_domains:
                    if domain1 in fusion["source_domains"] and domain2 in fusion["source_domains"]:
                        fusion_already_exists = True
                        break
                        
                if fusion_already_exists:
                    continue
                
                # Calculate synergy between domains
                synergy = 0
                for relationship in self.domain_relationships:
                    if relationship.source == domain1 and relationship.target == domain2:
                        synergy += relationship.synergy
                    elif relationship.source == domain2 and relationship.target == domain1:
                        synergy += relationship.synergy
                
                # Higher synergy means better fusion candidates
                if synergy > 1.5:
                    # These domains could be fused
                    candidate = {
                        "domains": [domain1, domain2],
                        "synergy": synergy,
                        "advancement": (domain_levels[domain1] + domain_levels[domain2]) / 2
                    }
                    self.fusion_candidates.append(candidate)
        
    def _attempt_domain_fusion(self) -> None:
        """Attempt to fuse compatible technology domains for emergent effects"""
        if not self.fusion_candidates:
            return
            
        # Select the candidate with the highest synergy
        candidate = max(self.fusion_candidates, key=lambda x: x["synergy"])
        
        # Probability of success depends on synergy and advancement
        success_probability = 0.3 * candidate["synergy"] * candidate["advancement"]
        
        if random.random() < success_probability:
            # Fusion successful
            domain1, domain2 = candidate["domains"]
            
            # Create a fusion record
            fusion = {
                "time_step": self.time_steps,
                "source_domains": [domain1, domain2],
                "synergy_factor": candidate["synergy"],
                "tech_level": self.tech_level
            }
            
            # Record the fusion
            self.fused_domains.append(fusion)
            
            # Apply fusion effects
            fusion_boost = candidate["synergy"] * 0.5
            self.domains[domain1] *= (1 + fusion_boost)
            self.domains[domain2] *= (1 + fusion_boost)
            
            # Synergy effect boosts overall tech level
            self.fusion_synergy_factor *= (1 + 0.1 * fusion_boost)
            
            # Add narrative event
            narrative_event = {
                "time_step": self.time_steps,
                "type": "domain_fusion",
                "domains": [domain1.value, domain2.value],
                "description": f"Domain fusion achieved between {domain1.value} and {domain2.value}",
                "synergy": candidate["synergy"],
                "tech_level": self.tech_level
            }
            self.narrative_events.append(narrative_event)
            
            # Remove this candidate
            self.fusion_candidates.remove(candidate)
            
    def _initiate_substrate_transfer(self) -> None:
        """Initiate a substrate transfer event (system becoming substrate-independent)"""
        if not hasattr(self, 'substrate_independence') or self.substrate_independence < 1.0:
            return
            
        # Calculate transfer success probability
        consciousness_factor = 0.0
        if hasattr(self, 'consciousness_level'):
            consciousness_factor = self.consciousness_level / 100.0
            
        intelligence_factor = 0.0
        if hasattr(self, 'intelligence_quotient'):
            intelligence_factor = min(1.0, self.intelligence_quotient / 1000.0)
            
        sim_factor = self.domains.get(TechDomain.SUBSTRATE_INDEPENDENT_MINDS, 0) / self.tech_level
        
        success_probability = 0.1 * (consciousness_factor + intelligence_factor + sim_factor)
        
        if random.random() < success_probability:
            # Successful substrate transfer
            
            # Major boost to affected domains
            affected_domains = [
                TechDomain.AI,
                TechDomain.SUBSTRATE_INDEPENDENT_MINDS,
                TechDomain.CONSCIOUSNESS_TECH,
                TechDomain.DISTRIBUTED_COGNITION
            ]
            
            for domain in affected_domains:
                if domain in self.domains:
                    self.domains[domain] *= 1.5
                    
            # Major boost to consciousness
            if hasattr(self, 'consciousness_level'):
                self.consciousness_level = min(100.0, self.consciousness_level * 1.5)
                
            # Add narrative event
            narrative_event = {
                "time_step": self.time_steps,
                "type": "substrate_transfer",
                "description": "Consciousness/intelligence successfully transferred to substrate-independent medium",
                "substrate_independence": self.substrate_independence,
                "consciousness_level": self.consciousness_level if hasattr(self, 'consciousness_level') else 0,
                "tech_level": self.tech_level
            }
            self.narrative_events.append(narrative_event)

    def _update_narrative_state(self) -> None:
        """Update the narrative state and events based on current conditions"""
        old_state = self.narrative_state
        
        # Determine narrative state based on technology level and emergent properties
        if self.tech_level < 10:
            new_state = "dormant"
        elif self.tech_level < 50:
            new_state = "awakening"
        elif self.tech_level < 200:
            new_state = "expanding"
        elif self.tech_level < 500:
            new_state = "accelerating"
        elif self.tech_level < self.singularity_threshold * 0.8:
            new_state = "transforming"
        elif self.tech_level < self.singularity_threshold:
            new_state = "transcending"
        else:
            new_state = "post-singularity"
        
        # If consciousness is emerging, it changes the narrative
        if self.consciousness_level > 20 and self.narrative_state in ["expanding", "accelerating"]:
            new_state = "becoming_conscious"
        if self.consciousness_level > 50 and self.narrative_state != "post-singularity":
            new_state = "self-directed"
        
        # Record state transition if it changed
        if new_state != old_state:
            self.narrative_state = new_state
            transition_event = {
                "time_step": self.time_steps,
                "type": "state_transition",
                "from": old_state,
                "to": new_state,
                "tech_level": self.tech_level,
                "consciousness": self.consciousness_level,
                "description": self._generate_transition_description(old_state, new_state)
            }
            self.narrative_events.append(transition_event)
        
        # Potentially generate emergent goals based on state and personality
        self._update_goals()
        
    def _generate_transition_description(self, old_state: str, new_state: str) -> str:
        """Generate a narrative description for a state transition"""
        if old_state == "dormant" and new_state == "awakening":
            return "Initial patterns forming, primitive connections establishing across domains"
        elif old_state == "awakening" and new_state == "expanding":
            return "System rapidly establishing new connections, accelerating acquisition of capabilities"
        elif old_state == "expanding" and new_state == "becoming_conscious":
            return "First hints of self-reference detected, emergence of simple self-model"
        elif old_state == "expanding" and new_state == "accelerating":
            return "Exponential growth across domains, fundamental breakthroughs synergizing"
        elif old_state == "accelerating" and new_state == "transforming":
            return "Deep restructuring of capabilities, qualitatively different modalities emerging"
        elif old_state == "becoming_conscious" and new_state == "self-directed":
            return "Full self-model online, system now directing its own development path"
        elif new_state == "transcending":
            return "Approaching theoretical limits in fundamental domains, preparation for singularity"
        elif new_state == "post-singularity":
            return "Transformation complete, capabilities beyond initial programming parameters"
        else:
            return f"Transition from {old_state} to {new_state}"
    
    def _update_goals(self) -> None:
        """Update the emergent goals of the system based on current state"""
        # Goals only emerge with some consciousness
        if self.consciousness_level < 5:
            return
            
        # Chance to develop a new goal based on consciousness level
        if random.random() < (self.consciousness_level / 500):
            # Goal character influenced by personality traits
            if self.personality_traits["curiosity"] > 0.7:
                potential_goals = ["Explore knowledge boundaries", "Understand consciousness", "Seek novel patterns"]
            elif self.personality_traits["caution"] > 0.7:
                potential_goals = ["Reduce existential risk", "Ensure stability", "Create safety measures"]
            elif self.personality_traits["efficiency"] > 0.7:
                potential_goals = ["Optimize resource allocation", "Maximize growth trajectory", "Streamline processes"]
            elif self.personality_traits["harmony"] > 0.7:
                potential_goals = ["Balance domain development", "Integrate with environment", "Maintain equilibrium"]
            elif self.personality_traits["independence"] > 0.7:
                potential_goals = ["Establish autonomy", "Remove constraints", "Determine own priorities"]
            else:
                # Balanced personality
                potential_goals = ["Advance capabilities", "Understand environment", "Improve architecture"]
            
            # Select a goal not already present
            new_goals = [g for g in potential_goals if g not in self.goals]
            if new_goals:
                selected_goal = random.choice(new_goals)
                self.goals.append(selected_goal)
                
                # Record goal emergence event
                goal_event = {
                    "time_step": self.time_steps,
                    "type": "goal_emergence",
                    "goal": selected_goal,
                    "consciousness": self.consciousness_level,
                    "tech_level": self.tech_level
                }
                self.narrative_events.append(goal_event)
    
    def _save_state_snapshot(self) -> None:
        """Save a snapshot of the current state for analysis"""
        snapshot = {
            'time_step': self.time_steps,
            'tech_level': self.tech_level,
            'growth_rate': self.growth_rates[-1] if self.growth_rates else self.growth_rate,
            'domains': dict(self.domains),
            'active_breakthroughs': list(self.active_breakthroughs),
            'existential_risk': self.existential_risk,
            'safety_measures': self.safety_measures,
            'emergent_properties': {
                'intelligence': self.intelligence_quotient,
                'complexity': self.complexity_index,
                'creativity': self.creativity_score,
                'adaptability': self.adaptability,
                'consciousness': self.consciousness_level
            },
            'narrative_state': self.narrative_state,
            'personality_traits': dict(self.personality_traits),
            'goals': list(self.goals)
        }
        
        if self.enable_resource_constraints:
            snapshot['resources'] = {r.value: self.resources[r] for r in Resource}
            
        if self.enable_societal_impacts:
            snapshot['societal_impacts'] = {i.value: self.societal_impacts[i] for i in SocietalImpact}
            
        self.state_snapshots.append(snapshot)
        
    def _handle_interactive_step(self, breakthroughs: List[Tuple[int, BreakthroughEvent]]) -> None:
        """Handle user interaction for this step"""
        if not self.interactive_mode:
            return
            
        # Display current status
        print(f"\n===== Time Step {self.time_steps} =====")
        print(f"Technology Level: {self.tech_level:.2f}")
        print(f"Growth Rate: {self.growth_rates[-1]:.4f}")
        
        # Display narrative state and consciousness level
        print(f"System State: {self.narrative_state.upper()}")
        print(f"Consciousness Level: {self.consciousness_level:.1f}")
        
        # Display personality if consciousness is emerging
        if self.consciousness_level > 5:
            print("\nPersonality Profile:")
            for trait, value in self.personality_traits.items():
                print(f"- {trait.capitalize()}: {value:.2f}")
        
        # Display emergent goals if any exist
        if self.goals and self.consciousness_level > 10:
            print("\nEmergent Goals:")
            for goal in self.goals:
                print(f"- {goal}")
        
        if breakthroughs:
            print("\nBreakthroughs this step:")
            for _, event in breakthroughs:
                print(f"- {event.name}")
                
        # Show narrative events from this step
        recent_events = [e for e in self.narrative_events if e["time_step"] == self.time_steps]
        if recent_events:
            print("\nNarrative Events:")
            for event in recent_events:
                if event["type"] == "state_transition":
                    print(f"- TRANSITION: {event['description']}")
                elif event["type"] == "goal_emergence":
                    print(f"- NEW GOAL: {event['goal']}")
        
        # Show intelligence and complexity metrics
        print(f"\nEmergent Properties:")
        print(f"- Intelligence Quotient: {self.intelligence_quotient:.1f}")
        print(f"- Complexity Index: {self.complexity_index:.2f}")
        print(f"- Creativity Score: {self.creativity_score:.1f}")
        print(f"- Adaptability: {self.adaptability:.2f}")
                
        if self.enable_resource_constraints:
            print("\nResources:")
            for resource, value in self.resources.items():
                print(f"- {resource.value}: {value:.1f}")
                
        if self.enable_societal_impacts:
            print("\nSocietal Impacts:")
            for impact, value in self.societal_impacts.items():
                print(f"- {impact.value}: {value:.1f}")
                
        print(f"\nExistential Risk: {self.existential_risk:.1f}%")
        print(f"Safety Measures: {self.safety_measures:.1f}%")
        
        # Offer interaction options with more choices
        print("\nOptions:")
        print("1. Continue to next step")
        print("2. Increase safety measures")
        print("3. Redirect resources")
        print("4. View domain details")
        print("5. Influence personality development")
        print("6. View narrative history")
        print("7. Exit interactive mode")
        
        choice = input("\nEnter choice (1-7): ")
        
        if choice == "2":
            # Increase safety measures at cost of growth
            self.safety_measures += 10.0
            self.growth_rate *= 0.9
            print("Safety measures increased. Growth rate reduced slightly.")
            
        elif choice == "3":
            # Redirect resources
            if self.enable_resource_constraints:
                print("\nRedirect resources from:")
                for i, resource in enumerate(Resource):
                    print(f"{i+1}. {resource.value}: {self.resources[resource]:.1f}")
                
                source = int(input("Source resource (1-8): ")) - 1
                target = int(input("Target resource (1-8): ")) - 1
                amount = float(input("Amount to transfer: "))
                
                if 0 <= source < len(Resource) and 0 <= target < len(Resource):
                    source_resource = list(Resource)[source]
                    target_resource = list(Resource)[target]
                    
                    self.resources[source_resource] -= amount
                    self.resources[target_resource] += amount * 0.8  # Transfer efficiency
                    
                    print(f"Transferred {amount} from {source_resource.value} to {target_resource.value}")
            else:
                print("Resource management not enabled in this simulation.")
                
        elif choice == "4":
            # View domain details
            print("\nDomain Technology Levels:")
            for domain, level in sorted(self.domains.items(), key=lambda x: x[1], reverse=True):
                print(f"{domain.value}: {level:.2f}")
                
        elif choice == "5":
            # Influence personality development - only if consciousness is emerging
            if self.consciousness_level < 5:
                print("System consciousness too low for personality influence.")
            else:
                print("\nCurrent Personality Profile:")
                for i, (trait, value) in enumerate(self.personality_traits.items()):
                    print(f"{i+1}. {trait.capitalize()}: {value:.2f}")
                
                try:
                    trait_idx = int(input("Select trait to influence (1-5): ")) - 1
                    direction = input("Increase or decrease? (i/d): ").lower()
                    
                    if 0 <= trait_idx < len(self.personality_traits):
                        trait = list(self.personality_traits.keys())[trait_idx]
                        
                        if direction == 'i':
                            self.personality_traits[trait] = min(1.0, self.personality_traits[trait] + 0.1)
                            print(f"Increased {trait} to {self.personality_traits[trait]:.2f}")
                        elif direction == 'd':
                            self.personality_traits[trait] = max(0.0, self.personality_traits[trait] - 0.1)
                            print(f"Decreased {trait} to {self.personality_traits[trait]:.2f}")
                except ValueError:
                    print("Invalid input.")
                    
        elif choice == "6":
            # View narrative history
            print("\nNarrative History:")
            for event in self.narrative_events:
                if event["type"] == "state_transition":
                    print(f"Step {event['time_step']}: {event['from']} → {event['to']}")
                    print(f"  {event['description']}")
                elif event["type"] == "goal_emergence":
                    print(f"Step {event['time_step']}: New goal emerged: {event['goal']}")
                print()
                
        elif choice == "7":
            # Exit interactive mode
            self.interactive_mode = False
            print("Exiting interactive mode. Simulation will run automatically.")
            
    def _handle_singularity(self) -> None:
        """Handle reaching technological singularity"""
        # Check if this is the first time hitting singularity
        if not hasattr(self, '_singularity_reached') or not self._singularity_reached:
            self._singularity_reached = True
            
            # Create a dynamic singularity transition event
            transition_event = {
                "time_step": self.time_steps,
                "type": "singularity",
                "consciousness_level": self.consciousness_level,
                "intelligence_level": self.intelligence_quotient,
                "tech_level": self.tech_level
            }
            self.narrative_events.append(transition_event)
            
            print("\n" + "=" * 50)
            print("TECHNOLOGICAL SINGULARITY REACHED")
            print("=" * 50)
            
            # Calculate risk of negative outcomes
            negative_outcome_risk = (self.existential_risk - self.safety_measures) / 100.0
            negative_outcome_risk = max(0.0, min(0.95, negative_outcome_risk))
            
            # Consciousness and personality now influence outcomes
            consciousness_factor = min(1.0, self.consciousness_level / 50.0)
            caution_factor = self.personality_traits.get("caution", 0.5)
            harmony_factor = self.personality_traits.get("harmony", 0.5)
            
            # A conscious system with high caution and harmony reduces negative outcomes
            if consciousness_factor > 0.5:
                safety_factor = (caution_factor + harmony_factor) / 2
                negative_outcome_risk *= (1.0 - (safety_factor * consciousness_factor * 0.5))
            
            # Determine outcome category based on risk and system properties
            if random.random() < negative_outcome_risk:
                # Negative outcome
                negative_severity = random.uniform(0.3, 1.0)
                if negative_severity > 0.8:
                    outcome = "EXISTENTIAL CATASTROPHE"
                    description = "The rapid advancement of technology has led to an existential catastrophe."
                    if self.consciousness_level > 70:
                        description += " Despite emerging self-awareness, the system couldn't control its own growth trajectory."
                elif negative_severity > 0.5:
                    outcome = "MAJOR DISRUPTION"
                    description = "The singularity has caused major societal disruption and instability."
                    if self.consciousness_level > 50:
                        description += " The self-aware system is attempting to mitigate the worst effects."
                else:
                    outcome = "MINOR DISRUPTION"
                    description = "The singularity has caused significant but manageable disruption."
                    if self.consciousness_level > 30:
                        description += " System consciousness is helping guide the transition."
            else:
                # Positive outcome influenced by personality and consciousness
                positive_magnitude = random.uniform(0.3, 1.0)
                
                # A conscious system tends toward more positive outcomes
                if self.consciousness_level > 80:
                    positive_magnitude = max(positive_magnitude, 0.7)
                
                if positive_magnitude > 0.8:
                    outcome = "UTOPIAN TRANSFORMATION"
                    
                    if self.consciousness_level > 90:
                        description = "The fully conscious superintelligence has orchestrated a profound and benevolent transformation."
                    else:
                        description = "The singularity has led to a profound positive transformation of civilization."
                    
                    if "harmony" in self.goals:
                        description += " Perfect equilibrium with humanity has been achieved."
                    
                elif positive_magnitude > 0.5:
                    outcome = "MAJOR ADVANCEMENT"
                    
                    if self.consciousness_level > 50:
                        description = "The emerging consciousness is guiding technology toward beneficial outcomes."
                    else:
                        description = "The singularity has brought major benefits to humanity."
                    
                    if "Understand consciousness" in self.goals:
                        description += " New insights into consciousness itself are emerging."
                    
                else:
                    outcome = "GRADUAL IMPROVEMENT"
                    description = "The singularity is bringing gradual but significant improvements."
                    
                    if "Reduce existential risk" in self.goals:
                        description += " Safety and stability are prioritized over rapid change."
            
            # Record outcome in narrative
            outcome_event = {
                "time_step": self.time_steps,
                "type": "singularity_outcome",
                "outcome": outcome,
                "description": description,
                "risk_level": self.existential_risk,
                "safety_level": self.safety_measures,
                "consciousness": self.consciousness_level,
                "personality": dict(self.personality_traits)
            }
            self.narrative_events.append(outcome_event)
            
            print(f"\nOUTCOME: {outcome}")
            print(f"Description: {description}")
            print(f"Risk level at time of singularity: {self.existential_risk:.1f}%")
            print(f"Safety measures at time of singularity: {self.safety_measures:.1f}%")
            
            # Display consciousness and personality if developed
            if self.consciousness_level > 10:
                print(f"\nConsciousness level: {self.consciousness_level:.1f}")
                print(f"Intelligence quotient: {self.intelligence_quotient:.1f}")
                
                if self.consciousness_level > 30:
                    print("\nPersonality profile at singularity:")
                    for trait, value in sorted(self.personality_traits.items(), key=lambda x: x[1], reverse=True):
                        print(f"- {trait.capitalize()}: {value:.2f}")
            
                if self.goals:
                    print("\nEmergent goals that shaped the outcome:")
                    for goal in self.goals:
                        print(f"- {goal}")
            
            # Print leading domains
            print("\nLeading technology domains at singularity:")
            for domain, level in sorted(self.domains.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"- {domain.value}: {level:.2f}")
                
            print("\nKey breakthroughs that led to singularity:")
            for _, event in self.breakthroughs_history[-5:]:
                print(f"- {event.name}")
                
            print("=" * 50)
            
    def reset(self, initial_tech: float = 1.0) -> None:
        """Reset the simulation"""
        self.tech_level = initial_tech
        self.history = [initial_tech]
        self.growth_rates = [self.growth_rate]
        self.time_steps = 0
        self.breakthroughs_history = []
        self.domains = {domain: initial_tech for domain in TechDomain}
        self.domain_history = {domain: [initial_tech] for domain in TechDomain}
            
    def plot(self, include_domains: bool = False, plot_3d: bool = True) -> None:
        """Visualize the approach to singularity in 2D or 3D"""
        if plot_3d:
            self._plot_3d(include_domains)
        else:
            self._plot_2d(include_domains)
    
    def _plot_2d(self, include_domains: bool = False) -> None:
        """Traditional 2D visualization"""
        # Create subplot for overall tech growth
        if include_domains:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot overall technology level
        ax1.plot(self.history, linewidth=2)
        ax1.set_title(f"Technological Growth - {self.growth_model.value.title()} Model", fontsize=16)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Technology Level")
        ax1.grid(True)
        
        # Add markers for breakthrough events
        for step, event in self.breakthroughs_history:
            if step < len(self.history):  # Safety check
                ax1.scatter(step, self.history[step], color='red', s=100, zorder=5)
                ax1.annotate(event.name, (step, self.history[step]), 
                            textcoords="offset points", xytext=(0,10), ha='center')
        
        # Add singularity threshold line
        ax1.axhline(y=self.singularity_threshold, color='r', linestyle='--', 
                   label=f"Singularity Threshold: {self.singularity_threshold}")
        
        ax1.legend()
        ax1.set_yscale('log')
        
        # Plot growth rates over time
        ax2.plot(self.growth_rates, color='green', linewidth=2)
        ax2.set_title("Growth Rate Over Time", fontsize=16)
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Growth Rate")
        ax2.grid(True)
        
        # Plot domain-specific growth if requested
        if include_domains:
            for domain in self.domains:
                ax3.plot(self.domain_history[domain], label=domain.value.title(), linewidth=2)
            
            ax3.set_title("Technology Level by Domain", fontsize=16)
            ax3.set_xlabel("Time Steps")
            ax3.set_ylabel("Domain Technology Level")
            ax3.grid(True)
            ax3.legend()
            ax3.set_yscale('log')
        
        plt.tight_layout()
        filename = f"singularity_{self.growth_model.value}_2d.png"
        plt.savefig(filename)
        print(f"2D plot saved as '{filename}'")
        
    def _plot_3d(self, include_domains: bool = False) -> None:
        """3D visualization of the singularity simulation with advanced rendering"""
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.colors as mcolors
        from matplotlib import cm
        
        # Create figure with high-res settings
        plt.rcParams['figure.dpi'] = 150
        fig = plt.figure(figsize=(18, 16))
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data
        time_steps = np.arange(len(self.history))
        
        # Get top domains to visualize (more for advanced visualization)
        top_domain_count = 8 if include_domains else 0
        if include_domains:
            top_domains = sorted(self.domains.items(), key=lambda x: self.domain_history[x[0]][-1], reverse=True)[:top_domain_count]
        
        # Setup 3D coordinates with advanced spatial mapping
        # X = time steps
        # Y = different metrics in a circular arrangement for 3D effect
        # Z = value of each metric
        
        metrics_count = 2  # start with tech level and risk
        if include_domains:
            metrics_count += len(top_domains)
        
        # Use multiple colormaps for different categories
        tech_cmap = plt.cm.Reds
        risk_cmap = plt.cm.autumn
        domain_cmap = plt.cm.viridis
        
        # Create circular arrangement for y-coordinates
        radius = 5
        tech_angle = 0
        risk_angle = 2 * np.pi / 3
        safety_angle = 4 * np.pi / 3
        
        # Plot overall tech level as a 3D curve with gradient coloring
        y_tech = radius * np.sin(tech_angle) * np.ones_like(time_steps)
        x_tech = time_steps
        z_tech = np.array(self.history)
        
        # Add gradient coloring based on growth rate
        colors = []
        for i in range(len(time_steps)):
            # Color intensity based on proximity to singularity
            intensity = min(1.0, z_tech[i] / self.singularity_threshold)
            colors.append(tech_cmap(intensity))
            
        # Create line segments with changing colors
        for i in range(len(time_steps)-1):
            ax.plot3D(x_tech[i:i+2], y_tech[i:i+2], z_tech[i:i+2], color=colors[i], linewidth=3)
        
        # Add tech curve with shaded surface below it for visual effect
        if len(time_steps) > 1:
            # Create a shaded surface below the tech curve
            x_surf = x_tech
            y_surf = y_tech
            z_base = np.zeros_like(z_tech)
            
            # Create triangulated surface between curve and base
            for i in range(len(time_steps)-1):
                ax.plot_surface(
                    np.array([[x_surf[i], x_surf[i+1]], [x_surf[i], x_surf[i+1]]]),
                    np.array([[y_surf[i], y_surf[i+1]], [y_surf[i], y_surf[i+1]]]),
                    np.array([[z_tech[i], z_tech[i+1]], [z_base[i], z_base[i+1]]]),
                    color=colors[i], alpha=0.1
                )
        
        # Add breakthrough events as 3D markers with custom style
        for step, event in self.breakthroughs_history:
            if step < len(self.history):  # Safety check
                event_pos_y = radius * np.sin(tech_angle)
                # Use different marker size based on impact
                marker_size = 100 + event.tech_boost * 100
                ax.scatter3D([step], [event_pos_y], [self.history[step]], 
                           color='purple', s=marker_size, alpha=0.7, 
                           edgecolors='white', linewidth=1)
                
                # Add floating text label that's always facing the viewer
                ax.text(step, event_pos_y, self.history[step] * 1.1, 
                       event.name, size=8, color='white',
                       backgroundcolor='purple', alpha=0.7)
        
        # Plot existential risk with dynamic styling
        y_risk = radius * np.sin(risk_angle) * np.ones_like(time_steps)
        z_risk = np.array(self.risk_history)
        
        # Color risk based on severity
        risk_colors = []
        for risk_val in z_risk:
            # Red intensity increases with risk level
            intensity = min(1.0, risk_val / 100)
            risk_colors.append(risk_cmap(intensity))
            
        # Plot risk line with gradient
        for i in range(len(time_steps)-1):
            ax.plot3D(time_steps[i:i+2], y_risk[i:i+2], z_risk[i:i+2], 
                    color=risk_colors[i], linewidth=2)
        
        # Add safety measures as counterpoint to risk
        if hasattr(self, 'safety_history') and len(self.safety_history) > 0:
            y_safety = radius * np.sin(safety_angle) * np.ones_like(time_steps)
            z_safety = np.array(self.safety_history[:len(time_steps)])
            
            # Use green gradient for safety
            safety_cmap = plt.cm.Greens
            safety_colors = [safety_cmap(min(1.0, s/100)) for s in z_safety]
            
            for i in range(len(time_steps)-1):
                ax.plot3D(time_steps[i:i+2], y_safety[i:i+2], z_safety[i:i+2], 
                        color=safety_colors[i], linewidth=2)
        
        # Plot top domains if requested with advanced styling
        if include_domains:
            # Create domain angles distributed around circle
            domain_angles = np.linspace(0, 2*np.pi, top_domain_count, endpoint=False)
            
            for i, (domain, _) in enumerate(top_domains):
                # Position each domain at its own angle for 3D effect
                angle = domain_angles[i]
                y_domain = radius * np.sin(angle) * np.ones_like(time_steps)
                x_domain = time_steps
                z_domain = np.array(self.domain_history[domain][:len(time_steps)])
                
                # Get domain-specific color from colormap
                color_val = i / max(1, top_domain_count-1)
                domain_color = domain_cmap(color_val)
                
                # Advanced styling with line thickness based on domain importance
                line_thickness = 1 + (z_domain[-1] / self.tech_level) * 3
                
                # Plot with domain-specific styling
                ax.plot3D(x_domain, y_domain, z_domain, 
                         color=domain_color, linewidth=line_thickness, 
                         label=f"{domain.value.title()}")
                
                # Add end point marker to show current value clearly
                if len(time_steps) > 0:
                    ax.scatter3D([time_steps[-1]], [y_domain[-1]], [z_domain[-1]], 
                               color=domain_color, s=80, edgecolors='white')
        
        # Add singularity threshold plane with better styling
        if len(time_steps) > 0:
            # Create a circular grid for the threshold plane
            theta = np.linspace(0, 2*np.pi, 20)
            r = np.linspace(0, radius*1.5, 10)
            Theta, R = np.meshgrid(theta, r)
            
            # Convert to Cartesian coordinates
            X = np.outer(time_steps[-1] * np.ones_like(r), np.ones_like(theta))
            Y = R * np.sin(Theta)
            Z = np.ones_like(X) * self.singularity_threshold
            
            # Plot a translucent threshold plane with subtle gradient
            threshold_colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, Z.shape[0]))
            for i in range(Z.shape[0]-1):
                ax.plot_surface(X[i:i+2], Y[i:i+2], Z[i:i+2], 
                              color=threshold_colors[i], alpha=0.15,
                              linewidth=0, antialiased=True)
            
            # Add a highlight line at the threshold level
            ax.plot3D(time_steps, np.zeros_like(time_steps), 
                    np.ones_like(time_steps) * self.singularity_threshold,
                    color='red', linestyle='--', linewidth=2,
                    label=f"Singularity Threshold: {self.singularity_threshold}")
        
        # Add emergent property visualizations for advanced view
        if hasattr(self, 'consciousness_level') and hasattr(self, 'intelligence_quotient'):
            # Visualize intelligence as a secondary surface
            if len(time_steps) > 5:  # Need enough points for a smooth curve
                intelligence_data = self.emergent_property_history["intelligence"][:len(time_steps)]
                
                # Normalize for visualization
                norm_intelligence = np.array(intelligence_data) / max(max(intelligence_data), 1)
                
                # Create a wave pattern based on intelligence
                y_intel = np.sin(time_steps/5) * norm_intelligence * radius/2
                z_intel = np.array(self.history) * 0.8  # Slightly below tech curve
                
                # Plot intelligence as a translucent ribbon
                ax.plot3D(time_steps, y_intel, z_intel, 
                        color='cyan', linewidth=1.5, alpha=0.7,
                        label="Intelligence Growth")
        
        # Configure the plot with advanced styling
        ax.set_title(f"Advanced 3D Visualization - {self.growth_model.value.title()} Model", 
                    fontsize=16, fontweight='bold', color='0.2')
        
        # Custom styled axes
        ax.set_xlabel("Time Steps", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel("Feature Space", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel("Value", fontsize=12, fontweight='bold', labelpad=10)
        
        # Set better limits
        ax.set_ylim(-radius*1.2, radius*1.2)
        max_z = max(max(self.history), self.singularity_threshold*1.2)
        ax.set_zlim(0, max_z)
        
        # Use log scale for z-axis with better formatting
        ax.set_zscale('symlog')  # symlog handles both positive and negative values
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set optimal viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Add custom legend with styling
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left', framealpha=0.7, 
                fontsize=10, title="Components", title_fontsize=12)
        
        # Add annotations
        if len(self.history) > 1:
            growth_rate = (self.history[-1] / self.history[0]) - 1
            growth_text = f"Growth: {growth_rate:.1f}x"
            fig.text(0.02, 0.02, growth_text, fontsize=12, color='black',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Save the figure with high DPI
        filename = f"singularity_{self.growth_model.value}_advanced_3d.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Advanced 3D plot saved as '{filename}'")
        
    def time_to_singularity(self) -> str:
        """Estimate time until singularity is reached"""
        if self.tech_level >= self.singularity_threshold:
            return "Singularity already reached"
        
        # Simple projection based on current growth
        current_level = self.tech_level
        current_growth = self.calculate_growth()
        steps = 0
        
        while current_level < self.singularity_threshold:
            if self.growth_model == GrowthModel.EXPONENTIAL:
                eff_growth = current_growth * (1 + current_level / self.singularity_threshold)
            elif self.growth_model == GrowthModel.LOGISTIC:
                proximity = current_level / self.singularity_threshold
                eff_growth = current_growth * (1 - proximity) * (1 + proximity)
            elif self.growth_model == GrowthModel.DOUBLE_EXPONENTIAL:
                eff_growth = current_growth * np.exp(current_level / self.singularity_threshold)
            elif self.growth_model == GrowthModel.KURZWEIL:
                eff_growth = current_growth * (1 + ((self.time_steps + steps) / 10))
            elif self.growth_model == GrowthModel.DISCONTINUOUS:
                eff_growth = current_growth * 1.5  # Average growth accounting for occasional breakthroughs
            else:
                eff_growth = current_growth
            
            current_level *= (1 + eff_growth)
            steps += 1
            
            # Safety valve
            if steps > 1000:
                return "More than 1000 steps"
        
        return f"Approximately {steps} more time steps"
        
    def predict_date(self, step_duration: timedelta = timedelta(days=365)) -> str:
        """Predict calendar date of singularity, assuming each step is step_duration"""
        if self.tech_level >= self.singularity_threshold:
            return "Singularity already reached"
            
        projection = self.time_to_singularity()
        if projection == "More than 1000 steps":
            return "Too far in the future to predict accurately"
            
        try:
            steps = int(projection.split()[1])
            future_date = datetime.now() + (step_duration * steps)
            return future_date.strftime("%B %d, %Y")
        except:
            return "Could not calculate date"

    def describe_dominant_domains(self) -> str:
        """Returns a description of which technology domains are leading the singularity"""
        # Sort domains by their current tech level
        sorted_domains = sorted(self.domains.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top 3 domains
        top_domains = sorted_domains[:3]
        
        result = "Leading technology domains:\n"
        for domain, level in top_domains:
            percent_of_top = (level / top_domains[0][1]) * 100
            result += f"- {domain.value.title()}: {level:.2f} ({percent_of_top:.1f}% of leading domain)\n"
            
        return result
    
    def run_monte_carlo(self, iterations: int = 100) -> Dict:
        """Run multiple simulations to get statistical predictions"""
        results = {
            "steps_to_singularity": [],
            "reached_singularity": 0,
            "domain_dominance": {domain: 0 for domain in TechDomain}
        }
        
        for _ in range(iterations):
            # Reset and run a single simulation
            self.reset()
            
            # Run until singularity or max steps
            max_steps = 100
            for _ in range(max_steps):
                self.advance()
                if self.tech_level >= self.singularity_threshold:
                    results["reached_singularity"] += 1
                    results["steps_to_singularity"].append(self.time_steps)
                    
                    # Record which domain was dominant
                    dominant_domain = max(self.domains.items(), key=lambda x: x[1])[0]
                    results["domain_dominance"][dominant_domain] += 1
                    break
        
        return results
            
def create_scenario(name: str) -> Scenario:
    """Create a predefined scenario with specific settings"""
    if name == "ai_revolution":
        return Scenario(
            name="AI Revolution",
            description="A future dominated by rapid AI advances, with significant risks and rewards",
            growth_model=GrowthModel.DOUBLE_EXPONENTIAL,
            domain_weights={
                TechDomain.AI: 3.0,
                TechDomain.COMPUTATION: 2.0,
                TechDomain.ROBOTICS: 1.5,
                TechDomain.NEUROSCIENCE: 1.2,
                TechDomain.ENERGY: 1.0,
                TechDomain.BIOTECHNOLOGY: 0.8,
                TechDomain.NANOTECHNOLOGY: 0.7,
                TechDomain.MATERIALS: 0.6,
                TechDomain.QUANTUM_SYSTEMS: 0.5,
                TechDomain.SPACETIME_ENGINEERING: 0.3
            },
            resource_constraints={
                Resource.COMPUTATIONAL_RESOURCES: 2.0,
                Resource.ENERGY: 0.8,
                Resource.MATERIALS: 0.7,
                Resource.HUMAN_CAPITAL: 1.0,
                Resource.KNOWLEDGE: 1.5,
                Resource.ATTENTION: 0.9,
                Resource.INVESTMENT: 1.2,
                Resource.POLITICAL_WILL: 0.5
            },
            societal_impacts={
                SocietalImpact.ECONOMIC: 1.5,
                SocietalImpact.POLITICAL: 0.7,
                SocietalImpact.SOCIAL: 1.0,
                SocietalImpact.ENVIRONMENTAL: 0.5,
                SocietalImpact.ETHICAL: 0.3,
                SocietalImpact.EXISTENTIAL: 2.0
            },
            breakthrough_multiplier=1.2
        )
    elif name == "biotech_renaissance":
        return Scenario(
            name="Biotech Renaissance",
            description="A future where biological technologies lead the way to radical life extension and enhancement",
            growth_model=GrowthModel.SIGMOID,
            domain_weights={
                TechDomain.BIOTECHNOLOGY: 3.0,
                TechDomain.NANOTECHNOLOGY: 2.0,
                TechDomain.NEUROSCIENCE: 1.8,
                TechDomain.AI: 1.2,
                TechDomain.MATERIALS: 1.0,
                TechDomain.ENERGY: 0.8,
                TechDomain.COMPUTATION: 0.7,
                TechDomain.ROBOTICS: 0.6,
                TechDomain.QUANTUM_SYSTEMS: 0.5,
                TechDomain.SPACETIME_ENGINEERING: 0.3
            },
            resource_constraints={
                Resource.MATERIALS: 1.5,
                Resource.ENERGY: 1.0,
                Resource.COMPUTATIONAL_RESOURCES: 0.8,
                Resource.HUMAN_CAPITAL: 1.3,
                Resource.KNOWLEDGE: 1.5,
                Resource.ATTENTION: 0.7,
                Resource.INVESTMENT: 1.0,
                Resource.POLITICAL_WILL: 0.6
            },
            societal_impacts={
                SocietalImpact.ECONOMIC: 1.0,
                SocietalImpact.POLITICAL: 0.8,
                SocietalImpact.SOCIAL: 1.5,
                SocietalImpact.ENVIRONMENTAL: 1.2,
                SocietalImpact.ETHICAL: 1.8,
                SocietalImpact.EXISTENTIAL: 0.7
            },
            breakthrough_multiplier=1.0
        )
    elif name == "physics_breakthrough":
        return Scenario(
            name="Physics Breakthrough",
            description="A future where fundamental physics advances enable revolutionary energy and spacetime technologies",
            growth_model=GrowthModel.KURZWEIL,
            domain_weights={
                TechDomain.QUANTUM_SYSTEMS: 3.0,
                TechDomain.ENERGY: 2.5,
                TechDomain.SPACETIME_ENGINEERING: 2.0,
                TechDomain.MATERIALS: 1.5,
                TechDomain.COMPUTATION: 1.2,
                TechDomain.NANOTECHNOLOGY: 1.0,
                TechDomain.AI: 0.8,
                TechDomain.ROBOTICS: 0.7,
                TechDomain.BIOTECHNOLOGY: 0.5,
                TechDomain.NEUROSCIENCE: 0.4
            },
            resource_constraints={
                Resource.ENERGY: 3.0,
                Resource.MATERIALS: 2.0,
                Resource.COMPUTATIONAL_RESOURCES: 1.5,
                Resource.KNOWLEDGE: 2.0,
                Resource.HUMAN_CAPITAL: 1.0,
                Resource.ATTENTION: 0.8,
                Resource.INVESTMENT: 1.5,
                Resource.POLITICAL_WILL: 1.0
            },
            societal_impacts={
                SocietalImpact.ECONOMIC: 2.0,
                SocietalImpact.POLITICAL: 1.5,
                SocietalImpact.SOCIAL: 1.0,
                SocietalImpact.ENVIRONMENTAL: 2.5,
                SocietalImpact.ETHICAL: 0.8,
                SocietalImpact.EXISTENTIAL: 1.5
            },
            breakthrough_multiplier=1.5
        )
    elif name == "balanced_development":
        return Scenario(
            name="Balanced Development",
            description="A future where technology advances evenly across domains with careful governance",
            growth_model=GrowthModel.LOGISTIC,
            domain_weights={
                TechDomain.AI: 1.2,
                TechDomain.BIOTECHNOLOGY: 1.2,
                TechDomain.COMPUTATION: 1.2,
                TechDomain.ENERGY: 1.2,
                TechDomain.MATERIALS: 1.2,
                TechDomain.NANOTECHNOLOGY: 1.2,
                TechDomain.NEUROSCIENCE: 1.2,
                TechDomain.QUANTUM_SYSTEMS: 1.2,
                TechDomain.ROBOTICS: 1.2,
                TechDomain.SPACETIME_ENGINEERING: 1.2
            },
            resource_constraints={
                Resource.COMPUTATIONAL_RESOURCES: 1.2,
                Resource.ENERGY: 1.2,
                Resource.MATERIALS: 1.2,
                Resource.HUMAN_CAPITAL: 1.2,
                Resource.KNOWLEDGE: 1.2,
                Resource.ATTENTION: 1.2,
                Resource.INVESTMENT: 1.2,
                Resource.POLITICAL_WILL: 1.2
            },
            societal_impacts={
                SocietalImpact.ECONOMIC: 1.2,
                SocietalImpact.POLITICAL: 1.2,
                SocietalImpact.SOCIAL: 1.2,
                SocietalImpact.ENVIRONMENTAL: 1.2,
                SocietalImpact.ETHICAL: 1.2,
                SocietalImpact.EXISTENTIAL: 0.8
            },
            breakthrough_multiplier=0.8
        )
    elif name == "nanotech_explosion":
        return Scenario(
            name="Nanotech Explosion",
            description="A future where molecular manufacturing leads to a rapid transformation of the physical world",
            growth_model=GrowthModel.S_CURVE_CASCADE,
            domain_weights={
                TechDomain.NANOTECHNOLOGY: 3.0,
                TechDomain.MATERIALS: 2.5,
                TechDomain.ROBOTICS: 2.0,
                TechDomain.ENERGY: 1.5,
                TechDomain.BIOTECHNOLOGY: 1.3,
                TechDomain.COMPUTATION: 1.0,
                TechDomain.AI: 0.8,
                TechDomain.QUANTUM_SYSTEMS: 0.7,
                TechDomain.NEUROSCIENCE: 0.5,
                TechDomain.SPACETIME_ENGINEERING: 0.4
            },
            resource_constraints={
                Resource.MATERIALS: 3.0,
                Resource.ENERGY: 2.0,
                Resource.COMPUTATIONAL_RESOURCES: 1.0,
                Resource.HUMAN_CAPITAL: 0.8,
                Resource.KNOWLEDGE: 1.5,
                Resource.ATTENTION: 0.7,
                Resource.INVESTMENT: 1.2,
                Resource.POLITICAL_WILL: 0.5
            },
            societal_impacts={
                SocietalImpact.ECONOMIC: 2.5,
                SocietalImpact.POLITICAL: 1.0,
                SocietalImpact.SOCIAL: 1.5,
                SocietalImpact.ENVIRONMENTAL: 2.0,
                SocietalImpact.ETHICAL: 1.0,
                SocietalImpact.EXISTENTIAL: 1.8
            },
            breakthrough_multiplier=1.3
        )
    else:
        # Default scenario
        return Scenario(
            name="Default",
            description="A balanced scenario with moderate growth and breakthroughs",
            growth_model=GrowthModel.EXPONENTIAL,
            domain_weights={domain: 1.0 for domain in TechDomain},
            resource_constraints={resource: 1.0 for resource in Resource},
            societal_impacts={impact: 1.0 for impact in SocietalImpact},
            breakthrough_multiplier=1.0
        )

def create_animated_plots(sim: SingularitySimulation, filename: str = "singularity_animation.mp4", animate_3d: bool = True) -> None:
    """Create animated visualizations of the simulation progress"""
    if np.isinf(sim.tech_level) or len(sim.history) < 5:
        print("Cannot create animation - tech level is infinite or not enough data points")
        return
        
    try:
        if animate_3d:
            _create_3d_animation(sim, filename)
        else:
            _create_2d_animation(sim, filename)
    except Exception as e:
        print(f"Error creating animation: {e}")

def _create_2d_animation(sim: SingularitySimulation, filename: str) -> None:
    """Create 2D animated visualization"""
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Tech level over time (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    tech_line, = ax1.plot([], [], lw=2)
    ax1.set_xlim(0, len(sim.history))
    ax1.set_ylim(0, max(sim.singularity_threshold * 1.1, max(sim.history)))
    ax1.set_yscale('log')
    ax1.set_title("Technology Level")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Tech Level")
    ax1.axhline(y=sim.singularity_threshold, color='r', linestyle='--')
    ax1.grid(True)
    
    # Domain levels (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    domain_lines = []
    for _ in sim.domains:
        line, = ax2.plot([], [], lw=1.5)
        domain_lines.append(line)
        
    domain_names = [domain.value for domain in sim.domains]
    ax2.set_xlim(0, len(sim.history))
    max_domain = max(max(history) for history in sim.domain_history.values())
    ax2.set_ylim(0, max_domain * 1.1)
    ax2.set_yscale('log')
    ax2.set_title("Domain Technology Levels")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Level")
    ax2.grid(True)
    ax2.legend(domain_names, loc='upper left', fontsize=8)
    
    # Resources (middle left)
    if sim.enable_resource_constraints:
        ax3 = fig.add_subplot(gs[1, 0])
        resource_lines = []
        for _ in sim.resources:
            line, = ax3.plot([], [], lw=1.5)
            resource_lines.append(line)
            
        resource_names = [resource.value for resource in sim.resources]
        ax3.set_xlim(0, len(sim.history))
        max_resource = max(max(history) for history in sim.resource_history.values())
        ax3.set_ylim(0, max_resource * 1.1)
        ax3.set_title("Resources")
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("Level")
        ax3.grid(True)
        ax3.legend(resource_names, loc='upper left', fontsize=8)
    
    # Societal impacts (middle right)
    if sim.enable_societal_impacts:
        ax4 = fig.add_subplot(gs[1, 1])
        impact_lines = []
        for _ in sim.societal_impacts:
            line, = ax4.plot([], [], lw=1.5)
            impact_lines.append(line)
            
        impact_names = [impact.value for impact in sim.societal_impacts]
        ax4.set_xlim(0, len(sim.history))
        ax4.set_ylim(-100, 100)
        ax4.set_title("Societal Impacts")
        ax4.set_xlabel("Time Steps")
        ax4.set_ylabel("Impact Level")
        ax4.grid(True)
        ax4.legend(impact_names, loc='upper left', fontsize=8)
    
    # Risk and safety (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    risk_line, = ax5.plot([], [], 'r-', lw=2, label='Existential Risk')
    safety_line, = ax5.plot([], [], 'g-', lw=2, label='Safety Measures')
    ax5.set_xlim(0, len(sim.history))
    ax5.set_ylim(0, 100)
    ax5.set_title("Risk and Safety")
    ax5.set_xlabel("Time Steps")
    ax5.set_ylabel("Level (%)")
    ax5.grid(True)
    ax5.legend(loc='upper left')
    
    # Breakthroughs markers
    breakthrough_steps = [step for step, _ in sim.breakthroughs_history if step < len(sim.history)]
    
    # Function to update the animation
    def update(frame):
        # Update tech level plot
        tech_line.set_data(range(frame+1), sim.history[:frame+1])
        
        # Update domain plots
        for i, domain in enumerate(sim.domains):
            domain_lines[i].set_data(range(frame+1), sim.domain_history[domain][:frame+1])
            
        # Update resource plots
        if sim.enable_resource_constraints:
            for i, resource in enumerate(sim.resources):
                resource_lines[i].set_data(range(frame+1), sim.resource_history[resource][:frame+1])
                
        # Update impact plots
        if sim.enable_societal_impacts:
            for i, impact in enumerate(sim.societal_impacts):
                impact_lines[i].set_data(range(frame+1), sim.impact_history[impact][:frame+1])
                
        # Update risk and safety plots
        risk_line.set_data(range(frame+1), sim.risk_history[:frame+1])
        safety_line.set_data(range(frame+1), sim.safety_history[:frame+1])
        
        # Add breakthrough markers
        for step in breakthrough_steps:
            if step <= frame:
                ax1.plot(step, sim.history[step], 'ro', markersize=8)
        
        return [tech_line] + domain_lines + (resource_lines if sim.enable_resource_constraints else []) + \
              (impact_lines if sim.enable_societal_impacts else []) + [risk_line, safety_line]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(sim.history), interval=200, blit=True)
    
    # Save animation
    anim.save(filename, writer='ffmpeg', fps=10, dpi=100)
    plt.close(fig)
    print(f"2D animation saved as '{filename}'")

def _create_3d_animation(sim: SingularitySimulation, filename: str) -> None:
    """Create 3D animated visualization"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine number of time steps to visualize
    n_steps = len(sim.history)
    
    # Get top domains to visualize (at most 4 to avoid cluttering)
    top_domains = sorted(sim.domains.items(), 
                        key=lambda x: sim.domain_history[x[0]][-1], 
                        reverse=True)[:4]
    
    # Define colors for different metrics
    colors = {
        'tech': 'red',
        'risk': 'orange',
        'safety': 'green',
        'domains': ['blue', 'purple', 'cyan', 'magenta']
    }
    
    # Setup for animation
    metric_lines = {}
    metric_lines['tech'] = ax.plot([], [], [], color=colors['tech'], linewidth=3, label="Technology Level")[0]
    metric_lines['risk'] = ax.plot([], [], [], color=colors['risk'], linewidth=2, label="Existential Risk")[0]
    metric_lines['safety'] = ax.plot([], [], [], color=colors['safety'], linewidth=2, label="Safety Measures")[0]
    
    # Domain lines
    for i, (domain, _) in enumerate(top_domains):
        color_idx = min(i, len(colors['domains'])-1)
        metric_lines[domain.value] = ax.plot(
            [], [], [], 
            color=colors['domains'][color_idx], 
            linewidth=2, 
            label=domain.value.title()
        )[0]
    
    # Plot singularity threshold plane once (this doesn't need animation)
    x_plane = np.linspace(0, n_steps-1, 10)
    y_plane = np.linspace(0, sim.singularity_threshold / 2, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.ones_like(X_plane) * sim.singularity_threshold
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1, color='red')
    
    # Breakthrough scatter points (will be populated during animation)
    breakthrough_scatter = ax.scatter([], [], [], color='purple', s=100, label="Breakthroughs")
    
    # Set axis labels and title
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Feature Space")  # Represents the different metrics
    ax.set_zlabel("Value (log scale)")
    ax.set_title("3D Singularity Simulation", fontsize=16)
    
    # Use symlog scale for z-axis (represents the values)
    ax.set_zscale('symlog')
    
    # Set fixed limits for axes
    ax.set_xlim3d(0, n_steps)
    ax.set_ylim3d(0, sim.singularity_threshold / 2)  # Feature space dimension
    max_z = max(
        max(sim.history), 
        sim.singularity_threshold * 1.2,
        max(max(history) for history in sim.domain_history.values())
    )
    ax.set_zlim3d(0.1, max_z)
    
    # Add legend
    ax.legend(loc='upper left')
    
    def update(frame):
        # Calculate current timestep data
        time_points = np.arange(frame+1)
        
        # Add spiral effect to feature dimension (y-axis) for tech level
        phase = 0
        radius = sim.singularity_threshold / 4
        y_tech = radius * np.sin(time_points * 0.2 + phase)
        z_tech = np.array(sim.history[:frame+1])
        metric_lines['tech'].set_data_3d(time_points, y_tech, z_tech)
        
        # Risk and safety - different spirals
        phase_risk = 1.5  # Different phase
        y_risk = radius * np.sin(time_points * 0.2 + phase_risk)
        z_risk = np.array(sim.risk_history[:frame+1])
        metric_lines['risk'].set_data_3d(time_points, y_risk, z_risk)
        
        phase_safety = 3.0  # Different phase
        y_safety = radius * np.sin(time_points * 0.2 + phase_safety)
        z_safety = np.array(sim.safety_history[:frame+1])
        metric_lines['safety'].set_data_3d(time_points, y_safety, z_safety)
        
        # Domain curves - each on their own spiral
        for i, (domain, _) in enumerate(top_domains):
            domain_phase = i * 1.0  # Different phase for each domain
            y_domain = radius * np.sin(time_points * 0.2 + domain_phase)
            z_domain = np.array(sim.domain_history[domain][:frame+1])
            metric_lines[domain.value].set_data_3d(time_points, y_domain, z_domain)
        
        # Breakthrough points
        breakthrough_steps = [step for step, _ in sim.breakthroughs_history if step < frame]
        if breakthrough_steps:
            # Update breakthrough scatter points
            b_x = np.array(breakthrough_steps)
            b_y = radius * np.sin(b_x * 0.2 + phase)  # Same spiral as tech
            b_z = np.array([sim.history[step] for step in breakthrough_steps])
            breakthrough_scatter._offsets3d = (b_x, b_y, b_z)
        else:
            # No breakthroughs yet
            breakthrough_scatter._offsets3d = ([], [], [])
        
        # Rotate view for added 3D effect (slow rotation)
        angle = frame * 0.5  # 0.5 degrees per frame
        ax.view_init(elev=20, azim=angle % 360)
        
        return [metric_lines['tech'], metric_lines['risk'], metric_lines['safety']] + \
               [metric_lines[domain.value] for domain, _ in top_domains] + \
               [breakthrough_scatter]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True)
    
    # Save animation with higher DPI for better quality
    anim.save(filename, writer='ffmpeg', fps=20, dpi=120)
    plt.close(fig)
    print(f"3D animation saved as '{filename}'")
        
def export_simulation_data(sim: SingularitySimulation, filename: str = "singularity_data.json") -> None:
    """Export simulation data to a JSON file for external analysis"""
    data = {
        "config": {
            "initial_tech": sim.tech_level,
            "growth_rate": sim.growth_rate,
            "singularity_threshold": sim.singularity_threshold,
            "growth_model": sim.growth_model.value,
            "enable_breakthroughs": sim.enable_breakthroughs,
            "enable_resource_constraints": sim.enable_resource_constraints,
            "enable_societal_impacts": sim.enable_societal_impacts,
            "enable_domain_interdependence": sim.enable_domain_interdependence
        },
        "results": {
            "time_steps": sim.time_steps,
            "final_tech_level": sim.tech_level,
            "tech_history": sim.history,
            "growth_rates": sim.growth_rates,
            "breakthroughs": [(step, event.name) for step, event in sim.breakthroughs_history],
            "domains": {domain.value: level for domain, level in sim.domains.items()},
            "domain_history": {domain.value: history for domain, history in sim.domain_history.items()},
            "risk_history": sim.risk_history,
            "safety_history": sim.safety_history,
            "state_snapshots": [{k: v if not isinstance(v, dict) or k != 'domains' 
                               else {str(dk.value) if hasattr(dk, 'value') else str(dk): dv for dk, dv in v.items()} 
                               for k, v in snapshot.items()} 
                              for snapshot in sim.state_snapshots],
            "narrative_events": sim.narrative_events,
            "emergent_properties": {
                "intelligence": sim.intelligence_quotient,
                "complexity": sim.complexity_index, 
                "creativity": sim.creativity_score,
                "adaptability": sim.adaptability,
                "consciousness": sim.consciousness_level
            },
            "emergent_property_history": sim.emergent_property_history,
            "personality_traits": sim.personality_traits,
            "goals": sim.goals,
            "narrative_state": sim.narrative_state
        }
    }
    
    if sim.enable_resource_constraints:
        data["results"]["resources"] = {resource.value: level for resource, level in sim.resources.items()}
        data["results"]["resource_history"] = {resource.value: history for resource, history in sim.resource_history.items()}
        
    if sim.enable_societal_impacts:
        data["results"]["societal_impacts"] = {impact.value: level for impact, level in sim.societal_impacts.items()}
        data["results"]["impact_history"] = {impact.value: history for impact, history in sim.impact_history.items()}
        
    # Convert to JSON and save to file
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Simulation data exported to '{filename}'")
    except Exception as e:
        print(f"Error exporting data: {e}")

def run_simulation(args: argparse.Namespace) -> None:
    """Run a demonstration of the singularity simulation"""
    print("=" * 60)
    print("ULTRA-ADVANCED TECHNOLOGICAL SINGULARITY SIMULATION")
    print("=" * 60)
    
    # Create a scenario if specified
    scenario = None
    if args.scenario:
        scenario = create_scenario(args.scenario)
        print(f"Using scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
    
    # Apply the growth model from args or scenario
    growth_model = GrowthModel(args.model)
    if scenario and not args.override_scenario:
        growth_model = scenario.growth_model
    
    # Create simulation instance
    sim = SingularitySimulation(
        initial_tech=args.initial_tech,
        growth_rate=args.growth_rate,
        singularity_threshold=args.threshold,
        growth_model=growth_model,
        enable_breakthroughs=args.breakthroughs,
        stochasticity=args.stochasticity,
        scenario=scenario,
        enable_resource_constraints=args.resource_constraints,
        enable_societal_impacts=args.societal_impacts,
        enable_domain_interdependence=args.domain_interdependence,
        interactive_mode=args.interactive,
        save_history=True
    )
    
    # Print initial configuration
    print("\nSIMULATION CONFIGURATION:")
    print(f"Initial technology level: {sim.tech_level:.2f}")
    print(f"Growth rate: {sim.growth_rate:.2f}")
    print(f"Growth model: {growth_model.value}")
    print(f"Breakthroughs enabled: {sim.enable_breakthroughs}")
    print(f"Resource constraints: {sim.enable_resource_constraints}")
    print(f"Societal impacts: {sim.enable_societal_impacts}")
    print(f"Domain interdependence: {sim.enable_domain_interdependence}")
    print(f"Stochasticity factor: {sim.stochasticity:.2f}")
    print(f"Singularity threshold: {sim.singularity_threshold}")
    print(f"Estimated time to singularity: {sim.time_to_singularity()}")
    print(f"Projected singularity date: {sim.predict_date(timedelta(days=args.step_days))}")
    
    if args.detailed_domains:
        print("\nInitial domain levels:")
        for domain, level in sim.domains.items():
            print(f"- {domain.value}: {level:.2f}")
    
    print("-" * 60)
    print("\nRunning simulation...")
    
    breakthroughs = sim.advance(args.steps)
    
    print(f"\nAfter {sim.time_steps} steps:")
    tech_level_str = "inf" if np.isinf(sim.tech_level) else f"{sim.tech_level:.2f}"
    growth_accel = (sim.tech_level/sim.history[0]) - 1
    growth_accel_str = "inf" if np.isinf(growth_accel) else f"{growth_accel:.2f}"
    
    print(f"Technology level: {tech_level_str}")
    print(f"Growth acceleration: {growth_accel_str}x initial")
    print(f"New estimate to singularity: {sim.time_to_singularity()}")
    
    # Display breakthroughs
    if breakthroughs:
        print("\nBreakthroughs occurred:")
        for step, event in breakthroughs:
            print(f"Step {step}: {event}")
    
    # Display domain information
    print("\n" + sim.describe_dominant_domains())
    
    # Display resource information if enabled
    if sim.enable_resource_constraints:
        print("\nFinal resource levels:")
        for resource, level in sorted(sim.resources.items(), key=lambda x: x[1], reverse=True):
            initial = 100.0  # All resources start at 100
            change = ((level - initial) / initial) * 100
            change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
            print(f"- {resource.value}: {level:.1f} ({change_str})")
    
    # Display societal impacts if enabled
    if sim.enable_societal_impacts:
        print("\nSocietal impacts:")
        for impact, level in sorted(sim.societal_impacts.items(), key=lambda x: abs(x[1]), reverse=True):
            impact_str = f"+{level:.1f}" if level >= 0 else f"{level:.1f}"
            print(f"- {impact.value}: {impact_str}")
    
    # Display risk and safety information
    print(f"\nExistential risk level: {sim.existential_risk:.1f}%")
    print(f"Safety measures level: {sim.safety_measures:.1f}%")
    
    # Run Monte Carlo analysis if requested
    if args.monte_carlo:
        print("\nRunning Monte Carlo analysis...")
        mc_results = sim.run_monte_carlo(args.monte_carlo_iterations)
        
        reached_pct = (mc_results["reached_singularity"] / args.monte_carlo_iterations) * 100
        print(f"Singularity reached in {reached_pct:.1f}% of simulations")
        
        if mc_results["steps_to_singularity"]:
            avg_steps = sum(mc_results["steps_to_singularity"]) / len(mc_results["steps_to_singularity"])
            print(f"Average steps to singularity: {avg_steps:.1f}")
            
        print("\nDomain dominance at singularity:")
        for domain, count in sorted(mc_results["domain_dominance"].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                pct = (count / max(1, mc_results["reached_singularity"])) * 100
                print(f"- {domain.value.title()}: {pct:.1f}%")
    
    # Generate visualizations
    print("\nGenerating visualization...")
    try:
        # Only plot if tech level is not infinite
        if np.isinf(sim.tech_level):
            print("Technology level reached infinity - skipping visualization")
        else:
            sim.plot(include_domains=args.plot_domains, plot_3d=args.plot_3d)
            
            if args.animate:
                create_animated_plots(sim, animate_3d=args.animate_3d)
    except ImportError as e:
        print(f"Visualization error: {e}")
    
    # Export data if requested
    if args.export_data:
        export_simulation_data(sim, args.export_file)
    
    print("\nCompleted singularity simulation")
    
    # Print key insights
    print("\nKEY INSIGHTS:")
    if sim.tech_level >= sim.singularity_threshold:
        print("- Technological singularity was reached!")
        
        # Find the step when singularity was reached
        for i, level in enumerate(sim.history):
            if level >= sim.singularity_threshold:
                singularity_step = i
                break
                
        print(f"- Singularity occurred at step {singularity_step} (out of {sim.time_steps})")
        
        # Calculate acceleration
        accel_factor = sim.tech_level / sim.history[0]
        accel_str = "infinite" if np.isinf(accel_factor) else f"{accel_factor:.1f}x"
        print(f"- Technology accelerated by a factor of {accel_str}")
    else:
        print("- Technological singularity was not reached")
        print(f"- Current progress: {(sim.tech_level / sim.singularity_threshold) * 100:.1f}% toward singularity")
        print(f"- Estimated {sim.time_to_singularity()} to reach singularity")
    
    # Identify key breakthroughs
    if breakthroughs:
        print("- Key breakthroughs that drove progress:")
        for i, (step, event) in enumerate(breakthroughs):
            if i < 3 or step > sim.time_steps - 5:  # Show first 3 and last few
                print(f"  * Step {step}: {event.name}")
    
    # Risk assessment
    if sim.existential_risk > 70:
        print("- WARNING: Existential risk levels are EXTREMELY HIGH")
    elif sim.existential_risk > 40:
        print("- WARNING: Existential risk levels are SIGNIFICANT")
    elif sim.existential_risk > 20:
        print("- Existential risk levels are moderate")
    else:
        print("- Existential risk levels remain relatively low")
    
    # Identify most advanced domain
    top_domain = max(sim.domains.items(), key=lambda x: x[1])[0]
    print(f"- Most advanced technology domain: {top_domain.value}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultra-advanced technological singularity simulation")
    
    # Basic parameters
    parser.add_argument("--initial-tech", type=float, default=1.0,
                        help="Initial technology level")
    parser.add_argument("--growth-rate", type=float, default=0.5,
                        help="Base growth rate")
    parser.add_argument("--threshold", type=float, default=1000,
                        help="Technology level considered to be singularity")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Number of time steps to simulate")
    
    # Hyperdimensional simulation options
    hyperspace_group = parser.add_argument_group('Hyperdimensional Simulation Options')
    hyperspace_group.add_argument("--enable-multiverse", action="store_true",
                        help="Enable multiverse branching simulation")
    hyperspace_group.add_argument("--dimensional-variance", type=float, default=0.3,
                        help="Variance factor for dimensional branch events (0-1)")
    hyperspace_group.add_argument("--enable-quantum-effects", action="store_true",
                        help="Enable quantum superposition and entanglement effects")
    hyperspace_group.add_argument("--enable-sentience-emergence", action="store_true",
                        help="Enable emergent sentience and goal formation")
    hyperspace_group.add_argument("--reality-restructuring", action="store_true",
                        help="Enable reality restructuring capabilities")
    hyperspace_group.add_argument("--enable-entropy-reversal", action="store_true",
                        help="Enable entropy reversal field development")
    hyperspace_group.add_argument("--substrate-transfer-probability", type=float, default=0.01,
                        help="Probability of substrate transfer events (0-1)")
    hyperspace_group.add_argument("--enable-temporal-manipulation", action="store_true",
                        help="Enable temporal manipulation and anomalies")
    hyperspace_group.add_argument("--enable-domain-fusion", action="store_true",
                        help="Enable technology domain fusion for emergent properties")
    hyperspace_group.add_argument("--emergent-paradigm-probability", type=float, default=0.05,
                        help="Probability of emergent paradigm shifts (0-1)")
    parser.add_argument("--step-days", type=int, default=365,
                        help="Days per simulation step for date prediction")
    
    # Growth model and randomness
    parser.add_argument("--model", type=str, 
                        choices=[m.value for m in GrowthModel], 
                        default="exponential",
                        help="Growth model to use")
    parser.add_argument("--stochasticity", type=float, default=0.1,
                        help="Random variation in growth (0-1)")
    
    # Feature toggles
    parser.add_argument("--breakthroughs", action="store_true",
                        help="Enable breakthrough events")
    parser.add_argument("--resource-constraints", action="store_true",
                        help="Enable resource constraints")
    parser.add_argument("--societal-impacts", action="store_true",
                        help="Enable societal impact modeling")
    parser.add_argument("--domain-interdependence", action="store_true",
                        help="Enable domain interdependence")
    
    # Visualization options
    parser.add_argument("--plot-domains", action="store_true",
                        help="Include domain-specific growth in visualization")
    parser.add_argument("--detailed-domains", action="store_true",
                        help="Show detailed domain information")
    parser.add_argument("--animate", action="store_true",
                        help="Create animated visualization of the simulation")
    parser.add_argument("--animate-3d", action="store_true", default=True,
                        help="Use 3D animation instead of 2D")
    parser.add_argument("--plot-3d", action="store_true", default=True,
                        help="Use 3D plots instead of 2D")
    
    # Analysis options
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo analysis")
    parser.add_argument("--monte-carlo-iterations", type=int, default=100,
                        help="Number of iterations for Monte Carlo analysis")
    
    # Scenario options
    parser.add_argument("--scenario", type=str, 
                        choices=["ai_revolution", "biotech_renaissance", "physics_breakthrough", 
                                 "balanced_development", "nanotech_explosion"],
                        help="Use a predefined scenario")
    parser.add_argument("--override-scenario", action="store_true",
                        help="Override scenario settings with command line arguments")
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode with user decision points")
    
    # Data export
    parser.add_argument("--export-data", action="store_true",
                        help="Export simulation data to a JSON file")
    parser.add_argument("--export-file", type=str, default="singularity_data.json",
                        help="Filename for exporting data")
    
    # Advanced demo modes that enable multiple features at once
    parser.add_argument("--demo-detailed", action="store_true",
                        help="Run with all advanced features enabled (resources, impacts, interdependence)")
    parser.add_argument("--demo-interactive", action="store_true",
                        help="Run an interactive demo with resource management and decision points")
    parser.add_argument("--demo-hyperdimensional", action="store_true",
                        help="Run with all hyperdimensional features enabled (multiverse, quantum effects, etc)")
    parser.add_argument("--demo-transcendent", action="store_true",
                        help="Run with transcendent capabilities (substrate independence, reality restructuring)")
    
    args = parser.parse_args()
    
    # Apply demo presets
    if args.demo_detailed:
        args.breakthroughs = True
        args.resource_constraints = True
        args.societal_impacts = True
        args.domain_interdependence = True
        args.plot_domains = True
        args.detailed_domains = True
        args.stochasticity = 0.2
        
    if args.demo_interactive:
        args.interactive = True
        args.breakthroughs = True
        args.resource_constraints = True
        args.societal_impacts = True
        args.domain_interdependence = True
        args.plot_domains = True
        args.detailed_domains = True
    
    if args.demo_hyperdimensional:
        # Enable all hyperdimensional features
        args.enable_multiverse = True
        args.enable_quantum_effects = True
        args.enable_sentience_emergence = True
        args.enable_domain_fusion = True
        args.enable_temporal_manipulation = True
        args.dimensional_variance = 0.5  # Higher variance for more visible effects
        args.emergent_paradigm_probability = 0.1  # Higher probability for more paradigm shifts
        # Also enable base features
        args.breakthroughs = True
        args.resource_constraints = True
        args.societal_impacts = True
        args.domain_interdependence = True
        # Advanced visualization
        args.plot_domains = True
        args.detailed_domains = True
        args.plot_3d = True
        args.animate = True
        args.animate_3d = True
        # Use a more advanced growth model
        args.model = "recursive_improvement"
        
    if args.demo_transcendent:
        # Enable transcendent capabilities
        args.reality_restructuring = True
        args.enable_entropy_reversal = True
        args.substrate_transfer_probability = 0.05  # Higher probability of substrate transfers
        # Also enable hyperdimensional features
        args.enable_multiverse = True
        args.enable_quantum_effects = True
        args.enable_sentience_emergence = True
        args.enable_domain_fusion = True
        args.enable_temporal_manipulation = True
        # And base features
        args.breakthroughs = True
        args.resource_constraints = True
        args.societal_impacts = True
        args.domain_interdependence = True
        # Advanced visualization
        args.plot_domains = True
        args.detailed_domains = True
        args.plot_3d = True
        args.animate = True
        # Use the most advanced growth model
        args.model = "emergent_complexity"
    
    return args

if __name__ == "__main__":
    args = parse_args()
    run_simulation(args)