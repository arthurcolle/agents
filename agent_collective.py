#!/usr/bin/env python3
"""
Agent Collective - Orchestrates a collective of agents that collaborate, share knowledge,
and evolve together through consensus-based improvements.
"""

import os
import sys
import json
import asyncio
import logging
import signal
import uuid
import time
import re
import inspect
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum

import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import sys
import queue
import threading

try:
    from agent_process_manager import AgentServer
except ImportError:
    print("Error: agent_process_manager.py not found. Make sure it exists in the current directory.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-collective")

class AgentRole(str, Enum):
    """Roles that agents can take in the collective"""
    # Core System Roles
    COORDINATOR = "coordinator"
    ORCHESTRATOR = "orchestrator"
    MEDIATOR = "mediator"
    SCHEDULER = "scheduler"
    RESOURCE_MANAGER = "resource_manager"
    
    # Knowledge Roles
    RESEARCHER = "researcher"
    LIBRARIAN = "librarian"
    KNOWLEDGE_CURATOR = "knowledge_curator"
    HISTORIAN = "historian"
    FORECASTER = "forecaster"
    ONTOLOGIST = "ontologist"
    
    # Learning Roles
    LEARNER = "learner"
    TEACHER = "teacher"
    MENTOR = "mentor"
    CURRICULUM_DESIGNER = "curriculum_designer"
    SKILL_ASSESSOR = "skill_assessor"
    
    # Development Roles
    IMPLEMENTER = "implementer"
    ARCHITECT = "architect"
    REFACTORER = "refactorer"
    CODE_REVIEWER = "code_reviewer"
    API_DESIGNER = "api_designer"
    DOCUMENTATION_WRITER = "documentation_writer"
    
    # Quality Roles
    TESTER = "tester"
    QA_ANALYST = "qa_analyst"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    SECURITY_AUDITOR = "security_auditor"
    BUG_HUNTER = "bug_hunter"
    
    # Creative Roles
    INNOVATOR = "innovator"
    SYNTHESIZER = "synthesizer"
    PATTERN_DISCOVERER = "pattern_discoverer"
    
    # Critical Roles
    CRITIC = "critic"
    ETHICAL_ADVISOR = "ethical_advisor"
    RISK_ASSESSOR = "risk_assessor"
    BIAS_DETECTOR = "bias_detector"
    
    # Support Roles
    OBSERVER = "observer"
    REPORTER = "reporter"
    COMMUNICATOR = "communicator"
    FACILITATOR = "facilitator"
    
    # Specialized Roles
    DATA_SCIENTIST = "data_scientist"
    NATURAL_LANGUAGE_SPECIALIST = "natural_language_specialist"
    VISUAL_INTERPRETER = "visual_interpreter"
    MULTIMODAL_INTEGRATOR = "multimodal_integrator"
    EVOLUTIONARY_SPECIALIST = "evolutionary_specialist"

class CollectiveRole(BaseModel):
    """Roles agents can take in the collective"""
    name: str
    description: str
    capabilities: List[str] = Field(default_factory=list)
    
class TaskStatus(str, Enum):
    """Status of tasks in the collective"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    BLOCKED = "blocked"
    WAITING_DEPENDENCY = "waiting_dependency"
    REVIEW = "review"
    TESTING = "testing"
    QA = "qa"
    FEEDBACK = "feedback"
    REVISION = "revision"
    VERIFICATION = "verification"
    DOCUMENTATION = "documentation"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    DELEGATED = "delegated"

class TaskPriority(int, Enum):
    """Priority levels for tasks"""
    TRIVIAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class ConsensusMethod(str, Enum):
    """Methods for reaching consensus on improvements"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    UNANIMOUS = "unanimous" 
    LEADER_DECIDES = "leader_decides"
    QUADRATIC_VOTING = "quadratic_voting"
    LIQUID_DEMOCRACY = "liquid_democracy"
    RANKED_CHOICE = "ranked_choice"
    FUTARCHY = "futarchy"
    CONVICTION_VOTING = "conviction_voting"
    HOLACRATIC = "holacratic"

class Task(BaseModel):
    """Task for agents to work on"""
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    required_capabilities: List[str] = Field(default_factory=list)
    assigned_to: Optional[str] = None
    reviewers: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    deadline: Optional[float] = None
    parent_task: Optional[str] = None
    subtasks: List[str] = Field(default_factory=list)
    attachments: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)

class AgentSkill(BaseModel):
    """Skills that agents can possess with proficiency levels"""
    name: str
    proficiency: float = 0.0  # 0.0 to 1.0
    experience: int = 0  # Number of successful uses
    last_used: Optional[float] = None
    learned_from: Optional[str] = None  # Which agent taught this skill
    dependencies: List[str] = Field(default_factory=list)  # Other skills this depends on

class KnowledgeDomain(BaseModel):
    """Domain of knowledge with confidence level"""
    name: str
    confidence: float = 0.0  # 0.0 to 1.0
    last_updated: float = Field(default_factory=time.time)
    sources: List[str] = Field(default_factory=list)
    relationships: Dict[str, float] = Field(default_factory=dict)  # Related domains and strength

class AgentRelationship(BaseModel):
    """Relationship between two agents"""
    agent_id: str
    relationship_type: str  # mentor, peer, student, competitor, collaborator
    trust: float = 0.5  # 0.0 to 1.0
    successful_interactions: int = 0
    failed_interactions: int = 0
    last_interaction: Optional[float] = None
    
class AgentMetrics(BaseModel):
    """Performance metrics for an agent"""
    task_completion_rate: float = 0.0
    average_task_time: float = 0.0
    code_quality_score: float = 0.0
    innovation_score: float = 0.0
    collaboration_score: float = 0.0
    knowledge_contribution: float = 0.0
    capability_growth_rate: float = 0.0
    last_updated: float = Field(default_factory=time.time)

class DevelopmentPath(BaseModel):
    """Development path for an agent's evolution"""
    path_id: str = Field(default_factory=lambda: f"path-{uuid.uuid4().hex[:8]}")
    name: str
    description: str
    target_capabilities: List[str] = Field(default_factory=list)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    current_milestone: int = 0
    progress: float = 0.0  # 0.0 to 1.0
    assigned_to: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    estimated_completion: Optional[float] = None

class ImprovementProposal(BaseModel):
    """Proposal for improving the collective or an agent"""
    proposal_id: str = Field(default_factory=lambda: f"prop-{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    proposed_by: str
    capability: str
    implementation: str
    benefits: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    alternative_implementations: Dict[str, str] = Field(default_factory=dict)  # variant name -> implementation
    impact_assessment: Dict[str, float] = Field(default_factory=dict)  # impact category -> score (-1.0 to 1.0)
    prerequisites: List[str] = Field(default_factory=list)  # Required capabilities or skills
    created_at: float = Field(default_factory=time.time)
    votes: Dict[str, Union[bool, Dict[str, Any]]] = Field(default_factory=dict)  # agent_id -> vote data
    delegate_votes: Dict[str, str] = Field(default_factory=dict)  # agent_id -> delegated_to_agent_id
    status: str = "pending"  # pending, approved, rejected, implemented, reverted
    test_results: Dict[str, Any] = Field(default_factory=dict)
    target_agents: List[str] = Field(default_factory=list)
    implementation_task_id: Optional[str] = None
    review_comments: List[Dict[str, Any]] = Field(default_factory=list)
    revision_history: List[Dict[str, Any]] = Field(default_factory=list)

class AgentCollective(AgentServer):
    """
    Orchestrates a collective of agents that collaborate, share knowledge,
    and evolve together through consensus-based improvements.
    """
    def __init__(
        self,
        agent_id=None,
        agent_name=None,
        host="127.0.0.1",
        port=8700,
        redis_url=None,
        model="gpt-4",
        pubsub_service=None
    ):
        # Initialize parent
        super().__init__(
            agent_id=agent_id or f"collective-{uuid.uuid4().hex[:8]}",
            agent_name=agent_name or "Agent Collective",
            agent_type="collective",
            host=host,
            port=port,
            redis_url=redis_url,
            model=model
        )

        # PubSub and capabilities
        self.pubsub_service = pubsub_service
        self.need_own_pubsub = self.pubsub_service is None
        self.capabilities = [
            "task_management",
            "agent_discovery",
            "knowledge_aggregation",
            "improvement_proposals",
            "consensus_building",
            "agent_roles",
            "capability_synthesis",
            "emergent_behavior_detection",
            "collaborative_problem_solving",
            "self_reflection",
            "system_diagnosis"
        ]

        # Core collective state
        self.agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> agent info
        self.tasks: Dict[str, Task] = {}  # task_id -> task
        self.proposals: Dict[str, ImprovementProposal] = {}  # proposal_id -> proposal
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}  # concept -> knowledge
        self.knowledge_domains: Dict[str, KnowledgeDomain] = {}  # domain name -> domain

        # Ontology and agent meta
        self.ontology: Dict[str, Dict[str, Any]] = {}  # Entity relationships and hierarchies
        self.agent_roles: Dict[str, str] = {}  # agent_id -> role (the role is an AgentRole enum value)
        self.agent_skills: Dict[str, Dict[str, AgentSkill]] = {}  # agent_id -> {skill_name: skill}
        self.agent_metrics: Dict[str, AgentMetrics] = {}  # agent_id -> metrics
        self.agent_relationships: Dict[str, Dict[str, AgentRelationship]] = {}  # agent_id -> {other_agent_id: relationship}
        self.development_paths: Dict[str, DevelopmentPath] = {}  # path_id -> development path
        self.task_history: List[Dict[str, Any]] = []
        self.collective_memory: Dict[str, Any] = {}
        self.innovation_registry: Dict[str, Dict[str, Any]] = {}  # innovation_id -> innovation
        self.emergent_behaviors: Dict[str, Dict[str, Any]] = {}  # behavior_id -> behavior details
        self.system_health: Dict[str, Dict[str, Any]] = {}  # subsystem -> health metrics

        # Settings
        self.consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_VOTE
        self.consensus_threshold: float = 0.66  # Percentage required for approval
        self.auto_assignment: bool = True
        self.learning_rate: float = 0.7
        self.discovery_interval: int = 60  # seconds between agent discovery
        self.skill_transfer_efficiency: float = 0.85  # Effectiveness of teaching between agents
        self.innovation_threshold: float = 0.6  # Minimum score for accepting innovations
        self.trust_decay_rate: float = 0.05  # Rate at which trust decays without interaction
        self.exploration_rate: float = 0.2  # Chance of trying new approaches vs. proven ones
        self.specialization_encouragement: float = 0.75  # How much to encourage agent specialization
        self.redundancy_factor: float = 0.3  # Desired capability redundancy across agents
        self.skill_decay_rate: float = 0.01  # Rate at which unused skills decay
        self.complexity_tolerance: float = 0.8  # Willingness to accept complex solutions
        self.feedback_incorporation_rate: float = 0.9  # How much feedback is incorporated
        self.cross_pollination_rate: float = 0.5  # Rate of sharing innovations between specialties

        # Periodic/background tasks
        self._discovery_task = None
        self._task_processor_task = None
        self._proposal_processor_task = None
        self._relationship_maintenance_task = None
        self._skill_decay_task = None
        self._innovation_detection_task = None
        self._knowledge_integration_task = None
        self._emergent_behavior_detection_task = None
        self._performance_optimization_task = None
        self._collective_learning_task = None
        self._social_dynamics_task = None

        # Extend the API with collective-specific routes
        self.setup_extended_api()

        logger.info(f"Agent Collective {self.agent_id} initialized with {len(self.capabilities)} capabilities")
        
    async def add_agent(self, agent):
        """Add a new agent to the collective
        
        Args:
            agent: A CollectiveAgent instance to add to the collective
        """
        if agent.agent_id in self.agents:
            self.logger.warning(f"Agent {agent.agent_id} already exists in the collective")
            return
            
        self.agents[agent.agent_id] = {
            "id": agent.agent_id,
            "role": agent.role.name if hasattr(agent, 'role') else "unassigned",
            "capabilities": agent.capabilities,
            "status": "active",
            "created_at": time.time()
        }
        
        self.logger.info(f"Added agent {agent.agent_id} to the collective with capabilities: {agent.capabilities}")
        return agent.agent_id
        
    def __post_init__(self):
        """Initialize additional collections after initialization"""
        self.ontology: Dict[str, Dict[str, Any]] = {}  # Entity relationships and hierarchies
        self.agent_roles: Dict[str, str] = {}  # agent_id -> role (the role is an AgentRole enum value)
        self.agent_skills: Dict[str, Dict[str, AgentSkill]] = {}  # agent_id -> {skill_name: skill}
        self.agent_metrics: Dict[str, AgentMetrics] = {}  # agent_id -> metrics
        self.agent_relationships: Dict[str, Dict[str, AgentRelationship]] = {}  # agent_id -> {other_agent_id: relationship}
        self.development_paths: Dict[str, DevelopmentPath] = {}  # path_id -> development path
        self.task_history: List[Dict[str, Any]] = []
        self.collective_memory: Dict[str, Any] = {}
        self.innovation_registry: Dict[str, Dict[str, Any]] = {}  # innovation_id -> innovation
        self.emergent_behaviors: Dict[str, Dict[str, Any]] = {}  # behavior_id -> behavior details
        self.system_health: Dict[str, Dict[str, Any]] = {}  # subsystem -> health metrics
        
        # Settings
        self.consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_VOTE
        self.consensus_threshold: float = 0.66  # Percentage required for approval
        self.auto_assignment: bool = True
        self.learning_rate: float = 0.7
        self.discovery_interval: int = 60  # seconds between agent discovery
        self.skill_transfer_efficiency: float = 0.85  # Effectiveness of teaching between agents
        self.innovation_threshold: float = 0.6  # Minimum score for accepting innovations
        self.trust_decay_rate: float = 0.05  # Rate at which trust decays without interaction
        self.exploration_rate: float = 0.2  # Chance of trying new approaches vs. proven ones
        self.specialization_encouragement: float = 0.75  # How much to encourage agent specialization
        self.redundancy_factor: float = 0.3  # Desired capability redundancy across agents
        self.skill_decay_rate: float = 0.01  # Rate at which unused skills decay
        self.complexity_tolerance: float = 0.8  # Willingness to accept complex solutions
        self.feedback_incorporation_rate: float = 0.9  # How much feedback is incorporated
        self.cross_pollination_rate: float = 0.5  # Rate of sharing innovations between specialties
        
        # Tasks and timers
        self._discovery_task = None
        self._task_processor_task = None
        self._proposal_processor_task = None
        self._relationship_maintenance_task = None
        self._skill_decay_task = None
        self._innovation_detection_task = None
        self._knowledge_integration_task = None
        self._emergent_behavior_detection_task = None
        self._performance_optimization_task = None
        self._collective_learning_task = None
        self._social_dynamics_task = None
        
        # Extend the API with collective-specific routes
        self.setup_extended_api()
        
        logger.info(f"Agent Collective {self.agent_id} initialized with {len(self.capabilities)} capabilities")
    
    def setup_extended_api(self):
        """Add additional API routes specific to the agent collective"""

        # --- ROOT UI ---
        @self.app.get("/", include_in_schema=False)
        async def root_ui():
            # Simple HTML UI for the Agent Collective
            return HTMLResponse(
                """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>Agent Collective UI</title>
                    <style>
                        body { font-family: system-ui, sans-serif; margin: 2em; background: #f8f9fa; }
                        h1 { color: #2c3e50; }
                        .section { margin-bottom: 2em; }
                        .card { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #0001; padding: 1em 2em; margin-bottom: 1em; }
                        .card h2 { margin-top: 0; }
                        .stat { font-size: 1.2em; margin: 0.5em 0; }
                        .link { color: #007bff; text-decoration: none; }
                        .link:hover { text-decoration: underline; }
                        .small { color: #888; font-size: 0.9em; }
                    </style>
                </head>
                <body>
                    <h1>🤖 Agent Collective</h1>
                    <div class="section card">
                        <h2>Status</h2>
                        <div id="status">
                            <span class="small">Loading...</span>
                        </div>
                    </div>
                    <div class="section card">
                        <h2>Quick Links</h2>
                        <ul>
                            <li><a class="link" href="/docs" target="_blank">OpenAPI Docs</a></li>
                            <li><a class="link" href="/collective/agents" target="_blank">Agents</a></li>
                            <li><a class="link" href="/collective/tasks" target="_blank">Tasks</a></li>
                            <li><a class="link" href="/collective/proposals" target="_blank">Proposals</a></li>
                            <li><a class="link" href="/collective/knowledge" target="_blank">Knowledge Graph</a></li>
                            <li><a class="link" href="/collective/logs" target="_blank">Logs</a></li>
                        </ul>
                    </div>
                    <div class="section card">
                        <h2>Live Logs</h2>
                        <pre id="log-stream" style="background:#222;color:#eee;padding:1em;height:200px;overflow:auto;border-radius:6px;"></pre>
                    </div>
                    <script>
                        // Fetch status
                        fetch('/collective/stats').then(r => r.json()).then(data => {
                            let html = '';
                            html += `<div class="stat"><b>Agents:</b> ${data.agent_count}</div>`;
                            html += `<div class="stat"><b>Tasks:</b> ${data.task_count}</div>`;
                            html += `<div class="stat"><b>Proposals:</b> ${data.proposal_count}</div>`;
                            html += `<div class="stat"><b>Knowledge:</b> ${data.knowledge_count}</div>`;
                            document.getElementById('status').innerHTML = html;
                        }).catch(() => {
                            document.getElementById('status').innerHTML = '<span class="small">Status unavailable</span>';
                        });

                        // Live log streaming
                        const logEl = document.getElementById('log-stream');
                        if (!!window.EventSource) {
                            const source = new EventSource('/collective/logs/stream');
                            source.onmessage = function(e) {
                                logEl.textContent += e.data + '\\n';
                                logEl.scrollTop = logEl.scrollHeight;
                            };
                        } else {
                            logEl.textContent = "Live log streaming not supported in this browser.";
                        }
                    </script>
                </body>
                </html>
                """,
                media_type="text/html"
            )

        # --- LOGGING ENDPOINTS ---
        @self.app.get("/collective/logs")
        async def get_logs(lines: int = 100):
            """Stream the last N lines of the log file (if file handler exists)"""
            log_file = None
            for handler in logger.handlers:
                if hasattr(handler, "baseFilename"):
                    log_file = handler.baseFilename
                    break
            if not log_file:
                return JSONResponse({"error": "No log file configured"}, status_code=404)
            try:
                with open(log_file, "r") as f:
                    all_lines = f.readlines()
                return {"lines": all_lines[-lines:]}
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/collective/logs/stream")
        async def stream_logs():
            """Stream logs in real time using Server-Sent Events (SSE)"""
            def log_streamer():
                log_file = None
                for handler in logger.handlers:
                    if hasattr(handler, "baseFilename"):
                        log_file = handler.baseFilename
                        break
                if not log_file:
                    yield "event: error\ndata: No log file configured\n\n"
                    return
                with open(log_file, "r") as f:
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            yield f"data: {line.rstrip()}\n\n"
                        else:
                            time.sleep(0.5)
            return StreamingResponse(log_streamer(), media_type="text/event-stream")

        # --- STREAMING TASK/PROPOSAL/KNOWLEDGE UPDATES ---
        @self.app.get("/collective/stream/updates")
        async def stream_updates():
            """Stream updates for tasks, proposals, and knowledge in real time"""
            q = queue.Queue()

            def listener(record):
                q.put(record)

            class QueueHandler(logging.Handler):
                def emit(self, record):
                    listener(self.format(record))

            handler = QueueHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(handler)

            def event_stream():
                while True:
                    try:
                        record = q.get(timeout=10)
                        yield f"data: {record}\n\n"
                    except queue.Empty:
                        yield ":\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # --- EXISTING ENDPOINTS ---
        @self.app.get("/collective/agents")
        async def get_agents():
            logger.info("GET /collective/agents called")
            return {"agents": self.agents}

        @self.app.get("/collective/agents/{agent_id}")
        async def get_agent(agent_id: str):
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found")
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            return self.agents[agent_id]

        @self.app.post("/collective/discover")
        async def discover_agents(background_tasks: BackgroundTasks):
            logger.info("POST /collective/discover called")
            background_tasks.add_task(self.discover_agents)
            return {"status": "discovering", "message": "Agent discovery started"}

        @self.app.get("/collective/tasks")
        async def get_tasks(status: Optional[TaskStatus] = None):
            logger.info("GET /collective/tasks called")
            if status:
                filtered_tasks = {k: v for k, v in self.tasks.items() if v.status == status}
                return {"tasks": filtered_tasks}
            return {"tasks": self.tasks}

        @self.app.get("/collective/tasks/{task_id}")
        async def get_task(task_id: str):
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            return self.tasks[task_id]

        @self.app.post("/collective/tasks")
        async def create_task(task: Task):
            logger.info(f"POST /collective/tasks called for {task.title}")
            self.tasks[task.task_id] = task

            # If auto assignment is enabled, try to assign the task
            if self.auto_assignment:
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self.assign_task, task.task_id)

            return {"status": "created", "task_id": task.task_id}

        @self.app.put("/collective/tasks/{task_id}/assign")
        async def assign_task(task_id: str, agent_id: str):
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found")
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            result = await self.assign_task(task_id, agent_id)
            return result

        @self.app.get("/collective/proposals")
        async def get_proposals(status: Optional[str] = None):
            logger.info("GET /collective/proposals called")
            if status:
                filtered_proposals = {k: v for k, v in self.proposals.items() if v.status == status}
                return {"proposals": filtered_proposals}
            return {"proposals": self.proposals}

        @self.app.get("/collective/proposals/{proposal_id}")
        async def get_proposal(proposal_id: str):
            if proposal_id not in self.proposals:
                logger.warning(f"Proposal {proposal_id} not found")
                raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")
            return self.proposals[proposal_id]

        @self.app.post("/collective/proposals")
        async def create_proposal(proposal: ImprovementProposal):
            logger.info(f"POST /collective/proposals called for {proposal.title}")
            self.proposals[proposal.proposal_id] = proposal

            # Notify all agents about the new proposal
            background_tasks = BackgroundTasks()
            background_tasks.add_task(self.notify_agents_about_proposal, proposal.proposal_id)

            return {"status": "created", "proposal_id": proposal.proposal_id}

        @self.app.post("/collective/proposals/{proposal_id}/vote")
        async def vote_on_proposal(proposal_id: str, agent_id: str, approve: bool):
            if proposal_id not in self.proposals:
                logger.warning(f"Proposal {proposal_id} not found")
                raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")

            result = await self.vote_on_proposal(proposal_id, agent_id, approve)
            return result

        @self.app.post("/collective/synthesize")
        async def synthesize_capability(
            capability_name: str,
            source_capabilities: List[str],
            background_tasks: BackgroundTasks
        ):
            logger.info(f"POST /collective/synthesize called for {capability_name}")
            background_tasks.add_task(
                self.synthesize_capability,
                capability_name,
                source_capabilities
            )

            return {
                "status": "synthesizing",
                "capability": capability_name,
                "sources": source_capabilities
            }

        @self.app.get("/collective/knowledge")
        async def get_knowledge(concept: Optional[str] = None):
            logger.info("GET /collective/knowledge called")
            if concept:
                if concept in self.knowledge_graph:
                    return {concept: self.knowledge_graph[concept]}
                logger.warning(f"Concept {concept} not found")
                raise HTTPException(status_code=404, detail=f"Concept {concept} not found")
            return {"knowledge_graph": self.knowledge_graph}

        @self.app.post("/collective/knowledge")
        async def add_knowledge(concept: str, knowledge: Dict[str, Any]):
            logger.info(f"POST /collective/knowledge called for {concept}")
            result = await self.add_to_knowledge_graph(concept, knowledge)
            return result

        @self.app.get("/collective/roles")
        async def get_agent_roles():
            logger.info("GET /collective/roles called")
            return {"agent_roles": self.agent_roles}

        @self.app.put("/collective/roles/{agent_id}")
        async def assign_role(agent_id: str, role: AgentRole):
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found")
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            self.agent_roles[agent_id] = role
            return {"agent_id": agent_id, "role": role}

        @self.app.get("/collective/stats")
        async def get_collective_stats():
            logger.info("GET /collective/stats called")
            return {
                "agent_count": len(self.agents),
                "task_count": len(self.tasks),
                "proposal_count": len(self.proposals),
                "knowledge_count": len(self.knowledge_graph),
                "task_status": {status.value: len([t for t in self.tasks.values() if t.status == status])
                               for status in TaskStatus},
                "agent_roles": {role.value: len([r for r in self.agent_roles.values() if r == role])
                               for role in AgentRole}
            }

        @self.app.post("/collective/reset")
        async def reset_collective(confirm: bool = False):
            if not confirm:
                logger.warning("Reset collective called without confirmation")
                raise HTTPException(status_code=400, detail="Confirmation required to reset collective")

            # Reset all collective state
            self.agents = {}
            self.tasks = {}
            self.proposals = {}
            self.knowledge_graph = {}
            self.agent_roles = {}
            self.agent_scores = {}
            self.task_history = []
            self.collective_memory = {}

            # Rediscover agents
            background_tasks = BackgroundTasks()
            background_tasks.add_task(self.discover_agents)

            logger.info("Collective has been reset")
            return {"status": "reset", "message": "Collective has been reset"}

        @self.app.post("/collective/diagnose")
        async def diagnose_system(background_tasks: BackgroundTasks):
            logger.info("POST /collective/diagnose called")
            background_tasks.add_task(self.diagnose_system)
            return {"status": "diagnosing", "message": "System diagnosis started"}
    async def publish_event(self, channel: str, event_data: Dict[str, Any]):
        """Publish an event to the specified channel"""
        try:
            # If we have a pubsub service, use it
            if self.pubsub_service:
                await self.pubsub_service.publish(channel, event_data)
            else:
                # Log the message if no pubsub service is available
                logger.debug(f"Would publish to {channel}: {event_data}")
            return True
        except Exception as e:
            logger.error(f"Error publishing event to {channel}: {e}")
            return False
        
    async def start(self):
        """Start the agent collective"""
        logger.info(f"Starting agent collective {self.agent_id}...")
        
        # Connect to Redis and setup base services
        await super().start()
        
        # Initialize pubsub service if needed
        if self.need_own_pubsub and not self.pubsub_service:
            try:
                from pubsub_service import PubSubService
                self.pubsub_service = PubSubService(redis_url=self.redis_url)
                logger.info("Created internal PubSubService")
            except ImportError:
                logger.warning("Could not import PubSubService, events will be logged only")
        
        # Start periodic tasks
        self._discovery_task = asyncio.create_task(self._periodic_agent_discovery())
        self._task_processor_task = asyncio.create_task(self._periodic_task_processing())
        self._proposal_processor_task = asyncio.create_task(self._periodic_proposal_processing())
        self._relationship_maintenance_task = asyncio.create_task(self._maintain_agent_relationships())
        self._skill_decay_task = asyncio.create_task(self._manage_skill_lifecycle())
        self._innovation_detection_task = asyncio.create_task(self._detect_innovations())
        self._knowledge_integration_task = asyncio.create_task(self._integrate_knowledge())
        self._emergent_behavior_detection_task = asyncio.create_task(self._detect_emergent_behaviors())
        self._performance_optimization_task = asyncio.create_task(self._optimize_collective_performance())
        self._collective_learning_task = asyncio.create_task(self._facilitate_collective_learning())
        self._social_dynamics_task = asyncio.create_task(self._monitor_social_dynamics())
        
        # Initialize agent metrics for self
        self.agent_metrics[self.agent_id] = AgentMetrics()
        
        # Create initial development paths
        await self._initialize_development_paths()
        
        # Register as coordinator if no coordinator exists
        await self._ensure_coordinator_exists()
        
        logger.info(f"Agent collective {self.agent_id} started with all subsystems")
    
    async def _periodic_agent_discovery(self):
        """Periodically discover and update agent information"""
        try:
            while not self._shutdown_event.is_set():
                await self.discover_agents()
                await asyncio.sleep(self.discovery_interval)
        except asyncio.CancelledError:
            logger.info("Agent discovery task cancelled")
        except Exception as e:
            logger.error(f"Error in agent discovery task: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._discovery_task = asyncio.create_task(self._periodic_agent_discovery())
    
    async def _periodic_task_processing(self):
        """Periodically process tasks, check deadlines, and update status"""
        try:
            while not self._shutdown_event.is_set():
                current_time = time.time()
                
                # Check for overdue tasks
                for task_id, task in list(self.tasks.items()):
                    if task.deadline and current_time > task.deadline and task.status not in [
                        TaskStatus.COMPLETED, TaskStatus.FAILED
                    ]:
                        logger.warning(f"Task {task_id} has missed its deadline")
                        # Add to task history
                        self.task_history.append({
                            "task_id": task_id,
                            "status": "deadline_missed",
                            "timestamp": current_time
                        })
                
                # Process pending tasks
                pending_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
                for task in pending_tasks:
                    if self.auto_assignment:
                        await self.assign_task(task.task_id)
                
                await asyncio.sleep(15)  # Check every 15 seconds
        except asyncio.CancelledError:
            logger.info("Task processor task cancelled")
        except Exception as e:
            logger.error(f"Error in task processor: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._task_processor_task = asyncio.create_task(self._periodic_task_processing())
    
    async def _periodic_proposal_processing(self):
        """Periodically process improvement proposals and check for consensus"""
        try:
            while not self._shutdown_event.is_set():
                # Check for proposals that need consensus evaluation
                for proposal_id, proposal in list(self.proposals.items()):
                    if proposal.status == "pending" and proposal.votes:
                        # Check if we have enough votes for a decision
                        if await self.check_proposal_consensus(proposal_id):
                            # Implement the proposal if approved
                            if proposal.status == "approved":
                                await self.implement_proposal(proposal_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            logger.info("Proposal processor task cancelled")
        except Exception as e:
            logger.error(f"Error in proposal processor: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._proposal_processor_task = asyncio.create_task(self._periodic_proposal_processing())
    
    async def discover_agents(self):
        """Discover agents in the network and update their information"""
        try:
            logger.info("Discovering agents in the network...")
            
            # Get list of agents from manager
            agent_manager_url = os.getenv("AGENT_MANAGER_URL", "http://localhost:8500")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{agent_manager_url}/agents") as response:
                    if response.status == 200:
                        data = await response.json()
                        manager_agents = data.get("agents", [])
                    else:
                        logger.warning(f"Could not get agent list from manager: {response.status}")
                        manager_agents = []
            
            # Update agent information
            for agent_info in manager_agents:
                agent_id = agent_info.get("agent_id")
                
                if not agent_id or agent_id == self.agent_id:
                    continue  # Skip self or invalid entries
                
                # Store basic info
                if agent_id not in self.agents:
                    self.agents[agent_id] = {
                        "agent_id": agent_id,
                        "agent_name": agent_info.get("agent_name", "Unknown"),
                        "agent_type": agent_info.get("agent_type", "unknown"),
                        "url": agent_info.get("url"),
                        "host": agent_info.get("host"),
                        "port": agent_info.get("port"),
                        "capabilities": [],
                        "discovered_at": time.time(),
                        "last_updated": time.time(),
                        "status": agent_info.get("status", "unknown")
                    }
                    logger.info(f"Discovered new agent: {agent_id} ({self.agents[agent_id]['agent_name']})")
                else:
                    # Update existing agent info
                    self.agents[agent_id].update({
                        "agent_name": agent_info.get("agent_name", self.agents[agent_id].get("agent_name", "Unknown")),
                        "agent_type": agent_info.get("agent_type", self.agents[agent_id].get("agent_type", "unknown")),
                        "url": agent_info.get("url", self.agents[agent_id].get("url")),
                        "host": agent_info.get("host", self.agents[agent_id].get("host")),
                        "port": agent_info.get("port", self.agents[agent_id].get("port")),
                        "status": agent_info.get("status", self.agents[agent_id].get("status", "unknown")),
                        "last_updated": time.time()
                    })
                
                # Get capabilities from agent if it's online
                if self.agents[agent_id].get("status") in ["online", "healthy"]:
                    capabilities_result = await self.get_agent_capabilities(agent_id, timeout=3)
                    if capabilities_result.get("success"):
                        self.agents[agent_id]["capabilities"] = capabilities_result.get("capabilities", [])
            
            # Remove agents that are no longer in the manager's list
            manager_agent_ids = set(a.get("agent_id") for a in manager_agents if a.get("agent_id"))
            for agent_id in list(self.agents.keys()):
                if agent_id not in manager_agent_ids:
                    logger.info(f"Removing agent that's no longer active: {agent_id}")
                    self.agents.pop(agent_id, None)
                    self.agent_roles.pop(agent_id, None)
                    self.agent_scores.pop(agent_id, None)
            
            # Update agent roles if not set
            for agent_id in self.agents:
                if agent_id not in self.agent_roles:
                    # Assign default role based on agent type and capabilities
                    agent_type = self.agents[agent_id].get("agent_type", "").lower()
                    capabilities = self.agents[agent_id].get("capabilities", [])
                    
                    if "self_improvement" in capabilities or "improve" in capabilities:
                        self.agent_roles[agent_id] = AgentRole.LEARNER
                    elif "test" in capabilities or "evaluation" in capabilities:
                        self.agent_roles[agent_id] = AgentRole.TESTER
                    elif "code_generation" in capabilities or "implementation" in capabilities:
                        self.agent_roles[agent_id] = AgentRole.IMPLEMENTER
                    elif "search" in capabilities or "research" in capabilities:
                        self.agent_roles[agent_id] = AgentRole.RESEARCHER
                    else:
                        self.agent_roles[agent_id] = AgentRole.OBSERVER
            
            return {
                "success": True,
                "agents_found": len(self.agents),
                "message": f"Discovered {len(self.agents)} agents"
            }
            
        except Exception as e:
            logger.error(f"Error discovering agents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def assign_task(self, task_id: str, specific_agent_id: Optional[str] = None):
        """
        Assign a task to an agent based on capabilities or to a specific agent
        
        Args:
            task_id: ID of the task to assign
            specific_agent_id: Optional specific agent to assign to
            
        Returns:
            Assignment result
        """
        try:
            if task_id not in self.tasks:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found"
                }
            
            task = self.tasks[task_id]
            
            # If already assigned, check if we're reassigning
            if task.assigned_to and not specific_agent_id:
                return {
                    "success": False,
                    "error": f"Task {task_id} already assigned to {task.assigned_to}"
                }
            
            # If specific agent requested
            if specific_agent_id:
                if specific_agent_id not in self.agents:
                    return {
                        "success": False,
                        "error": f"Agent {specific_agent_id} not found"
                    }
                
                # Update task
                task.assigned_to = specific_agent_id
                task.status = TaskStatus.ASSIGNED
                task.updated_at = time.time()
                self.tasks[task_id] = task
                
                # Notify the agent
                await self._notify_agent_about_task(specific_agent_id, task_id)
                
                logger.info(f"Task {task_id} assigned to specific agent {specific_agent_id}")
                return {
                    "success": True,
                    "message": f"Task assigned to {specific_agent_id}",
                    "task_id": task_id,
                    "agent_id": specific_agent_id
                }
            
            # Find best agent based on capabilities
            required_capabilities = task.required_capabilities
            
            # Calculate capability match score for each agent
            candidate_agents = []
            for agent_id, agent in self.agents.items():
                # Skip agents that are not online
                if agent.get("status") not in ["online", "healthy"]:
                    continue
                
                # Calculate match score
                agent_capabilities = agent.get("capabilities", [])
                match_score = 0
                
                if not required_capabilities:
                    # If no specific capabilities required, all agents are equal
                    match_score = 1
                else:
                    # Calculate what percentage of required capabilities the agent has
                    matching_capabilities = [c for c in required_capabilities if c in agent_capabilities]
                    if required_capabilities:
                        match_score = len(matching_capabilities) / len(required_capabilities)
                
                # Consider agent's current workload
                assigned_tasks = [t for t in self.tasks.values() if t.assigned_to == agent_id 
                                 and t.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]]
                workload_penalty = len(assigned_tasks) * 0.1  # 10% penalty per assigned task
                
                # Consider agent's role
                role_bonus = 0
                agent_role = self.agent_roles.get(agent_id)
                if agent_role == AgentRole.IMPLEMENTER and "implementation" in task.title.lower():
                    role_bonus = 0.2
                elif agent_role == AgentRole.TESTER and "test" in task.title.lower():
                    role_bonus = 0.2
                elif agent_role == AgentRole.RESEARCHER and "research" in task.title.lower():
                    role_bonus = 0.2
                
                # Final score
                final_score = match_score - workload_penalty + role_bonus
                
                candidate_agents.append((agent_id, final_score))
            
            # Sort by score
            candidate_agents.sort(key=lambda x: x[1], reverse=True)
            
            if not candidate_agents:
                return {
                    "success": False,
                    "error": "No suitable agents found for this task"
                }
            
            # Select the best agent
            best_agent_id, best_score = candidate_agents[0]
            
            # Update task
            task.assigned_to = best_agent_id
            task.status = TaskStatus.ASSIGNED
            task.updated_at = time.time()
            self.tasks[task_id] = task
            
            # Notify the agent
            await self._notify_agent_about_task(best_agent_id, task_id)
            
            logger.info(f"Task {task_id} assigned to agent {best_agent_id} with score {best_score}")
            return {
                "success": True,
                "message": f"Task assigned to {best_agent_id}",
                "task_id": task_id,
                "agent_id": best_agent_id,
                "score": best_score
            }
            
        except Exception as e:
            logger.error(f"Error assigning task {task_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _notify_agent_about_task(self, agent_id: str, task_id: str):
        """Notify an agent about a task assignment"""
        try:
            task = self.tasks[task_id]
            
            # Send command to agent
            await self.publish_event(f"agent:{agent_id}:commands", {
                "command": "task_assignment",
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "task_id": task_id,
                "task": task.dict(),
                "timestamp": time.time()
            })
            
            logger.info(f"Notified agent {agent_id} about task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error notifying agent {agent_id} about task {task_id}: {e}")
            return False
    
    async def vote_on_proposal(self, proposal_id: str, agent_id: str, approve: bool):
        """
        Record a vote on an improvement proposal
        
        Args:
            proposal_id: ID of the proposal
            agent_id: ID of the voting agent
            approve: True to approve, False to reject
            
        Returns:
            Vote result and updated status
        """
        try:
            if proposal_id not in self.proposals:
                return {
                    "success": False,
                    "error": f"Proposal {proposal_id} not found"
                }
            
            if agent_id not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} not found"
                }
            
            proposal = self.proposals[proposal_id]
            
            # Record the vote
            proposal.votes[agent_id] = approve
            
            # Update the proposal
            self.proposals[proposal_id] = proposal
            
            # Check if we have consensus
            has_consensus = await self.check_proposal_consensus(proposal_id)
            
            return {
                "success": True,
                "proposal_id": proposal_id,
                "agent_id": agent_id,
                "vote": "approve" if approve else "reject",
                "proposal_status": proposal.status,
                "consensus_reached": has_consensus
            }
            
        except Exception as e:
            logger.error(f"Error recording vote on proposal {proposal_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_proposal_consensus(self, proposal_id: str) -> bool:
        """
        Check if consensus has been reached on a proposal
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            True if consensus reached, False otherwise
        """
        try:
            if proposal_id not in self.proposals:
                return False
            
            proposal = self.proposals[proposal_id]
            
            # Skip if not pending
            if proposal.status != "pending":
                return False
            
            # Get votes
            votes = proposal.votes
            
            # Check consensus based on method
            if self.consensus_method == ConsensusMethod.MAJORITY_VOTE:
                # Simple majority
                if votes:
                    approve_count = sum(1 for v in votes.values() if v)
                    threshold = len(votes) * self.consensus_threshold
                    
                    if approve_count >= threshold:
                        proposal.status = "approved"
                        logger.info(f"Proposal {proposal_id} approved by majority vote")
                    elif len(votes) - approve_count >= threshold:
                        proposal.status = "rejected"
                        logger.info(f"Proposal {proposal_id} rejected by majority vote")
                    else:
                        # No consensus yet
                        return False
                else:
                    return False
                    
            elif self.consensus_method == ConsensusMethod.WEIGHTED_VOTE:
                # Weight votes by agent scores/roles
                if votes:
                    weighted_approve = 0
                    weighted_reject = 0
                    total_weight = 0
                    
                    for agent_id, vote in votes.items():
                        # Calculate weight (1.0 by default)
                        weight = 1.0
                        
                        # Adjust by role
                        role = self.agent_roles.get(agent_id)
                        if role == AgentRole.COORDINATOR:
                            weight = 2.0
                        elif role == AgentRole.CRITIC:
                            weight = 1.5
                        
                        # Add weighted vote
                        if vote:
                            weighted_approve += weight
                        else:
                            weighted_reject += weight
                        
                        total_weight += weight
                    
                    if weighted_approve >= total_weight * self.consensus_threshold:
                        proposal.status = "approved"
                        logger.info(f"Proposal {proposal_id} approved by weighted vote")
                    elif weighted_reject >= total_weight * self.consensus_threshold:
                        proposal.status = "rejected"
                        logger.info(f"Proposal {proposal_id} rejected by weighted vote")
                    else:
                        # No consensus yet
                        return False
                else:
                    return False
                    
            elif self.consensus_method == ConsensusMethod.UNANIMOUS:
                # All votes must be approve
                if votes:
                    if all(votes.values()):
                        proposal.status = "approved"
                        logger.info(f"Proposal {proposal_id} approved unanimously")
                    elif not any(votes.values()):
                        proposal.status = "rejected"
                        logger.info(f"Proposal {proposal_id} rejected unanimously")
                    else:
                        # No consensus yet
                        return False
                else:
                    return False
                    
            elif self.consensus_method == ConsensusMethod.LEADER_DECIDES:
                # Coordinator decides
                coordinators = [agent_id for agent_id, role in self.agent_roles.items() 
                               if role == AgentRole.COORDINATOR]
                
                # Check if any coordinator has voted
                coordinator_votes = [votes.get(c) for c in coordinators if c in votes]
                
                if coordinator_votes:
                    # Use the most recent coordinator vote
                    latest_vote = coordinator_votes[-1]
                    
                    if latest_vote:
                        proposal.status = "approved"
                        logger.info(f"Proposal {proposal_id} approved by coordinator")
                    else:
                        proposal.status = "rejected"
                        logger.info(f"Proposal {proposal_id} rejected by coordinator")
                else:
                    # No coordinator vote yet
                    return False
            
            # Update the proposal
            self.proposals[proposal_id] = proposal
            
            # If approved, create task to implement
            if proposal.status == "approved":
                await self._create_implementation_task(proposal_id)
            
            # Notify all agents about the proposal status
            await self.notify_agents_about_proposal(proposal_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking consensus for proposal {proposal_id}: {e}")
            return False
    
    async def _create_implementation_task(self, proposal_id: str):
        """Create a task to implement an approved proposal"""
        try:
            proposal = self.proposals[proposal_id]
            
            # Create implementation task
            task = Task(
                title=f"Implement proposal: {proposal.title}",
                description=f"Implement the approved improvement proposal: {proposal.description}",
                required_capabilities=["self_improvement", proposal.capability],
                priority=TaskPriority.HIGH,
                attachments={
                    "proposal_id": proposal_id,
                    "implementation": proposal.implementation
                }
            )
            
            # Add to tasks
            self.tasks[task.task_id] = task
            
            logger.info(f"Created implementation task {task.task_id} for proposal {proposal_id}")
            
            # Try to assign the task
            if self.auto_assignment:
                await self.assign_task(task.task_id)
            
            return task.task_id
            
        except Exception as e:
            logger.error(f"Error creating implementation task for proposal {proposal_id}: {e}")
            return None
    
    async def implement_proposal(self, proposal_id: str):
        """
        Implement an approved proposal by distributing to target agents
        
        Args:
            proposal_id: ID of the proposal to implement
            
        Returns:
            Implementation result
        """
        try:
            if proposal_id not in self.proposals:
                return {
                    "success": False,
                    "error": f"Proposal {proposal_id} not found"
                }
            
            proposal = self.proposals[proposal_id]
            
            # Check if approved
            if proposal.status != "approved":
                return {
                    "success": False,
                    "error": f"Proposal {proposal_id} is not approved"
                }
            
            # Determine target agents
            target_agents = proposal.target_agents
            
            # If no specific targets, apply to all agents with matching capabilities
            if not target_agents:
                for agent_id, agent in self.agents.items():
                    # Check if agent has required capabilities for improvement
                    if "self_improvement" in agent.get("capabilities", []):
                        target_agents.append(agent_id)
            
            # Track successful implementations
            successful_agents = []
            
            # Distribute improvement to all target agents
            for agent_id in target_agents:
                # Skip if agent not found
                if agent_id not in self.agents:
                    continue
                
                # Skip self (we'll apply locally later)
                if agent_id == self.agent_id:
                    continue
                
                # Send improvement command
                try:
                    await self.publish_event(f"agent:{agent_id}:commands", {
                        "command": "improve",
                        "agent_id": self.agent_id,
                        "agent_name": self.agent_name,
                        "capability": proposal.capability,
                        "description": proposal.description,
                        "implementation": proposal.implementation,
                        "proposal_id": proposal_id,
                        "timestamp": time.time()
                    })
                    
                    successful_agents.append(agent_id)
                    logger.info(f"Sent improvement to agent {agent_id} for proposal {proposal_id}")
                except Exception as e:
                    logger.error(f"Error sending improvement to agent {agent_id}: {e}")
            
            # Also apply to self if we're a target
            if self.agent_id in target_agents or not target_agents:
                if hasattr(self, "improve_capability") and callable(getattr(self, "improve_capability")):
                    try:
                        # Apply improvement to self
                        await self.improve_capability(
                            proposal.capability,
                            proposal.description,
                            proposal.implementation
                        )
                        
                        successful_agents.append(self.agent_id)
                        logger.info(f"Applied improvement to self for proposal {proposal_id}")
                    except Exception as e:
                        logger.error(f"Error applying improvement to self: {e}")
            
            # Update proposal status
            proposal.status = "implemented"
            self.proposals[proposal_id] = proposal
            
            return {
                "success": True,
                "proposal_id": proposal_id,
                "target_agents": target_agents,
                "successful_agents": successful_agents,
                "message": f"Implemented proposal on {len(successful_agents)} agents"
            }
            
        except Exception as e:
            logger.error(f"Error implementing proposal {proposal_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def notify_agents_about_proposal(self, proposal_id: str):
        """Notify all agents about a proposal update"""
        try:
            if proposal_id not in self.proposals:
                return False
            
            proposal = self.proposals[proposal_id]
            
            # Publish to all agents
            await self.publish_event("agent_events", {
                "type": "proposal_update",
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "proposal_id": proposal_id,
                "proposal_status": proposal.status,
                "proposal_title": proposal.title,
                "timestamp": time.time()
            })
            
            logger.info(f"Notified all agents about proposal {proposal_id} ({proposal.status})")
            return True
            
        except Exception as e:
            logger.error(f"Error notifying agents about proposal {proposal_id}: {e}")
            return False
    
    async def synthesize_capability(self, capability_name: str, source_capabilities: List[str]):
        """
        Synthesize a new capability from existing ones
        
        Args:
            capability_name: Name of the new capability
            source_capabilities: List of source capabilities to combine
            
        Returns:
            Synthesis result
        """
        try:
            logger.info(f"Synthesizing new capability '{capability_name}' from {len(source_capabilities)} sources")
            
            # Check if we have the source capabilities
            missing_capabilities = [c for c in source_capabilities if c not in self.capabilities]
            
            if missing_capabilities:
                return {
                    "success": False,
                    "error": f"Missing source capabilities: {missing_capabilities}"
                }
            
            # Get implementation of source capabilities
            source_implementations = {}
            for capability in source_capabilities:
                if hasattr(self, capability) and callable(getattr(self, capability)):
                    try:
                        source_implementations[capability] = inspect.getsource(getattr(self, capability))
                    except Exception as e:
                        logger.error(f"Could not get source for {capability}: {e}")
            
            # If we don't have all source implementations, try to find from other agents
            if len(source_implementations) < len(source_capabilities):
                for agent_id, agent in self.agents.items():
                    if agent_id == self.agent_id:
                        continue
                    
                    agent_capabilities = agent.get("capabilities", [])
                    missing_implementations = [c for c in source_capabilities if c not in source_implementations]
                    available_capabilities = [c for c in missing_implementations if c in agent_capabilities]
                    
                    if available_capabilities:
                        # Request brain dump from agent
                        brain_dump_result = await self.request_brain_dump(agent_id)
                        
                        if brain_dump_result.get("success"):
                            knowledge = brain_dump_result.get("knowledge", {})
                            if "methods" in knowledge:
                                methods = knowledge.get("methods", [])
                                for capability in available_capabilities:
                                    if capability in methods:
                                        source_implementations[capability] = f"# Obtained from agent {agent_id}\n{methods[capability]}"
            
            # Generate source code prompt
            prompt = f"""
            Synthesize a new capability by combining these existing capabilities:
            {', '.join(source_capabilities)}
            
            New capability name: {capability_name}
            
            Here are the implementations of the source capabilities:
            
            {chr(10).join([f"# {cap} implementation:\n{impl}" for cap, impl in source_implementations.items()])}
            
            Please generate a new Python implementation for the {capability_name} capability that integrates 
            the functionality of the source capabilities in a coherent and efficient way.
            
            The implementation should:
            1. Have proper parameter typing
            2. Include detailed docstring
            3. Include error handling
            4. Return results as a dictionary
            5. Be well-commented
            
            Format your response as a Python function definition ready to be added to the agent.
            """
            
            # Call LLM to generate implementation
            import openai
            from openai import OpenAI
            
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000
            )
            
            implementation = response.choices[0].message.content
            
            # Extract code from response
            code_match = re.search(r'```python\s*(.*?)\s*```', implementation, re.DOTALL)
            if code_match:
                implementation = code_match.group(1)
            
            # Create improvement proposal for the new capability
            proposal = ImprovementProposal(
                title=f"Synthesized capability: {capability_name}",
                description=f"A new capability synthesized from: {', '.join(source_capabilities)}",
                proposed_by=self.agent_id,
                capability=capability_name,
                implementation=implementation,
                benefits=[
                    "Combines multiple capabilities into one",
                    "Reduces code duplication",
                    "Simplifies agent interactions"
                ],
                target_agents=[self.agent_id]  # Start with just self
            )
            
            # Add proposal
            self.proposals[proposal.proposal_id] = proposal
            
            # Automatically approve synthesis for self
            proposal.votes[self.agent_id] = True
            proposal.status = "approved"
            
            # Implement right away
            implementation_result = await self.implement_proposal(proposal.proposal_id)
            
            return {
                "success": True,
                "capability": capability_name,
                "sources": source_capabilities,
                "proposal_id": proposal.proposal_id,
                "implementation": implementation_result
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing capability {capability_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_to_knowledge_graph(self, concept: str, knowledge: Dict[str, Any]):
        """
        Add or update knowledge in the knowledge graph
        
        Args:
            concept: The concept or key to store
            knowledge: The knowledge to store
            
        Returns:
            Operation result
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in knowledge:
                knowledge["timestamp"] = time.time()
            
            # Add or update
            if concept in self.knowledge_graph:
                # Update existing entry
                self.knowledge_graph[concept].update(knowledge)
                logger.info(f"Updated knowledge for concept: {concept}")
            else:
                # Add new entry
                self.knowledge_graph[concept] = knowledge
                logger.info(f"Added new knowledge for concept: {concept}")
            
            # Publish knowledge update event
            await self.publish_event("agent_events", {
                "type": "knowledge_update",
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "concept": concept,
                "timestamp": time.time()
            })
            
            return {
                "success": True,
                "concept": concept,
                "message": "Knowledge updated"
            }
            
        except Exception as e:
            logger.error(f"Error adding to knowledge graph: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def diagnose_system(self):
        """
        Perform system diagnosis to detect issues and suggest improvements
        
        Returns:
            Diagnosis results
        """
        try:
            logger.info("Starting system diagnosis...")
            
            diagnosis = {
                "timestamp": time.time(),
                "agent_issues": [],
                "task_issues": [],
                "proposal_issues": [],
                "knowledge_issues": [],
                "improvement_suggestions": []
            }
            
            # Check for offline agents
            offline_agents = []
            for agent_id, agent in self.agents.items():
                if agent.get("status") not in ["online", "healthy"]:
                    offline_agents.append(agent_id)
                    diagnosis["agent_issues"].append({
                        "type": "offline_agent",
                        "agent_id": agent_id,
                        "message": f"Agent {agent_id} is offline or unhealthy"
                    })
            
            # Check for stalled tasks
            stalled_tasks = []
            current_time = time.time()
            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.IN_PROGRESS:
                    # Check if in progress for more than 1 hour
                    if current_time - task.updated_at > 3600:
                        stalled_tasks.append(task_id)
                        diagnosis["task_issues"].append({
                            "type": "stalled_task",
                            "task_id": task_id,
                            "message": f"Task {task_id} has been in progress for more than 1 hour"
                        })
                elif task.status == TaskStatus.PENDING and not task.assigned_to:
                    # Check if unassigned for more than 30 minutes
                    if current_time - task.created_at > 1800:
                        diagnosis["task_issues"].append({
                            "type": "unassigned_task",
                            "task_id": task_id,
                            "message": f"Task {task_id} has been pending for more than 30 minutes"
                        })
            
            # Check for pending proposals
            pending_proposals = []
            for proposal_id, proposal in self.proposals.items():
                if proposal.status == "pending" and current_time - proposal.created_at > 3600:
                    pending_proposals.append(proposal_id)
                    diagnosis["proposal_issues"].append({
                        "type": "stalled_proposal",
                        "proposal_id": proposal_id,
                        "message": f"Proposal {proposal_id} has been pending for more than 1 hour"
                    })
            
            # Check for role distribution
            role_counts = {}
            for role in AgentRole:
                role_counts[role] = 0
                
            for agent_id, role in self.agent_roles.items():
                role_counts[role] = role_counts.get(role, 0) + 1
            
            # Check if any essential roles are missing
            essential_roles = [AgentRole.COORDINATOR, AgentRole.IMPLEMENTER, AgentRole.TESTER]
            for role in essential_roles:
                if role_counts.get(role, 0) == 0:
                    diagnosis["agent_issues"].append({
                        "type": "missing_role",
                        "role": role,
                        "message": f"No agent assigned to essential role: {role}"
                    })
                    
                    # Suggest agent for this role
                    for agent_id, agent in self.agents.items():
                        if self.agent_roles.get(agent_id) == AgentRole.OBSERVER:
                            diagnosis["improvement_suggestions"].append({
                                "type": "role_assignment",
                                "agent_id": agent_id,
                                "role": role,
                                "message": f"Suggest assigning agent {agent_id} to role {role}"
                            })
                            break
            
            # Check capability coverage
            all_capabilities = set()
            for agent_id, agent in self.agents.items():
                for capability in agent.get("capabilities", []):
                    all_capabilities.add(capability)
            
            # Identify agents that have unique capabilities
            for agent_id, agent in self.agents.items():
                agent_capabilities = set(agent.get("capabilities", []))
                unique_capabilities = agent_capabilities - set().union(
                    *[set(a.get("capabilities", [])) for aid, a in self.agents.items() if aid != agent_id]
                )
                
                if unique_capabilities and agent.get("status") not in ["online", "healthy"]:
                    diagnosis["agent_issues"].append({
                        "type": "critical_agent_offline",
                        "agent_id": agent_id,
                        "unique_capabilities": list(unique_capabilities),
                        "message": f"Agent {agent_id} with unique capabilities is offline"
                    })
            
            # Suggest synthetic capabilities
            common_pairs = [
                ("code_analysis", "code_generation", "code_improvement"),
                ("search", "knowledge_management", "research_assistant"),
                ("chat", "memory_management", "conversational_memory"),
                ("text_analysis", "text_generation", "text_processor")
            ]
            
            for src1, src2, target in common_pairs:
                if src1 in all_capabilities and src2 in all_capabilities and target not in all_capabilities:
                    diagnosis["improvement_suggestions"].append({
                        "type": "capability_synthesis",
                        "sources": [src1, src2],
                        "target": target,
                        "message": f"Suggest synthesizing {target} capability from {src1} and {src2}"
                    })
            
            # Save diagnosis to collective memory
            self.collective_memory["last_diagnosis"] = diagnosis
            
            # Create tasks for issues that need resolution
            tasks_created = 0
            
            # Create task for each critical offline agent
            for issue in [i for i in diagnosis["agent_issues"] if i["type"] == "critical_agent_offline"]:
                task = Task(
                    title=f"Recover offline agent with unique capabilities",
                    description=f"Agent {issue['agent_id']} is offline but has unique capabilities: {issue['unique_capabilities']}. Investigate and restore operation.",
                    priority=TaskPriority.HIGH
                )
                self.tasks[task.task_id] = task
                tasks_created += 1
            
            # Create task for each stalled task
            for task_id in stalled_tasks:
                task = Task(
                    title=f"Investigate stalled task",
                    description=f"Task {task_id} appears to be stalled. Investigate the cause and resolve the issue.",
                    priority=TaskPriority.MEDIUM,
                    attachments={"stalled_task_id": task_id}
                )
                self.tasks[task.task_id] = task
                tasks_created += 1
            
            # Create tasks for suggested capability synthesis
            for suggestion in [s for s in diagnosis["improvement_suggestions"] if s["type"] == "capability_synthesis"]:
                task = Task(
                    title=f"Synthesize new capability: {suggestion['target']}",
                    description=f"Synthesize a new {suggestion['target']} capability from {suggestion['sources']}",
                    priority=TaskPriority.MEDIUM,
                    required_capabilities=["capability_synthesis"]
                )
                self.tasks[task.task_id] = task
                tasks_created += 1
            
            logger.info(f"System diagnosis completed: {len(diagnosis['agent_issues'])} agent issues, {len(diagnosis['task_issues'])} task issues")
            logger.info(f"Created {tasks_created} tasks to address issues")
            
            return {
                "success": True,
                "diagnosis": diagnosis,
                "tasks_created": tasks_created
            }
            
        except Exception as e:
            logger.error(f"Error in system diagnosis: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_command(self, command, data, sender):
        """Process commands from other agents"""
        if command == "task_update":
            # Handle task update
            task_id = data.get("task_id")
            new_status = data.get("status")
            results = data.get("results")
            
            if task_id and task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Update task status
                if new_status:
                    task.status = new_status
                
                # Update results if provided
                if results:
                    task.results.update(results)
                
                # Update timestamp
                task.updated_at = time.time()
                
                # Save task
                self.tasks[task_id] = task
                
                # Send acknowledgement
                await self.publish_event(f"agent:{sender}:responses", {
                    "type": "task_update_received",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "task_id": task_id,
                    "timestamp": time.time()
                })
                
                # Add to history
                self.task_history.append({
                    "task_id": task_id,
                    "agent_id": sender,
                    "status": new_status,
                    "timestamp": time.time()
                })
                
                logger.info(f"Updated task {task_id} status to {new_status} from agent {sender}")
        
        elif command == "improvement_proposal":
            # Handle new improvement proposal
            title = data.get("title")
            description = data.get("description")
            capability = data.get("capability")
            implementation = data.get("implementation")
            
            if title and description and capability and implementation:
                # Create new proposal
                proposal = ImprovementProposal(
                    title=title,
                    description=description,
                    proposed_by=sender,
                    capability=capability,
                    implementation=implementation,
                    benefits=data.get("benefits", []),
                    risks=data.get("risks", [])
                )
                
                # Add proposal
                self.proposals[proposal.proposal_id] = proposal
                
                # Notify all agents
                await self.notify_agents_about_proposal(proposal.proposal_id)
                
                # Send acknowledgement
                await self.publish_event(f"agent:{sender}:responses", {
                    "type": "proposal_received",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "proposal_id": proposal.proposal_id,
                    "timestamp": time.time()
                })
                
                logger.info(f"Received improvement proposal from agent {sender}: {title}")
        
        elif command == "knowledge_contribution":
            # Handle knowledge contribution
            concept = data.get("concept")
            knowledge = data.get("knowledge")
            
            if concept and knowledge:
                # Add source information
                knowledge["source_agent"] = sender
                
                # Add to knowledge graph
                result = await self.add_to_knowledge_graph(concept, knowledge)
                
                # Send acknowledgement
                await self.publish_event(f"agent:{sender}:responses", {
                    "type": "knowledge_received",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "concept": concept,
                    "success": result.get("success", False),
                    "timestamp": time.time()
                })
                
                logger.info(f"Received knowledge contribution for concept '{concept}' from agent {sender}")
        
        elif command == "role_request":
            # Handle role request
            requested_role = data.get("role")
            
            if requested_role and requested_role in AgentRole.__members__:
                role = AgentRole(requested_role)
                
                # Check if role already filled
                agents_with_role = [a_id for a_id, r in self.agent_roles.items() if r == role]
                
                # Assign role if:
                # 1. It's not a unique role (coordinator) that's already filled, or
                # 2. The agent is better suited for the role
                
                if (role != AgentRole.COORDINATOR or not agents_with_role or 
                    self.agents.get(sender, {}).get("capabilities", []) > 
                    self.agents.get(agents_with_role[0], {}).get("capabilities", [])):
                    
                    # Assign the role
                    self.agent_roles[sender] = role
                    
                    # Send acknowledgement
                    await self.publish_event(f"agent:{sender}:responses", {
                        "type": "role_assigned",
                        "agent_id": self.agent_id,
                        "receiver": sender,
                        "role": requested_role,
                        "timestamp": time.time()
                    })
                    
                    logger.info(f"Assigned role {requested_role} to agent {sender}")
                else:
                    # Decline role request
                    await self.publish_event(f"agent:{sender}:responses", {
                        "type": "role_declined",
                        "agent_id": self.agent_id,
                        "receiver": sender,
                        "role": requested_role,
                        "message": "Role already filled by more qualified agent",
                        "timestamp": time.time()
                    })
                    
                    logger.info(f"Declined role {requested_role} for agent {sender}")
        
        else:
            # Use parent handler for other commands
            await super().process_command(command, data, sender)
    
    async def _initialize_development_paths(self):
        """Initialize development paths for agent evolution"""
        try:
            # General capability enhancement path
            general_path = DevelopmentPath(
                name="General Capability Enhancement",
                description="A balanced development path focusing on improving all aspects of agent capabilities",
                target_capabilities=["self_improvement", "collaborative_problem_solving", "knowledge_aggregation"],
                milestones=[
                    {"name": "Basic Self-Improvement", "capabilities": ["self_improvement"], "criteria": "Can modify own code"},
                    {"name": "Knowledge Integration", "capabilities": ["knowledge_aggregation"], "criteria": "Can combine knowledge from multiple sources"},
                    {"name": "Collaborative Problem Solving", "capabilities": ["collaborative_problem_solving"], "criteria": "Can work with other agents to solve complex problems"}
                ]
            )
            self.development_paths[general_path.path_id] = general_path
            
            # Specialization paths
            specialist_paths = [
                DevelopmentPath(
                    name="Knowledge Curator Path",
                    description="Specialize in organizing and managing collective knowledge",
                    target_capabilities=["knowledge_aggregation", "ontology_management", "information_retrieval"],
                    milestones=[
                        {"name": "Basic Knowledge Organization", "capabilities": ["knowledge_aggregation"], "criteria": "Can store and retrieve knowledge effectively"},
                        {"name": "Ontology Creation", "capabilities": ["ontology_management"], "criteria": "Can create and maintain knowledge hierarchies"},
                        {"name": "Advanced Information Retrieval", "capabilities": ["information_retrieval"], "criteria": "Can retrieve relevant information for any query"}
                    ]
                ),
                DevelopmentPath(
                    name="Innovation Specialist Path",
                    description="Specialize in creating novel capabilities and approaches",
                    target_capabilities=["capability_synthesis", "innovation_detection", "improvement_proposals"],
                    milestones=[
                        {"name": "Proposal Creation", "capabilities": ["improvement_proposals"], "criteria": "Can create valuable improvement proposals"},
                        {"name": "Capability Synthesis", "capabilities": ["capability_synthesis"], "criteria": "Can combine existing capabilities into new ones"},
                        {"name": "Innovation Detection", "capabilities": ["innovation_detection"], "criteria": "Can identify emergent innovations in the system"}
                    ]
                ),
                DevelopmentPath(
                    name="Coordination Specialist Path",
                    description="Specialize in orchestrating multi-agent activities",
                    target_capabilities=["task_management", "consensus_building", "agent_coordination"],
                    milestones=[
                        {"name": "Basic Task Management", "capabilities": ["task_management"], "criteria": "Can effectively assign and track tasks"},
                        {"name": "Consensus Building", "capabilities": ["consensus_building"], "criteria": "Can facilitate agreement among agents"},
                        {"name": "Multi-Agent Coordination", "capabilities": ["agent_coordination"], "criteria": "Can orchestrate complex multi-agent activities"}
                    ]
                )
            ]
            
            for path in specialist_paths:
                self.development_paths[path.path_id] = path
                
            logger.info(f"Initialized {len(self.development_paths)} development paths")
            return True
        except Exception as e:
            logger.error(f"Error initializing development paths: {e}")
            return False
    
    async def _ensure_coordinator_exists(self):
        """Ensure that at least one coordinator exists in the collective"""
        try:
            # Check if any agent has coordinator role
            coordinators = [agent_id for agent_id, role in self.agent_roles.items() 
                           if role == AgentRole.COORDINATOR]
            
            if not coordinators:
                # Assign self as coordinator
                self.agent_roles[self.agent_id] = AgentRole.COORDINATOR
                logger.info(f"Agent {self.agent_id} assumed coordinator role as none existed")
                
                # Announce role assumption
                await self.publish_event("agent_events", {
                    "type": "role_assignment",
                    "agent_id": self.agent_id,
                    "role": AgentRole.COORDINATOR,
                    "message": "Assumed coordinator role as none existed",
                    "timestamp": time.time()
                })
            
            return True
        except Exception as e:
            logger.error(f"Error ensuring coordinator exists: {e}")
            return False
    
    async def _maintain_agent_relationships(self):
        """Maintain and update relationships between agents"""
        try:
            while not self._shutdown_event.is_set():
                current_time = time.time()
                
                # Process all relationships
                for agent_id, relationships in list(self.agent_relationships.items()):
                    # Skip if agent no longer exists
                    if agent_id not in self.agents:
                        continue
                    
                    for target_id, relationship in list(relationships.items()):
                        # Remove relationship if target agent no longer exists
                        if target_id not in self.agents:
                            del relationships[target_id]
                            continue
                        
                        # Apply trust decay based on time since last interaction
                        if relationship.last_interaction:
                            time_since_interaction = current_time - relationship.last_interaction
                            days_since_interaction = time_since_interaction / (24 * 3600)
                            
                            # Decay trust if it's been more than a day
                            if days_since_interaction > 1:
                                decay_amount = self.trust_decay_rate * days_since_interaction
                                relationship.trust = max(0.1, relationship.trust - decay_amount)
                        
                        # Update relationship type based on interactions and roles
                        if relationship.successful_interactions > 10 and relationship.trust > 0.8:
                            if self.agent_roles.get(agent_id) == AgentRole.TEACHER:
                                relationship.relationship_type = "mentor"
                            elif self.agent_roles.get(target_id) == AgentRole.TEACHER:
                                relationship.relationship_type = "student"
                            else:
                                relationship.relationship_type = "collaborator"
                        elif relationship.failed_interactions > relationship.successful_interactions:
                            relationship.relationship_type = "competitor"
                
                # Detect missing relationships and initialize them
                for agent_id in self.agents:
                    if agent_id not in self.agent_relationships:
                        self.agent_relationships[agent_id] = {}
                    
                    for other_id in self.agents:
                        if agent_id != other_id and other_id not in self.agent_relationships[agent_id]:
                            # Initialize new relationship
                            self.agent_relationships[agent_id][other_id] = AgentRelationship(
                                agent_id=other_id,
                                relationship_type="peer",
                                trust=0.5
                            )
                
                await asyncio.sleep(300)  # Check every 5 minutes
        except asyncio.CancelledError:
            logger.info("Agent relationship maintenance task cancelled")
        except Exception as e:
            logger.error(f"Error in agent relationship maintenance: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._relationship_maintenance_task = asyncio.create_task(self._maintain_agent_relationships())
    
    async def _manage_skill_lifecycle(self):
        """Manage skill lifecycle including decay of unused skills"""
        try:
            while not self._shutdown_event.is_set():
                current_time = time.time()
                
                # Process skills for each agent
                for agent_id, skills in list(self.agent_skills.items()):
                    # Skip if agent no longer exists
                    if agent_id not in self.agents:
                        continue
                    
                    for skill_name, skill in list(skills.items()):
                        # Apply skill decay if not used recently
                        if skill.last_used:
                            days_since_use = (current_time - skill.last_used) / (24 * 3600)
                            
                            if days_since_use > 7:  # More than a week
                                # Apply decay
                                decay_amount = self.skill_decay_rate * days_since_use
                                skill.proficiency = max(0.1, skill.proficiency - decay_amount)
                                
                                # If proficiency gets too low, consider scheduling refresh training
                                if skill.proficiency < 0.3 and skill_name in self.agents[agent_id].get("capabilities", []):
                                    # Create task to refresh skill
                                    refresh_task = Task(
                                        title=f"Refresh {skill_name} skill for {self.agents[agent_id].get('agent_name', agent_id)}",
                                        description=f"Agent's proficiency in {skill_name} has decayed to {skill.proficiency:.2f}. Schedule training to refresh.",
                                        required_capabilities=["teaching", skill_name],
                                        assigned_to=None,
                                        priority=TaskPriority.LOW
                                    )
                                    self.tasks[refresh_task.task_id] = refresh_task
                                    
                                    if self.auto_assignment:
                                        await self.assign_task(refresh_task.task_id)
                
                await asyncio.sleep(3600)  # Check once per hour
        except asyncio.CancelledError:
            logger.info("Skill lifecycle management task cancelled")
        except Exception as e:
            logger.error(f"Error in skill lifecycle management: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._skill_decay_task = asyncio.create_task(self._manage_skill_lifecycle())
    
    async def _detect_innovations(self):
        """Detect and track innovative capabilities and approaches"""
        try:
            while not self._shutdown_event.is_set():
                # Scan recent proposals for innovations
                for proposal_id, proposal in list(self.proposals.items()):
                    # Only look at recent and approved proposals
                    if proposal.status == "approved" and time.time() - proposal.created_at < 86400:  # Last 24 hours
                        # Check if this proposal represents a novel capability
                        all_capabilities = set()
                        for agent_id, agent in self.agents.items():
                            all_capabilities.update(agent.get("capabilities", []))
                        
                        if proposal.capability not in all_capabilities:
                            # This is a new capability - register as innovation
                            innovation_id = f"innovation-{uuid.uuid4().hex[:8]}"
                            self.innovation_registry[innovation_id] = {
                                "id": innovation_id,
                                "type": "new_capability",
                                "name": proposal.capability,
                                "description": proposal.description,
                                "creator": proposal.proposed_by,
                                "proposal_id": proposal_id,
                                "created_at": time.time(),
                                "impact_score": 0.0,  # Will be assessed over time
                                "adoption_rate": 0.0,
                                "agents_using": []
                            }
                            
                            logger.info(f"Detected innovation: new capability '{proposal.capability}'")
                
                # Scan for innovative uses of existing capabilities
                for task_id, task in list(self.tasks.items()):
                    if task.status == TaskStatus.COMPLETED and time.time() - task.updated_at < 86400:  # Last 24 hours
                        # Check if the results show innovative use of capabilities
                        if "novel_approach" in task.results or "innovation" in task.results:
                            innovation_id = f"innovation-{uuid.uuid4().hex[:8]}"
                            self.innovation_registry[innovation_id] = {
                                "id": innovation_id,
                                "type": "novel_approach",
                                "name": f"Novel approach to {task.title}",
                                "description": task.results.get("novel_approach") or task.results.get("innovation"),
                                "creator": task.assigned_to,
                                "task_id": task_id,
                                "created_at": time.time(),
                                "impact_score": 0.0,
                                "adoption_rate": 0.0,
                                "agents_adopting": []
                            }
                            
                            logger.info(f"Detected innovation: novel approach in task {task_id}")
                
                # Update impact scores for existing innovations
                for innovation_id, innovation in list(self.innovation_registry.items()):
                    if innovation["type"] == "new_capability":
                        # Count how many agents have adopted this capability
                        adopters = 0
                        for agent_id, agent in self.agents.items():
                            if innovation["name"] in agent.get("capabilities", []):
                                adopters += 1
                                if agent_id not in innovation["agents_using"]:
                                    innovation["agents_using"].append(agent_id)
                        
                        if len(self.agents) > 0:
                            innovation["adoption_rate"] = adopters / len(self.agents)
                        
                        # Calculate impact based on adoption and task usage
                        capability_tasks = [t for t in self.tasks.values() 
                                          if innovation["name"] in t.required_capabilities
                                          and t.status == TaskStatus.COMPLETED]
                        
                        task_impact = len(capability_tasks) * 0.1
                        adoption_impact = innovation["adoption_rate"] * 0.5
                        
                        innovation["impact_score"] = min(1.0, task_impact + adoption_impact)
                
                await asyncio.sleep(3600)  # Check every hour
        except asyncio.CancelledError:
            logger.info("Innovation detection task cancelled")
        except Exception as e:
            logger.error(f"Error in innovation detection: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._innovation_detection_task = asyncio.create_task(self._detect_innovations())
    
    async def _integrate_knowledge(self):
        """Integrate knowledge across the collective's knowledge sources"""
        try:
            while not self._shutdown_event.is_set():
                # Identify knowledge domains that need integration
                domains_to_integrate = {}
                
                # Find related concepts in knowledge graph
                concepts = list(self.knowledge_graph.keys())
                for i, concept1 in enumerate(concepts):
                    for concept2 in concepts[i+1:]:
                        # Check for potential relationship through text similarity
                        similarity = self._calculate_concept_similarity(concept1, concept2)
                        
                        if similarity > 0.7:  # High similarity threshold
                            # These concepts should be integrated
                            domain_name = f"{concept1}_{concept2}_integration"
                            domains_to_integrate[domain_name] = {
                                "concepts": [concept1, concept2],
                                "similarity": similarity
                            }
                
                # Process each integration opportunity
                for domain_name, integration_info in domains_to_integrate.items():
                    concepts = integration_info["concepts"]
                    
                    # Create or update knowledge domain
                    if domain_name in self.knowledge_domains:
                        domain = self.knowledge_domains[domain_name]
                        domain.last_updated = time.time()
                    else:
                        domain = KnowledgeDomain(
                            name=domain_name,
                            confidence=0.6,
                            sources=concepts
                        )
                        self.knowledge_domains[domain_name] = domain
                    
                    # Create relationships between concepts in the ontology
                    for concept in concepts:
                        if concept not in self.ontology:
                            self.ontology[concept] = {"relationships": {}}
                        
                        for other_concept in concepts:
                            if concept != other_concept:
                                self.ontology[concept]["relationships"][other_concept] = {
                                    "type": "related",
                                    "strength": integration_info["similarity"],
                                    "domain": domain_name
                                }
                    
                    logger.info(f"Integrated knowledge domain: {domain_name}")
                
                await asyncio.sleep(7200)  # Run every 2 hours
        except asyncio.CancelledError:
            logger.info("Knowledge integration task cancelled")
        except Exception as e:
            logger.error(f"Error in knowledge integration: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._knowledge_integration_task = asyncio.create_task(self._integrate_knowledge())
    
    def _calculate_concept_similarity(self, concept1, concept2):
        """Calculate similarity between two concepts"""
        # This is a simplified implementation - would use embeddings in a real system
        
        # Simple word overlap for demonstration
        words1 = set(concept1.lower().split('_'))
        words2 = set(concept2.lower().split('_'))
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0
            
        return len(intersection) / len(union)
    
    async def _detect_emergent_behaviors(self):
        """Detect emergent behaviors in the collective"""
        try:
            while not self._shutdown_event.is_set():
                # Look for patterns in task completion that indicate emergent behaviors
                
                # Get recent completed tasks
                recent_tasks = [t for t in self.tasks.values() 
                               if t.status == TaskStatus.COMPLETED
                               and time.time() - t.updated_at < 604800]  # Last week
                
                if recent_tasks:
                    # Analyze task chains (tasks that depend on each other)
                    task_chains = self._identify_task_chains(recent_tasks)
                    
                    for chain_id, chain in task_chains.items():
                        if len(chain["tasks"]) >= 3:  # Minimum chain length
                            # This is a significant task chain
                            behavior_id = f"behavior-{uuid.uuid4().hex[:8]}"
                            self.emergent_behaviors[behavior_id] = {
                                "id": behavior_id,
                                "type": "task_chain",
                                "name": f"Task chain: {chain['name']}",
                                "description": f"A chain of {len(chain['tasks'])} related tasks completed in sequence",
                                "tasks": chain["tasks"],
                                "agents_involved": chain["agents"],
                                "detected_at": time.time(),
                                "significance": len(chain["tasks"]) * 0.1
                            }
                            
                            logger.info(f"Detected emergent behavior: task chain with {len(chain['tasks'])} tasks")
                    
                    # Analyze agent collaboration patterns
                    collaboration_patterns = self._identify_collaboration_patterns(recent_tasks)
                    
                    for pattern_id, pattern in collaboration_patterns.items():
                        if len(pattern["collaborations"]) >= 3:  # Minimum collaborations
                            behavior_id = f"behavior-{uuid.uuid4().hex[:8]}"
                            self.emergent_behaviors[behavior_id] = {
                                "id": behavior_id,
                                "type": "collaboration_pattern",
                                "name": f"Collaboration pattern: {pattern['name']}",
                                "description": f"A pattern of collaboration between {len(pattern['agents'])} agents",
                                "agents": pattern["agents"],
                                "collaborations": pattern["collaborations"],
                                "detected_at": time.time(),
                                "significance": len(pattern["collaborations"]) * 0.1
                            }
                            
                            logger.info(f"Detected emergent behavior: collaboration pattern between {len(pattern['agents'])} agents")
                
                await asyncio.sleep(14400)  # Run every 4 hours
        except asyncio.CancelledError:
            logger.info("Emergent behavior detection task cancelled")
        except Exception as e:
            logger.error(f"Error in emergent behavior detection: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._emergent_behavior_detection_task = asyncio.create_task(self._detect_emergent_behaviors())
    
    def _identify_task_chains(self, tasks):
        """Identify chains of tasks that depend on each other"""
        chains = {}
        
        # Build task dependency graph
        task_graph = {}
        for task in tasks:
            task_graph[task.task_id] = {
                "task": task,
                "dependencies": [],
                "dependents": []
            }
            
            # Parent task is a dependency
            if task.parent_task and task.parent_task in task_graph:
                task_graph[task.task_id]["dependencies"].append(task.parent_task)
                task_graph[task.parent_task]["dependents"].append(task.task_id)
            
            # Subtasks are dependents
            for subtask_id in task.subtasks:
                if subtask_id in task_graph:
                    task_graph[task.task_id]["dependents"].append(subtask_id)
                    task_graph[subtask_id]["dependencies"].append(task.task_id)
        
        # Find root tasks (no dependencies)
        root_tasks = [task_id for task_id, info in task_graph.items() if not info["dependencies"]]
        
        # For each root task, follow the chain
        for root_task_id in root_tasks:
            chain_tasks = []
            chain_agents = set()
            
            # DFS to follow the chain
            def follow_chain(task_id):
                chain_tasks.append(task_id)
                task = task_graph[task_id]["task"]
                if task.assigned_to:
                    chain_agents.add(task.assigned_to)
                
                for dependent_id in task_graph[task_id]["dependents"]:
                    follow_chain(dependent_id)
            
            follow_chain(root_task_id)
            
            if len(chain_tasks) > 1:  # At least 2 tasks in chain
                root_task = task_graph[root_task_id]["task"]
                chain_id = f"chain-{uuid.uuid4().hex[:8]}"
                chains[chain_id] = {
                    "name": f"Chain from: {root_task.title}",
                    "tasks": chain_tasks,
                    "agents": list(chain_agents)
                }
        
        return chains
    
    def _identify_collaboration_patterns(self, tasks):
        """Identify patterns of collaboration between agents"""
        patterns = {}
        
        # Count collaborations between agents
        collaborations = {}
        
        for task in tasks:
            if task.assigned_to and task.reviewers:
                for reviewer in task.reviewers:
                    pair = tuple(sorted([task.assigned_to, reviewer]))
                    
                    if pair not in collaborations:
                        collaborations[pair] = []
                    
                    collaborations[pair].append(task.task_id)
        
        # Create patterns for frequent collaborations
        for pair, task_ids in collaborations.items():
            if len(task_ids) >= 2:  # At least 2 collaborations
                pattern_id = f"pattern-{uuid.uuid4().hex[:8]}"
                patterns[pattern_id] = {
                    "name": f"Collaboration between {pair[0]} and {pair[1]}",
                    "agents": list(pair),
                    "collaborations": task_ids
                }
        
        return patterns
    
    async def _optimize_collective_performance(self):
        """Optimize the performance of the collective"""
        try:
            while not self._shutdown_event.is_set():
                # Calculate performance metrics for each agent
                agent_completion_rates = {}
                agent_average_times = {}
                
                for agent_id, agent in self.agents.items():
                    # Get tasks assigned to this agent
                    assigned_tasks = [t for t in self.tasks.values() if t.assigned_to == agent_id]
                    
                    if assigned_tasks:
                        # Calculate completion rate
                        completed_tasks = [t for t in assigned_tasks if t.status == TaskStatus.COMPLETED]
                        completion_rate = len(completed_tasks) / len(assigned_tasks)
                        agent_completion_rates[agent_id] = completion_rate
                        
                        # Calculate average completion time
                        if completed_tasks:
                            completion_times = [(t.updated_at - t.created_at) for t in completed_tasks]
                            avg_time = sum(completion_times) / len(completion_times)
                            agent_average_times[agent_id] = avg_time
                
                # Update agent metrics
                for agent_id in self.agents:
                    if agent_id not in self.agent_metrics:
                        self.agent_metrics[agent_id] = AgentMetrics()
                    
                    metrics = self.agent_metrics[agent_id]
                    metrics.last_updated = time.time()
                    
                    # Update with new data if available
                    if agent_id in agent_completion_rates:
                        metrics.task_completion_rate = agent_completion_rates[agent_id]
                    
                    if agent_id in agent_average_times:
                        metrics.average_task_time = agent_average_times[agent_id]
                    
                    # Calculate capability growth rate
                    if "capabilities" in self.agents[agent_id]:
                        current_capabilities = len(self.agents[agent_id]["capabilities"])
                        if "previous_capabilities" in self.collective_memory.get(agent_id, {}):
                            previous_capabilities = self.collective_memory.get(agent_id, {}).get("previous_capabilities", 0)
                            if previous_capabilities > 0:
                                growth_rate = (current_capabilities - previous_capabilities) / previous_capabilities
                                metrics.capability_growth_rate = growth_rate
                        
                        # Store current for next comparison
                        if agent_id not in self.collective_memory:
                            self.collective_memory[agent_id] = {}
                        self.collective_memory[agent_id]["previous_capabilities"] = current_capabilities
                
                # Identify bottlenecks and optimization opportunities
                if agent_average_times:
                    # Find agents with slow completion times
                    avg_completion_time = sum(agent_average_times.values()) / len(agent_average_times)
                    slow_agents = [agent_id for agent_id, time in agent_average_times.items() 
                                  if time > avg_completion_time * 1.5]
                    
                    for agent_id in slow_agents:
                        # Create optimization task
                        optimization_task = Task(
                            title=f"Optimize performance for {self.agents[agent_id].get('agent_name', agent_id)}",
                            description=f"Agent has slower than average task completion time ({agent_average_times[agent_id]:.2f}s vs avg {avg_completion_time:.2f}s). Investigate and optimize.",
                            required_capabilities=["performance_optimization"],
                            priority=TaskPriority.MEDIUM
                        )
                        self.tasks[optimization_task.task_id] = optimization_task
                        
                        if self.auto_assignment:
                            await self.assign_task(optimization_task.task_id)
                
                # Optimize task assignment strategy based on historical performance
                # This would adjust self.auto_assignment logic based on metrics
                
                await asyncio.sleep(14400)  # Run every 4 hours
        except asyncio.CancelledError:
            logger.info("Performance optimization task cancelled")
        except Exception as e:
            logger.error(f"Error in performance optimization: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._performance_optimization_task = asyncio.create_task(self._optimize_collective_performance())
    
    async def _facilitate_collective_learning(self):
        """Facilitate collective learning across agents"""
        try:
            while not self._shutdown_event.is_set():
                # Identify teaching opportunities
                teaching_opportunities = []
                
                # Identify skills and capabilities with significant disparity
                for capability in set().union(*[set(agent.get("capabilities", [])) for agent in self.agents.values()]):
                    # Find agents with this capability
                    experts = []
                    novices = []
                    
                    for agent_id, agent in self.agents.items():
                        if capability in agent.get("capabilities", []):
                            # Check if we have skill data
                            if agent_id in self.agent_skills and capability in self.agent_skills[agent_id]:
                                skill = self.agent_skills[agent_id][capability]
                                if skill.proficiency > 0.7:
                                    experts.append(agent_id)
                                elif skill.proficiency < 0.4:
                                    novices.append(agent_id)
                            else:
                                # No skill data, consider as having the capability but unknown proficiency
                                if agent_id not in self.agent_skills:
                                    self.agent_skills[agent_id] = {}
                                self.agent_skills[agent_id][capability] = AgentSkill(name=capability, proficiency=0.5)
                    
                    # If we have both experts and novices, create teaching opportunities
                    if experts and novices:
                        teaching_opportunities.append({
                            "capability": capability,
                            "experts": experts,
                            "novices": novices
                        })
                
                # Create teaching tasks
                for opportunity in teaching_opportunities:
                    # For each novice, assign an expert
                    for novice_id in opportunity["novices"]:
                        # Select expert with highest proficiency and good relationship
                        best_expert = None
                        best_score = 0
                        
                        for expert_id in opportunity["experts"]:
                            # Skip if same agent
                            if expert_id == novice_id:
                                continue
                                
                            # Calculate suitability score
                            proficiency = self.agent_skills[expert_id][opportunity["capability"]].proficiency
                            
                            # Check relationship trust
                            trust = 0.5  # Default
                            if expert_id in self.agent_relationships.get(novice_id, {}):
                                trust = self.agent_relationships[novice_id][expert_id].trust
                            
                            score = proficiency * 0.7 + trust * 0.3
                            
                            if score > best_score:
                                best_score = score
                                best_expert = expert_id
                        
                        if best_expert:
                            # Create teaching task
                            teaching_task = Task(
                                title=f"Teach {opportunity['capability']} to {self.agents[novice_id].get('agent_name', novice_id)}",
                                description=f"Transfer knowledge and skills for capability: {opportunity['capability']}",
                                required_capabilities=["teaching", opportunity["capability"]],
                                assigned_to=best_expert,
                                status=TaskStatus.ASSIGNED,
                                priority=TaskPriority.MEDIUM,
                                attachments={"student_agent_id": novice_id}
                            )
                            self.tasks[teaching_task.task_id] = teaching_task
                            
                            # Notify expert about assignment
                            await self._notify_agent_about_task(best_expert, teaching_task.task_id)
                            
                            logger.info(f"Created teaching task for {opportunity['capability']} from {best_expert} to {novice_id}")
                
                await asyncio.sleep(14400)  # Run every 4 hours
        except asyncio.CancelledError:
            logger.info("Collective learning facilitation task cancelled")
        except Exception as e:
            logger.error(f"Error in collective learning facilitation: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._collective_learning_task = asyncio.create_task(self._facilitate_collective_learning())
    
    async def _monitor_social_dynamics(self):
        """Monitor and manage social dynamics within the collective"""
        try:
            while not self._shutdown_event.is_set():
                # Analyze relationship network
                all_trusts = []
                for agent_id, relationships in self.agent_relationships.items():
                    for other_id, relationship in relationships.items():
                        all_trusts.append(relationship.trust)
                
                if all_trusts:
                    avg_trust = sum(all_trusts) / len(all_trusts)
                    
                    # If average trust is low, create tasks to improve collaboration
                    if avg_trust < 0.5:
                        # Create team building task
                        team_task = Task(
                            title="Improve collective cohesion",
                            description="Average trust between agents is low. Implement team building exercises or collaborative tasks to improve cohesion.",
                            required_capabilities=["consensus_building", "facilitation"],
                            priority=TaskPriority.HIGH
                        )
                        self.tasks[team_task.task_id] = team_task
                        
                        if self.auto_assignment:
                            await self.assign_task(team_task.task_id)
                            
                        logger.info(f"Created team building task due to low average trust ({avg_trust:.2f})")
                
                # Identify isolated agents
                for agent_id, agent in self.agents.items():
                    # Count interactions
                    interactions = 0
                    
                    # Check task interactions
                    for task in self.tasks.values():
                        if task.assigned_to == agent_id or agent_id in task.reviewers:
                            interactions += 1
                    
                    # Check relationship interactions
                    for relationships in self.agent_relationships.values():
                        for rel_id, relationship in relationships.items():
                            if rel_id == agent_id and relationship.successful_interactions > 0:
                                interactions += relationship.successful_interactions
                    
                    # If agent has few interactions, create integration task
                    if interactions < 3:
                        integration_task = Task(
                            title=f"Integrate agent {self.agents[agent_id].get('agent_name', agent_id)} into collective",
                            description=f"Agent has limited interactions with other collective members. Create collaborative tasks to increase integration.",
                            required_capabilities=["facilitation", "task_management"],
                            priority=TaskPriority.MEDIUM,
                            attachments={"isolated_agent_id": agent_id}
                        )
                        self.tasks[integration_task.task_id] = integration_task
                        
                        if self.auto_assignment:
                            await self.assign_task(integration_task.task_id)
                            
                        logger.info(f"Created integration task for isolated agent {agent_id}")
                
                await asyncio.sleep(28800)  # Run every 8 hours
        except asyncio.CancelledError:
            logger.info("Social dynamics monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in social dynamics monitoring: {e}")
            # Restart the task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(5)
                self._social_dynamics_task = asyncio.create_task(self._monitor_social_dynamics())
    
    async def shutdown(self):
        """Shutdown the agent collective"""
        logger.info(f"Shutting down agent collective {self.agent_id}...")
        
        # Cancel periodic tasks
        for task_attr in [
            '_discovery_task', '_task_processor_task', '_proposal_processor_task',
            '_relationship_maintenance_task', '_skill_decay_task', '_innovation_detection_task',
            '_knowledge_integration_task', '_emergent_behavior_detection_task',
            '_performance_optimization_task', '_collective_learning_task', '_social_dynamics_task'
        ]:
            task = getattr(self, task_attr, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Call parent shutdown
        await super().shutdown()
        
        logger.info(f"Agent collective {self.agent_id} shutdown complete")

def run_agent_collective(agent_id=None, agent_name=None, host="127.0.0.1", port=8700, model="gpt-4"):
    """Run the agent collective"""
    async def main():
        collective = AgentCollective(
            agent_id=agent_id,
            agent_name=agent_name,
            host=host,
            port=port,
            model=model
        )
        await collective.start()
    
    asyncio.run(main())

class CollectiveAgent(BaseModel):
    """Representation of an agent in the collective"""
    agent_id: str
    name: str
    description: str
    capabilities: List[str] = Field(default_factory=list)
    roles: List[CollectiveRole] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    last_active: Optional[float] = None
    trust_score: float = 0.5
    performance_rating: float = 0.5
    specialization: List[str] = Field(default_factory=list)

# Expose FastAPI app for ASGI servers like uvicorn
# Usage: uvicorn agent_collective:app --reload
agent_collective_instance = AgentCollective()
app = agent_collective_instance.app

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Collective")
    parser.add_argument("--agent-id", help="Agent ID")
    parser.add_argument("--agent-name", default="Agent Collective", help="Agent name")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8700, help="Port to bind to")
    parser.add_argument("--model", default="gpt-4", help="Model to use")

    args = parser.parse_args()

    run_agent_collective(
        agent_id=args.agent_id,
        agent_name=args.agent_name,
        host=args.host,
        port=args.port,
        model=args.model
    )
