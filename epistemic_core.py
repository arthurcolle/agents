#!/usr/bin/env python3
"""
Epistemic Core - Foundational knowledge management system for AI agents

This module provides abstractions for knowledge representation, storage, and reasoning
that can be used by AI agents without them needing to reimplement knowledge management.
"""

import os
import json
import time
import sqlite3
import numpy as np
import uuid
import logging
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("epistemic-core")

# Try to import vector database capabilities
try:
    from embedding_service import EmbeddingServiceClient, compute_embedding
    HAS_EMBEDDING_SERVICE = True
except ImportError:
    HAS_EMBEDDING_SERVICE = False
    logger.warning("EmbeddingService not available. Vector operations will be limited.")


@dataclass
class EpistemicUnit:
    """
    The fundamental unit of knowledge with epistemological properties
    
    This represents a single piece of knowledge with its associated provenance, 
    confidence, and justifications.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    confidence: float = 0.5  # 0-1 scale
    source: str = ""
    source_type: str = "external"  # external, derived, observed
    creation_time: float = field(default_factory=time.time)
    last_verified: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)  # IDs of supporting units
    contradictions: List[str] = field(default_factory=list)  # IDs of contradicting units
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_confidence(self, new_confidence: float, reason: str = None):
        """Update confidence score with justification"""
        old_confidence = self.confidence
        self.confidence = max(0.0, min(1.0, new_confidence))  # Clamp to [0,1]
        
        if reason:
            if "confidence_history" not in self.metadata:
                self.metadata["confidence_history"] = []
                
            self.metadata["confidence_history"].append({
                "timestamp": time.time(),
                "old_value": old_confidence,
                "new_value": self.confidence,
                "reason": reason
            })
    
    def add_evidence(self, evidence_id: str):
        """Add supporting evidence"""
        if evidence_id not in self.evidence:
            self.evidence.append(evidence_id)
    
    def add_contradiction(self, contradiction_id: str):
        """Add contradicting evidence"""
        if contradiction_id not in self.contradictions:
            self.contradictions.append(contradiction_id)
    
    def conflicts_with(self, other_unit: 'EpistemicUnit') -> bool:
        """Check if this unit conflicts with another unit"""
        # Simple direct contradiction
        return other_unit.id in self.contradictions or self.id in other_unit.contradictions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpistemicUnit':
        """Create from dictionary representation"""
        return cls(**data)


class EpistemicStore:
    """
    Storage system for EpistemicUnits with SQLite backend
    
    This provides persistence for knowledge units with efficient querying
    and relationship tracking.
    """
    
    def __init__(self, db_path: str):
        """Initialize the epistemicstore with the specified database path"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.cursor = None
        self._initialize_db()
        
        # Initialize embedding service if available
        self.embedding_client = EmbeddingServiceClient() if HAS_EMBEDDING_SERVICE else None
    
    def _initialize_db(self):
        """Set up database tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create units table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS epistemic_units (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL,
                source TEXT,
                source_type TEXT,
                creation_time REAL,
                last_verified REAL,
                domain TEXT,
                metadata TEXT
            )
            ''')
            
            # Create table for relationships between units
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS unit_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_unit_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                to_unit_id TEXT NOT NULL,
                confidence REAL,
                metadata TEXT,
                FOREIGN KEY (from_unit_id) REFERENCES epistemic_units(id),
                FOREIGN KEY (to_unit_id) REFERENCES epistemic_units(id)
            )
            ''')
            
            # Create table for embeddings
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS unit_embeddings (
                unit_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (unit_id) REFERENCES epistemic_units(id)
            )
            ''')
            
            # Create indexes for efficient queries
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_units_domain ON epistemic_units(domain)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_units_source ON epistemic_units(source)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_units_confidence ON epistemic_units(confidence)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_from ON unit_relations(from_unit_id)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_to ON unit_relations(to_unit_id)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_relations_type ON unit_relations(relation_type)')
            
            self.conn.commit()
            logger.info(f"EpistemicStore initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            raise
    
    def store_unit(self, unit: EpistemicUnit) -> str:
        """Store an epistemic unit in the database"""
        try:
            # Convert metadata to JSON
            metadata_json = json.dumps(unit.metadata)
            
            # Insert or update the unit
            self.cursor.execute(
                'INSERT OR REPLACE INTO epistemic_units (id, content, confidence, source, source_type, '
                'creation_time, last_verified, domain, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (unit.id, unit.content, unit.confidence, unit.source, unit.source_type,
                 unit.creation_time, unit.last_verified, unit.domain, metadata_json)
            )
            
            # Store relationships for evidence and contradictions
            for evidence_id in unit.evidence:
                self._add_relation(unit.id, "evidence", evidence_id)
            
            for contradiction_id in unit.contradictions:
                self._add_relation(unit.id, "contradiction", contradiction_id)
            
            # Generate and store embedding if embedding service is available
            if self.embedding_client:
                try:
                    embedding = self.embedding_client.embed_text(unit.content)
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    
                    self.cursor.execute(
                        'INSERT OR REPLACE INTO unit_embeddings (unit_id, embedding) VALUES (?, ?)',
                        (unit.id, embedding_bytes)
                    )
                except Exception as e:
                    logger.warning(f"Error generating embedding: {e}")
            
            self.conn.commit()
            return unit.id
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing epistemic unit: {e}")
            raise
    
    def get_unit(self, unit_id: str) -> Optional[EpistemicUnit]:
        """Retrieve an epistemic unit by ID"""
        try:
            self.cursor.execute(
                'SELECT id, content, confidence, source, source_type, creation_time, '
                'last_verified, domain, metadata FROM epistemic_units WHERE id = ?',
                (unit_id,)
            )
            
            row = self.cursor.fetchone()
            if not row:
                return None
            
            unit_data = {
                "id": row[0],
                "content": row[1],
                "confidence": row[2],
                "source": row[3],
                "source_type": row[4],
                "creation_time": row[5],
                "last_verified": row[6],
                "domain": row[7],
                "metadata": json.loads(row[8])
            }
            
            # Get evidence relationships
            self.cursor.execute(
                'SELECT to_unit_id FROM unit_relations WHERE from_unit_id = ? AND relation_type = ?',
                (unit_id, "evidence")
            )
            evidence = [row[0] for row in self.cursor.fetchall()]
            unit_data["evidence"] = evidence
            
            # Get contradiction relationships
            self.cursor.execute(
                'SELECT to_unit_id FROM unit_relations WHERE from_unit_id = ? AND relation_type = ?',
                (unit_id, "contradiction")
            )
            contradictions = [row[0] for row in self.cursor.fetchall()]
            unit_data["contradictions"] = contradictions
            
            return EpistemicUnit.from_dict(unit_data)
        except Exception as e:
            logger.error(f"Error retrieving epistemic unit: {e}")
            return None
    
    def query_units(self, query: str, top_k: int = 5, domain: str = None, min_confidence: float = 0.0) -> List[EpistemicUnit]:
        """Search for units using vector similarity or keyword matching"""
        if self.embedding_client:
            return self._vector_search(query, top_k, domain, min_confidence)
        else:
            return self._keyword_search(query, top_k, domain, min_confidence)
    
    def _vector_search(self, query: str, top_k: int = 5, domain: str = None, min_confidence: float = 0.0) -> List[EpistemicUnit]:
        """Search using vector similarity"""
        try:
            # Generate embedding for query
            query_embedding = np.array(self.embedding_client.embed_text(query), dtype=np.float32)
            
            # Build SQL query
            sql_filters = []
            sql_params = []
            
            if domain:
                sql_filters.append("u.domain = ?")
                sql_params.append(domain)
            
            if min_confidence > 0:
                sql_filters.append("u.confidence >= ?")
                sql_params.append(min_confidence)
            
            where_clause = "WHERE " + " AND ".join(sql_filters) if sql_filters else ""
            
            # Get all embeddings with their units
            sql_query = f"""
            SELECT e.embedding, u.id, u.content, u.confidence, u.source, u.source_type,
                u.creation_time, u.last_verified, u.domain, u.metadata
            FROM unit_embeddings e
            JOIN epistemic_units u ON e.unit_id = u.id
            {where_clause}
            """
            
            self.cursor.execute(sql_query, sql_params)
            rows = self.cursor.fetchall()
            
            # Calculate similarities
            results = []
            for row in rows:
                vector_bytes = row[0]
                unit_id = row[1]
                
                # Convert bytes to vector
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, vector)
                
                # Create unit object
                unit_data = {
                    "id": unit_id,
                    "content": row[2],
                    "confidence": row[3],
                    "source": row[4],
                    "source_type": row[5],
                    "creation_time": row[6],
                    "last_verified": row[7],
                    "domain": row[8],
                    "metadata": json.loads(row[9])
                }
                
                # Store similarity separately (not part of EpistemicUnit constructor)
                similarity_value = similarity
                
                # Get evidence relationships
                self.cursor.execute(
                    'SELECT to_unit_id FROM unit_relations WHERE from_unit_id = ? AND relation_type = ?',
                    (unit_id, "evidence")
                )
                evidence = [row[0] for row in self.cursor.fetchall()]
                unit_data["evidence"] = evidence
                
                # Get contradiction relationships
                self.cursor.execute(
                    'SELECT to_unit_id FROM unit_relations WHERE from_unit_id = ? AND relation_type = ?',
                    (unit_id, "contradiction")
                )
                contradictions = [row[0] for row in self.cursor.fetchall()]
                unit_data["contradictions"] = contradictions
                
                unit = EpistemicUnit.from_dict(unit_data)
                results.append((unit, similarity_value))
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            return [unit for unit, _ in results[:top_k]]
        
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            # Fallback to keyword search
            return self._keyword_search(query, top_k, domain, min_confidence)
    
    def _keyword_search(self, query: str, top_k: int = 5, domain: str = None, min_confidence: float = 0.0) -> List[EpistemicUnit]:
        """Search using keyword matching as fallback"""
        try:
            # Split query into terms
            query_terms = query.lower().split()
            
            # Build SQL query
            sql_filters = []
            sql_params = []
            
            # Add LIKE conditions for each term
            for term in query_terms:
                sql_filters.append("(LOWER(u.content) LIKE ?)")
                sql_params.append(f"%{term}%")
            
            if domain:
                sql_filters.append("u.domain = ?")
                sql_params.append(domain)
            
            if min_confidence > 0:
                sql_filters.append("u.confidence >= ?")
                sql_params.append(min_confidence)
            
            where_clause = "WHERE " + " AND ".join(sql_filters) if sql_filters else ""
            
            # Get matching units
            sql_query = f"""
            SELECT u.id, u.content, u.confidence, u.source, u.source_type,
                u.creation_time, u.last_verified, u.domain, u.metadata
            FROM epistemic_units u
            {where_clause}
            """
            
            self.cursor.execute(sql_query, sql_params)
            rows = self.cursor.fetchall()
            
            # Calculate relevance scores
            results = []
            for row in rows:
                unit_id = row[0]
                content = row[1]
                
                # Calculate relevance score based on term frequency
                score = sum(content.lower().count(term) for term in query_terms)
                
                # Create unit object
                unit_data = {
                    "id": unit_id,
                    "content": content,
                    "confidence": row[2],
                    "source": row[3],
                    "source_type": row[4],
                    "creation_time": row[5],
                    "last_verified": row[6],
                    "domain": row[7],
                    "metadata": json.loads(row[8]),
                }
                
                # Get evidence relationships
                self.cursor.execute(
                    'SELECT to_unit_id FROM unit_relations WHERE from_unit_id = ? AND relation_type = ?',
                    (unit_id, "evidence")
                )
                evidence = [row[0] for row in self.cursor.fetchall()]
                unit_data["evidence"] = evidence
                
                # Get contradiction relationships
                self.cursor.execute(
                    'SELECT to_unit_id FROM unit_relations WHERE from_unit_id = ? AND relation_type = ?',
                    (unit_id, "contradiction")
                )
                contradictions = [row[0] for row in self.cursor.fetchall()]
                unit_data["contradictions"] = contradictions
                
                unit = EpistemicUnit.from_dict(unit_data)
                
                # Store score separately (not part of EpistemicUnit constructor)
                relevance_score = score
                
                results.append((unit, relevance_score))
            
            # Sort by relevance score (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            return [unit for unit, _ in results[:top_k]]
        
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _add_relation(self, from_id: str, relation_type: str, to_id: str, confidence: float = 1.0, metadata: Dict = None):
        """Add a relationship between two units"""
        metadata_json = json.dumps(metadata or {})
        
        self.cursor.execute(
            'INSERT OR REPLACE INTO unit_relations (from_unit_id, relation_type, to_unit_id, confidence, metadata) '
            'VALUES (?, ?, ?, ?, ?)',
            (from_id, relation_type, to_id, confidence, metadata_json)
        )
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_related_units(self, unit_id: str, relation_type: str = None) -> List[EpistemicUnit]:
        """Get units related to the given unit"""
        try:
            sql_params = [unit_id]
            relation_filter = ""
            
            if relation_type:
                relation_filter = "AND relation_type = ?"
                sql_params.append(relation_type)
            
            # Get related units
            self.cursor.execute(
                f'SELECT to_unit_id FROM unit_relations WHERE from_unit_id = ? {relation_filter}',
                sql_params
            )
            
            related_ids = [row[0] for row in self.cursor.fetchall()]
            
            # Load the actual units
            related_units = []
            for related_id in related_ids:
                unit = self.get_unit(related_id)
                if unit:
                    related_units.append(unit)
            
            return related_units
        
        except Exception as e:
            logger.error(f"Error getting related units: {e}")
            return []
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()


class KnowledgeGraph:
    """
    A semantic knowledge graph for representing complex relationships 
    between EpistemicUnits and domain concepts
    """
    
    def __init__(self, epistemic_store: EpistemicStore):
        """Initialize with an existing epistemic store"""
        self.store = epistemic_store
    
    def add_node(self, concept: str, properties: Dict[str, Any], 
                 confidence: float, source: str) -> str:
        """Add a concept node to the knowledge graph"""
        content = f"CONCEPT: {concept}\n"
        content += f"PROPERTIES: {json.dumps(properties)}"
        
        unit = EpistemicUnit(
            content=content,
            confidence=confidence,
            source=source,
            source_type="derived",
            domain="knowledge_graph",
            metadata={"node_type": "concept", "concept": concept, "properties": properties}
        )
        
        return self.store.store_unit(unit)
    
    def add_edge(self, source_node_id: str, relation_type: str, 
                 target_node_id: str, confidence: float, 
                 evidence_ids: List[str] = None) -> str:
        """Add a relation between two concept nodes"""
        # Get the source and target nodes
        source_node = self.store.get_unit(source_node_id)
        target_node = self.store.get_unit(target_node_id)
        
        if not source_node or not target_node:
            raise ValueError("Source or target node not found")
        
        source_concept = source_node.metadata.get("concept", "unknown")
        target_concept = target_node.metadata.get("concept", "unknown")
        
        content = f"RELATION: {source_concept} --[{relation_type}]--> {target_concept}"
        
        unit = EpistemicUnit(
            content=content,
            confidence=confidence,
            source="graph_inference",
            source_type="derived",
            domain="knowledge_graph",
            metadata={
                "node_type": "relation",
                "relation_type": relation_type,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id
            }
        )
        
        # Add evidence if provided
        if evidence_ids:
            for evidence_id in evidence_ids:
                unit.add_evidence(evidence_id)
        
        # Store the relation
        relation_id = self.store.store_unit(unit)
        
        # Also create a direct database relation for efficient querying
        self.store._add_relation(
            from_id=source_node_id,
            relation_type=relation_type,
            to_id=target_node_id,
            confidence=confidence,
            metadata={"relation_unit_id": relation_id}
        )
        
        return relation_id
    
    def query_subgraph(self, central_concept: str, max_distance: int = 2) -> Dict[str, Any]:
        """Extract a subgraph around a central concept"""
        # Find nodes matching the central concept
        central_nodes = self._find_concept_nodes(central_concept)
        
        if not central_nodes:
            return {"nodes": [], "edges": []}
        
        # Initialize the subgraph
        nodes = {}
        edges = []
        explored = set()
        queue = [(node.id, 0) for node in central_nodes]  # (node_id, distance)
        
        # BFS to explore the graph
        while queue:
            node_id, distance = queue.pop(0)
            
            if node_id in explored or distance > max_distance:
                continue
            
            explored.add(node_id)
            unit = self.store.get_unit(node_id)
            
            if unit and "node_type" in unit.metadata:
                # Add node to subgraph
                if unit.metadata["node_type"] == "concept":
                    nodes[node_id] = {
                        "id": node_id,
                        "type": "concept",
                        "label": unit.metadata.get("concept", "unknown"),
                        "properties": unit.metadata.get("properties", {})
                    }
                
                # Expand outgoing relations
                related_units = self.store.get_related_units(node_id)
                
                for related in related_units:
                    if related.metadata.get("node_type") == "relation":
                        target_id = related.metadata.get("target_node_id")
                        
                        # Add edge to subgraph
                        edges.append({
                            "source": node_id,
                            "target": target_id,
                            "type": related.metadata.get("relation_type", "unknown"),
                            "confidence": related.confidence
                        })
                        
                        # Add target to BFS queue if not explored
                        if target_id not in explored:
                            queue.append((target_id, distance + 1))
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }
    
    def _find_concept_nodes(self, concept: str) -> List[EpistemicUnit]:
        """Find nodes matching a concept"""
        # Simple keyword search implementation
        units = self.store.query_units(f"CONCEPT: {concept}", top_k=20)
        
        # Filter to only include concept nodes
        return [
            unit for unit in units 
            if unit.metadata.get("node_type") == "concept" and 
            unit.metadata.get("concept", "").lower() == concept.lower()
        ]


class TemporalKnowledgeState:
    """
    Manages knowledge states over time, allowing for snapshots, 
    diffs, and reconstructing past states
    """
    
    def __init__(self, epistemic_store: EpistemicStore):
        """Initialize with an existing epistemic store"""
        self.store = epistemic_store
        self.snapshots = {}  # timestamp -> set of unit_ids
    
    def create_snapshot(self, name: str = None) -> str:
        """Create a snapshot of the current knowledge state"""
        timestamp = time.time()
        snapshot_id = name or f"snapshot_{timestamp}"
        
        # Get all unit IDs
        self.store.cursor.execute('SELECT id FROM epistemic_units')
        unit_ids = [row[0] for row in self.store.cursor.fetchall()]
        
        # Store snapshot
        self.snapshots[snapshot_id] = {
            "timestamp": timestamp,
            "unit_ids": set(unit_ids)
        }
        
        logger.info(f"Created knowledge snapshot '{snapshot_id}' with {len(unit_ids)} units")
        return snapshot_id
    
    def diff_snapshots(self, snapshot1_id: str, snapshot2_id: str) -> Dict[str, Any]:
        """Compare two snapshots and return the differences"""
        if snapshot1_id not in self.snapshots or snapshot2_id not in self.snapshots:
            raise ValueError("One or both snapshot IDs not found")
        
        # Get unit IDs for each snapshot
        units1 = self.snapshots[snapshot1_id]["unit_ids"]
        units2 = self.snapshots[snapshot2_id]["unit_ids"]
        
        # Calculate differences
        added = units2 - units1
        removed = units1 - units2
        common = units1.intersection(units2)
        
        # Get details for added units
        added_units = []
        for unit_id in added:
            unit = self.store.get_unit(unit_id)
            if unit:
                added_units.append(unit.to_dict())
        
        # Get details for removed units
        removed_units = []
        for unit_id in removed:
            # We might not be able to get removed units if they're truly gone
            try:
                unit = self.store.get_unit(unit_id)
                if unit:
                    removed_units.append(unit.to_dict())
            except:
                removed_units.append({"id": unit_id, "status": "not_retrievable"})
        
        return {
            "snapshot1": snapshot1_id,
            "snapshot2": snapshot2_id,
            "time_delta": self.snapshots[snapshot2_id]["timestamp"] - self.snapshots[snapshot1_id]["timestamp"],
            "added_count": len(added),
            "removed_count": len(removed),
            "common_count": len(common),
            "added_units": added_units,
            "removed_units": removed_units
        }


class KnowledgeAPI:
    """
    High-level API for agents to interact with the knowledge system
    
    This provides a clean, intuitive interface for AI agents to interact
    with the knowledge system without worrying about the underlying implementation.
    """
    
    def __init__(self, db_path: str = "./knowledge/epistemic.db"):
        """Initialize the knowledge API with a database path"""
        self.store = EpistemicStore(db_path)
        self.graph = KnowledgeGraph(self.store)
        self.temporal = TemporalKnowledgeState(self.store)
        
        # Create initial snapshot
        self.temporal.create_snapshot("initial")
    
    def ask(self, query: str, reasoning_depth: int = 2, 
            min_confidence: float = 0.3, domain: str = None) -> Dict[str, Any]:
        """Query the knowledge system with justifications"""
        
        # Start with direct knowledge search
        direct_hits = self.store.query_units(
            query=query, 
            top_k=5, 
            domain=domain,
            min_confidence=min_confidence
        )
        
        direct_results = [unit.to_dict() for unit in direct_hits]
        
        # If we need to do deeper reasoning
        supporting_evidence = []
        all_evidence_ids = set()
        
        if reasoning_depth > 0 and direct_hits:
            # Gather evidence for each direct hit
            for unit in direct_hits:
                # First-level evidence
                evidence_units = self.store.get_related_units(unit.id, "evidence")
                
                for evidence in evidence_units:
                    if evidence.id not in all_evidence_ids:
                        supporting_evidence.append(evidence.to_dict())
                        all_evidence_ids.add(evidence.id)
                        
                        # Second-level evidence (if requested)
                        if reasoning_depth > 1:
                            second_level = self.store.get_related_units(evidence.id, "evidence")
                            for second in second_level:
                                if second.id not in all_evidence_ids:
                                    supporting_evidence.append(second.to_dict())
                                    all_evidence_ids.add(second.id)
        
        # We might also want to include conceptual knowledge
        conceptual_knowledge = None
        major_concepts = self._extract_concepts(query)
        
        if major_concepts:
            # Get the most significant concept
            main_concept = major_concepts[0]
            conceptual_knowledge = self.graph.query_subgraph(main_concept, max_distance=1)
        
        return {
            "query": query,
            "direct_results": direct_results,
            "supporting_evidence": supporting_evidence,
            "conceptual_knowledge": conceptual_knowledge,
            "confidence": self._aggregate_confidence(direct_hits)
        }
    
    def tell(self, content: str, source: str, confidence: float = 0.7, 
             domain: str = None, metadata: Dict[str, Any] = None) -> str:
        """Add new knowledge to the system"""
        # Create a new epistemic unit
        unit = EpistemicUnit(
            content=content,
            confidence=confidence,
            source=source,
            domain=domain or "general",
            metadata=metadata or {}
        )
        
        # Store the unit
        unit_id = self.store.store_unit(unit)
        
        # Check for potential conflicts
        conflicts = self._find_potential_conflicts(unit)
        if conflicts:
            for conflict in conflicts:
                unit.add_contradiction(conflict.id)
                # Update the stored unit
                self.store.store_unit(unit)
            
            logger.info(f"Added knowledge unit {unit_id} with {len(conflicts)} potential conflicts")
        else:
            logger.info(f"Added knowledge unit {unit_id}")
        
        return unit_id
    
    def explore(self, concept: str, max_distance: int = 2) -> Dict[str, Any]:
        """Explore a concept and its relationships"""
        # Get the concept subgraph
        subgraph = self.graph.query_subgraph(concept, max_distance)
        
        # Also get relevant knowledge units
        related_units = self.store.query_units(concept, top_k=10)
        
        return {
            "concept": concept,
            "graph": subgraph,
            "related_knowledge": [unit.to_dict() for unit in related_units]
        }
    
    def explain(self, unit_id: str) -> Dict[str, Any]:
        """Provide explanation for why we believe a unit to be true"""
        unit = self.store.get_unit(unit_id)
        if not unit:
            return {"error": "Unit not found"}
        
        # Get supporting evidence
        evidence_units = self.store.get_related_units(unit_id, "evidence")
        
        # Get contradicting evidence
        contradiction_units = self.store.get_related_units(unit_id, "contradiction")
        
        # Build explanation
        explanation = {
            "unit": unit.to_dict(),
            "supporting_evidence": [e.to_dict() for e in evidence_units],
            "contradicting_evidence": [c.to_dict() for c in contradiction_units],
            "confidence_history": unit.metadata.get("confidence_history", [])
        }
        
        return explanation
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using simple heuristics"""
        # This is a very simplified implementation
        # In a real system, this would use NLP techniques
        
        words = text.lower().replace("?", "").replace(".", "").split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "is", "are", "in", "on", "of", "and", "or", "to", "for", "with"}
        filtered = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequencies
        word_counts = {}
        for word in filtered:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_concepts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top concepts
        return [concept for concept, _ in sorted_concepts[:3]]
    
    def _find_potential_conflicts(self, unit: EpistemicUnit) -> List[EpistemicUnit]:
        """Find potential conflicting knowledge units"""
        # This is a very simplified implementation
        # In a real system, this would use more sophisticated conflict detection
        
        # Search for similar content
        similar_units = self.store.query_units(unit.content, top_k=10)
        
        conflicts = []
        for similar in similar_units:
            # Skip if it's the same unit
            if similar.id == unit.id:
                continue
            
            # Very basic conflict detection - in reality this would be much more sophisticated
            if "not" in unit.content.lower() and "not" not in similar.content.lower():
                conflicts.append(similar)
            elif "not" in similar.content.lower() and "not" not in unit.content.lower():
                conflicts.append(similar)
        
        return conflicts
    
    def _aggregate_confidence(self, units: List[EpistemicUnit]) -> float:
        """Aggregate confidence from multiple units"""
        if not units:
            return 0.0
        
        # Simple weighted average by relevance
        # In a real system this would be more sophisticated
        total_confidence = sum(unit.confidence for unit in units)
        return total_confidence / len(units)
    
    def close(self):
        """Close the knowledge system"""
        self.store.close()


class ReasoningWorkspace:
    """
    Temporary workspace for agents to perform complex reasoning
    
    This provides a sandbox for step-by-step reasoning without
    immediately committing derived knowledge to the main store.
    """
    
    def __init__(self, knowledge_api: KnowledgeAPI, goal: str):
        """Initialize a reasoning workspace with a goal"""
        self.api = knowledge_api
        self.goal = goal
        self.steps = []
        self.working_memory = {}
        self.derived_units = []
        
        # Create a timestamp for this reasoning session
        self.started_at = time.time()
        self.id = f"reasoning_{self.started_at}"
    
    def pull_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        """Pull knowledge relevant to the reasoning task"""
        result = self.api.ask(query, reasoning_depth=2)
        
        # Store in working memory
        self.working_memory[query] = result
        
        return result
    
    def add_reasoning_step(self, step_type: str, content: str, 
                          evidence_ids: List[str] = None) -> Dict[str, Any]:
        """Add a reasoning step with optional evidence"""
        step = {
            "step_number": len(self.steps) + 1,
            "step_type": step_type,
            "content": content,
            "evidence_ids": evidence_ids or [],
            "timestamp": time.time()
        }
        
        self.steps.append(step)
        return step
    
    def derive_knowledge(self, content: str, confidence: float, 
                        evidence_ids: List[str] = None) -> Dict[str, Any]:
        """Derive new knowledge from reasoning"""
        derived = {
            "content": content,
            "confidence": confidence,
            "evidence_ids": evidence_ids or [],
            "derived_at": time.time(),
            "committed": False
        }
        
        self.derived_units.append(derived)
        return derived
    
    def commit_knowledge(self) -> List[str]:
        """Commit derived knowledge to the main knowledge store"""
        committed_ids = []
        
        for derived in self.derived_units:
            if not derived["committed"]:
                # Create a new epistemic unit
                unit = EpistemicUnit(
                    content=derived["content"],
                    confidence=derived["confidence"],
                    source=f"reasoning:{self.id}",
                    source_type="derived",
                    metadata={
                        "reasoning_goal": self.goal,
                        "derivation_steps": self.steps
                    }
                )
                
                # Add evidence
                for evidence_id in derived["evidence_ids"]:
                    unit.add_evidence(evidence_id)
                
                # Store the unit
                unit_id = self.api.store.store_unit(unit)
                committed_ids.append(unit_id)
                
                # Mark as committed
                derived["committed"] = True
        
        return committed_ids
    
    def get_reasoning_chain(self) -> Dict[str, Any]:
        """Get the complete reasoning chain"""
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": self.steps,
            "derived_knowledge": self.derived_units,
            "duration": time.time() - self.started_at
        }


# Example usage
if __name__ == "__main__":
    # Create the knowledge API
    api = KnowledgeAPI("./knowledge/test.db")
    
    # Add some knowledge
    unit_id1 = api.tell(
        content="Quantum computing leverages quantum mechanical phenomena like superposition and entanglement.",
        source="quantum_textbook",
        confidence=0.9,
        domain="quantum_computing"
    )
    
    unit_id2 = api.tell(
        content="Quantum machine learning combines quantum computing with machine learning techniques.",
        source="research_paper",
        confidence=0.85,
        domain="quantum_ml"
    )
    
    # Create a relationship in the knowledge graph
    api.graph.add_node("Quantum Computing", {"field": "computing"}, 0.9, "taxonomy")
    api.graph.add_node("Machine Learning", {"field": "ai"}, 0.9, "taxonomy")
    api.graph.add_edge(unit_id1, "related_to", unit_id2, 0.8)
    
    # Query the knowledge
    result = api.ask("How does quantum computing relate to machine learning?")
    print(f"Found {len(result['direct_results'])} direct results")
    
    # Create a reasoning workspace
    workspace = ReasoningWorkspace(api, "Understand quantum ML applications")
    workspace.pull_relevant_knowledge("quantum machine learning applications")
    workspace.add_reasoning_step("observation", "Quantum ML can leverage quantum properties for speedup")
    
    # Close the API
    api.close()