#!/usr/bin/env python3
"""
Epistemic Visualization - Interactive visualization interface for the epistemic knowledge system

This module provides visualization and interaction capabilities for the epistemic knowledge system:
1. Knowledge graph visualization
2. Temporal knowledge evolution timelines
3. Reasoning workspace visualization
4. Interactive knowledge exploration
5. Contradiction visualization and resolution
"""

import os
import json
import time
import logging
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import dash_cytoscape as cyto
    import plotly.express as px
    import plotly.graph_objects as go
    
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization dependencies not available. Install with: pip install matplotlib networkx dash dash-cytoscape plotly")

from epistemic_core import (
    EpistemicUnit,
    EpistemicStore,
    KnowledgeGraph,
    TemporalKnowledgeState,
    KnowledgeAPI,
    ReasoningWorkspace
)

from epistemic_tools import (
    initialize_knowledge_system,
    query_knowledge,
    explore_concept,
    workspace_get_chain,
    create_temporal_snapshot
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("epistemic-viz")


class KnowledgeGraphVisualizer:
    """Visualizes the epistemic knowledge graph with interactive capabilities"""
    
    def __init__(self, knowledge_api=None, db_path: str = "./knowledge/epistemic.db"):
        """Initialize the knowledge graph visualizer"""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Visualization dependencies not available")
        
        # Initialize knowledge connection
        if knowledge_api:
            self.knowledge_api = knowledge_api
        else:
            initialize_knowledge_system(db_path)
            self.knowledge_api = KnowledgeAPI(db_path)
        
        self.graph = KnowledgeGraph(db_path)
        self.temp_state = TemporalKnowledgeState(db_path)
    
    def visualize_graph(self, concept: str, depth: int = 2, max_nodes: int = 30,
                       save_path: Optional[str] = None) -> None:
        """Visualize a knowledge graph centered on a concept"""
        # Get the concept exploration data
        exploration = explore_concept(concept, depth=depth)
        
        # Create a NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        nodes_added = set()
        
        # Start with the central concept
        G.add_node(concept, node_type="concept", size=20, color="red")
        nodes_added.add(concept)
        
        # Add relationships
        for rel in exploration.get("relationships", []):
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("relation_type", "")
            confidence = rel.get("confidence", 0.5)
            
            # Truncate long node labels
            source_label = source[:25] + "..." if len(source) > 25 else source
            target_label = target[:25] + "..." if len(target) > 25 else target
            
            # Add nodes if not already added
            if source not in nodes_added:
                node_type = "domain" if source.startswith("domain:") else "concept"
                G.add_node(source, node_type=node_type, label=source_label, size=10, 
                          color="blue" if node_type == "domain" else "green")
                nodes_added.add(source)
            
            if target not in nodes_added:
                node_type = "domain" if target.startswith("domain:") else "concept"
                G.add_node(target, node_type=node_type, label=target_label, size=10,
                          color="blue" if node_type == "domain" else "green")
                nodes_added.add(target)
            
            # Add edge
            G.add_edge(source, target, relation=rel_type, weight=confidence*5, 
                      label=f"{rel_type} ({confidence:.2f})")
        
        # Limit to max_nodes
        if len(G.nodes) > max_nodes:
            # Keep central node and highest confidence connections
            central_edges = list(G.edges(concept, data=True))
            if not central_edges:  # If concept isn't directly in the graph
                central_edges = list(G.edges(data=True))
            
            # Sort by weight (confidence)
            sorted_edges = sorted(central_edges, key=lambda x: x[2].get('weight', 0), reverse=True)
            
            # Create a new graph with limited nodes
            limited_G = nx.DiGraph()
            limited_G.add_node(concept, **G.nodes[concept])
            
            # Add top edges up to max_nodes-1 (account for central node)
            edges_to_add = sorted_edges[:max_nodes-1]
            for source, target, attr in edges_to_add:
                if source not in limited_G.nodes:
                    limited_G.add_node(source, **G.nodes[source])
                if target not in limited_G.nodes:
                    limited_G.add_node(target, **G.nodes[target])
                limited_G.add_edge(source, target, **attr)
            
            G = limited_G
        
        # Prepare the visualization
        plt.figure(figsize=(12, 10))
        
        # Use a force-directed layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw nodes with different colors and sizes
        for node_type, color in [("concept", "green"), ("domain", "blue")]:
            nodes = [node for node, attr in G.nodes(data=True) 
                    if attr.get("node_type") == node_type]
            
            if nodes:
                sizes = [G.nodes[node].get("size", 10) for node in nodes]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes, 
                                     node_color=color, alpha=0.8)
        
        # Highlight the central concept
        nx.draw_networkx_nodes(G, pos, nodelist=[concept], node_size=800, 
                             node_color="red", alpha=0.9)
        
        # Draw edges with width based on confidence
        for (u, v, attr) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=attr.get("weight", 1),
                                 alpha=0.6, arrows=True, arrowsize=15,
                                 connectionstyle="arc3,rad=0.1")
        
        # Draw labels
        labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")
        
        # Draw edge labels for important edges
        important_edges = [(u, v) for u, v, attr in G.edges(data=True) 
                          if attr.get("weight", 0) > 3]
        edge_labels = {(u, v): attr.get("relation", "") 
                     for u, v, attr in G.edges(data=True)
                     if (u, v) in important_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Knowledge Graph for: {concept}", fontsize=16)
        plt.axis("off")
        
        # Save or show the visualization
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Knowledge graph visualization saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def visualize_temporal_evolution(self, concept: str, save_path: Optional[str] = None) -> None:
        """Visualize the temporal evolution of knowledge about a concept"""
        # Get temporal evolution data
        evolution = self.temp_state.get_concept_evolution(concept)
        
        if not evolution:
            logger.warning(f"No temporal data found for concept: {concept}")
            return
        
        # Prepare data for visualization
        snapshots = []
        confidences = []
        dates = []
        
        for entry in evolution:
            snapshot_id = entry.get("snapshot_id", "unknown")
            confidence = entry.get("confidence", 0.5)
            timestamp = entry.get("timestamp", 0)
            
            snapshots.append(snapshot_id)
            confidences.append(confidence)
            dates.append(datetime.fromtimestamp(timestamp))
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        
        # Plot confidence over time
        plt.plot(dates, confidences, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Highlight significant changes
        significant_changes = []
        for i in range(1, len(confidences)):
            if abs(confidences[i] - confidences[i-1]) > 0.1:
                significant_changes.append(i)
        
        if significant_changes:
            sig_dates = [dates[i] for i in significant_changes]
            sig_confs = [confidences[i] for i in significant_changes]
            plt.scatter(sig_dates, sig_confs, c='red', s=100, zorder=5)
        
        plt.title(f"Temporal Evolution of Knowledge: {concept}", fontsize=16)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Confidence", fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for significant changes
        for i in significant_changes:
            plt.annotate(f"Snapshot: {snapshots[i][:8]}...",
                        (dates[i], confidences[i]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
        
        # Save or show the visualization
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Temporal evolution visualization saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def visualize_reasoning_workspace(self, workspace_id: str, save_path: Optional[str] = None) -> None:
        """Visualize a reasoning workspace with its chain of steps"""
        # Get the workspace chain
        chain = workspace_get_chain(workspace_id)
        
        if not chain or "chain" not in chain:
            logger.warning(f"No chain found for workspace: {workspace_id}")
            return
        
        steps = chain["chain"].get("steps", [])
        derived = chain["chain"].get("derived_knowledge", [])
        
        if not steps:
            logger.warning(f"No steps found in workspace: {workspace_id}")
            return
        
        # Create a directed graph for the reasoning chain
        G = nx.DiGraph()
        
        # Add steps as nodes
        for i, step in enumerate(steps):
            step_id = step.get("step_id", f"step_{i}")
            step_type = step.get("step_type", "unknown")
            content = step.get("content", "No content")
            
            # Truncate content for label
            short_content = content[:50] + "..." if len(content) > 50 else content
            
            G.add_node(step_id, 
                      label=f"Step {i+1}: {short_content}",
                      type=step_type,
                      content=content,
                      node_type="step")
            
            # Connect to previous step
            if i > 0:
                prev_step_id = steps[i-1].get("step_id", f"step_{i-1}")
                G.add_edge(prev_step_id, step_id, relation="next")
        
        # Add derived knowledge
        for i, unit in enumerate(derived):
            unit_id = unit.get("unit_id", f"unit_{i}")
            content = unit.get("content", "No content")
            confidence = unit.get("confidence", 0.5)
            
            # Truncate content for label
            short_content = content[:50] + "..." if len(content) > 50 else content
            
            G.add_node(unit_id,
                      label=f"Knowledge: {short_content}",
                      confidence=confidence,
                      content=content,
                      node_type="knowledge")
            
            # Connect to last step
            if steps:
                last_step_id = steps[-1].get("step_id", f"step_{len(steps)-1}")
                G.add_edge(last_step_id, unit_id, relation="derives")
        
        # Visualize the graph
        plt.figure(figsize=(14, 10))
        
        # Use hierarchical layout for reasoning chains
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        
        # Draw steps
        step_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "step"]
        if step_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=step_nodes,
                                 node_color="lightblue", 
                                 node_size=700,
                                 alpha=0.8)
        
        # Draw knowledge units
        knowledge_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "knowledge"]
        if knowledge_nodes:
            # Color by confidence
            confidence_values = [G.nodes[n].get("confidence", 0.5) for n in knowledge_nodes]
            cmap = plt.cm.RdYlGn  # Red to yellow to green
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=knowledge_nodes,
                                 node_color=confidence_values,
                                 cmap=cmap,
                                 node_size=900,
                                 alpha=0.8,
                                 vmin=0, vmax=1)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             width=2, 
                             alpha=0.7, 
                             arrows=True,
                             arrowsize=20)
        
        # Draw labels
        labels = {n: attr.get("label", n) for n, attr in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, 
                              labels=labels, 
                              font_size=10,
                              font_weight="bold")
        
        plt.title(f"Reasoning Workspace: {workspace_id}", fontsize=16)
        plt.axis("off")
        
        # Save or show the visualization
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Reasoning workspace visualization saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def visualize_contradiction(self, units: List[Dict[str, Any]], resolution: Optional[Dict[str, Any]] = None,
                              save_path: Optional[str] = None) -> None:
        """Visualize contradictory knowledge units and their resolution"""
        if not units:
            logger.warning("No units provided for contradiction visualization")
            return
        
        # Create a figure with subplots
        fig, axs = plt.subplots(1, 2 if resolution else 1, figsize=(14, 8))
        
        # If only one subplot, convert to array for consistent indexing
        if not resolution:
            axs = [axs]
        
        # Draw contradictory units
        ax = axs[0]
        
        # Prepare data
        contents = [unit.get("content", "No content")[:50] + "..." for unit in units]
        confidences = [unit.get("confidence", 0) for unit in units]
        sources = [unit.get("source", "Unknown") for unit in units]
        
        # Create bar chart of contradictory statements
        y_pos = range(len(contents))
        bars = ax.barh(y_pos, confidences, align='center')
        
        # Color bars by confidence
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(confidences[i]))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(contents)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Confidence')
        ax.set_title('Contradictory Knowledge Statements')
        
        # Add source annotations
        for i, conf in enumerate(confidences):
            ax.text(conf + 0.01, i, sources[i], va='center')
        
        # Draw resolution if provided
        if resolution:
            ax = axs[1]
            
            # Create a node for the resolution
            G = nx.DiGraph()
            
            # Add contradictory units as nodes
            for i, unit in enumerate(units):
                content = unit.get("content", "No content")
                short_content = content[:50] + "..." if len(content) > 50 else content
                confidence = unit.get("confidence", 0)
                
                G.add_node(f"unit_{i}", 
                          label=short_content,
                          confidence=confidence,
                          node_type="contradictory")
            
            # Add resolution node
            resolution_content = resolution.get("resolved_statement", "No resolution")
            short_resolution = resolution_content[:50] + "..." if len(resolution_content) > 50 else resolution_content
            resolution_confidence = resolution.get("confidence", 0)
            
            G.add_node("resolution", 
                      label=short_resolution,
                      confidence=resolution_confidence,
                      node_type="resolution")
            
            # Add edges from units to resolution
            for i in range(len(units)):
                G.add_edge(f"unit_{i}", "resolution", relation="resolves_to")
            
            # Draw the graph
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            
            # Draw contradictory nodes
            contra_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "contradictory"]
            node_colors = [plt.cm.RdYlGn(G.nodes[n].get("confidence", 0)) for n in contra_nodes]
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=contra_nodes,
                                 node_color=node_colors,
                                 node_size=700,
                                 alpha=0.8)
            
            # Draw resolution node
            resolution_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "resolution"]
            res_color = plt.cm.RdYlGn(resolution_confidence)
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=resolution_nodes,
                                 node_color=[res_color],
                                 node_size=1000,
                                 alpha=0.9)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos,
                                 width=2,
                                 alpha=0.7,
                                 arrows=True,
                                 arrowsize=15)
            
            # Draw labels
            labels = {n: attr.get("label", n) for n, attr in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos,
                                  labels=labels,
                                  font_size=9,
                                  font_weight="bold")
            
            ax.set_title('Contradiction Resolution')
            ax.axis("off")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the visualization
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Contradiction visualization saved to {save_path}")
        else:
            plt.show()


class InteractiveKnowledgeExplorer:
    """Interactive web interface for exploring the epistemic knowledge system"""
    
    def __init__(self, db_path: str = "./knowledge/epistemic.db"):
        """Initialize the interactive knowledge explorer"""
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Visualization dependencies not available")
        
        # Initialize knowledge system
        initialize_knowledge_system(db_path)
        self.db_path = db_path
        self.knowledge_api = KnowledgeAPI(db_path)
        self.graph = KnowledgeGraph(db_path)
        self.temp_state = TemporalKnowledgeState(db_path)
        
        # Initialize the app
        self.app = dash.Dash(__name__, title="Epistemic Knowledge Explorer")
        
        # Load Cytoscape extensions
        cyto.load_extra_layouts()
        
        # Setup the layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = html.Div([
            html.H1("Epistemic Knowledge Explorer", style={"textAlign": "center"}),
            
            html.Div([
                html.Div([
                    html.H3("Knowledge Graph Exploration"),
                    html.Div([
                        html.Label("Search Concept:"),
                        dcc.Input(
                            id="concept-search",
                            type="text",
                            placeholder="Enter a concept...",
                            value="",
                            style={"width": "100%", "marginBottom": "10px"}
                        ),
                        html.Button("Explore", id="explore-button", n_clicks=0),
                        html.Div(id="search-results")
                    ], style={"padding": "15px"})
                ], className="card", style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
                
                html.Div([
                    html.H3("Knowledge Visualization"),
                    dcc.Tabs([
                        dcc.Tab(label="Knowledge Graph", children=[
                            cyto.Cytoscape(
                                id="knowledge-graph",
                                layout={"name": "cose"},
                                style={"width": "100%", "height": "600px"},
                                elements=[],
                                stylesheet=[
                                    {
                                        "selector": "node",
                                        "style": {
                                            "content": "data(label)",
                                            "text-wrap": "wrap",
                                            "text-max-width": "80px"
                                        }
                                    },
                                    {
                                        "selector": ".concept",
                                        "style": {
                                            "background-color": "#6FB1FC",
                                            "shape": "ellipse"
                                        }
                                    },
                                    {
                                        "selector": ".domain",
                                        "style": {
                                            "background-color": "#F5A45D",
                                            "shape": "round-rectangle"
                                        }
                                    },
                                    {
                                        "selector": ".evidence",
                                        "style": {
                                            "background-color": "#86B342",
                                            "shape": "diamond"
                                        }
                                    },
                                    {
                                        "selector": "edge",
                                        "style": {
                                            "label": "data(label)",
                                            "width": "data(weight)",
                                            "line-color": "#999",
                                            "curve-style": "bezier",
                                            "target-arrow-shape": "triangle",
                                            "target-arrow-color": "#999",
                                            "text-background-opacity": 1,
                                            "text-background-color": "#fff",
                                            "text-background-padding": "3px"
                                        }
                                    }
                                ]
                            )
                        ]),
                        dcc.Tab(label="Temporal View", children=[
                            dcc.Graph(id="temporal-graph")
                        ]),
                        dcc.Tab(label="Reasoning Trace", children=[
                            html.Div([
                                html.Label("Workspace ID:"),
                                dcc.Input(
                                    id="workspace-id-input",
                                    type="text",
                                    placeholder="Enter workspace ID...",
                                    value="",
                                    style={"width": "70%", "marginRight": "10px"}
                                ),
                                html.Button("Load", id="load-workspace-button", n_clicks=0),
                            ], style={"marginBottom": "15px"}),
                            dcc.Graph(id="reasoning-graph")
                        ])
                    ])
                ], className="card", style={"width": "68%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "2%"})
            ]),
            
            html.Div([
                html.H3("Knowledge Details"),
                html.Div(id="knowledge-details", style={"padding": "15px"})
            ], className="card", style={"marginTop": "20px"}),
            
            # Hidden div for storing current state
            html.Div(id="current-concept", style={"display": "none"})
        ], style={"padding": "20px"})
    
    def _setup_callbacks(self):
        """Setup the Dash app callbacks"""
        # Callback for search button
        @self.app.callback(
            [Output("knowledge-graph", "elements"),
             Output("search-results", "children"),
             Output("current-concept", "children")],
            [Input("explore-button", "n_clicks")],
            [State("concept-search", "value")]
        )
        def update_graph(n_clicks, concept):
            if not concept:
                return [], "Enter a concept to explore.", ""
            
            # Get concept exploration
            exploration = explore_concept(concept, depth=2)
            
            # Prepare graph elements
            elements = []
            
            # Add central concept node
            elements.append({
                "data": {"id": concept, "label": concept},
                "classes": "concept"
            })
            
            # Add relationship edges and related nodes
            added_nodes = {concept}
            
            for rel in exploration.get("relationships", []):
                source = rel.get("source", "")
                target = rel.get("target", "")
                rel_type = rel.get("relation_type", "")
                confidence = rel.get("confidence", 0.5)
                
                # Add source node if not added
                if source not in added_nodes:
                    node_class = "domain" if source.startswith("domain:") else "concept"
                    elements.append({
                        "data": {"id": source, "label": source},
                        "classes": node_class
                    })
                    added_nodes.add(source)
                
                # Add target node if not added
                if target not in added_nodes:
                    node_class = "domain" if target.startswith("domain:") else "concept"
                    elements.append({
                        "data": {"id": target, "label": target},
                        "classes": node_class
                    })
                    added_nodes.add(target)
                
                # Add edge
                elements.append({
                    "data": {
                        "source": source, 
                        "target": target,
                        "label": rel_type,
                        "weight": max(1, confidence * 5)
                    }
                })
            
            # Results summary
            results_text = [
                html.P(f"Found {len(exploration.get('direct_results', []))} direct results"),
                html.P(f"Found {len(exploration.get('relationships', []))} relationships")
            ]
            
            return elements, results_text, concept
        
        # Callback for temporal view
        @self.app.callback(
            Output("temporal-graph", "figure"),
            [Input("current-concept", "children")]
        )
        def update_temporal_view(concept):
            if not concept:
                return go.Figure()
            
            # Get temporal evolution data
            evolution = self.temp_state.get_concept_evolution(concept)
            
            if not evolution:
                # Empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="No temporal data available for this concept",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                return fig
            
            # Prepare data
            timestamps = []
            confidences = []
            snapshot_ids = []
            
            for entry in evolution:
                timestamp = entry.get("timestamp", 0)
                timestamps.append(datetime.fromtimestamp(timestamp))
                confidences.append(entry.get("confidence", 0.5))
                snapshot_ids.append(entry.get("snapshot_id", "unknown"))
            
            # Create the figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=confidences,
                mode="lines+markers",
                name="Confidence",
                marker=dict(size=10, color="blue"),
                line=dict(width=3, color="blue"),
                text=snapshot_ids,
                hovertemplate="<b>Snapshot</b>: %{text}<br>" +
                              "<b>Confidence</b>: %{y:.2f}<br>" +
                              "<b>Time</b>: %{x}"
            ))
            
            fig.update_layout(
                title=f"Temporal Evolution of: {concept}",
                xaxis_title="Time",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1.05]),
                hovermode="closest"
            )
            
            return fig
        
        # Callback for reasoning workspace
        @self.app.callback(
            Output("reasoning-graph", "figure"),
            [Input("load-workspace-button", "n_clicks")],
            [State("workspace-id-input", "value")]
        )
        def update_reasoning_view(n_clicks, workspace_id):
            if not workspace_id:
                # Empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="Enter a workspace ID to view reasoning trace",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                return fig
            
            # Get workspace chain
            chain = workspace_get_chain(workspace_id)
            
            if not chain or "chain" not in chain:
                # Empty figure with error message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No chain found for workspace: {workspace_id}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                return fig
            
            steps = chain["chain"].get("steps", [])
            derived = chain["chain"].get("derived_knowledge", [])
            
            if not steps:
                # Empty figure with error message
                fig = go.Figure()
                fig.add_annotation(
                    text="No steps found in workspace",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                return fig
            
            # Create nodes for steps
            step_nodes = []
            for i, step in enumerate(steps):
                step_id = step.get("step_id", f"step_{i}")
                step_type = step.get("step_type", "unknown")
                content = step.get("content", "No content")
                
                # Truncate content for display
                short_content = content[:50] + "..." if len(content) > 50 else content
                
                step_nodes.append({
                    "x": i,
                    "y": 0,
                    "id": step_id,
                    "label": f"Step {i+1}: {short_content}",
                    "type": step_type
                })
            
            # Create nodes for derived knowledge
            knowledge_nodes = []
            for i, unit in enumerate(derived):
                unit_id = unit.get("unit_id", f"unit_{i}")
                content = unit.get("content", "No content")
                confidence = unit.get("confidence", 0.5)
                
                # Truncate content for display
                short_content = content[:50] + "..." if len(content) > 50 else content
                
                knowledge_nodes.append({
                    "x": len(steps) - 1,  # Position at last step
                    "y": -(i + 1),  # Position below the steps
                    "id": unit_id,
                    "label": f"Knowledge: {short_content}",
                    "confidence": confidence
                })
            
            # Create edges
            edges = []
            
            # Connect steps
            for i in range(len(steps) - 1):
                source_id = steps[i].get("step_id", f"step_{i}")
                target_id = steps[i+1].get("step_id", f"step_{i+1}")
                
                edges.append({
                    "source": source_id,
                    "target": target_id
                })
            
            # Connect last step to derived knowledge
            if derived and steps:
                last_step_id = steps[-1].get("step_id", f"step_{len(steps)-1}")
                for unit in derived:
                    unit_id = unit.get("unit_id", "unknown")
                    edges.append({
                        "source": last_step_id,
                        "target": unit_id
                    })
            
            # Create the figure
            fig = go.Figure()
            
            # Add step nodes
            fig.add_trace(go.Scatter(
                x=[node["x"] for node in step_nodes],
                y=[node["y"] for node in step_nodes],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=30,
                    color="skyblue",
                    line=dict(color="royalblue", width=2)
                ),
                text=[node["label"] for node in step_nodes],
                textposition="top center",
                hoverinfo="text",
                hovertext=[node["label"] for node in step_nodes],
                name="Reasoning Steps"
            ))
            
            # Add knowledge nodes
            if knowledge_nodes:
                # Color by confidence
                cmap = px.colors.sequential.Viridis
                colors = [px.colors.sample_colorscale(
                    cmap, node["confidence"])[0] for node in knowledge_nodes]
                
                fig.add_trace(go.Scatter(
                    x=[node["x"] for node in knowledge_nodes],
                    y=[node["y"] for node in knowledge_nodes],
                    mode="markers+text",
                    marker=dict(
                        symbol="diamond",
                        size=25,
                        color=colors,
                        line=dict(color="darkgreen", width=2)
                    ),
                    text=[node["label"] for node in knowledge_nodes],
                    textposition="middle right",
                    hoverinfo="text",
                    hovertext=[f"{node['label']} (Confidence: {node['confidence']:.2f})" 
                             for node in knowledge_nodes],
                    name="Derived Knowledge"
                ))
            
            # Add edges as shapes
            for edge in edges:
                # Find source and target nodes
                source_node = None
                target_node = None
                
                for node in step_nodes + knowledge_nodes:
                    if node["id"] == edge["source"]:
                        source_node = node
                    if node["id"] == edge["target"]:
                        target_node = node
                
                if source_node and target_node:
                    fig.add_shape(
                        type="line",
                        x0=source_node["x"],
                        y0=source_node["y"],
                        x1=target_node["x"],
                        y1=target_node["y"],
                        line=dict(color="gray", width=2, dash="solid"),
                        layer="below"
                    )
            
            # Update layout
            fig.update_layout(
                title=f"Reasoning Workspace: {workspace_id}",
                showlegend=True,
                hovermode="closest",
                xaxis=dict(
                    title="Reasoning Steps",
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                plot_bgcolor="white"
            )
            
            return fig
        
        # Callback for node selection
        @self.app.callback(
            Output("knowledge-details", "children"),
            [Input("knowledge-graph", "tapNodeData")]
        )
        def display_node_details(node_data):
            if not node_data:
                return "Click on a node to see details"
            
            node_id = node_data.get("id", "")
            
            # Query knowledge about this node
            query_result = query_knowledge(node_id, reasoning_depth=1)
            
            # Direct results
            direct_results = query_result.get("direct_results", [])
            
            if direct_results:
                # Display the knowledge units
                details = [html.H4(f"Knowledge about: {node_id}")]
                
                for i, unit in enumerate(direct_results[:5]):  # Limit to 5 units
                    content = unit.get("content", "No content")
                    confidence = unit.get("confidence", 0)
                    source = unit.get("source", "Unknown source")
                    evidence = unit.get("evidence", "No evidence")
                    
                    unit_card = html.Div([
                        html.H5(f"Knowledge Unit {i+1}"),
                        html.Div([
                            html.P(content),
                            html.Div([
                                html.Span("Confidence: ", style={"fontWeight": "bold"}),
                                html.Span(f"{confidence:.2f}")
                            ]),
                            html.Div([
                                html.Span("Source: ", style={"fontWeight": "bold"}),
                                html.Span(source)
                            ]),
                            html.Div([
                                html.Span("Evidence: ", style={"fontWeight": "bold"}),
                                html.Span(evidence)
                            ])
                        ], style={"padding": "10px", "backgroundColor": "#f8f9fa", "borderRadius": "5px"})
                    ], style={"marginBottom": "15px"})
                    
                    details.append(unit_card)
                
                if len(direct_results) > 5:
                    details.append(html.P(f"... and {len(direct_results) - 5} more units"))
                
                return details
            else:
                return html.P(f"No knowledge units found for: {node_id}")
    
    def run_server(self, debug=False, port=8050):
        """Run the Dash server"""
        self.app.run_server(debug=debug, port=port)
        
        # Open browser automatically
        webbrowser.open(f"http://localhost:{port}")


def generate_visualizations(knowledge_path: str = "./knowledge/epistemic.db"):
    """
    Generate and save a set of visualizations for the epistemic knowledge system
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization dependencies not available. Skipping visualization generation.")
        return
    
    # Create output directory
    output_dir = Path("./knowledge/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations in {output_dir}...")
    
    # Initialize knowledge system
    initialize_knowledge_system(knowledge_path)
    
    # Create visualizer
    visualizer = KnowledgeGraphVisualizer(db_path=knowledge_path)
    
    # Generate knowledge graph visualization
    concepts = ["ai ethics", "epistemology", "knowledge management"]
    for concept in concepts:
        output_path = output_dir / f"knowledge_graph_{concept.replace(' ', '_')}.png"
        print(f"Generating knowledge graph for: {concept}")
        try:
            visualizer.visualize_graph(concept, save_path=str(output_path))
        except Exception as e:
            print(f"Error generating knowledge graph for {concept}: {e}")
    
    # Generate temporal evolution visualization
    for concept in concepts:
        output_path = output_dir / f"temporal_{concept.replace(' ', '_')}.png"
        print(f"Generating temporal evolution for: {concept}")
        try:
            visualizer.visualize_temporal_evolution(concept, save_path=str(output_path))
        except Exception as e:
            print(f"Error generating temporal visualization for {concept}: {e}")
    
    print("Visualization generation complete!")


def run_interactive_explorer(knowledge_path: str = "./knowledge/epistemic.db", port: int = 8050):
    """
    Run the interactive knowledge explorer web interface
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization dependencies not available. Cannot run interactive explorer.")
        return
    
    print(f"Starting interactive knowledge explorer on port {port}...")
    print(f"Using knowledge database: {knowledge_path}")
    
    try:
        explorer = InteractiveKnowledgeExplorer(db_path=knowledge_path)
        explorer.run_server(debug=True, port=port)
    except Exception as e:
        print(f"Error running interactive explorer: {e}")


if __name__ == "__main__":
    # Check if visualization dependencies are available
    if not VISUALIZATION_AVAILABLE:
        print("Visualization dependencies not available. Install with:")
        print("pip install matplotlib networkx dash dash-cytoscape plotly")
        exit(1)
    
    # Ensure knowledge directory exists
    os.makedirs("./knowledge", exist_ok=True)
    
    # Default knowledge path
    knowledge_path = "./knowledge/epistemic.db"
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Epistemic Knowledge System Visualization")
    parser.add_argument("--generate", action="store_true", help="Generate static visualizations")
    parser.add_argument("--interactive", action="store_true", help="Run interactive explorer")
    parser.add_argument("--db", type=str, default=knowledge_path, help="Path to knowledge database")
    parser.add_argument("--port", type=int, default=8050, help="Port for interactive server")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_visualizations(args.db)
    
    if args.interactive:
        run_interactive_explorer(args.db, args.port)
    
    if not args.generate and not args.interactive:
        print("No action specified. Use --generate or --interactive")
        print("Example: python epistemic_visualization.py --interactive --db ./knowledge/epistemic.db")