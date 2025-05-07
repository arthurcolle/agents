#!/bin/bash
# Script to execute the revenue-focused cross-domain integration process

# Create output directory
mkdir -p revenue_opportunities

# Run the integration generator
echo "Starting revenue integration process..."
python3 revenue_integration_plan.py --count 40 --kb-dir knowledge_bases --output-dir revenue_opportunities

# Generate visualization of integration opportunities
echo "Generating integration visualization..."
python3 - << EOF
import matplotlib.pyplot as plt
import networkx as nx
import os
import re
from pathlib import Path

# Create a graph of domain connections
G = nx.Graph()

# Find all generated opportunity files
output_dir = "revenue_opportunities"
opp_files = list(Path(output_dir).glob("revenue_opportunity_*.md"))

# Track domain relationships and scores
domain_connections = {}
domain_scores = {}

# Process each file to extract domains and scores
for file_path in opp_files:
    # Extract domains from filename
    file_num = int(re.search(r'revenue_opportunity_(\d+)', file_path.name).group(1))
    
    # Try to extract information from file
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Extract title and score information
        title_match = re.search(r'# (.*?)\n', content)
        title = title_match.group(1) if title_match else f"Opportunity {file_num}"
        
        # Look for domain mentions in content
        domains_mentioned = []
        with open('domains_list.txt', 'w') as dl:
            for line in content.split('\n'):
                if "**Domains**:" in line:
                    domains_mentioned = line.split("**Domains**:")[1].strip().split(', ')
                    dl.write(f"Found domains: {domains_mentioned}\n")
        
        # Extract scores if available
        score_matches = re.findall(r'- ([\w\s]+): (\d+)', content)
        scores = {}
        for score_name, score_val in score_matches:
            scores[score_name.strip()] = int(score_val)
            
        # Add domains to graph
        if domains_mentioned:
            for domain in domains_mentioned:
                if domain not in G:
                    G.add_node(domain, score=0)
                if "Revenue Potential" in scores:
                    if domain not in domain_scores:
                        domain_scores[domain] = []
                    domain_scores[domain].append(scores["Revenue Potential"])
            
            # Add connections
            for i in range(len(domains_mentioned)):
                for j in range(i+1, len(domains_mentioned)):
                    d1, d2 = domains_mentioned[i], domains_mentioned[j]
                    G.add_edge(d1, d2)
                    
                    key = tuple(sorted([d1, d2]))
                    if key not in domain_connections:
                        domain_connections[key] = 0
                    domain_connections[key] += 1

# Calculate average revenue scores for domains
for domain, scores in domain_scores.items():
    avg_score = sum(scores) / len(scores)
    G.nodes[domain]['score'] = avg_score

# Create network visualization
plt.figure(figsize=(20, 16))

# Set node colors based on revenue potential score
node_colors = [G.nodes[n]['score'] * 10 if 'score' in G.nodes[n] else 50 for n in G.nodes]

# Set node sizes based on number of connections
node_sizes = [G.degree(n) * 100 + 500 for n in G.nodes]

# Set edge widths based on connection frequency
edge_widths = [domain_connections.get(tuple(sorted([u, v])), 1) for u, v in G.edges]

# Create layout
pos = nx.spring_layout(G, k=0.15, iterations=50)

# Draw network
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, cmap='viridis')
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')

plt.title("Domain Integration Network: Revenue Opportunity Map", fontsize=20)
plt.axis('off')

# Add a colorbar legend
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=10))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Average Revenue Potential Score (0-10)', fontsize=14)

# Save the visualization
plt.tight_layout()
plt.savefig(f"{output_dir}/domain_integration_network.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated network visualization at {output_dir}/domain_integration_network.png")

# Create a summary table of top opportunities
with open(f"{output_dir}/top_opportunities.md", 'w') as f:
    f.write("# Top Revenue Integration Opportunities\n\n")
    f.write("| Rank | Domain Combination | Connection Strength | Avg Revenue Score |\n")
    f.write("|------|-------------------|---------------------|------------------|\n")
    
    # Sort connections by frequency
    sorted_connections = sorted(domain_connections.items(), key=lambda x: x[1], reverse=True)
    
    for rank, ((d1, d2), strength) in enumerate(sorted_connections[:20], 1):
        d1_score = G.nodes[d1]['score'] if 'score' in G.nodes[d1] else 0
        d2_score = G.nodes[d2]['score'] if 'score' in G.nodes[d2] else 0
        avg_score = (d1_score + d2_score) / 2
        
        f.write(f"| {rank} | {d1} + {d2} | {strength} | {avg_score:.1f} |\n")
EOF

echo "Process complete. Check revenue_opportunities/ directory for results."
echo "Key files:"
echo "  - revenue_opportunities/revenue_opportunities_summary.md"
echo "  - revenue_opportunities/domain_integration_network.png"
echo "  - revenue_opportunities/top_opportunities.md"