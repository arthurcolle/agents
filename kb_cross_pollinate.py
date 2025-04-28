#!/usr/bin/env python3
"""
Advanced knowledge base cross-pollination tool for generating research topics using chain of thought decomposition.
"""

import os
import json
import random
import argparse
import time
import datetime
import csv
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

# Check for openai package, default to using Anthropic if not available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

@dataclass
class ResearchTopic:
    """A research topic with metadata and chain of thought decomposition"""
    title: str
    source_domains: Set[str]
    confidence: float
    chain_of_thought: List[Dict[str, Any]] = field(default_factory=list)
    subtopics: List[str] = field(default_factory=list)
    potential_applications: List[str] = field(default_factory=list)
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    impact_score: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())
    
    def add_thought_step(self, step_data: Dict[str, Any]) -> None:
        """Add a step to the chain of thought reasoning"""
        self.chain_of_thought.append(step_data)
        
    def compute_overall_score(self) -> float:
        """Compute an overall score based on novelty, feasibility, and impact"""
        return (self.novelty_score * 0.4 + 
                self.feasibility_score * 0.3 + 
                self.impact_score * 0.3)
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "title": self.title,
            "source_domains": list(self.source_domains),
            "confidence": self.confidence,
            "chain_of_thought": self.chain_of_thought,
            "subtopics": self.subtopics,
            "potential_applications": self.potential_applications,
            "novelty_score": self.novelty_score,
            "feasibility_score": self.feasibility_score,
            "impact_score": self.impact_score,
            "overall_score": self.compute_overall_score(),
            "timestamp": self.timestamp
        }

def load_knowledge_base(file_path: str) -> Dict:
    """Load a knowledge base from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_knowledge_bases(directory: str) -> List[str]:
    """Get all JSON files in the knowledge_bases directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if f.endswith('.json') and os.path.isfile(os.path.join(directory, f))]

def select_random_knowledge_bases(kb_list: List[str], count: int = 2) -> List[str]:
    """Randomly select a specified number of knowledge bases."""
    if len(kb_list) < count:
        raise ValueError(f"Not enough knowledge bases. Found {len(kb_list)}, needed {count}")
    return random.sample(kb_list, count)

def get_chain_of_thought_prompt(domains: List[Dict], num_steps: int = 5) -> str:
    """Generate a prompt that requests a chain of thought reasoning process."""
    domain_descriptions = "\n\n".join([
        f"Domain {i+1}: {domain.get('name')}\n"
        f"Description: {domain.get('content', 'No content available')[:500]}..."  # Truncate long descriptions
        for i, domain in enumerate(domains)
    ])
    
    return f"""
You are a creative interdisciplinary researcher. You've been given information about {len(domains)} different domains of knowledge.

{domain_descriptions}

Please generate an interdisciplinary research topic that combines these domains in a novel and meaningful way. 
Use a chain of thought reasoning process to decompose your thinking into {num_steps} clear steps:

Step 1: Problem Statement
- Identify a specific problem or opportunity that could benefit from combining these domains
- Explain why this problem is important and interesting

Step 2: Core Concepts Identification
- Identify the key concepts from each domain that could be relevant
- Explain how these concepts might interact or complement each other

Step 3: Conceptual Integration
- Describe how these concepts could be integrated into a unified framework
- Identify any potential challenges in integrating these concepts

Step 4: Research Methods
- Suggest specific research methodologies that would be appropriate
- Explain how these methods would leverage the strengths of each domain

Step 5: Potential Applications & Impact
- Describe potential real-world applications of this research
- Assess the potential scientific and societal impact of this work

For each step, provide detailed reasoning and show your thought process explicitly.

Finally, after completing these steps, provide:
1. A clear, concise title for this research topic (10-15 words)
2. 3-5 specific subtopics or research questions within this broader topic
3. 2-4 potential real-world applications
4. Scores (0-10) for:
   - Novelty: How original is this research direction?
   - Feasibility: How practical would it be to pursue this research?
   - Impact: What potential impact could this research have if successful?

Format your response with clear headings for each step and section.
"""

def generate_cross_domain_research(domains: List[Dict], 
                                 model: str,
                                 chain_of_thought_steps: int = 5,
                                 temperature: float = 0.8) -> Dict[str, Any]:
    """
    Generate a research topic combining multiple domains using chain of thought reasoning.
    
    Args:
        domains: List of domain dictionaries
        model: LLM model to use
        chain_of_thought_steps: Number of reasoning steps to request
        temperature: Temperature setting for generation (higher = more creative)
        
    Returns:
        Parsed research topic with chain of thought
    """
    # Generate the prompt
    prompt = get_chain_of_thought_prompt(domains, chain_of_thought_steps)
    
    # Use OpenAI if available, otherwise try Anthropic
    if HAS_OPENAI and os.environ.get('OPENAI_API_KEY'):
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        response = client.chat.completions.create(
            model=model if "gpt" in model else "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=4000
        )
        result = response.choices[0].message.content
    
    elif HAS_ANTHROPIC and os.environ.get('ANTHROPIC_API_KEY'):
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        response = client.messages.create(
            model=model if "claude" in model else "claude-3-opus-20240229",
            max_tokens=4000,
            temperature=temperature,
            system="You are a creative interdisciplinary researcher who specializes in finding connections between different domains through structured chain of thought reasoning.",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.content[0].text
    
    else:
        raise RuntimeError("No LLM API available. Please install openai or anthropic package and set the API key.")
    
    # Parse the result to extract structured information
    return parse_chain_of_thought_response(result, domains)

def parse_chain_of_thought_response(response: str, domains: List[Dict]) -> Dict[str, Any]:
    """
    Parse the LLM response to extract structured information about the research topic.
    
    Args:
        response: The raw LLM response
        domains: The list of domain dictionaries
        
    Returns:
        Dictionary with parsed information
    """
    # Extract chain of thought steps
    steps = []
    step_markers = ["Step 1:", "Step 2:", "Step 3:", "Step 4:", "Step 5:"]
    
    for i, marker in enumerate(step_markers):
        start_idx = response.find(marker)
        if start_idx == -1:
            continue
            
        # Find the end of this step (start of next step or end of text)
        if i < len(step_markers) - 1:
            end_idx = response.find(step_markers[i+1])
            if end_idx == -1:
                end_idx = len(response)
        else:
            end_idx = len(response)
            
            # Try to find common section headers that might appear after the steps
            for header in ["Title:", "Research Title:", "Subtopics:", "Applications:"]:
                header_idx = response.find(header, start_idx)
                if header_idx != -1 and header_idx < end_idx:
                    end_idx = header_idx
        
        # Extract and clean the step content
        step_content = response[start_idx:end_idx].strip()
        step_title = step_content.split("\n")[0].replace(marker, "").strip()
        step_content = step_content.replace(step_title, "", 1).strip()
        
        steps.append({
            "step": i + 1,
            "title": step_title,
            "content": step_content
        })
    
    # Extract title
    title = "Interdisciplinary Research Topic"  # Default if we can't find it
    title_markers = ["Title:", "Research Title:"]
    for marker in title_markers:
        start_idx = response.find(marker)
        if start_idx != -1:
            end_idx = response.find("\n", start_idx)
            if end_idx == -1:
                end_idx = len(response)
            title = response[start_idx + len(marker):end_idx].strip()
            break
    
    # Extract subtopics
    subtopics = []
    subtopic_section = False
    for line in response.split("\n"):
        line = line.strip()
        if "Subtopic" in line or "Research Question" in line:
            subtopic_section = True
            continue
        if subtopic_section and line and not line.startswith("Application") and not "impact" in line.lower():
            # Skip empty lines and headers
            if line.strip() and not line.endswith(":") and len(line) > 10:
                subtopics.append(line)
        if "Application" in line:
            subtopic_section = False
    
    # Clean up subtopics (remove numbers and bullet points)
    cleaned_subtopics = []
    for subtopic in subtopics:
        if subtopic.startswith(("- ", "• ", "* ")):
            subtopic = subtopic[2:].strip()
        if subtopic.startswith(("1.", "2.", "3.", "4.", "5.")):
            subtopic = subtopic[2:].strip()
        cleaned_subtopics.append(subtopic)
    
    # Extract applications
    applications = []
    application_section = False
    for line in response.split("\n"):
        line = line.strip()
        if "Application" in line and not "Impact" in line:
            application_section = True
            continue
        if application_section and line and not "Score" in line and not "Novelty" in line:
            # Skip empty lines and headers
            if line.strip() and not line.endswith(":") and len(line) > 10:
                applications.append(line)
        if "Score" in line or "Novelty" in line:
            application_section = False
    
    # Clean up applications (remove numbers and bullet points)
    cleaned_applications = []
    for app in applications:
        if app.startswith(("- ", "• ", "* ")):
            app = app[2:].strip()
        if app.startswith(("1.", "2.", "3.", "4.", "5.")):
            app = app[2:].strip()
        cleaned_applications.append(app)
    
    # Extract scores
    novelty = feasibility = impact = 7.0  # Default scores
    score_markers = ["Novelty:", "Feasibility:", "Impact:"]
    for marker in score_markers:
        start_idx = response.find(marker)
        if start_idx != -1:
            end_idx = response.find("\n", start_idx)
            if end_idx == -1:
                end_idx = len(response)
            score_text = response[start_idx + len(marker):end_idx].strip()
            # Extract numeric value
            try:
                score = float([s for s in score_text.split() if s.replace(".", "").isdigit()][0])
                if marker == "Novelty:":
                    novelty = min(10.0, max(0.0, score))
                elif marker == "Feasibility:":
                    feasibility = min(10.0, max(0.0, score))
                elif marker == "Impact:":
                    impact = min(10.0, max(0.0, score))
            except (IndexError, ValueError):
                pass
    
    # Create a research topic object
    topic = ResearchTopic(
        title=title,
        source_domains=set(domain["name"] for domain in domains),
        confidence=0.85,  # Default confidence
        subtopics=cleaned_subtopics[:5],  # Limit to 5 subtopics
        potential_applications=cleaned_applications[:4],  # Limit to 4 applications
        novelty_score=novelty / 10.0,  # Convert to 0-1 scale
        feasibility_score=feasibility / 10.0,
        impact_score=impact / 10.0
    )
    
    # Add chain of thought steps
    for step in steps:
        topic.add_thought_step(step)
    
    return topic.to_dict()

def save_research_topics(topics: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save research topics to a file in JSON format.
    
    Args:
        topics: List of research topic dictionaries
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=2)

def save_research_topics_csv(topics: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save research topics to a CSV file for easy import into spreadsheets.
    
    Args:
        topics: List of research topic dictionaries
        output_file: Path to output CSV file
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['title', 'domains', 'novelty', 'feasibility', 'impact', 'overall_score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for topic in topics:
            writer.writerow({
                'title': topic['title'],
                'domains': ', '.join(topic['source_domains']),
                'novelty': f"{topic['novelty_score']:.2f}",
                'feasibility': f"{topic['feasibility_score']:.2f}",
                'impact': f"{topic['impact_score']:.2f}",
                'overall_score': f"{topic['overall_score']:.2f}"
            })

def format_research_topic_markdown(topic: Dict[str, Any]) -> str:
    """
    Format a research topic as markdown for readable output.
    
    Args:
        topic: Research topic dictionary
        
    Returns:
        Markdown formatted string
    """
    md = f"# {topic['title']}\n\n"
    md += f"**Domains**: {', '.join(topic['source_domains'])}\n\n"
    md += f"**Scores**:\n- Novelty: {topic['novelty_score']:.2f}\n- Feasibility: {topic['feasibility_score']:.2f}\n- Impact: {topic['impact_score']:.2f}\n- Overall: {topic['overall_score']:.2f}\n\n"
    
    md += "## Chain of Thought Reasoning\n\n"
    for step in topic['chain_of_thought']:
        md += f"### Step {step['step']}: {step['title']}\n\n{step['content']}\n\n"
    
    md += "## Subtopics\n\n"
    for i, subtopic in enumerate(topic['subtopics'], 1):
        md += f"{i}. {subtopic}\n"
    
    md += "\n## Potential Applications\n\n"
    for i, application in enumerate(topic['potential_applications'], 1):
        md += f"{i}. {application}\n"
    
    return md

def main():
    parser = argparse.ArgumentParser(description="Generate research topics using chain of thought decomposition across multiple knowledge domains")
    parser.add_argument("--dir", type=str, default="knowledge_bases", 
                        help="Directory containing knowledge base JSON files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to save results (default: print to console)")
    parser.add_argument("--output-dir", type=str, default="research_topics",
                        help="Directory to save all generated topics (when using --count > 1)")
    parser.add_argument("--model", type=str, default="claude-3-opus-20240229",
                        help="LLM model to use (default: claude-3-opus-20240229)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of research topics to generate")
    parser.add_argument("--domains-per-topic", type=int, default=2,
                        help="Number of domains to combine for each topic (default: 2)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of chain of thought reasoning steps (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for generation (0.0-1.0, higher = more creative, default: 0.8)")
    parser.add_argument("--kb1", type=str, default=None,
                        help="Specify first knowledge base (optional)")
    parser.add_argument("--kb2", type=str, default=None,
                        help="Specify second knowledge base (optional)")
    parser.add_argument("--kb3", type=str, default=None,
                        help="Specify third knowledge base (optional)")
    parser.add_argument("--format", type=str, default="md", choices=["md", "json", "csv"],
                        help="Output format (md, json, or csv, default: md)")
    parser.add_argument("--batch-name", type=str, default=None,
                        help="Name for this batch of generated topics (default: timestamp)")
    
    args = parser.parse_args()
    
    # Get absolute path to the knowledge bases directory
    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dir)
    
    # Create output directory if needed
    if args.count > 1:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all knowledge bases
    all_kbs = get_all_knowledge_bases(kb_dir)
    print(f"Found {len(all_kbs)} knowledge bases")
    
    # Prepare to collect all generated topics
    all_topics = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = args.batch_name or f"batch_{timestamp}"
    
    # Generate the requested number of topics
    for i in range(args.count):
        # Select knowledge bases for this topic
        if i == 0 and (args.kb1 or args.kb2 or args.kb3):
            # Use specified knowledge bases for the first topic if provided
            specified_kbs = []
            
            if args.kb1:
                kb_path1 = os.path.join(kb_dir, args.kb1 if args.kb1.endswith('.json') else f"{args.kb1}.json")
                if not os.path.exists(kb_path1):
                    raise FileNotFoundError(f"Knowledge base not found: {kb_path1}")
                specified_kbs.append(kb_path1)
                
            if args.kb2:
                kb_path2 = os.path.join(kb_dir, args.kb2 if args.kb2.endswith('.json') else f"{args.kb2}.json")
                if not os.path.exists(kb_path2):
                    raise FileNotFoundError(f"Knowledge base not found: {kb_path2}")
                specified_kbs.append(kb_path2)
                
            if args.kb3:
                kb_path3 = os.path.join(kb_dir, args.kb3 if args.kb3.endswith('.json') else f"{args.kb3}.json")
                if not os.path.exists(kb_path3):
                    raise FileNotFoundError(f"Knowledge base not found: {kb_path3}")
                specified_kbs.append(kb_path3)
                
            selected_kbs = specified_kbs
        else:
            # Randomly select knowledge bases
            kb_count = args.domains_per_topic
            if kb_count < 2:
                kb_count = 2  # Ensure at least 2 domains
            if kb_count > 5:
                kb_count = 5  # Cap at 5 domains to avoid overwhelming the LLM
                
            selected_kbs = select_random_knowledge_bases(all_kbs, kb_count)
        
        # Load the selected knowledge bases
        domains = [load_knowledge_base(kb) for kb in selected_kbs]
        
        domain_names = [domain.get('name', 'Unknown') for domain in domains]
        print(f"\nGenerating topic {i+1}/{args.count}: Combining {', '.join(domain_names)}")
        
        # Generate research topic
        try:
            research_topic = generate_cross_domain_research(
                domains=domains,
                model=args.model,
                chain_of_thought_steps=args.steps,
                temperature=args.temperature
            )
            
            all_topics.append(research_topic)
            
            # Print a summary
            print(f"Created research topic: {research_topic['title']}")
            print(f"Scores: Novelty={research_topic['novelty_score']:.2f}, Feasibility={research_topic['feasibility_score']:.2f}, Impact={research_topic['impact_score']:.2f}, Overall={research_topic['overall_score']:.2f}")
            
            # Save individual topic if generating multiple
            if args.count > 1:
                domain_slug = "_".join([d.split(" ")[0].lower() for d in research_topic['source_domains']])
                topic_slug = research_topic['title'].lower().replace(" ", "_")[:30]
                
                if args.format == "md":
                    topic_path = os.path.join(args.output_dir, f"{domain_slug}_{topic_slug}.md")
                    with open(topic_path, 'w', encoding='utf-8') as f:
                        f.write(format_research_topic_markdown(research_topic))
                    print(f"Saved topic to {topic_path}")
                    
                elif args.format == "json":
                    topic_path = os.path.join(args.output_dir, f"{domain_slug}_{topic_slug}.json")
                    with open(topic_path, 'w', encoding='utf-8') as f:
                        json.dump(research_topic, f, indent=2)
                    print(f"Saved topic to {topic_path}")
        
        except Exception as e:
            print(f"Error generating research topic: {str(e)}")
    
    # Output all results to specified format
    if args.output:
        # Save to specified output file
        if args.format == "md":
            with open(args.output, 'w', encoding='utf-8') as f:
                for i, topic in enumerate(all_topics, 1):
                    f.write(f"{format_research_topic_markdown(topic)}\n\n")
                    if i < len(all_topics):
                        f.write("---\n\n")
            print(f"All topics saved to {args.output}")
            
        elif args.format == "json":
            save_research_topics(all_topics, args.output)
            print(f"All topics saved to {args.output}")
            
        elif args.format == "csv":
            save_research_topics_csv(all_topics, args.output)
            print(f"All topics saved to {args.output}")
    
    # Additionally, save all topics to a batch file if generating multiple
    if args.count > 1:
        batch_file = os.path.join(args.output_dir, f"{batch_name}_all.json")
        save_research_topics(all_topics, batch_file)
        print(f"All {len(all_topics)} topics saved to {batch_file}")
        
        # Also save a CSV with the summary
        csv_file = os.path.join(args.output_dir, f"{batch_name}_summary.csv")
        save_research_topics_csv(all_topics, csv_file)
        print(f"Topic summary saved to {csv_file}")
    
    # Print the full output for a single topic if not saving to file
    if not args.output and len(all_topics) == 1:
        print("\n" + "="*80)
        print(format_research_topic_markdown(all_topics[0]))
        print("="*80)

if __name__ == "__main__":
    main()