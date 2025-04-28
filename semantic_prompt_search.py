#!/usr/bin/env python3
"""
Semantic Prompt Search Tool

This tool demonstrates the use of the embedding service with the polymorphic prompts system.
It allows for semantic search of prompts based on text similarity.
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional
import yaml
import json

# Import from our modules
from embedding_service import EmbeddingServiceClient
import advanced_polymorphic_prompts as app


def load_all_prompts(prompt_dir: str = "./prompts") -> List[app.Prompt]:
    """Load all prompts from the prompts directory"""
    prompt_manager = app.AdvancedPromptManager(prompts_dir=prompt_dir)
    prompt_names = prompt_manager.list_prompts()
    
    prompts = []
    for name in prompt_names:
        prompt = prompt_manager.load_prompt(name)
        if prompt:
            prompts.append(prompt)
    
    return prompts


def convert_to_yaml(prompts: List[app.Prompt], output_dir: str = "./prompts") -> None:
    """Convert all text prompts to YAML format with metadata"""
    prompt_manager = app.AdvancedPromptManager(prompts_dir=output_dir)
    
    for prompt in prompts:
        # Detect prompt type if not already set
        if prompt.metadata.prompt_type == "standard":
            detected_type = prompt_manager.detect_prompt_type(prompt.content)
            if detected_type != "standard":
                prompt.metadata.prompt_type = detected_type
                print(f"Detected prompt type for {prompt.name}: {detected_type}")
        
        # Extract variables
        if not prompt.variables:
            prompt.variables = prompt_manager.extract_variables(prompt.content)
            
        # Save in YAML format
        prompt_manager.save_prompt(prompt, format="yaml")
        
    print(f"Converted {len(prompts)} prompts to YAML format")


def generate_embeddings(prompts: List[app.Prompt]) -> None:
    """Generate embeddings for all prompts"""
    client = EmbeddingServiceClient()
    prompt_manager = app.AdvancedPromptManager()
    
    print("Generating embeddings for all prompts...")
    
    # Prepare texts for batch embedding
    texts = [prompt.content for prompt in prompts]
    prompt_ids = [prompt.name for prompt in prompts]
    
    try:
        # Generate embeddings in batch
        embeddings_result = client.generate_embeddings(texts)
        
        # Store embeddings back in prompts
        for i, prompt in enumerate(prompts):
            if 'embeddings' in embeddings_result and i < len(embeddings_result['embeddings']):
                prompt.embeddings["main"] = embeddings_result['embeddings'][i]['embedding']
                print(f"Generated embedding for {prompt.name} ({len(prompt.embeddings['main'])} dimensions)")
                
                # Save the prompt with embedding
                prompt_manager.save_prompt(prompt)
            else:
                print(f"Failed to generate embedding for {prompt.name}")
    
    except Exception as e:
        print(f"Error generating embeddings: {e}")


def search_prompts(query: str, prompts: List[app.Prompt], top_k: int = 3) -> List[Tuple[app.Prompt, float]]:
    """Search for prompts semantically similar to the query"""
    client = EmbeddingServiceClient()
    
    # Generate query embedding
    try:
        query_embedding = client.embed_text(query)
        
        # Calculate similarities
        results = []
        for prompt in prompts:
            if "main" in prompt.embeddings and prompt.embeddings["main"]:
                similarity = client._cosine_similarity(query_embedding, prompt.embeddings["main"])
                results.append((prompt, similarity))
            
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []


def print_prompt_details(prompt: app.Prompt, similarity: Optional[float] = None) -> None:
    """Print details of a prompt with optional similarity score"""
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt.name}")
    
    if similarity is not None:
        print(f"Similarity: {similarity:.4f}")
        
    print(f"Type: {prompt.metadata.prompt_type}")
    
    if prompt.metadata.description:
        print(f"Description: {prompt.metadata.description}")
        
    if prompt.metadata.tags:
        print(f"Tags: {', '.join(prompt.metadata.tags)}")
        
    # Print variables if any
    if prompt.variables:
        print("\nVariables:")
        for var_name, var_value in prompt.variables.items():
            print(f"  {var_name}: {var_value or '(empty)'}")
    
    # Print a preview of the content
    print("\nContent Preview:")
    preview = prompt.content[:500]
    if len(prompt.content) > 500:
        preview += "..."
    print(preview)
    print("=" * 50)


def main():
    """Main entry point for the semantic prompt search tool"""
    parser = argparse.ArgumentParser(
        description="Semantic Prompt Search Tool"
    )
    
    parser.add_argument(
        "--generate-embeddings", 
        action="store_true",
        help="Generate embeddings for all prompts"
    )
    
    parser.add_argument(
        "--convert-yaml", 
        action="store_true",
        help="Convert all prompts to YAML format with metadata"
    )
    
    parser.add_argument(
        "--search", 
        type=str,
        help="Search for semantically similar prompts"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int,
        default=3,
        help="Number of top results to return (default: 3)"
    )
    
    parser.add_argument(
        "--create", 
        action="store_true",
        help="Create example prompts from our template set"
    )
    
    parser.add_argument(
        "--compose",
        nargs="+",
        help="Compose multiple prompts (provide prompt names)"
    )
    
    parser.add_argument(
        "--composition-type",
        type=str,
        default="sequence",
        choices=["sequence", "nested", "parallel"],
        help="Type of composition to create"
    )
    
    parser.add_argument(
        "--prompts-dir", 
        type=str,
        default="./prompts",
        help="Directory containing prompts (default: ./prompts)"
    )
    
    args = parser.parse_args()
    
    # Load all prompts
    prompts = load_all_prompts(args.prompts_dir)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_dir}")
    
    # Create example prompts if requested
    if args.create and not prompts:
        print("Creating example prompts...")
        prompt_manager = app.AdvancedPromptManager(prompts_dir=args.prompts_dir)
        
        # Create holographic prompt
        holographic = prompt_manager.create_prompt(
            name="holographic_analysis",
            content="""<holographic_prompt>
  <level depth="1">
    Summarize the key points of this research paper.
  </level>
  <level depth="2">
    Analyze this research paper, extracting its methodology, findings, and limitations. 
    Provide a structured summary with sections for background, methods, results, and discussion.
  </level>
  <level depth="3">
    Conduct a comprehensive analysis of this research paper. Include:
    
    1. Background context and how it fits into the broader literature
    2. Detailed methodology assessment including statistical approaches
    3. Critical evaluation of results with attention to statistical significance
    4. Discussion of limitations, potential biases, and alternative interpretations
    5. Implications for theory and practice
    6. Suggestions for future research directions
    
    Organize your response with clear headings and use specific examples from the paper to support your analysis.
  </level>
</holographic_prompt>""",
            prompt_type="holographic",
            description="Multi-level research paper analysis prompt",
            tags=["research", "analysis", "academic"]
        )
        prompt_manager.save_prompt(holographic)
        
        # Create temporal prompt
        temporal = prompt_manager.create_prompt(
            name="temporal_market_analysis",
            content="""<temporal_prompt>
  <timeframe period="immediate">
    Analyze the current market conditions and immediate trends for the {{industry}} industry. 
    Focus on data from the last 7 days, including recent news, price movements, and social media sentiment.
  </timeframe>
  <timeframe period="recent" range="past-30-days">
    Examine the past month's developments in this industry. 
    Identify emerging patterns, comparative performance metrics, and noteworthy events. 
    Consider how these recent developments might influence immediate decisions.
  </timeframe>
  <timeframe period="historical" range="past-5-years">
    Provide historical context by analyzing the industry's performance over the past five years. 
    Identify cyclical patterns, long-term trends, and structural changes. 
    Compare current metrics against historical benchmarks.
  </timeframe>
  <timeframe period="future" range="next-6-months">
    Forecast likely industry developments over the next six months. 
    Consider scheduled events, announced changes, seasonal factors, and extrapolate from current trends. 
    Identify potential disruptions, opportunities, and strategic pivots.
  </timeframe>
</temporal_prompt>""",
            prompt_type="temporal",
            description="Time-aware market analysis prompt",
            tags=["market", "analysis", "forecasting"],
            variables={"industry": ""}
        )
        prompt_manager.save_prompt(temporal)
        
        # Create multi-agent prompt
        multi_agent = prompt_manager.create_prompt(
            name="multi_agent_data_analysis",
            content="""<multi_agent_prompt>
  <agent role="analyst">
    Examine the provided {{data_type}} data set and identify key patterns, trends, and statistical insights. 
    Focus on numerical analysis and quantitative relationships. Generate visualizations if appropriate.
  </agent>
  <agent role="critic">
    Evaluate the methodology, identify potential biases, weaknesses, and limitations in the data. 
    Consider what might be missing or misrepresented. Challenge assumptions and identify alternative interpretations.
  </agent>
  <agent role="creative">
    Generate novel hypotheses and unexpected connections based on the data. 
    Consider unusual patterns or outliers as potential innovation sources. 
    Suggest creative approaches to address limitations.
  </agent>
  <agent role="implementer">
    Develop a practical action plan based on the insights. 
    Include specific, actionable steps, timeframes, resource requirements, and potential obstacles. 
    Focus on translating insights into real-world application.
  </agent>
  <integration>
    Synthesize the perspectives from all agents to create a comprehensive, balanced analysis 
    that combines analytical rigor, critical thinking, creative insights, and practical implementation.
  </integration>
</multi_agent_prompt>""",
            prompt_type="multi-agent",
            description="Multi-perspective data analysis prompt",
            tags=["data", "analysis", "multi-agent"],
            variables={"data_type": ""}
        )
        prompt_manager.save_prompt(multi_agent)
        
        # Create self-calibrating prompt
        self_calibrating = prompt_manager.create_prompt(
            name="self_calibrating_recommendation",
            content="""<self_calibrating_prompt>
  <instruction>
    Provide recommendations for {{topic}} based on the information provided.
    Analyze the available data and offer strategic advice tailored to the specific context.
  </instruction>
  <confidence_requirements>
    <requirement>Estimate your confidence for each recommendation on a scale of 1-10</requirement>
    <requirement>For any confidence below 7, provide alternative recommendations</requirement>
    <requirement>For any confidence below 5, explicitly state what information would improve confidence</requirement>
  </confidence_requirements>
  <verification_steps>
    <step>Check factual claims against reliable sources</step>
    <step>Verify logical consistency of all arguments</step>
    <step>Ensure comprehensive coverage of all aspects of the question</step>
  </verification_steps>
  <output_format>
    <answer>Main recommendations with confidence markers</answer>
    <alternatives>Alternative possibilities for low-confidence items</alternatives>
    <information_needs>Additional information that would improve confidence</information_needs>
  </output_format>
</self_calibrating_prompt>""",
            prompt_type="self-calibrating",
            description="Confidence-aware recommendation prompt",
            tags=["recommendation", "confidence", "decision-making"],
            variables={"topic": ""}
        )
        prompt_manager.save_prompt(self_calibrating)
        
        # Create knowledge-synthesis prompt
        knowledge_synthesis = prompt_manager.create_prompt(
            name="knowledge_synthesis_education",
            content="""<knowledge_synthesis_prompt>
  <domain name="education">
    Consider educational theories, learning methodologies, and pedagogical approaches
    relevant to {{educational_level}} education. Focus on evidence-based practices
    and established frameworks in the field of education.
  </domain>
  <domain name="psychology">
    Incorporate insights from cognitive psychology, developmental psychology,
    and motivational theory that explain how students learn, develop, and stay engaged.
  </domain>
  <domain name="technology">
    Examine how educational technology, digital tools, and innovative platforms
    can enhance learning experiences and outcomes in modern educational settings.
  </domain>
  <connection_points>
    <connection>
      <from>Learning theories from education</from>
      <to>Cognitive processing models from psychology</to>
      <relationship>Foundation for understanding how information is acquired and retained</relationship>
    </connection>
    <connection>
      <from>Pedagogical methodologies from education</from>
      <to>Educational technology applications</to>
      <relationship>Implementation vehicles for teaching strategies</relationship>
    </connection>
    <connection>
      <from>Motivational frameworks from psychology</from>
      <to>Engagement design in educational technology</to>
      <relationship>Creating sustained interest and participation</relationship>
    </connection>
  </connection_points>
  <synthesis_goal>
    Create an integrated understanding that leverages insights from education, psychology,
    and technology to develop a comprehensive approach for {{educational_context}}.
  </synthesis_goal>
</knowledge_synthesis_prompt>""",
            prompt_type="knowledge-synthesis",
            description="Cross-domain educational synthesis prompt",
            tags=["education", "psychology", "technology", "synthesis"],
            variables={"educational_level": "", "educational_context": ""}
        )
        prompt_manager.save_prompt(knowledge_synthesis)
        
        print("Created 5 example prompts")
        
        # Reload prompts
        prompts = load_all_prompts(args.prompts_dir)
    
    # Convert prompts to YAML if requested
    if args.convert_yaml:
        convert_to_yaml(prompts, args.prompts_dir)
        # Reload prompts after conversion
        prompts = load_all_prompts(args.prompts_dir)
    
    # Generate embeddings if requested
    if args.generate_embeddings:
        generate_embeddings(prompts)
        # Reload prompts with embeddings
        prompts = load_all_prompts(args.prompts_dir)
    
    # Compose prompts if requested
    if args.compose and len(args.compose) >= 2:
        prompt_manager = app.AdvancedPromptManager(prompts_dir=args.prompts_dir)
        
        # Find the requested prompts
        compose_prompts = []
        for name in args.compose:
            found = False
            for prompt in prompts:
                if prompt.name == name:
                    compose_prompts.append(prompt)
                    found = True
                    break
            if not found:
                print(f"Warning: Prompt '{name}' not found")
        
        if len(compose_prompts) >= 2:
            # Create the composition
            composition = prompt_manager.create_composition(
                compose_prompts, 
                args.composition_type
            )
            
            # Save the composition
            prompt_manager.save_prompt(composition)
            
            print(f"Created composition '{composition.name}' using {args.composition_type} method")
            print_prompt_details(composition)
    
    # Search for prompts if requested
    if args.search:
        search_results = search_prompts(args.search, prompts, args.top_k)
        
        if search_results:
            print(f"\nTop {len(search_results)} results for query: '{args.search}'")
            
            for prompt, similarity in search_results:
                print_prompt_details(prompt, similarity)
        else:
            print(f"No matching prompts found for query: '{args.search}'")


if __name__ == "__main__":
    main()