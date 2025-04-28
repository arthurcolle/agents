#!/usr/bin/env python3
"""
Embeddings CLI - Command-line interface for the Embeddings API
"""

import argparse
import json
import logging
import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np

# Import the embeddings client
from modules.embeddings_client import EmbeddingsClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("embeddings-cli")

def generate_embeddings(client: EmbeddingsClient, args: argparse.Namespace):
    """Generate embeddings for input texts"""
    # Get input texts
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Either --text or --file must be provided")
        return 1
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} texts...")
    response = client.generate_embeddings(texts, batch_size=args.batch_size, max_length=args.max_length)
    
    if "error" in response:
        print(f"Error: {response.get('message', 'Unknown error')}")
        return 1
    
    # Process results
    embeddings = response["embeddings"]
    print(f"Generated {len(embeddings)} embeddings")
    
    if args.output:
        # Save to file
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.format == 'json':
                json.dump(response, f, indent=2)
            else:
                # Simple text format
                for i, embedding in enumerate(embeddings):
                    f.write(f"Embedding {i+1}:\n")
                    f.write(f"  ID: {embedding['id']}\n")
                    f.write(f"  Content: {embedding['content']}\n")
                    f.write(f"  Vector: {embedding['embedding'][:5]}... (first 5 elements)\n")
                    f.write("\n")
        
        print(f"Results saved to {args.output}")
    else:
        # Print to console
        for i, embedding in enumerate(embeddings):
            if i < 3:  # Limit output
                print(f"\nEmbedding {i+1}:")
                print(f"  ID: {embedding['id']}")
                print(f"  Content: {embedding['content']}")
                print(f"  Vector: {embedding['embedding'][:5]}... (first 5 elements)")
        
        if len(embeddings) > 3:
            print(f"\n... and {len(embeddings) - 3} more embeddings")
    
    return 0

def compute_similarity(client: EmbeddingsClient, args: argparse.Namespace):
    """Compute similarity between texts"""
    if not args.text1 or not args.text2:
        print("Error: Both --text1 and --text2 must be provided")
        return 1
    
    # Compute similarity
    print("Computing similarity...")
    response = client.compute_cosine_similarity(args.text1, args.text2)
    
    if "error" in response:
        print(f"Error: {response.get('message', 'Unknown error')}")
        return 1
    
    # Display results
    similarity = response["similarity"]
    print(f"\nSimilarity: {similarity:.4f}")
    print(f"Text 1: {args.text1}")
    print(f"Text 2: {args.text2}")
    
    return 0

def pairwise_similarity(client: EmbeddingsClient, args: argparse.Namespace):
    """Compute pairwise similarities for a list of texts"""
    # Get input texts
    texts = []
    if args.texts:
        texts = args.texts
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Either --texts or --file must be provided")
        return 1
    
    if len(texts) < 2:
        print("Error: At least 2 texts are required for pairwise similarity")
        return 1
    
    # Compute pairwise similarity
    print(f"Computing pairwise similarity for {len(texts)} texts...")
    response = client.compute_pairwise_similarity(texts)
    
    if "error" in response:
        print(f"Error: {response.get('message', 'Unknown error')}")
        return 1
    
    # Process results
    similarity_matrix = response["similarity"]
    
    if args.output:
        # Save to file
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.format == 'json':
                json.dump(response, f, indent=2)
            else:
                # Text format with matrix
                f.write("Pairwise Similarity Matrix:\n\n")
                
                # Write header
                f.write("     ")
                for i in range(len(texts)):
                    f.write(f"{i+1:4d} ")
                f.write("\n")
                
                # Write matrix
                for i in range(len(texts)):
                    f.write(f"{i+1:4d} ")
                    for j in range(len(texts)):
                        f.write(f"{similarity_matrix[i][j]:4.2f} ")
                    f.write("\n")
                
                # Write texts
                f.write("\nTexts:\n")
                for i, text in enumerate(texts):
                    f.write(f"{i+1}: {text}\n")
        
        print(f"Results saved to {args.output}")
    else:
        # Print to console
        print("\nPairwise Similarity Matrix:")
        
        # Print header
        print("     ", end="")
        for i in range(len(texts)):
            print(f"{i+1:4d} ", end="")
        print()
        
        # Print matrix
        for i in range(len(texts)):
            print(f"{i+1:4d} ", end="")
            for j in range(len(texts)):
                print(f"{similarity_matrix[i][j]:4.2f} ", end="")
            print()
        
        # Print texts
        print("\nTexts:")
        for i, text in enumerate(texts):
            print(f"{i+1}: {text}")
    
    return 0

def find_similar(client: EmbeddingsClient, args: argparse.Namespace):
    """Find most similar texts to a query"""
    if not args.query:
        print("Error: --query must be provided")
        return 1
    
    # Get corpus texts
    corpus = []
    if args.corpus_file:
        with open(args.corpus_file, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
    elif args.corpus:
        corpus = args.corpus
    else:
        print("Error: Either --corpus or --corpus-file must be provided")
        return 1
    
    # Find similar texts
    print(f"Finding texts similar to: {args.query}")
    print(f"Corpus size: {len(corpus)} texts")
    
    similar_texts = client.find_most_similar(args.query, corpus, top_k=args.top_k)
    
    if not similar_texts:
        print("No similar texts found or an error occurred")
        return 1
    
    # Display results
    print("\nMost similar texts:")
    for i, (text, similarity, idx) in enumerate(similar_texts):
        print(f"{i+1}. [{similarity:.4f}] {text}")
    
    return 0

def list_models(client: EmbeddingsClient, args: argparse.Namespace):
    """List available embedding models"""
    print("Listing available embedding models...")
    response = client.list_available_models()
    
    if "error" in response:
        print(f"Error: {response.get('message', 'Unknown error')}")
        return 1
    
    # Display results
    print(f"\nCurrent model: {response.get('current_model', 'None')}")
    print("\nAvailable models:")
    for model in response.get("models", []):
        print(f"- {model}")
    
    return 0

def load_model(client: EmbeddingsClient, args: argparse.Namespace):
    """Load a specific embedding model"""
    if not args.model:
        print("Error: --model must be provided")
        return 1
    
    print(f"Loading model: {args.model}...")
    response = client.load_model(args.model)
    
    if "error" in response:
        print(f"Error: {response.get('message', 'Unknown error')}")
        return 1
    
    # Display results
    print(f"Status: {response.get('status', 'Unknown')}")
    print(f"Message: {response.get('message', 'No message provided')}")
    
    return 0

def main():
    """Main function"""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Embeddings CLI")
    parser.add_argument("--url", type=str, default="https://arthurcolle--embeddings.modal.run",
                      help="URL for the Embeddings API")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # generate-embeddings command
    generate_parser = subparsers.add_parser("generate", help="Generate embeddings")
    generate_parser.add_argument("--text", type=str, help="Text to embed")
    generate_parser.add_argument("--file", type=str, help="File with texts to embed (one per line)")
    generate_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    generate_parser.add_argument("--max-length", type=int, default=8192, help="Maximum text length")
    generate_parser.add_argument("--output", type=str, help="Output file for results")
    generate_parser.add_argument("--format", type=str, choices=["json", "text"], default="text",
                               help="Output format (json or text)")
    
    # similarity command
    similarity_parser = subparsers.add_parser("similarity", help="Compute similarity between texts")
    similarity_parser.add_argument("--text1", type=str, help="First text")
    similarity_parser.add_argument("--text2", type=str, help="Second text")
    
    # pairwise-similarity command
    pairwise_parser = subparsers.add_parser("pairwise", help="Compute pairwise similarity")
    pairwise_parser.add_argument("--texts", type=str, nargs="+", help="List of texts")
    pairwise_parser.add_argument("--file", type=str, help="File with texts (one per line)")
    pairwise_parser.add_argument("--output", type=str, help="Output file for results")
    pairwise_parser.add_argument("--format", type=str, choices=["json", "text"], default="text",
                               help="Output format (json or text)")
    
    # find-similar command
    find_parser = subparsers.add_parser("find-similar", help="Find similar texts")
    find_parser.add_argument("--query", type=str, help="Query text")
    find_parser.add_argument("--corpus", type=str, nargs="+", help="List of corpus texts")
    find_parser.add_argument("--corpus-file", type=str, help="File with corpus texts (one per line)")
    find_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    # list-models command
    list_parser = subparsers.add_parser("list-models", help="List available embedding models")
    
    # load-model command
    load_parser = subparsers.add_parser("load-model", help="Load a specific embedding model")
    load_parser.add_argument("--model", type=str, help="Model name to load")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize client
    client = EmbeddingsClient(base_url=args.url)
    
    # Execute command
    if args.command == "generate":
        return generate_embeddings(client, args)
    elif args.command == "similarity":
        return compute_similarity(client, args)
    elif args.command == "pairwise":
        return pairwise_similarity(client, args)
    elif args.command == "find-similar":
        return find_similar(client, args)
    elif args.command == "list-models":
        return list_models(client, args)
    elif args.command == "load-model":
        return load_model(client, args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())