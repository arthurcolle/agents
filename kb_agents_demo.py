#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional

# Import our knowledge base components
from knowledge_base_dispatcher import dispatcher as kb_dispatcher
from kb_agent_connector import connector as kb_connector
from kb_cli_integration import apply_kb_integration_patches

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kb_agents_demo")

async def list_knowledge_bases():
    """List all available knowledge bases"""
    kb_list = kb_connector.get_available_knowledge_bases()
    
    print(f"Found {len(kb_list)} knowledge bases:")
    for i, kb in enumerate(kb_list):
        print(f"{i+1}. {kb['name']}")
    
    return kb_list

async def search_knowledge_base(kb_name: str, query: str):
    """Search a specific knowledge base"""
    print(f"Searching knowledge base '{kb_name}' for: {query}")
    
    result = await kb_connector.dispatch_to_kb_agent(kb_name, f"search {query}")
    
    if result.get("success", False):
        print(f"Search results:")
        data = result.get("data", [])
        if isinstance(data, list):
            for i, item in enumerate(data):
                print(f"Result {i+1}:")
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key != "content" or len(str(value)) < 100:
                            print(f"  - {key}: {value}")
                        else:
                            print(f"  - {key}: {str(value)[:100]}...")
                else:
                    print(f"  {item}")
        else:
            print(data)
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def search_all_knowledge_bases(query: str):
    """Search across all knowledge bases"""
    print(f"Searching all knowledge bases for: {query}")
    
    result = await kb_connector.dispatch_query_to_all_kbs(query)
    
    if result.get("success", False):
        results = result.get("results", [])
        print(f"Found {len(results)} results across all knowledge bases")
        
        for i, item in enumerate(results[:5]):  # Show first 5 results
            print(f"Result {i+1} from {item.get('source_kb', 'unknown')}:")
            for key, value in item.items():
                if key not in ["source_kb", "content"] or len(str(value)) < 100:
                    print(f"  - {key}: {value}")
                else:
                    print(f"  - {key}: {str(value)[:100]}...")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def get_kb_info(kb_name: str):
    """Get information about a knowledge base"""
    print(f"Getting information about knowledge base: {kb_name}")
    
    result = await kb_connector.dispatch_to_kb_agent(kb_name, "info")
    
    if result.get("success", False):
        print(f"Knowledge base information:")
        info = result.get("data", {})
        for key, value in info.items():
            print(f"  - {key}: {value}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def list_kb_entries(kb_name: str, limit: int = 5):
    """List entries in a knowledge base"""
    print(f"Listing entries in knowledge base: {kb_name} (limit: {limit})")
    
    result = await kb_connector.dispatch_to_kb_agent(kb_name, f"list_entries {limit}")
    
    if result.get("success", False):
        entries = result.get("data", [])
        print(f"Found {len(entries)} entries:")
        for i, entry in enumerate(entries):
            print(f"Entry {i+1}:")
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if key != "content" or len(str(value)) < 100:
                        print(f"  - {key}: {value}")
                    else:
                        print(f"  - {key}: {str(value)[:100]}...")
            else:
                print(f"  {entry}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def get_kb_entry(kb_name: str, entry_id: str):
    """Get a specific entry from a knowledge base"""
    print(f"Getting entry {entry_id} from knowledge base: {kb_name}")
    
    result = await kb_connector.dispatch_to_kb_agent(kb_name, f"get_entry {entry_id}")
    
    if result.get("success", False):
        entry = result.get("data", {})
        print(f"Entry content:")
        if isinstance(entry, dict):
            for key, value in entry.items():
                if key != "content" or len(str(value)) < 200:
                    print(f"  - {key}: {value}")
                else:
                    print(f"  - {key}: {str(value)[:200]}...")
                    print(f"     (content truncated, {len(str(value))} characters total)")
        else:
            print(f"  {entry}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def interactive_mode():
    """Run in interactive mode, allowing the user to explore knowledge bases"""
    print("Knowledge Base Agents Demo - Interactive Mode")
    print("--------------------------------------------")
    
    while True:
        print("\nOptions:")
        print("1. List all knowledge bases")
        print("2. Search a specific knowledge base")
        print("3. Search all knowledge bases")
        print("4. Get knowledge base information")
        print("5. List knowledge base entries")
        print("6. Get a specific knowledge base entry")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            break
        
        try:
            if choice == '1':
                await list_knowledge_bases()
                
            elif choice == '2':
                kb_list = await list_knowledge_bases()
                if not kb_list:
                    print("No knowledge bases found")
                    continue
                
                kb_index = input("Enter knowledge base number: ").strip()
                try:
                    kb_index = int(kb_index) - 1
                    if kb_index < 0 or kb_index >= len(kb_list):
                        print(f"Invalid index. Must be between 1 and {len(kb_list)}")
                        continue
                    
                    kb_name = kb_list[kb_index]["name"]
                    query = input("Enter search query: ").strip()
                    await search_knowledge_base(kb_name, query)
                except ValueError:
                    print("Invalid input. Please enter a number.")
                
            elif choice == '3':
                query = input("Enter search query: ").strip()
                await search_all_knowledge_bases(query)
                
            elif choice == '4':
                kb_list = await list_knowledge_bases()
                if not kb_list:
                    print("No knowledge bases found")
                    continue
                
                kb_index = input("Enter knowledge base number: ").strip()
                try:
                    kb_index = int(kb_index) - 1
                    if kb_index < 0 or kb_index >= len(kb_list):
                        print(f"Invalid index. Must be between 1 and {len(kb_list)}")
                        continue
                    
                    kb_name = kb_list[kb_index]["name"]
                    await get_kb_info(kb_name)
                except ValueError:
                    print("Invalid input. Please enter a number.")
                
            elif choice == '5':
                kb_list = await list_knowledge_bases()
                if not kb_list:
                    print("No knowledge bases found")
                    continue
                
                kb_index = input("Enter knowledge base number: ").strip()
                try:
                    kb_index = int(kb_index) - 1
                    if kb_index < 0 or kb_index >= len(kb_list):
                        print(f"Invalid index. Must be between 1 and {len(kb_list)}")
                        continue
                    
                    kb_name = kb_list[kb_index]["name"]
                    limit = input("Enter limit (or press Enter for default): ").strip()
                    limit = int(limit) if limit.isdigit() else 5
                    await list_kb_entries(kb_name, limit)
                except ValueError:
                    print("Invalid input. Please enter a number.")
                
            elif choice == '6':
                kb_list = await list_knowledge_bases()
                if not kb_list:
                    print("No knowledge bases found")
                    continue
                
                kb_index = input("Enter knowledge base number: ").strip()
                try:
                    kb_index = int(kb_index) - 1
                    if kb_index < 0 or kb_index >= len(kb_list):
                        print(f"Invalid index. Must be between 1 and {len(kb_list)}")
                        continue
                    
                    kb_name = kb_list[kb_index]["name"]
                    
                    # First list entries so user can see available IDs
                    await list_kb_entries(kb_name, 5)
                    
                    entry_id = input("Enter entry ID: ").strip()
                    await get_kb_entry(kb_name, entry_id)
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            else:
                print("Invalid choice. Please try again.")
        
        except Exception as e:
            print(f"Error: {e}")

async def main():
    """Main entry point for the knowledge base agents demo"""
    parser = argparse.ArgumentParser(description="Knowledge Base Agents Demo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list", "-l", action="store_true", help="List all knowledge bases")
    parser.add_argument("--search", "-s", nargs=2, metavar=("KB_NAME", "QUERY"), help="Search a knowledge base")
    parser.add_argument("--search-all", "-a", metavar="QUERY", help="Search all knowledge bases")
    parser.add_argument("--info", "-f", metavar="KB_NAME", help="Get information about a knowledge base")
    parser.add_argument("--entries", "-e", nargs="+", metavar=("KB_NAME", "LIMIT"), help="List entries in a knowledge base")
    parser.add_argument("--entry", "-n", nargs=2, metavar=("KB_NAME", "ENTRY_ID"), help="Get a specific entry from a knowledge base")
    
    args = parser.parse_args()
    
    # Apply patches to CLIAgent if it's available
    try:
        apply_kb_integration_patches()
        print("CLIAgent integration applied successfully")
    except Exception as e:
        print(f"CLIAgent integration failed: {e}")
        print("Continuing with standalone demo...")
    
    # Run requested command or interactive mode
    if args.interactive or len(sys.argv) == 1:
        await interactive_mode()
    elif args.list:
        await list_knowledge_bases()
    elif args.search:
        kb_name, query = args.search
        await search_knowledge_base(kb_name, query)
    elif args.search_all:
        await search_all_knowledge_bases(args.search_all)
    elif args.info:
        await get_kb_info(args.info)
    elif args.entries:
        kb_name = args.entries[0]
        limit = int(args.entries[1]) if len(args.entries) > 1 and args.entries[1].isdigit() else 5
        await list_kb_entries(kb_name, limit)
    elif args.entry:
        kb_name, entry_id = args.entry
        await get_kb_entry(kb_name, entry_id)

if __name__ == "__main__":
    asyncio.run(main())