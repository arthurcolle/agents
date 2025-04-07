#!/usr/bin/env python3
import asyncio
import os
import json
from pathlib import Path
from dynamic_agents import registry, AgentContext, execute_agent_command

async def run_file_operations_example():
    """Example of using a file agent to perform file operations"""
    print("=== File Operations Example ===")
    
    # Create a file agent
    file_agent_id = "file_example"
    registry.create_agent(file_agent_id, "file")
    
    # Create a context for the agent
    context = AgentContext(agent_id=file_agent_id)
    
    # List current directory
    print("\nListing current directory:")
    result = await execute_agent_command(file_agent_id, "ls .", context)
    print(json.dumps(result, indent=2))
    
    # Create a test file
    print("\nCreating a test file:")
    result = await execute_agent_command(
        file_agent_id, 
        "write test.txt This is a test file created by the dynamic file agent.", 
        context
    )
    print(json.dumps(result, indent=2))
    
    # Read the file
    print("\nReading the file:")
    result = await execute_agent_command(file_agent_id, "read test.txt", context)
    print(json.dumps(result, indent=2))

async def run_data_analysis_example():
    """Example of using a data analysis agent"""
    print("\n=== Data Analysis Example ===")
    
    # Create a data analysis agent
    data_agent_id = "data_example"
    registry.create_agent(data_agent_id, "data_analysis")
    
    # Create a context for the agent
    context = AgentContext(agent_id=data_agent_id)
    
    # Create a sample CSV file using the file agent
    file_agent_id = "file_helper"
    registry.create_agent(file_agent_id, "file")
    file_context = AgentContext(agent_id=file_agent_id)
    
    # Create a directory for sample data
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a sample CSV file
    csv_content = """date,temperature,humidity,pressure
2023-01-01,32,45,1012
2023-01-02,28,50,1010
2023-01-03,30,48,1011
2023-01-04,33,42,1013
2023-01-05,35,40,1012
"""
    
    await execute_agent_command(
        file_agent_id,
        f"write {sample_dir}/weather.csv {csv_content}",
        file_context
    )
    
    # Analyze the CSV file
    print("\nAnalyzing CSV file:")
    result = await execute_agent_command(
        data_agent_id,
        f"analyze_csv {sample_dir}/weather.csv",
        context
    )
    print(json.dumps(result, indent=2))
    
    # Create a visualization
    print("\nCreating visualization:")
    options = {
        "column": "temperature",
        "title": "Temperature Distribution"
    }
    
    result = await execute_agent_command(
        data_agent_id,
        f"visualize weather histogram {json.dumps(options)}",
        context
    )
    print(json.dumps(result, indent=2))

async def main():
    """Run all examples"""
    await run_file_operations_example()
    await run_data_analysis_example()
    
    print("\nExamples completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
