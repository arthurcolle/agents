#!/usr/bin/env python3
"""
Start Agent System - Launches the agent process manager and multiple agents
with FastAPI servers and Redis PubSub communication.
"""

import os
import sys
import asyncio
import logging
import argparse
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Union
import subprocess
import signal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-system")

def start_manager(host="0.0.0.0", port=8500):
    """Start the agent process manager"""
    try:
        cmd = [
            sys.executable, 
            "agent_process_manager.py",
            "--mode", "manager",
            "--host", host,
            "--port", str(port)
        ]
        
        logger.info(f"Starting agent process manager: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Start background threads to handle stdout/stderr
        def handle_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                print(f"{prefix}: {line.strip()}")
        
        import threading
        threading.Thread(target=handle_output, args=(process.stdout, "MANAGER"), daemon=True).start()
        threading.Thread(target=handle_output, args=(process.stderr, "MANAGER ERROR"), daemon=True).start()
        
        return process
    except Exception as e:
        logger.error(f"Error starting agent process manager: {e}")
        return None

def start_agent(agent_type, agent_id=None, agent_name=None, host="127.0.0.1", 
               port=None, model="gpt-4", env_vars=None):
    """Start an agent process"""
    try:
        # Create unique ID if not provided
        if not agent_id:
            agent_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        
        # Create default name if not provided
        if not agent_name:
            agent_name = f"{agent_type.capitalize()} Agent {agent_id[-4:]}"
        
        # Default port based on agent type if not provided
        if not port:
            base_port = {
                "self_improving": 8600,
                "cli": 8700,
                "web": 8800,
                "openai": 8900
            }.get(agent_type, 9000)
            # Find available port
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            port = base_port
            while True:
                try:
                    s.bind((host, port))
                    break
                except OSError:
                    port += 1
            s.close()
        
        # Prepare command based on agent type
        if agent_type == "self_improving":
            cmd = [
                sys.executable, 
                "self_improving_agent.py",
                "--agent-id", agent_id,
                "--agent-name", agent_name,
                "--host", host,
                "--port", str(port),
                "--model", model
            ]
        elif agent_type == "collective":
            cmd = [
                sys.executable, 
                "agent_collective.py",
                "--agent-id", agent_id,
                "--agent-name", agent_name,
                "--host", host,
                "--port", str(port),
                "--model", model
            ]
        elif agent_type == "pubsub_service":
            cmd = [
                sys.executable,
                "pubsub_service.py"
                # Environment variables handle the rest of the configuration
            ]
        else:
            # Generic agent launching through agent process manager
            cmd = [
                sys.executable, 
                "agent_process_manager.py",
                "--mode", "agent",
                "--agent-id", agent_id,
                "--agent-name", agent_name,
                "--agent-type", agent_type,
                "--host", host,
                "--port", str(port),
                "--model", model
            ]
        
        # Prepare environment with additional vars
        env = os.environ.copy()
        if env_vars:
            for key, value in env_vars.items():
                env[key] = value
        
        logger.info(f"Starting {agent_type} agent: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Start background threads to handle stdout/stderr
        def handle_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                print(f"{prefix}: {line.strip()}")
        
        import threading
        threading.Thread(target=handle_output, args=(process.stdout, f"AGENT {agent_id}"), daemon=True).start()
        threading.Thread(target=handle_output, args=(process.stderr, f"AGENT {agent_id} ERROR"), daemon=True).start()
        
        return {
            "process": process,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_type": agent_type,
            "port": port
        }
    except Exception as e:
        logger.error(f"Error starting {agent_type} agent: {e}")
        return None

def start_system(config_file=None):
    """Start the entire agent system from a config file or defaults"""
    try:
        # Load config if provided
        if config_file:
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "manager": {
                    "host": "0.0.0.0",
                    "port": 8500
                },
                "agents": [
                    {
                        "agent_type": "self_improving",
                        "agent_name": "Self-Improving Agent",
                        "port": 8600,
                        "model": "gpt-4"
                    },
                    {
                        "agent_type": "cli",
                        "agent_name": "CLI Agent",
                        "port": 8700,
                        "model": "gpt-4"
                    }
                ]
            }
        
        # Start manager
        manager_config = config.get("manager", {})
        manager_process = start_manager(
            host=manager_config.get("host", "0.0.0.0"),
            port=manager_config.get("port", 8500)
        )
        
        # Wait for manager to start
        time.sleep(2)
        
        # Start agents
        agent_processes = []
        for agent_config in config.get("agents", []):
            agent_process = start_agent(
                agent_type=agent_config.get("agent_type"),
                agent_id=agent_config.get("agent_id"),
                agent_name=agent_config.get("agent_name"),
                host=agent_config.get("host", "127.0.0.1"),
                port=agent_config.get("port"),
                model=agent_config.get("model", "gpt-4"),
                env_vars=agent_config.get("env_vars")
            )
            
            if agent_process:
                agent_processes.append(agent_process)
            
            # Wait a bit between agent starts
            time.sleep(1)
        
        # Setup signal handling for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down agent system...")
            
            # Stop all agents
            for agent in agent_processes:
                process = agent["process"]
                logger.info(f"Stopping agent {agent['agent_id']}...")
                process.terminate()
            
            # Stop manager
            if manager_process:
                logger.info("Stopping manager...")
                manager_process.terminate()
            
            # Give processes time to shutdown
            time.sleep(2)
            
            # Force kill any remaining processes
            for agent in agent_processes:
                process = agent["process"]
                if process.poll() is None:
                    process.kill()
            
            if manager_process and manager_process.poll() is None:
                manager_process.kill()
            
            logger.info("Agent system shutdown complete")
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Print system info
        logger.info("Agent system started!")
        logger.info(f"Manager running on port {manager_config.get('port', 8500)}")
        for agent in agent_processes:
            logger.info(f"{agent['agent_name']} ({agent['agent_id']}) running on port {agent['port']}")
        
        # Keep running until interrupted
        while True:
            # Check if any processes have died
            alive = True
            if manager_process and manager_process.poll() is not None:
                logger.error(f"Manager process exited with code {manager_process.returncode}")
                alive = False
            
            for agent in agent_processes:
                process = agent["process"]
                if process.poll() is not None:
                    logger.error(f"Agent {agent['agent_id']} process exited with code {process.returncode}")
                    alive = False
            
            if not alive and not config.get("keep_running_on_failure", False):
                logger.error("One or more processes have died, shutting down...")
                signal_handler(None, None)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        # Let the signal handler handle this
    except Exception as e:
        logger.error(f"Error starting agent system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Agent System")
    parser.add_argument("--config", help="Path to config file (JSON)")
    
    args = parser.parse_args()
    
    sys.exit(start_system(args.config))