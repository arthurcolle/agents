#!/usr/bin/env python3
"""
Run Multi-Agent System - Launches all components of the multi-agent system
with FastAPI servers and Redis PubSub communication.
"""

import os
import sys
import asyncio
import logging
import argparse
import time
import json
import signal
import subprocess
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi-agent-system")

def check_redis():
    """Check if Redis is running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        logger.info("Redis is running")
        return True
    except Exception as e:
        logger.error(f"Redis check failed: {e}")
        return False

def check_dependencies():
    """Check for required dependencies"""
    try:
        import fastapi
        import uvicorn
        import redis
        import aiohttp
        import pydantic
        import rich
        logger.info("All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def run_system(config_file='agent_config.json', skip_checks=False, redis_url=None):
    """Run the entire multi-agent system"""
    # Check dependencies and Redis
    if not skip_checks:
        if not check_dependencies():
            logger.error("Missing dependencies. Please run: pip install -r requirements.txt")
            return 1
        
        if not check_redis():
            logger.error("Redis is not running. Please start Redis before running the system.")
            return 1
    
    # Set Redis URL environment variable if provided
    if redis_url:
        os.environ["REDIS_URL"] = redis_url
    
    # Start the agent system using the existing script
    try:
        cmd = [
            sys.executable,
            "start_agent_system.py",
            "--config", config_file
        ]
        
        logger.info(f"Starting agent system with command: {' '.join(cmd)}")
        
        # Run the process and wait for it to complete
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Handle output in real-time
        def handle_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                print(f"{prefix}: {line.strip()}")
        
        # Start threads to handle output
        import threading
        stdout_thread = threading.Thread(target=handle_output, args=(process.stdout, "SYSTEM"), daemon=True)
        stderr_thread = threading.Thread(target=handle_output, args=(process.stderr, "ERROR"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete or for user interrupt
        try:
            process.wait()
            return process.returncode
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return 0
            
    except Exception as e:
        logger.error(f"Error running agent system: {e}")
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Multi-Agent System")
    parser.add_argument("--config", default="agent_config.json", help="Path to config file (JSON)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and Redis checks")
    parser.add_argument("--redis-url", help="Redis URL (e.g., redis://localhost:6379/0)")
    
    args = parser.parse_args()
    
    return run_system(args.config, args.skip_checks, args.redis_url)

if __name__ == "__main__":
    sys.exit(main())