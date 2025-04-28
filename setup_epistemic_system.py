#!/usr/bin/env python3
"""
Setup script for the Epistemic Knowledge Management System

This script will:
1. Check and install required dependencies
2. Create necessary directories
3. Initialize the knowledge database
4. Run basic tests to verify functionality
5. Optionally generate sample visualizations
"""

import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("epistemic_setup.log")
    ]
)
logger = logging.getLogger("epistemic-setup")

# Required dependencies
CORE_DEPENDENCIES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "sqlite3",
]

# Visualization dependencies (optional)
VIZ_DEPENDENCIES = [
    "matplotlib",
    "networkx",
    "dash",
    "dash-cytoscape",
    "plotly"
]


def check_dependencies(dependencies, optional=False):
    """Check if dependencies are installed"""
    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep.replace("-", "_"))
            logger.info(f"✓ {dep} is installed")
        except ImportError:
            if dep == "sqlite3":
                # sqlite3 is part of the standard library, but the import might fail
                # if Python wasn't compiled with SQLite support
                try:
                    import sqlite3
                    logger.info(f"✓ {dep} is installed (built-in)")
                    continue
                except ImportError:
                    pass
            
            logger.warning(f"✗ {dep} is NOT installed")
            missing.append(dep)
    
    if missing:
        if optional:
            logger.warning(f"Missing optional dependencies: {', '.join(missing)}")
        else:
            logger.error(f"Missing required dependencies: {', '.join(missing)}")
        
        return missing
    else:
        if optional:
            logger.info("All optional dependencies are installed ✓")
        else:
            logger.info("All required dependencies are installed ✓")
        return []


def install_dependencies(dependencies):
    """Install missing dependencies using pip"""
    if not dependencies:
        return True
    
    logger.info(f"Installing dependencies: {', '.join(dependencies)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *dependencies
        ])
        logger.info("Dependencies installed successfully ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories for the knowledge system"""
    directories = [
        "./knowledge",
        "./knowledge/long_context_sessions",
        "./knowledge/visualizations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True


def initialize_database():
    """Initialize the knowledge database"""
    try:
        from epistemic_tools import initialize_knowledge_system, shutdown_knowledge_system
        
        db_path = "./knowledge/epistemic.db"
        initialize_knowledge_system(db_path)
        logger.info(f"Knowledge database initialized at: {db_path}")
        
        shutdown_knowledge_system()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def run_tests():
    """Run basic tests to verify functionality"""
    try:
        logger.info("Running basic tests...")
        import test_epistemic_system
        
        result = test_epistemic_system.run_tests()
        
        if result:
            logger.info("All tests passed ✓")
        else:
            logger.warning("Some tests failed. Check test_epistemic_system.py for details.")
        
        return result
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


def generate_sample_visualizations():
    """Generate sample visualizations"""
    try:
        # Check if visualization dependencies are installed
        missing = check_dependencies(VIZ_DEPENDENCIES, optional=True)
        if missing:
            logger.warning("Cannot generate visualizations due to missing dependencies")
            return False
        
        logger.info("Generating sample visualizations...")
        
        from epistemic_visualization import generate_visualizations
        generate_visualizations()
        
        logger.info("Sample visualizations generated ✓")
        return True
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return False


def check_components():
    """Check if all required components are present"""
    required_files = [
        "epistemic_core.py",
        "epistemic_tools.py",
        "epistemic_long_context.py",
        "epistemic_visualization.py",
        "epistemic_complex_demo.py",
        "test_epistemic_system.py",
        "EPISTEMIC_README.md"
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            logger.warning(f"Missing component: {file}")
            missing.append(file)
    
    if missing:
        logger.error(f"Missing components: {', '.join(missing)}")
        return False
    else:
        logger.info("All required components are present ✓")
        return True


def setup_epistemic_system(install_deps=True, run_test=True, generate_viz=True):
    """Set up the Epistemic Knowledge Management System"""
    logger.info("Starting setup of Epistemic Knowledge Management System...")
    
    # Check if all components are present
    if not check_components():
        logger.error("Setup failed: Missing components")
        return False
    
    # Check core dependencies
    missing_core = check_dependencies(CORE_DEPENDENCIES)
    
    # Install missing core dependencies if needed
    if missing_core and install_deps:
        if not install_dependencies(missing_core):
            logger.error("Setup failed: Could not install core dependencies")
            return False
    elif missing_core:
        logger.error("Setup failed: Missing core dependencies")
        return False
    
    # Create directories
    if not create_directories():
        logger.error("Setup failed: Could not create directories")
        return False
    
    # Initialize database
    if not initialize_database():
        logger.error("Setup failed: Could not initialize database")
        return False
    
    # Run tests if requested
    if run_test:
        if not run_tests():
            logger.warning("Setup completed with warnings: Some tests failed")
    
    # Generate visualizations if requested
    if generate_viz:
        # Check visualization dependencies
        missing_viz = check_dependencies(VIZ_DEPENDENCIES, optional=True)
        
        # Install missing visualization dependencies if needed
        if missing_viz and install_deps:
            install_dependencies(missing_viz)
            # Generate visualizations only if installation was successful
            if not check_dependencies(VIZ_DEPENDENCIES, optional=True):
                generate_sample_visualizations()
        elif not missing_viz:
            generate_sample_visualizations()
    
    logger.info("Epistemic Knowledge Management System setup completed ✓")
    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Setup for Epistemic Knowledge Management System")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--no-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--no-viz", action="store_true", help="Skip generating visualizations")
    
    args = parser.parse_args()
    
    # Run setup with specified options
    success = setup_epistemic_system(
        install_deps=not args.no_deps,
        run_test=not args.no_tests,
        generate_viz=not args.no_viz
    )
    
    if success:
        print("\n===============================================")
        print("✅ Epistemic Knowledge Management System is ready!")
        print("===============================================")
        print("\nTo get started:")
        print("1. Read EPISTEMIC_README.md for documentation")
        print("2. Run python epistemic_complex_demo.py for a comprehensive demo")
        print("3. Run python epistemic_visualization.py --interactive for the web interface")
        sys.exit(0)
    else:
        print("\n===============================================")
        print("❌ Setup encountered errors. Check epistemic_setup.log for details.")
        print("===============================================")
        sys.exit(1)