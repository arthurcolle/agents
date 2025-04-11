#!/usr/bin/env python3
"""
Dynamic Environment Example

Demonstrates the dynamic code reloading system with a more complex example
involving class inheritance, state persistence, and advanced use cases.
"""

import time
import logging
from dynamic_env import ModuleBuffer, CodeRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_env_example")

# Initialize the environment
buffer = ModuleBuffer()
registry = CodeRegistry(buffer.registry)

# Create a base module with a parent class
base_module_code = """
class BaseProcessor:
    def __init__(self, name):
        self.name = name
        self.processed_count = 0
        print(f"Created BaseProcessor: {name}")
        
    def process(self, data):
        self.processed_count += 1
        return f"Base processed: {data}"
        
    def get_stats(self):
        return {
            "name": self.name,
            "processed_count": self.processed_count,
            "processor_type": "base"
        }
"""

# Create a module that depends on the base module
processor_module_code = """
from __dynamic_modules__.base_module import BaseProcessor

class AdvancedProcessor(BaseProcessor):
    def __init__(self, name, prefix="ADV"):
        super().__init__(name)
        self.prefix = prefix
        
    def process(self, data):
        self.processed_count += 1
        return f"{self.prefix}: {data}"
        
    def get_stats(self):
        stats = super().get_stats()
        stats["processor_type"] = "advanced"
        stats["prefix"] = self.prefix
        return stats
        
def create_processor(name, prefix="ADV"):
    return AdvancedProcessor(name, prefix)
"""

# Create a user module that uses both modules
user_module_code = """
from __dynamic_modules__.base_module import BaseProcessor
from __dynamic_modules__.processor_module import create_processor, AdvancedProcessor

class ProcessorManager:
    def __init__(self):
        self.processors = {}
        
    def add_processor(self, name, processor_type="base", prefix="ADV"):
        if processor_type == "base":
            processor = BaseProcessor(name)
        else:
            processor = create_processor(name, prefix)
            
        self.processors[name] = processor
        return processor
        
    def process_data(self, processor_name, data):
        if processor_name not in self.processors:
            raise ValueError(f"Processor not found: {processor_name}")
            
        return self.processors[processor_name].process(data)
        
    def get_all_stats(self):
        return {name: processor.get_stats() for name, processor in self.processors.items()}
"""

def run_demo():
    print("\n=== Dynamic Environment Advanced Example ===\n")
    
    # Create the modules
    print("Creating modules...")
    base_mod = buffer.create_module("base_module", base_module_code)
    proc_mod = buffer.create_module("processor_module", processor_module_code)
    user_mod = buffer.create_module("user_module", user_module_code)
    
    # Use the modules
    print("\nUsing the modules...")
    
    # Create a processor manager
    manager = user_mod.ProcessorManager()
    
    # Add processors
    print("\nCreating processors:")
    base_proc = manager.add_processor("base-1", "base")
    adv_proc = manager.add_processor("advanced-1", "advanced", "SUPER")
    
    # Process some data
    print("\nProcessing data...")
    print(f"Base result: {manager.process_data('base-1', 'hello')}")
    print(f"Advanced result: {manager.process_data('advanced-1', 'world')}")
    
    # Get stats
    print("\nProcessor stats:")
    for name, stats in manager.get_all_stats().items():
        print(f"- {name}: {stats}")
    
    # Update the processor module with new capabilities
    print("\nUpdating processor module...")
    new_processor_code = """
from __dynamic_modules__.base_module import BaseProcessor

class AdvancedProcessor(BaseProcessor):
    def __init__(self, name, prefix="ADV"):
        super().__init__(name)
        self.prefix = prefix
        self.extra_data = []
        
    def process(self, data):
        self.processed_count += 1
        self.extra_data.append(data)
        return f"{self.prefix}: {data} [{self.processed_count}]"
        
    def get_stats(self):
        stats = super().get_stats()
        stats["processor_type"] = "advanced"
        stats["prefix"] = self.prefix
        stats["data_points"] = len(self.extra_data)
        return stats
        
    def get_history(self):
        return self.extra_data
        
def create_processor(name, prefix="ADV"):
    return AdvancedProcessor(name, prefix)
"""
    
    buffer.update_module("processor_module", new_processor_code)
    
    # Process more data using the updated module
    print("\nProcessing data with updated module...")
    print(f"Base result: {manager.process_data('base-1', 'testing base')}")
    print(f"Advanced result: {manager.process_data('advanced-1', 'testing advanced')}")
    
    # Check that the state was maintained
    print("\nProcessor stats after update:")
    for name, stats in manager.get_all_stats().items():
        print(f"- {name}: {stats}")
    
    # Try the new capabilities
    print("\nUsing new capabilities:")
    if hasattr(adv_proc, 'get_history'):
        print(f"Advanced processor history: {adv_proc.get_history()}")
    else:
        print("Method 'get_history' not found - reference updating may have failed")
    
    # List all modules
    print("\nAll modules in the environment:")
    for info in buffer.list_modules():
        print(f"- {info['name']} (v{info.get('version', '?')})")

if __name__ == "__main__":
    run_demo()